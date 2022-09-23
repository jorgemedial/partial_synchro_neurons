import pickle
import json
import os
import datetime

import brian2 as b2
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import scipy as sp

from scipy.signal import find_peaks
from scipy.ndimage import gaussian_filter1d
from scipy.integrate import solve_ivp, quad
from sklearn.metrics import r2_score
from scipy.stats import shapiro
from statsmodels.graphics.gofplots import qqplot


class NeuralSimulator:
    """
    This class simulates the neural network and stores the output
    """

    def __init__(self, filename="config.json"):
        self.__config_filename = filename
        self.__dirname = os.path.dirname(__file__)
        self.__configurate()

    def __configurate(self):
        """
        Reads config.json file
        """
        self.__sim_results_path = os.path.join(self.__dirname, "simulation_results/", "results.P")
        self.__config_path = os.path.join(self.__dirname, "config/", self.__config_filename)
        self.__config = json.load(open(self.__config_path, "r"))
        self.__load_constants()
        self.__load_recording_configuration()

    def __load_constants(self):
        """
        Computes neural constants (i.e. potentials, conductivities) from the data in config.json
        """
        self.__sim_constants = self.__config.get("simulation_constants")
        self._N = self.__sim_constants.get("N_neurons")
        self._v_L = self.__sim_constants.get("v_L")
        self._v_I = self.__sim_constants.get("v_I")
        self._v_thr = self.__sim_constants.get("v_thr")
        self._v_res = self.__sim_constants.get("v_res")
        self._duration = self.__sim_constants.get("duration")
        self._sigma = self.__sim_constants.get("sigma")
        self._I_dc = self.__sim_constants.get("I_dc")
        self._g_L = self.__sim_constants.get("g_L")
        self._tau = self.__sim_constants.get("tau")
        self._tau_D = self.__sim_constants.get("tau_D")
        self._tau_R = self.__sim_constants.get("tau_R")
        self._w = self.__sim_constants.get("weights")
        self._correction = self.__sim_constants.get("correction")
        self._initial_g = self.__sim_constants.get("initial_g")

        # The initial voltage of the neurons is uniformly distributed between the reset potential and the threshold
        self._initial_v = np.random.uniform(low=self._v_res, high=self._v_thr, size=self._N)

        # These attributes are filled in self.simulate()
        self._interspike_times = None
        self._spikes = None
        self._t = None
        self._g_i = None
        self._v = None
        self._neuron_model = None
        self._global_rate = None

    def __load_recording_configuration(self):
        """

        :return:
        """
        self.__recording_config = self.__config.get("recording_config")
        self.__recorded_variables = self.__recording_config.get("recorded_variables")
        # Set recorded neurons
        self.__N_recorded = self.__recording_config.get("N_recorded")
        if self.__N_recorded > self._N:
            self.__N_recorded = self._N
        rng = np.random.default_rng()
        self.__recorded_neurons_id = rng.choice(np.arange(self._N), size=self.__N_recorded)

    def simulate(self):
        """
        Simulates the system and records output
        """

        b2.start_scope()
        N = self._N
        v_L = self._v_L * b2.volt
        v_I = self._v_I * b2.volt
        v_thr = self._v_thr * b2.volt
        v_res = self._v_res * b2.volt
        duration = self._duration * b2.second
        sigma = self._sigma * b2.volt
        I_dc = self._I_dc * b2.volt * b2.hertz
        g_L = self._g_L * b2.hertz
        tau = self._tau * b2.second
        tau_D = self._tau_D * b2.second
        tau_R = self._tau_R * b2.second
        w = self._w
        self.__neuron_model_eqs = '''
            dv/dt = g_L*(v_L-v) + I_dc + w*g*(v_I - v) + sigma*xi*tau**(-0.5) : volt
            dg/dt = (R-g)/tau_D : hertz
            dR/dt = -R/tau_R : hertz
        '''

        # Neuron group
        G = b2.NeuronGroup(
            N=self._N,
            model=self.__neuron_model_eqs,
            threshold='v > v_thr',
            reset='v = v_res',
            method='heun',
        )
        G.v = self._initial_v * b2.volt
        G.g = self._initial_g * b2.hertz

        # Synapses
        S = b2.Synapses(G, G, model='''delta : hertz # synaptic weight''', on_pre='R+=delta')
        S.connect(condition='True')
        S.delta = 1 * b2.hertz

        # Preparing monitors and network
        spike_monitor = b2.SpikeMonitor(G)
        monitor = b2.StateMonitor(G, variables=["v", "g", "R"], record=True)
        rate_monitor = b2.PopulationRateMonitor(G)
        net = b2.Network(b2.collect())  # automatically include G and S
        net.add(monitor)  # manually add the monitors

        # Simulation
        print("Running simulation")
        start_time = datetime.datetime.now()
        net.run(self._duration * b2.second)
        elapsed_time = datetime.datetime.now() - start_time
        print(f"Elapsed time: {elapsed_time.total_seconds()}s")

        self._v = np.array(monitor.v)
        self._g_i = np.array(monitor.g)
        self._R_i = np.array(monitor.R)
        self.g_t = np.mean(self.g_i, axis=0)
        self.R_t = np.mean(self._R_i, axis=0)
        self._mean_v = np.mean(self.v, axis=0)
        self._std_v = np.std(self.v, axis=0)
        self._t = pd.Series(np.array(monitor.t))

        total_length = len(self.g_t)
        self._mu_sim_g_sim = np.empty(total_length)
        self._mu_sim_g_exp = np.empty(total_length)
        self._mu_sim_spiked = np.empty(total_length)
        self._var_sim_g_sim = np.empty(total_length)
        self._var_sim_g_exp = np.empty(total_length)
        self._var_spiked = np.empty(total_length)

        self._mu_f_sim = np.empty(total_length)
        self._var_f_sim = np.empty(total_length)
        self._g_f_sim = np.empty(total_length)
        self._R_f_sim = np.empty(total_length)
        self._f_sim = np.empty(total_length)

        self._mu_f_cycle = np.empty(total_length)
        self._var_f_cycle = np.empty(total_length)
        self._g_f_cycle = np.empty(total_length)
        self._R_f_cycle = np.empty(total_length)
        self._f_cycle = np.empty(total_length)

        self._mu_R_exp = np.empty(total_length)
        self._var_R_exp = np.empty(total_length)
        self._g_R_exp = np.empty(total_length)

        self._a_lag = np.empty(total_length)
        self._b_lag = np.empty(total_length)
        self._mu_inf = np.empty(total_length)
        self._var_inf = np.empty(total_length)

        spikes = spike_monitor.values('t')
        spikes = pd.DataFrame.from_dict(spikes, orient='index').stack()
        self._spikes = spikes.map(lambda time: time * b2.hertz).to_frame(name="time")
        self._spikes["cycle"] = 0
        self._interspike_times = self._spikes["time"].groupby(level=0).diff()
        self._global_rate = self.get_global_rate(rate_monitor)
        self._global_spikes = self.get_global_spikes()
        self._cycles = self._global_spikes.index[2:-2]
        self._compute_non_spiking_moments()

    def save_simulation(self):
        with open(self.__sim_results_path, "w") as file:
            pickle.dump(self, file)

    def get_global_rate(self, rate_monitor):
        # Store monitor data
        global_rate = pd.DataFrame(
            {
                "rate": rate_monitor.rate / b2.hertz,
                "time": rate_monitor.t / b2.second
            },
        )
        # Smooth spike rate to find the maxima
        global_rate["rate_smooth"] = gaussian_filter1d(
            input=global_rate["rate"],
            sigma=20,
        )

        return global_rate

    def get_global_spikes(self):
        """

        :return:
        """
        peak_indices = find_peaks(self._global_rate["rate_smooth"])[0]
        global_spikes = pd.DataFrame({"peak_time": self._global_rate["time"].iloc[peak_indices]})

        global_spikes = pd.concat(
            [
                pd.DataFrame({global_spikes.columns[0]: [0]}),
                global_spikes,
                pd.DataFrame({global_spikes.columns[0]: [self.duration]}),
            ],
            ignore_index=True,
        )

        global_spikes["next_cycle"] = global_spikes["peak_time"].rolling(2).mean().shift(-1)
        global_spikes["next_cycle"].iat[-1] = global_spikes["peak_time"].iat[-1]
        global_spikes["start_cycle"] = global_spikes["next_cycle"].shift(1)

        for cycle in global_spikes.index[1:-1]:
            cycle_index = self._spikes["time"].astype(float).between(
                global_spikes["next_cycle"].iat[cycle - 1],
                global_spikes["next_cycle"].iat[cycle],
            )

            global_spikes.loc[cycle, ["%_spiking_neurons"]] = 100 * cycle_index.sum() / self.N
            self._spikes.loc[cycle_index, "cycle"] = int(cycle)
            # global_spikes.loc[cycle, ["spiking_neurons"]] = self.spikes.loc[cycle_index].index.get_level_values(0))

            # Experimental onset of spiking. First spike
            start_time = self._spikes["time"].loc[cycle_index].min()
            global_spikes.loc[cycle, ['start_spikes']] = start_time

            # Experimental end of spiking. Last spike
            global_spikes.loc[cycle, ['end_spikes']] = self._spikes["time"].loc[cycle_index].max() + self.t[1]

            # Get mean and std of the start time
            try:
                start_time_id = self.time_id(start_time)
                global_spikes.loc[cycle, ['mu']] = self.mean[start_time_id - 1]
                global_spikes.loc[cycle, ['var']] = self.std[start_time_id - 1] ** 2
            except:
                global_spikes.loc[cycle, ['mu']] = np.nan
                global_spikes.loc[cycle, ['var']] = np.nan

            # Theoretical onset of spiking
            self.theo_onset(global_spikes, cycle)
            self.theo_onset_inf(global_spikes, cycle)

        # Redefine next_cycle at the next start_spikes
        global_spikes.loc[:, "next_cycle"] = global_spikes["start_spikes"].shift(-1)

        # Reorder dataframe columns
        global_spikes = global_spikes[
            [
                "start_cycle",
                "start_spikes",
                "pred_start_spikes",
                "pred_start_spikes_inf",
                "peak_time",
                "end_spikes",
                "next_cycle",
                "%_spiking_neurons",
                "mu",
                "var",
                "pred_p_th",
                "pred_p_th_inf",
            ]
        ]
        return global_spikes

    def theo_onset(self, global_spikes, cycle):
        onset_time_id = self.time_id(global_spikes["start_cycle"].iat[cycle])
        time_id_end = self.time_id(global_spikes["peak_time"].iat[cycle])
        found = False
        p_th = 0
        while (onset_time_id < time_id_end) & (not found):
            mu = self._mean_v[onset_time_id]
            std = self._std_v[onset_time_id]
            Z_th = (mu - self.v_thr) / (np.sqrt(2) * std)
            p_th = self.N * (1 + sp.special.erf(Z_th))/2
            if p_th > 1:
                found = True
            else:
                onset_time_id += 1

        onset_time = self.t[onset_time_id-1]
        global_spikes.at[cycle, "pred_start_spikes"] = onset_time
        global_spikes.at[cycle, "pred_p_th"] = p_th

    def theo_onset_inf(self, global_spikes, cycle):
        onset_time_id = self.time_id(global_spikes["start_cycle"].iat[cycle])
        time_id_end = self.time_id(global_spikes["peak_time"].iat[cycle])
        found = False
        p_th = 0
        R_0 = self.R_t[onset_time_id]
        g_0 = self.g_t[onset_time_id]
        t_0 = self.t[onset_time_id]
        while (onset_time_id < time_id_end) & (not found):
            t = self.t[onset_time_id]
            g = self.g_compute(t , t_0, R_0, g_0)
            a = self.a(g)
            g_lagged = self.g_compute(t - 1/a, t_0, R_0, g_0)
            a_lag = self.a(g_lagged)
            b_lag = self.b(g_lagged)
            mu = b_lag/a_lag
            std = np.sqrt(self._sigma**2/np.sqrt(2*a_lag))
            Z_th = (mu - self.v_thr) / std
            p_th = self.N * (1 + sp.special.erf(Z_th/np.sqrt(2)))/2
            if p_th > 1:
                found = True
            else:
                onset_time_id += 1

        onset_time = self.t[onset_time_id]
        global_spikes.at[cycle, "pred_start_spikes_inf"] = onset_time + 1/self.a(self.g_compute(t-1/self.a(g), t_0, R_0, g_0))
        global_spikes.at[cycle, "pred_p_th_inf"] = p_th

    def time_id(self, time):  # very inefficient
        """
        Gets the index of the element in "self.time" that has the same value as "time".
        If such element does not exist, the immediate smaller is returned.
        :param time:
        :return:
        """
        if hasattr(time, '__iter__'):
            id_t0 = self.time_id(time[0])
            id_tf = self.time_id(time[-1])
            time_ids = np.arange(id_t0, id_tf+1)
            return time_ids

        else:
            time = float(time)
            time_index = 0
            while self.t.iat[time_index] < time:
                time_index += 1
            return time_index

    def simulate_reduced_model(self):
        for cycle in self._cycles:
            self.simulate_cycle(cycle)
        self.simulate_spiking_phase()
        self.simulate_full_dynamics()

    def g_compute(self, t, t_0, R_0, g_0):
        dt = t - t_0
        try:
            tau_S = self._tau_R*self._tau_D/(self._tau_R - self._tau_D)
            g = np.exp(-dt/self._tau_D)*(g_0 + R_0*tau_S*(np.exp(dt/tau_S)-1))/self._tau_D
        except ZeroDivisionError:
            g = np.exp(-dt/self._tau_D)*(g_0 + R_0*dt/self._tau_D)

        return g

    def a(self, g):
        return self.g_L + g*self._w

    def b(self, g):
        return self.g_L * self._v_L + self._I_dc + self._v_I * g*self._w

    def simulate_silent_phase(self, ids_t_silent):
        t = self.t[ids_t_silent].to_numpy()
        id_t_0 = ids_t_silent[0]
        t_0 = self.t[ids_t_silent[0]]

        g = self.g_compute(t, t_0, self.R_t[id_t_0], self.g_t[id_t_0])
        a = self.a(g)
        time_ids = np.vectorize(self.time_id)
        lag_t_id = time_ids(t-1/a)
        g_lag = self.g_t[lag_t_id]
        a_lag = self.a(g_lag)
        b_lag = self.b(g_lag)
        R = self.R_t[id_t_0]*np.exp(-(t-t_0)/self._tau_R)

        self._g_f_sim[ids_t_silent] = g
        self._R_f_sim[ids_t_silent] = R
        self._f_sim[ids_t_silent] = 0
        self._a_lag[ids_t_silent] = a_lag
        self._b_lag[ids_t_silent] = b_lag
        self._mu_inf[ids_t_silent] = b_lag/a_lag
        self._var_inf[ids_t_silent] = self._sigma ** 2 / (2 * a)

        def d_silent_dt_g_sim(t, y):
            mu = y[0]
            var = y[1]
            g_t = self.g_compute(t, t_0, R_0=self.R_t[id_t_0], g_0=self.g_t[id_t_0])
            a_t = self.a(g_t)
            b_t = self.b(g_t)
            return [
                b_t - a_t*mu,
                -2*a_t*var + self._sigma**2
            ]

        def d_silent_dt_g_exp(t, y):
            mu = y[0]
            var = y[1]
            g_t = self.g_t[self.time_id(t)]
            a_t = self.a(g_t)
            b_t = self.b(g_t)
            return [
                b_t - a_t*mu,
                -2*a_t*var + self._sigma**2
            ]

        solver_g_sim = solve_ivp(
            fun=d_silent_dt_g_sim,
            t_span=(t[0], t[-1]),
            y0=(
                self._mean_silent[id_t_0],
                self._std_silent[id_t_0] ** 2,
            ),
            method='RK23',
            t_eval=t,
        )

        solver_g_exp = solve_ivp(
            fun=d_silent_dt_g_exp,
            t_span=(t[0], t[-1]),
            y0=(
                self._mean_silent[id_t_0],
                self._std_silent[id_t_0] ** 2,
            ),
            method='RK23',
            t_eval=t,
        )

        solver_spiked = solve_ivp(
            fun=d_silent_dt_g_sim,
            t_span=(t[0], t[-1]),
            y0=(
                self._mean_spike[id_t_0],
                self._std_spike[id_t_0] ** 2,
            ),
            method='RK23',
            t_eval=t,
        )
        self._mu_sim_g_sim[ids_t_silent] = solver_g_sim.y[0]
        self._var_sim_g_sim[ids_t_silent] = solver_g_sim.y[1]
        self._mu_sim_g_exp[ids_t_silent] = solver_g_exp.y[0]
        self._var_sim_g_exp[ids_t_silent] = solver_g_exp.y[1]
        self._mu_sim_spiked[ids_t_silent] = solver_spiked.y[0]
        self._var_spiked[ids_t_silent] = solver_spiked.y[1]

    def simulate_full_dynamics(self):
        t_start_spikes = self._global_spikes.loc[self._cycles[0], 'start_spikes']
        t_end_cycle = self._global_spikes.loc[self._cycles[-1], 'next_cycle']

        self.ids_t_full = self.time_id([t_start_spikes, t_end_cycle])
        self.t_full = self.t[self.ids_t_full].to_numpy()
        d_spiking_dt_f_cycle = self.d_spiking_dt_generator(mode="full_sim")
        solver_f_cycle = solve_ivp(
            fun=d_spiking_dt_f_cycle,
            t_span=(self.t_full[0], self.t_full[-1]),
            y0=(
                self._mean_silent[self.ids_t_full[0]],
                self._std_silent[self.ids_t_full[0]] ** 2,
                self.g_t[self.ids_t_full[0]],
                self.R_t[self.ids_t_full[0]],
            ),
            method='RK23',
            # Use the same time steps as the brian simulation
            t_eval=self.t_full
        )
        self._mu_f_cycle[self.ids_t_full] = solver_f_cycle.y[0]
        self._var_f_cycle[self.ids_t_full] = solver_f_cycle.y[1]
        self._g_f_cycle[self.ids_t_full] = solver_f_cycle.y[2]
        self._R_f_cycle[self.ids_t_full] = solver_f_cycle.y[3]
        a = self.a(solver_f_cycle.y[2])
        b = self.b(solver_f_cycle.y[2])
        self._f_cycle[self.ids_t_full] = self.get_activity_time_range(b, a, solver_f_cycle.y[0],
                                                                      np.sqrt(solver_f_cycle.y[1]))

    def simulate_spiking_phase(self):
        t_start_spikes = self._global_spikes.loc[self._cycles[0], 'start_spikes']
        t_end_cycle = self._global_spikes.loc[self._cycles[-1], 'next_cycle']
        ids_t_spiking = self.time_id([t_start_spikes, t_end_cycle])
        id_t_start_spikes = ids_t_spiking[0]
        t_spiking = list(self.t[ids_t_spiking])
        d_spiking_dt_R_exp = self.d_spiking_dt_generator(mode="R_exp")
        d_spiking_dt_f_sim = self.d_spiking_dt_generator(mode="f_sim")

        # solver_f_cycle = solve_ivp(
        #     fun=d_spiking_dt_f_cycle,
        #     t_span=(t_spiking[0], t_spiking[-1]),
        #     y0=(
        #         self._mean_silent[id_t_start_spikes],
        #         self._std_silent[id_t_start_spikes] ** 2,
        #         self.g_t[id_t_start_spikes],
        #         self.R_t[id_t_start_spikes],
        #     ),
        #     method='RK23',
        #     # Use the same time steps as the brian simulation
        #     t_eval=t_spiking
        # )
        #
        # self._mu_f_cycle[ids_t_spiking] = solver_f_cycle.y[0]
        # self._var_f_cycle[ids_t_spiking] = solver_f_cycle.y[1]
        # self._g_f_cycle[ids_t_spiking] = solver_f_cycle.y[2]
        # self._R_f_cycle[ids_t_spiking] = solver_f_cycle.y[3]
        # a = self.a(solver_f_cycle.y[2])
        # b = self.b(solver_f_cycle.y[2])
        # self._f_cycle[ids_t_spiking] = self.get_activity_time_range(b, a, solver_f_cycle.y[0], np.sqrt(solver_f_cycle.y[1]))

        solver_f_sim = solve_ivp(
            fun=d_spiking_dt_f_sim,
            t_span=(t_spiking[0], t_spiking[-1]),
            y0=(
                self._mean_silent[id_t_start_spikes],
                self._std_silent[id_t_start_spikes] ** 2,
                self.g_t[id_t_start_spikes],
                self.R_t[id_t_start_spikes],
            ),
            method='RK23',
            # Use the same time steps as the brian simulation
            t_eval=t_spiking
        )

        self._mu_f_sim[ids_t_spiking] = solver_f_sim.y[0]
        self._var_f_sim[ids_t_spiking] = solver_f_sim.y[1]
        self._g_f_sim[ids_t_spiking] = solver_f_sim.y[2]
        self._R_f_sim[ids_t_spiking] = solver_f_sim.y[3]
        g_exp = self.g_t[ids_t_spiking]
        b_exp = self.b(g_exp)
        a_exp = self.a(g_exp)
        mu_exp = self._mean_silent[ids_t_spiking]
        std_exp = self._std_silent[ids_t_spiking]
        self._f_sim[ids_t_spiking] = self.get_activity_time_range(b_exp, a_exp, mu_exp, std_exp)

        solver_R_exp = solve_ivp(
            fun=d_spiking_dt_R_exp,
            t_span=(t_spiking[0], t_spiking[-1]),
            y0=(
                self._mean_silent[id_t_start_spikes],
                self._std_silent[id_t_start_spikes] ** 2,
                self.g_t[id_t_start_spikes],
            ),
            method='RK23',
            # Use the same time steps as the brian simulation
            t_eval=t_spiking
        )

        self._mu_R_exp[ids_t_spiking] = solver_R_exp.y[0]
        self._var_R_exp[ids_t_spiking] = solver_R_exp.y[1]
        self._g_R_exp[ids_t_spiking] = solver_R_exp.y[2]


    def simulate_cycle(self, cycle):
        t_start_spikes = self._global_spikes.loc[cycle, 'start_spikes']
        t_end_spikes = self._global_spikes.loc[cycle, 'end_spikes']
        t_end_cycle = self._global_spikes.loc[cycle, 'next_cycle']

        id_t_start_spikes = self.time_id(t_start_spikes)
        id_t_end_spikes = self.time_id(t_end_spikes)
        id_t_end_cycle = self.time_id(t_end_cycle)

        id_t_spikes = np.arange(id_t_start_spikes, id_t_end_cycle)
        id_t_silent = np.arange(id_t_end_spikes, id_t_end_cycle)

        self.simulate_silent_phase(id_t_silent)



    def d_spiking_dt_generator(self, mode):
        """
        Generates a function d_spiking_dt that returns the derivative of mu in the spiking stage
        """
        if mode == "f_sim":
            def d_spiking_dt(t, y):
                mu = y[0]
                var = y[1]
                g = y[2]
                R = y[3]
                a = self._g_L + g * self._w
                b = self._g_L * self._v_L + self._I_dc + g * self._v_I * self._w

                id_t = self.time_id(t)
                g_exp = self.g_t[id_t]
                b_exp = self.b(g_exp)
                a_exp = self.a(g_exp)
                mu_exp = self._mean_silent[id_t]
                std_exp = self._std_silent[id_t]
                f = self.N * self.get_activity(b_exp, a_exp, mu_exp, std_exp)
                return [
                    -a * mu + b,  # d_mu/dt
                    2 * var * (-a) + self._sigma ** 2,  # d_var/dt
                    (R - g) / self._tau_D,  # dg/dt
                    f - R / self._tau_R,  # dR/dt
                ]

        elif mode == "R_exp":
            def d_spiking_dt(t, y):
                mu = y[0]
                var = y[1]
                g = y[2]
                R = self.R_t[self.time_id(t)]
                a = self._g_L + g * self._w
                b = self._g_L * self._v_L + self._I_dc + g * self._v_I * self._w
                return [
                    -a * mu + b,  # d_mu/dt
                    2*var*(-a) + self._sigma**2,  # d_var/dt
                    (R - g)/self._tau_D,  # dg/dt
                ]
        elif mode == "g_exp":
            def d_spiking_dt(t, y):
                mu = y[0]
                var = y[1]
                g = self.g_t[self.time_id(t)]
                a = self._g_L + g * self._w
                b = self._g_L * self._v_L + self._I_dc + g * self._v_I * self._w
                return [
                    -a * mu + b,  # d_mu/dt
                    2 * var * (-a) + self._sigma ** 2,  # d_var/dt
                ]
        elif mode == "full_sim":
            def d_spiking_dt(t, y):
                mu = y[0]
                var = y[1]
                g = y[2]
                R = y[3]
                a = self._g_L + g * self._w
                b = self._g_L * self._v_L + self._I_dc + g * self._v_I * self._w
                f = self.N * self.get_activity(b, a, mu, var**0.5)

                return [
                    -a * mu + b,  # d_mu/dt
                    2 * var * (-a) + self._sigma ** 2,  # d_var/dt
                    (R - g) / self._tau_D,  # dg/dt
                    f - R / self._tau_R,  # dR/dt
                ]
        else:
            raise Exception("No valid mode")

        return d_spiking_dt


    def get_activity_time_range(self, b_t, a_t, mu_t, std_t):
        get_activity = np.vectorize(self.get_activity)
        return get_activity(b_t, a_t, mu_t, std_t)

    def get_activity(self, b, a, mu, std):
        """
        Gets the activity f(t) depending on mu and std
        """
        if mu > self.v_thr - 0.0025:
            density_thr = np.exp(-0.5*((self._v_thr-mu)/std)**2)/(std*np.sqrt(2*np.pi))
            act = density_thr * self.mean_dot_v(a, b, mu) #/ density_thr
        else:
            act = 0
        return act

    def mean_dot_v(self, a, b, mu):
        def p_dot_V_dot_V(dot_V):
            return dot_V * np.exp(-0.5 * ((dot_V-b+a*self._v_thr) / (self._sigma * self._correction)) ** 2) / (np.sqrt(2 * np.pi) * self._sigma * self._correction)

        return quad(p_dot_V_dot_V, 0, np.inf)[0]

    def _compute_non_spiking_moments(self):
        """
        Computes the mean and std in time of the non spiking population at each cycle
        :return:
        """
        self._mean_silent = self._mean_v.copy()  # hacer en función del tiempo
        self._std_silent = self._std_v.copy()
        self._std_spike = self._std_v.copy()
        self._mean_spike = self._mean_v.copy()
        self._spike_neurons = {cycle: [] for cycle in self._cycles}
        self._silent_neurons = {cycle: [] for cycle in self._cycles}
        for cycle in self._cycles:
            # Select non spiking neurons of the cycle
            spike_neurons = self._spikes[self._spikes["cycle"] == cycle].index.get_level_values(0)
            all_neurons = set(range(self.N))
            silent_neurons = list(all_neurons - set(spike_neurons))

            # Select time indices of the cycle
            t0_spikes_id = self.time_id(self._global_spikes.at[cycle, "start_spikes"])
            t0_silent_id = self.time_id(self._global_spikes.at[cycle, "end_spikes"])
            t_end_id = self.time_id(self._global_spikes.at[cycle, "next_cycle"]) - 1

            # Compute mean and std
            self._mean_silent[t0_silent_id:t_end_id] = self._v[silent_neurons, t0_silent_id:t_end_id].mean(axis=0)
            self._std_silent[t0_silent_id:t_end_id] = self._v[silent_neurons, t0_silent_id:t_end_id].std(axis=0)

            self._mean_spike[t0_silent_id:t_end_id] = self._v[spike_neurons, t0_silent_id:t_end_id].mean(axis=0)
            self._std_spike[t0_silent_id:t_end_id] = self._v[spike_neurons,  t0_silent_id:t_end_id].std(axis=0)
            self._spike_neurons[cycle] = spike_neurons
            self._silent_neurons[cycle] = silent_neurons

            for id_t in np.arange(t0_spikes_id+1, t0_silent_id):
                silent_neurons = self._v[:, id_t] > np.mean([self._v_thr, self._v_res])
                spiked_neurons = self._v[:, id_t] <= np.mean([self._v_thr, self._v_res])
                self._mean_silent[id_t] = self._v[silent_neurons, id_t].mean(axis=0)
                self._std_silent[id_t] = self._v[silent_neurons, id_t].std(axis=0)
                self._mean_spike[id_t] = self._v[spiked_neurons, id_t].mean(axis=0)
                self._std_spike[id_t] = self._v[spiked_neurons, id_t].std(axis=0)




    @property
    def N(self) -> int:
        return self._N

    @property
    def duration(self):
        return self._duration

    @property
    def v_res(self):
        return self._v_res

    @property
    def v_thr(self):
        return self._v_thr

    @property
    def v(self):
        return self._v

    @property
    def g_i(self):
        return self._g_i

    @property
    def t(self):
        return self._t

    @property
    def spikes(self):
        return self._spikes

    @property
    def output_path(self):
        return self.__sim_results_path

    @property
    def initial_v(self):
        return self._initial_v

    @property
    def interspike_times(self):
        return self._interspike_times

    @property
    def g_L(self):
        return self._g_L


class NeuralAnalyzer(NeuralSimulator):
    def __init__(self, filename="config.json"):
        super().__init__(filename)
        super().simulate()
        super().simulate_reduced_model()
        self._init_NeuralAnalyzer()
    def _init_NeuralAnalyzer(self):
        self.__time_steps = self.v.shape[1]
        self._voltage_resolution = 500
        self.__v_density = self.compute_voltage_distribution()
        self.mean = np.mean(self.v, axis=0)
        self.std = np.std(self.v, axis=0)

    def compute_voltage_distribution(self):
        """
        Computes the voltage distribution over time and stores it
        :return:
        """
        v_density = np.empty((self._voltage_resolution, self.__time_steps))

        for t in range(self.__time_steps):
            v_density[:, t] = np.histogram(
                self.v[:, t],
                range=(self.v_res, self.v_thr),
                bins=self._voltage_resolution,
            )[0]
        return v_density

    def plot_activities(self, ax=None, fig=None, start_cycle=4, end_cycle=4):

        if ax is None:
            fig, ax = plt.subplots()

        start_time = self._global_spikes.at[start_cycle, "start_spikes"]
        end_time = self._global_spikes.at[end_cycle, "next_cycle"]

        id_start_time = self.time_id(start_time)
        id_end_time = self.time_id(end_time) - 1

        exp_activity = self._global_rate.loc[id_start_time:id_end_time, "rate"]
        smooth_activity = self._global_rate.loc[id_start_time:id_end_time, "rate_smooth"]

        cycle_time = np.linspace(start_time, end_time, len(exp_activity))
        ax.plot(cycle_time, exp_activity, color="navy", linewidth=0.8, label="Activity")
        ax.plot(cycle_time, smooth_activity, color="darkorange", linewidth=2,  label="Smoothed activity")

        ax.set_xlabel("Time (s)")
        ax.legend()
        ax.set_yscale("linear")
        return fig, ax

    def plot_density(self, cycle=None, t0:float=0, t_end:float=None, ax=None, fig=None):
        """

        :param ax:  Axes
                    The axes to draw to
        :param t0:  Start time
                    Time where the x-axis starts from
        :param t_end:   End time
                        Time where the x-axis ends
        :return:
        """
        sns.set()
        if ax is None:
            fig, ax = plt.subplots()

        if cycle is not None:
            t0 = self._global_spikes.at[cycle, "start_spikes"]
            t_end = self._global_spikes.at[cycle, "next_cycle"]
            t0_silent = self.get_global_spikes().at[cycle, "end_spikes"]
        if t_end is None:
            t_end = self.duration

        id_t0_silent = self.time_id(t0_silent)
        id_t_end = self.time_id(t_end)

        id_t_silent = np.arange(id_t0_silent, id_t_end)
        t_silent = self.t[id_t_silent]

        # Reduced Model
        color_sim = "salmon"
        color_exp = "tab:blue"
        mean_width = 1
        var_width = 0.5
        ax.plot(self.t, self._mean_silent, linewidth=mean_width, color=color_exp, label="Reduced model in the silent stage")
        ax.plot(t_silent, self._mu_sim_g_sim[id_t_silent], linewidth=mean_width, color=color_sim, label="Reduced model in the spiking stage")
        ax.axvspan(t0, t0_silent, alpha=0.2, color="lightcoral")

        for neuron in self._silent_neurons[cycle]:
            sns.scatterplot(
                x=self.t,
                y=self._v[neuron, :],
                color="navy",
                s=1,
                alpha=0.15,
                markers=False,
                ax=ax,
            )

        ax.set_xlim(t0, t_end)
        ax.set_ylim(-0.052, -0.04)

        ax.set_xlabel("Time (s)")
        ax.set_ylabel("Potential (V)")
        ax.legend(loc=8)
        return ax

    def plot_mean_silent(self, cycle, ax, zoom=False, g_exp=False, t0: float=0, t_end:float=None, fig=None):
        t0 = self._global_spikes.at[cycle, "start_spikes"]
        t_end_silent = self._global_spikes.at[cycle, "next_cycle"]
        t_end = t_end_silent + 0.001
        t0_silent = self.get_global_spikes().at[cycle, "end_spikes"]

        id_t0_silent = self.time_id(t0_silent)
        id_t_end_silent = self.time_id(t_end_silent)

        id_t_silent = np.arange(id_t0_silent, id_t_end_silent)
        t_silent = self.t[id_t_silent]

        # Reduced Model
        mean_width = 1.5
        mu_exp = self._mean_silent[id_t_silent]
        mu_sim_g_sim = self._mu_sim_g_sim[id_t_silent]
        mu_inf = self._mu_inf[id_t_silent]
        mu_sim_spiked = self._mu_sim_spiked[id_t_silent]
        mu_exp_spike = self._mean_spike[id_t_silent]

        r2_mu_sim_g_sim = r2_score(mu_exp, mu_sim_g_sim)
        r2_mu_spiked = r2_score(mu_exp_spike, mu_sim_spiked)

        rel_error_sim = np.abs((self._mean_v[id_t_silent[-1]] - self._mu_sim_spiked[id_t_silent[-1]])/self._mean_v[id_t_silent[-1]])
        rel_error_inf = np.abs((self._mean_v[id_t_silent[-1]] - self._mu_inf[id_t_silent[-1]])/self._mean_v[id_t_silent[-1]])

        if zoom:
            ax.set_xlim(t_end - self.t[40], t_end)
            ax.set_ylim(-0.045, -0.04)
        else:
            ax.set_xlim(t0, t_end)
            ax.set_ylim(-0.06, -0.04)

        ax.set_xlabel("Time (s)")
        ax.set_ylabel("Potential (V)")
        ax.axvspan(t0, t0_silent, alpha=0.2, color="lightcoral")
        ax.axvline(self.t[id_t_end_silent - 1], alpha=0.5, color="navy", linewidth=2, )
        ax.axvspan(self.t[id_t_end_silent - 1], t_end, alpha=0.2, color="lightcoral")

        base_color = "skyblue"
        color_dist_silent = "slateblue"
        color_dist_spikes = "orangered"
        size_scatter = {"silent": 0.2, "spikes": 0.2}

        if zoom:
            ax.plot(self.t, self._mean_v, linewidth=mean_width, color="salmon",
                    label="$\mu_{exp}$")
            ax.plot(t_silent, self._mu_sim_g_sim[id_t_silent], '--', linewidth=mean_width, color="tab:blue",
                    label="$\mu_{sim}(t)$" + f" (Rel. Error = {100*rel_error_sim: .4f}%)")
            ax.plot(t_silent, mu_inf, linewidth=mean_width, color="tab:green",
                    label="$\mu_{\infty}(t-\Delta t)$" + f" (Rel. Error = {100*rel_error_inf: .4-f}%)")

        else:
            ax.plot(self.t, self._mean_silent, linewidth=mean_width, color="salmon",
                    label="$\mu_{exp}$ (silent branch)")
            ax.plot(t_silent, self._mu_sim_g_sim[id_t_silent], '--', linewidth=mean_width, color="tab:blue",
                    label="$\mu_{sim}(t)$ $(r^2$" + f" score = {r2_mu_sim_g_sim: .4f})")
            ax.plot(t_silent, self._mean_spike[id_t_silent], linewidth=mean_width, color="goldenrod",
                    label="$\mu_{exp}$ (spiked branch)")
            ax.plot(t_silent, self._mu_sim_spiked[id_t_silent], '--', linewidth=mean_width, color="mediumaquamarine",
                    label="$\mu_{sim}(t)$ $(r^2$" + f" score = {r2_mu_spiked: .4f})")
            ax.plot(t_silent, mu_inf, linewidth=mean_width, color="tab:green",
                    label="$\mu_{\infty}(t-\Delta t)$")

        size_scatter["silent"] = 1
        size_scatter["spikes"] = 1.5

        for neuron in self._silent_neurons[cycle]:
            sns.scatterplot(
                x=self.t,
                y=self._v[neuron, :],
                color=color_dist_silent,
                s=25 if zoom else size_scatter["silent"],
                alpha=0.01 if zoom else 0.15,
                markers=False,
                ax=ax
            )
        for neuron in self._spike_neurons[cycle]:
            sns.scatterplot(
                x=t_silent,
                y=self._v[neuron, id_t_silent],
                color=color_dist_spikes,
                s=0 if zoom else size_scatter["spikes"],
                alpha=0.15,
                markers=False,
                ax=ax
            )

        loc = "lower right" if zoom else "best"
        ax.legend(loc=loc)
        return ax

    def plot_mean_spikes(self, cycle, g_exp=False, t0:float=0, t_end:float=None, ax=None, fig=None):
        """
        :param ax:  Axes
                    The axes to draw to
        :param t0:  Start time
                    Time where the x-axis starts from
        :param t_end:   End time
                        Time where the x-axis ends
        :return:
        """
        cycles = None
        if hasattr(cycle, "__iter__"):
            cycles = np.arange(cycle[0], cycle[-1]+1)
            for cycle in cycles:
                start_time = self._global_spikes.at[cycle, "start_spikes"]
                end_spikes = self._global_spikes.at[cycle, "end_spikes"] - self.t[1]
                ax.axvspan(start_time, end_spikes, color="lightcoral", alpha=0.2)
            start_time = self._global_spikes.at[cycles[0], "start_spikes"]
            end_spikes = self._global_spikes.at[cycles[-1], "end_spikes"] - self.t[1]
            id_t_spikes = self.time_id([start_time, end_spikes])
            id_t_total = self.time_id([start_time - self.t[10], end_spikes + self.t[20]])
        else:
            start_time = self._global_spikes.at[cycle, "start_spikes"]
            end_spikes = self._global_spikes.at[cycle, "end_spikes"] - self.t[1]
            id_t_spikes = self.time_id([start_time, end_spikes])
            id_t_total = self.time_id([start_time - self.t[10], end_spikes + self.t[20]])
            ax.axvspan(start_time, end_spikes, color="lightcoral", alpha=0.2)

        t_spikes = self.t[id_t_spikes]
        t_total = self.t[id_t_total] - self.t[1]

        # Reduced Model
        mean_width = 1.5
        mu_exp = self._mean_silent[id_t_spikes]
        mu_f_sim = self._mu_f_sim[id_t_spikes]

        r2 = self.get_r2(variable_name="mu", cycles=cycles)
        ax.plot(t_total, self._mean_silent[id_t_total], linewidth=mean_width, color="salmon",
                label="$\mu_{exp}(t)$")
        ax.plot(t_spikes, self._mu_f_cycle[id_t_spikes], '--',  linewidth=mean_width,
                color="tab:blue", label="$\mu_{sim}(t)$ ($r^2$"+ f" score = {r2: .3f})")

        for neuron in self._silent_neurons[cycle][:50]:
            sns.scatterplot(
                x=t_total,
                y=self._v[neuron, id_t_total],
                color="navy",
                alpha=0.15,
                s=2,
                markers=False,
                ax=ax
            )
        #
        ax.set_ylim(-0.06, -0.04)

        ax.set_xlabel("Time (s)")
        ax.set_ylabel("Potential (V)")
        ax.legend(loc="best")
        return ax

    def plot_mean_zoom(self, cycle=None, ax=None):
        return self.plot_mean_silent(cycle=cycle, zoom=True, ax=ax)

    def plot_std_silent(self, cycle, ax, zoom=False, g_exp=False, t0: float=0, t_end:float=None, fig=None):
        t0 = self._global_spikes.at[cycle, "start_spikes"]
        t_end_silent = self._global_spikes.at[cycle, "next_cycle"]
        t_end = t_end_silent + 0.001
        t0_silent = self.get_global_spikes().at[cycle, "end_spikes"]

        id_t0_silent = self.time_id(t0_silent)
        id_t_end_silent = self.time_id(t_end_silent)

        id_t_silent = np.arange(id_t0_silent, id_t_end_silent)
        t_silent = self.t[id_t_silent]

        # Reduced Model
        mean_width = 1

        std_sim_g_sim = self._var_sim_g_sim[id_t_silent] ** 0.5
        std_sim_spiked = self._var_spiked[id_t_silent] ** 0.5

        r2_std_sim_g_sim = r2_score(self._std_silent[id_t_silent], std_sim_g_sim)
        r2_std_spiked = r2_score(self._std_spike[id_t_silent], std_sim_spiked)

        rel_error_sim = np.abs((self._std_v[id_t_silent[-1]] - self._var_sim_g_sim[id_t_silent[-1]]**0.5)/self._std_v[id_t_silent[-1]])
        rel_error_inf = np.abs((self._std_v[id_t_silent[-1]] - self._var_inf[id_t_silent[-1]]**0.5)/self._std_v[id_t_silent[-1]])

        if zoom:
            ax.set_xlim(t_end - self.t[40], t_end)
            ax.set_ylim(top=0.0007, bottom=0.0006)
        else:
            ax.set_xlim(t0, t_end)
            ax.set_ylim(top=1.5 * max(self._std_silent[id_t_silent]), bottom=0.0004)

        ax.set_xlabel("Time (s)")
        ax.set_ylabel("Potential (V)")
        ax.axvspan(t0, t0_silent, alpha=0.2, color="lightcoral")
        ax.axvline(self.t[id_t_end_silent - 1], alpha=0.5, color="navy", linewidth=2, )
        ax.axvspan(self.t[id_t_end_silent - 1], t_end, alpha=0.2, color="lightcoral")

        base_color = "skyblue"
        color_dist_silent = "slateblue"
        color_dist_spikes = "orangered"
        size_scatter = {"silent": 0.2, "spikes": 0.2}

        if zoom:
            ax.plot(t_silent, self._std_v[id_t_silent], linewidth=mean_width, color="salmon",
                    label="$\epsilon_{exp}$")
            ax.plot(t_silent, self._var_sim_g_sim[id_t_silent] ** 0.5, '--', linewidth=mean_width, color="tab:blue",
                    label="$\epsilon_{sim}(t)$" + f"(Rel. Error = {100*rel_error_sim: .2f}%)")
            ax.plot(t_silent, self._var_inf[id_t_silent] ** 0.5, linewidth=mean_width, color="tab:green",
                    label="$\epsilon_{\infty}(t-\Delta t)$" + f"(Rel. Error = {100*rel_error_inf: .2f}%)")
        else:
            ax.plot(t_silent, self._std_silent[id_t_silent], linewidth=mean_width, color="salmon",
                    label="$\epsilon_{exp}(t)$ (silent branch)")
            ax.plot(t_silent, self._var_sim_g_sim[id_t_silent] ** 0.5, '--', linewidth=mean_width, color="tab:blue",
                    label="$\epsilon_{sim}(t)$ ($r^2$" + f" score = {r2_std_sim_g_sim: .3f})")
            ax.plot(t_silent[:-1], self._std_spike[id_t_silent[:-1]], linewidth=mean_width, color="goldenrod",
                    label="$\epsilon_{exp}(t)$ (spiked branch)")
            ax.plot(t_silent, self._var_spiked[id_t_silent] ** 0.5, '--', linewidth=mean_width, color="mediumaquamarine",
                    label="$\epsilon_{sim}(t)$ ($r^2$" + f" score = {r2_std_spiked: .3f})")
            ax.plot(t_silent, self._var_inf[id_t_silent] ** 0.5, linewidth=mean_width, color="tab:green",
                    label="$\epsilon_{\infty}(t - \Delta t)$")

        size_scatter["silent"] = 1
        size_scatter["spikes"] = 1.5

        loc = "lower right" if zoom else "best"
        ax.legend(loc=loc)
        return ax

    def plot_std_zoom(self, cycle=None, ax=None):
        return self.plot_std_silent(cycle=cycle, zoom=True, ax=ax)

    def plot_std_inf(self, cycle=None, zoom=False, t0:float=0, t_end_silent:float=None, ax=None, fig=None):
        """

        :param ax:  Axes
                    The axes to draw to
        :param t0:  Start time
                    Time where the x-axis starts from
        :param t_end_silent:   End time
                        Time where the x-axis ends
        :return:
        """
        sns.set()
        if ax is None:
            fig, ax = plt.subplots()

        if cycle is not None:
            t0 = self._global_spikes.at[cycle, "start_spikes"]
            t_end_silent = self._global_spikes.at[cycle, "next_cycle"]
            t_end = t_end_silent + 0.001
            t0_silent = self.get_global_spikes().at[cycle, "end_spikes"]
        if t_end_silent is None:
            t_end_silent = self.duration

        id_t0_silent = self.time_id(t0_silent)
        id_t_end_silent = self.time_id(t_end_silent)

        id_t_silent = np.arange(id_t0_silent, id_t_end_silent)
        t_silent = self.t[id_t_silent]

        mean_width = 1.5
        std_exp = self._std_silent[id_t_silent]
        std_inf = self._var_inf[id_t_silent] ** 0.5

        diff = np.abs((std_exp[-1] - std_inf[-1])/std_exp[-1])

        ax.plot(self.t, self._std_silent, linewidth=mean_width, color="tab:blue", label="$\epsilon_{exp}$ ")
        ax.plot(t_silent, std_inf, linewidth=mean_width, color="salmon", label="$\epsilon_{\infty}$")
        # ax.plot(t_silent, self._mu_sim_g_exp[id_t_silent], linewidth=mean_width, color="tab:green", label="$\mu_{sim}(g_{sim})$ (r^2"+ f" score = {r2_mu_sim_g_sim: .3f})")

        ax.axvspan(t0, t0_silent, alpha=0.2, color="lightcoral")
        ax.axvline(self.t[id_t_end_silent-1], alpha=0.5, color="navy", linewidth=2, label="$t_0^{spk}=$"+f"{t_end_silent: .3f}, Relative error" + f" {diff: .3e})")
        ax.axvspan(self.t[id_t_end_silent-1], t_end, alpha=0.2, color="lightcoral")

        if zoom:
            ax.set_xlim(t_end-0.01, t_end)
        else:
            ax.set_xlim(t0, t_end)
            ax.set_ylim(top=0.001, bottom=0.0005)

        ax.set_xlabel("Time (s)")
        ax.set_ylabel("Potential (V)")
        ax.legend(loc=9)
        return ax

    def plot_std_spikes(self, cycle=None, g_exp=False, t0: float = 0, t_end: float = None, ax=None, fig=None):
        """

        :param ax:  Axes
                    The axes to draw to
        :param t0:  Start time
                    Time where the x-axis starts from
        :param t_end:   End time
                        Time where the x-axis ends
        :return:
        """
        sns.set()
        if ax is None:
            fig, ax = plt.subplots()
        t0 = self._global_spikes.at[cycle, "start_spikes"]
        t_end_spikes = self.get_global_spikes().at[cycle, "end_spikes"] - self.t[1]
        t_end = t_end_spikes + 0.4 * (t_end_spikes - t0)

        id_t_spikes = self.time_id([t0, t_end_spikes])
        id_t_total = self.time_id([t0 - self.t[10], t_end])

        t_spikes = self.t[id_t_spikes]
        t_total = self.t[id_t_total]

        # Reduced Model
        color_sim = "salmon"
        color_exp = "tab:blue"
        mean_width = 1.5
        std_exp = self._std_silent[id_t_spikes]
        std_f_sim = self._var_f_sim[id_t_spikes] ** 0.5
        r2 = self.get_r2(variable_name="eps")

        ax.plot(t_total, self._std_silent[id_t_total], linewidth=mean_width, color="tab:blue",
                label="$\epsilon_{exp}$ (experimental)")
        ax.plot(t_spikes, std_f_sim, linewidth=mean_width, color="salmon",
                label="$\epsilon_{sim}(g_{model}$) ($r^2$" + f" score = {r2: .3f})")
        ax.axvspan(t0, t_end_spikes, alpha=0.2, color="lightcoral")

        ax.set_xlabel("Time (s)")
        ax.set_ylabel("Potential (V)")
        ax.set_ylim(bottom=0.00055, top=0.0007)
        ax.legend(loc=1)
        return ax

    def plot_g_R_silent(self, cycle, ax=None):
        sns.set()
        if ax is None:
            fig, ax = plt.subplots()

        if cycle is not None:
            t0 = self._global_spikes.at[cycle, "start_spikes"]
            t_end = self._global_spikes.at[cycle, "next_cycle"]
            t0_silent = self._global_spikes.at[cycle, "end_spikes"]
        if t_end is None:
            t_end = self.duration

        id_t0_silent = self.time_id(t0_silent)
        id_t_end = self.time_id(t_end)

        id_t_silent = np.arange(id_t0_silent, id_t_end)
        t_silent = self.t[id_t_silent]


        # Reduced Model
        color_sim = "salmon"
        color_exp = "tab:blue"
        mean_width = 1

        r2_g = r2_score(self.g_t[id_t_silent], self._g_f_sim[id_t_silent])

        r2_R = r2_score(self.R_t[id_t_silent], self._R_f_sim[id_t_silent])

        ax.plot(self.t, self.g_t, linewidth=mean_width, color=color_exp, label="$g_{exp}$ (experimental)")
        ax.plot(t_silent, self._g_f_sim[id_t_silent], '--', linewidth=mean_width, color=color_sim, label="$g_{sim}$ (model) ($r^2$ score =" + f"{r2_g: .5f})")

        ax.plot(self.t, self.R_t, linewidth=mean_width, color="tab:green", label="$R_{exp}$ (experimental)")
        ax.plot(t_silent, self._R_f_sim[id_t_silent], '--', linewidth=mean_width, color=color_sim, label="$R_{sim}$ (model) ($r^2$ score =" + f"{r2_R: .5f})")

        ax.axvspan(t0, t0_silent, alpha=0.2, color="lightcoral")
        ax.set_xlim(t0, t_end)
        ax.set_ylim(0, 1.05 * max(self._R_f_sim[id_t_silent]))

        ax.set_xlabel("Time (s)")
        ax.set_ylabel("Conductancy ($S/cm^2$)")
        ax.legend(loc=1)
        return ax

    def rel_error(self, variable_name, time_tag):
        time = self._global_spikes[time_tag]
        pred_variables = {
            "mu": self._mu_f_sim,
            "eps": self._var_f_sim**0.5,
            "g": self._g_f_sim,
            "R": self._R_f_sim,
            "f": self._f_sim,
        }
        exp_variables = {
            "mu": self._mean_silent,
            "eps": self._std_silent,
            "g": self.g_t,
            "R": self.R_t,
            "f": gaussian_filter1d(
                input=self._global_rate.loc[:, "rate"],
                sigma=4,
            )
        }
        pred_variable = pred_variables[variable_name]
        exp_variable = exp_variables[variable_name]
        rel_errors = np.array([])
        for cycle in self._cycles:
            time_id = self.time_id(time[cycle]) - 1
            pred = pred_variable[time_id]
            exp = exp_variable[time_id]
            rel_error = np.abs((pred-exp)/exp)
            rel_errors = np.append(rel_errors, rel_error)
        return rel_errors.mean()

    def get_r2(self, variable_name, phase=None, cycles=np.arange(2, 6)):
        predicted_global = np.array([])
        exp_global = np.array([])
        pred_variables = {
            "mu": self._mu_f_cycle,
            "eps": self._var_f_cycle**0.5,
            "g": self._g_f_cycle,
            "R": self._R_f_cycle,
            "f": self._f_cycle,
        }
        exp_variables = {
            "mu": self._mean_silent,
            "eps": self._std_silent,
            "g": self.g_t,
            "R": self.R_t,
            "f": gaussian_filter1d(
                input=self._global_rate.loc[:, "rate"],
                sigma=4,
            )
        }
        pred_variable = pred_variables[variable_name]
        exp_variable = exp_variables[variable_name]
        for cycle in cycles:
            if phase is None:
                start_time = self._global_spikes.at[cycle, "start_spikes"]
                end_spikes = self._global_spikes.at[cycle, "next_cycle"] - self.t[1]
            elif phase == "spiking":
                start_time = self._global_spikes.at[cycle, "start_spikes"]
                end_spikes = self._global_spikes.at[cycle, "end_spikes"] - self.t[1]
            elif phase == "silent":
                start_time = self._global_spikes.at[cycle, "end_spikes"]
                end_spikes = self._global_spikes.at[cycle, "next_cycle"] - self.t[1]
            id_t_phase = self.time_id([start_time, end_spikes])
            predicted_global = np.concatenate((predicted_global, pred_variable[id_t_phase]))
            exp_global = np.concatenate((exp_global, exp_variable[id_t_phase]))
        return r2_score(exp_global, predicted_global)

    def plot_g_R_spikes(self, cycle, ax=None, sim=True):
        if hasattr(cycle, "__iter__"):
            cycles = np.arange(cycle[0], cycle[-1]+1)
            for cycle in cycles:
                start_time = self._global_spikes.at[cycle, "start_spikes"]
                end_spikes = self._global_spikes.at[cycle, "end_spikes"] - self.t[1]
                id_t_spikes = self.time_id([start_time, end_spikes])
                id_t_total = self.time_id([start_time - self.t[10], end_spikes + self.t[20]])
                total_time = self.t[id_t_total]
                ax.axvspan(start_time, end_spikes, color="lightcoral", alpha=0.2)
            start_time = self._global_spikes.at[cycles[0], "start_spikes"]
            end_spikes = self._global_spikes.at[cycles[-1], "next_cycle"] - self.t[1]
            id_t_spikes = self.time_id([start_time, end_spikes])
            id_t_total = self.time_id([start_time - self.t[10], end_spikes + self.t[20]])
            total_time = self.t[id_t_total]
        else:
            start_time = self._global_spikes.at[cycle, "start_spikes"]
            end_spikes = self._global_spikes.at[cycle, "end_spikes"] - self.t[1]
            id_t_spikes = self.time_id([start_time, end_spikes])
            id_t_total = self.time_id([start_time - self.t[10], end_spikes + self.t[20]])
            total_time = self.t[id_t_total]
            ax.axvspan(start_time, end_spikes, color="lightcoral", alpha=0.2)

        # Reduced Model
        color_sim = "salmon"
        color_exp = "tab:blue"
        mean_width = 1
        # rel_error_g = self.rel_error(variable_name="g", time_tag="end_spikes")
        # rel_error_R = self.rel_error(variable_name="R", time_tag="end_spikes")
        ax.plot(total_time, self.g_t[id_t_total], linewidth=mean_width, color="goldenrod", label="$g_{exp}(t)$")
        ax.plot(total_time, self.R_t[id_t_total], linewidth=mean_width, color="salmon", label="$R_{exp}(t)$")

        if sim:
            r2_g = self.get_r2(variable_name="g")
            r2_R = self.get_r2(variable_name="R")
            ax.plot(total_time, self._g_f_cycle[id_t_total], '--', linewidth=mean_width, color="teal",
                label="$g_{sim}(t)$ (Avg. $r^2$ =" + f"{r2_g: .2f})")
        # ax.plot(spike_time, self._g_R_exp[id_t_spikes], linewidth=mean_width, color="tab:purple",
        #         label="$g_{sim}$ (f exp)")

            ax.plot(total_time, self._R_f_cycle[id_t_total], '--', linewidth=mean_width, color="tab:blue",
                    label="$R_{sim}(t)$ (Avg. $r^2$ =" + f"{r2_R: .2f})")
        ax.legend(loc="upper left")
        ax.set_xlabel("Time (s)")
        ax.set_ylabel("Conductivity ($\Omega^{-1}/cm^2$)")

    def plot_activity_pred(self, cycle, ax, sim=True):
        if hasattr(cycle, "__iter__"):
            cycles = np.arange(cycle[0], cycle[-1]+1)
            for cycle in cycles:
                start_time = self._global_spikes.at[cycle, "start_spikes"]
                end_spikes = self._global_spikes.at[cycle, "end_spikes"] - self.t[1]
                ax.axvspan(start_time, end_spikes, color="lightcoral", alpha=0.2)
            start_time = self._global_spikes.at[cycles[0], "start_spikes"]
            end_spikes = self._global_spikes.at[cycles[-1], "next_cycle"] - self.t[1]
            id_t_spikes = self.time_id([start_time, end_spikes])
            id_t_total = self.time_id([start_time, end_spikes])
            cycle_time = self.t[id_t_total]
            barsize = 0.0005
        else:
            start_time = self._global_spikes.at[cycle, "start_spikes"]
            end_spikes = self._global_spikes.at[cycle, "end_spikes"] - self.t[1]
            id_t_spikes = self.time_id([start_time, end_spikes])
            id_t_total = self.time_id([start_time, end_spikes])
            cycle_time = self.t[id_t_total]
            ax.axvspan(start_time, end_spikes, color="lightcoral", alpha=0.2)
            barsize = 0.0001
        exp_activity = self._global_rate.loc[id_t_total, "rate"]
        smooth_activity = gaussian_filter1d(
            input=self._global_rate.loc[id_t_spikes, "rate"],
            sigma=4,
        )
        if sim:
            r2 = self.get_r2(variable_name="f", cycles=[2])
        spike_time = self.t[id_t_spikes]
        if sim:
            sns.lineplot(x=spike_time, ax=ax, y=smooth_activity, color="salmon", linewidth=3, label="$f_{smooth}(t)$")
            ax.bar(cycle_time, exp_activity, color="lightsteelblue", width=barsize, label="$f_{exp}(t)$", alpha=1)
        else:
            ax.bar(cycle_time, exp_activity, color="navy", edgecolor=None, width=barsize, label="$f_{exp}(t)$", alpha=0.5)
        if sim:
            sns.lineplot(x=cycle_time, y=self._f_cycle[id_t_total], color="salmon", label="$f_{sim}(t)$ (global $r^2=$" + f"{r2: 0.3f})")
        ax.set_xlabel("Time (s)")
        ax.set_ylabel("Activity")
        ax.legend()

        ax.set_ylim(top=min(60, max(exp_activity)))
        return ax

    def plot_all_neurons(self, cycle: int = None, t0: float = 0, t_end: float = None, ax=None, fig=None):
        if ax is None:
            fig, ax = plt.subplots()

        if cycle is not None:
            t0 = self._global_spikes.at[cycle, "start_spikes"]
            t_end = self._global_spikes.at[cycle, "next_cycle"]

        if t_end is None:
            t_end = self.duration

        xgrid = np.linspace(0, self.duration, self.__time_steps)
        ygrid = np.linspace(self.v_res, self.v_thr, self._voltage_resolution)
        for neuron in range(20):
            sns.scatterplot(
                x=xgrid,
                y=self._v[neuron, :],
                color="navy",
                s=10,
                alpha=0.015,
                markers=False,
                ax=ax,
            )

        ax.set_xlim(t0, t_end)
        ax.set_ylim(self.v_res, self.v_thr+0.01)

        sns.lineplot(
            x=xgrid,
            y=self._mean_silent,
            ax=ax,
            label="Mean voltage of non-spiking neurons",
        )

        sns.lineplot(
            x=xgrid,
            y=self._mean_spike,
            ax=ax,
            label="Mean voltage of spiking neurons"
        )

        ax.set_xlabel("Time (s)")
        ax.set_ylabel("Potential (V)")
        ax.legend(loc="upper right")
        return ax

    def plot_neurons(self, t0: float = 0, cycle=None, t_end: float = None, ax=None, fig=None):
        if ax is None:
            fig, ax = plt.subplots()

        if cycle is not None:
            t0 = self._global_spikes.at[cycle, "start_spikes"]
            t_end = self._global_spikes.at[cycle, "next_cycle"]

        if t_end is None:
            t_end = self.duration

        xgrid = np.linspace(0, self.duration, self.__time_steps)

        ax.set_xlim(t0, t_end)
        ax.set_ylim(self.v_res, self.v_thr+0.01)

        for neuron in range(50):
            sns.lineplot(
                x=xgrid,
                y=self._v[neuron],
                ax=ax,
                color="navy",
            )

        ax.set_xlabel("Time (s)")
        ax.set_ylabel("Potential (V)")
        ax.legend(loc="upper right")
        return ax

    def plot_rasterplot(self, ax, cycles):
        start_time = self._global_spikes.at[cycles[0], "start_spikes"]
        end_spikes = self._global_spikes.at[cycles[-1]+1, "end_spikes"] - self.t[1]
        id_t_total = self.time_id([start_time - self.t[10], end_spikes + self.t[20]])

        N = 70
        ax.set_xlabel("Time (s)")
        ax.set_ylabel("Neuron Cell")
        xgrid = self.t[id_t_total]
        ygrid = np.array(range(30, N+30))
        X, Y = np.meshgrid(xgrid, ygrid)
        ax.pcolormesh(X, Y, -self.v[:N, id_t_total], shading="auto", cmap="coolwarm")
        return ax

    def plot_qq(self, cycle=4, ax=None):
        if ax is None:
            fig, ax = plt.subplots()
        time_spikes = self._global_spikes.at[cycle, "start_cycle"] - self.t[2]
        voltages = self._v[self._silent_neurons[cycle], self.time_id(time_spikes)]
        stat, pvalue = shapiro(voltages)
        qqplot(voltages, fit=True, ax=ax, line='s', label=f"Shapiro p-value = {pvalue: .3e}")
        ax.legend(loc="best")
        return ax

    def plot_voltage_dist_onset(self, ax=None, cycle=4):
        if hasattr(cycle, "__iter__"):
            cycle = cycle[0]
        if ax is None:
            fig, ax = plt.subplots()

        t_0_silent = self._global_spikes.at[cycle, "start_spikes"] - self.t[1]
        neurons = self._silent_neurons[cycle]
        voltages = self._v[neurons, self.time_id(t_0_silent)]
        mu = self._mean_silent[self.time_id(t_0_silent)]
        std = self._std_silent[self.time_id(t_0_silent)]
        exp_tag = " ($\mu = $" + f"{mu: .5f}, " + "$\epsilon = $" + f"{std: .3e})"


        bins = ax.hist(voltages, density=1, bins="auto", label="Potential distribution" + exp_tag)
        max_height = max(bins[0])

        x = np.linspace(bins[1][0] - 0.001, bins[1][-1] + 0.001, 100)
        gaussian_curve_emp = np.exp(-0.5 * ((x - mu) / std) ** 2) / (std * np.sqrt(2 * np.pi))

        def gaus(x, a, x0, sigma, offset):
            return a * np.exp(-(x - x0) ** 2 / (2 * sigma ** 2)) + offset

        x_hist = (bins[1][1:] + bins[1][:-1])/2
        y_hist = bins[0]
        popt, pcov = sp.optimize.curve_fit(gaus, x_hist, y_hist, p0=[1, mu, std, 0.0])
        mu_gauss = popt[1]
        std_gauss = popt[2]
        gauss_tag = " ($\mu = $" + f"{mu_gauss: .5f}, " + "$\epsilon = $" + f"{std_gauss: .3e})"

        sns.lineplot(x=x, y=gaussian_curve_emp, color="salmon", linewidth=1.5, label="Gaussian fit" + gauss_tag, alpha=1)

        ax.set_ylim(bottom=-5)
        ax.set_ylabel("Probability density")
        ax.set_xlabel("Potential (V)")

        return ax

    def plot_voltage_dist_spiking(self, ax=None, cycle=4):
        if ax is None:
            fig, ax = plt.subplots()

        time_spikes = self._global_spikes.at[cycle, "start_spikes"]
        voltages = self._v[:, self.time_id(time_spikes)]
        bins = ax.hist(voltages, density=1, bins=70, label="Potential distribution")
        max_height = max(bins[0])
        ax.vlines(
            x=self._v_thr,
            ymin=0,
            ymax=max_height,
            linewidth=2,
            color="orange",
            label="V threshold",
        )

        mu = self._mean_v[self.time_id(time_spikes)]
        std = self._std_v[self.time_id(time_spikes)]
        x = np.linspace(bins[1][0] - 0.001, bins[1][-1] + 0.001, 100)
        gaussian_curve_emp = np.exp(-0.5 * ((x - mu) / std) ** 2) / (std * np.sqrt(2 * np.pi))

        def gaus(x, a, x0, sigma, offset):
            return a * np.exp(-(x - x0) ** 2 / (2 * sigma ** 2)) + offset

        x_hist = (bins[1][1:] + bins[1][:-1])/2
        y_hist = bins[0]
        popt, pcov = sp.optimize.curve_fit(gaus, x_hist, y_hist, p0=[1, mu, std, 0.0])

        sns.lineplot(x=x, y=gaussian_curve_emp+11, color="darkorange", linewidth=2, label="Gaussian fit", alpha=1)
        ax.fill_between(
            x=x[x >= (self._v_thr-0.0001)],
            y1=gaussian_curve_emp[x >= (self._v_thr-0.0001)]+11,
            color="coral",
            alpha=0.5,
            label="Probability $V>V_{thr}$"
        )

        ax.set_ylim(bottom=-10)
        ax.set_ylabel("Probability density")
        ax.set_xlabel("Potential (V)")
        ax.legend()
        return ax

    def plot_attractor(self, ax, sim=True):


        V = self._mu_f_cycle[1000:-1000]
        g = self._g_f_cycle[1000:-1000]
        R = self._R_f_cycle[1000:-1000]
        color = "navy"
        mark="--"
        label="Attractor (Experimental)"
        ax.plot3D(V, g, R, alpha=0.5, color=color, label=label)

        V = self._mean_silent[1000:]
        g = self.g_t[1000:]
        R = self.R_t[1000:]
        color = "goldenrod"
        mark="-"
        label = "Attractor (simulated)"

        ax.plot3D(V, g, R, alpha=0.5, color=color, label=label)

        ax.set_xlabel("V")
        ax.set_ylabel("g")
        ax.set_zlabel("R")

        # ax.set_yticklabels([])
        # ax.set_xticklabels([])
        # ax.set_zticklabels([])
        ax.legend(loc="best")

        return ax

    def plot_sample_lag_dynamics(self, ax, lagged=True):
        t_eval = np.linspace(0, 30, 10000)
        def pred_speed(t):
            return 2 + 0.7*np.sin(t)

        def prey(t):
            return t*np.sin(0.5 * t) + 10*np.cos(t)

        def d_mu_dt(t, mu):
            return prey(t) - pred_speed(t) * mu[0]

        mu = solve_ivp(
            fun=d_mu_dt,
            t_span=(0, 150),
            y0=np.array([0]),
            method='RK23',
            t_eval=t_eval
        )

        y_prey = prey(t_eval)
        if lagged:
            lag = 1/pred_speed(t_eval)
            label = "$\mu_\infty(t-1/a(t))$"
        else:
            lag = 0
            label = "$\mu_\infty(t) = b(t)/a(t)$"
        ax.plot(t_eval-lag, mu.y[0], label="$\mu(t)$")
        ax.plot(t_eval, y_prey/pred_speed(t_eval), label=label)
        ax.legend(loc="lower left")
        if not lagged:
            ax.set_xlabel("t")
        ax.set_ylabel("$\mu$")
        return ax

    def plot_onset_times(self, ax, cycle):
        self._global_spikes["start_cycle"] = self._global_spikes["end_spikes"].shift(1)
        self._global_spikes["Experimental onset"] = self._global_spikes["start_spikes"] - \
                                                       self._global_spikes["start_cycle"]
        self._global_spikes["Predicted onset (Exact method)"] = self._global_spikes["pred_start_spikes"] - \
                                                    self._global_spikes["start_cycle"]
        self._global_spikes["Predicted onset (asymptotic method)"] = self._global_spikes["pred_start_spikes_inf"] - \
                                                    self._global_spikes["start_cycle"]

        exp_onset = self._global_spikes["Experimental onset"][1:-2]
        pred_onset_exact = self._global_spikes["Predicted onset (Exact method)"][1:-2]
        pred_onset_inf = self._global_spikes["Predicted onset (asymptotic method)"][1:-2]

        rel_error_exact = np.mean(np.abs((exp_onset - pred_onset_exact)/exp_onset))
        rel_error_inf = np.mean(np.abs((exp_onset - pred_onset_inf) / exp_onset))
        cycles = list(self._global_spikes.index[2: 9])
        self._global_spikes.loc[cycles, [
            "Experimental onset",
            "Predicted onset (Exact method)",
            "Predicted onset (asymptotic method)"]
        ].plot(
            kind='bar',
            ax=ax,
        )
        ax.legend([
                    "Experimental onset",
                    f"Predicted onset (Exact method | Rel. Error {100*rel_error_exact: .2f}%)",
                    f"Predicted onset (Asymptotic method | Rel. Error {100*rel_error_inf: .2f}%)"
        ])
        ax.set_xlabel("Cycle")
        ax.set_ylabel("Time (s)")
        ax.set_ylim(top=0.06)

    def plot_numeric_noise(self, ax, cycle):
        start_time = self._global_spikes.at[cycle, "start_spikes"]
        end_spikes = self._global_spikes.at[cycle, "end_spikes"] - self.t[1]
        id_t_spikes = self.time_id([start_time, end_spikes])
        t_spikes = self.t[id_t_spikes]

        for neuron in self._silent_neurons[cycle][:30]:
            ax.plot(t_spikes, self.v[neuron, id_t_spikes], color="powderblue")

        for neuron in self._spike_neurons[cycle][:7]:
            ax.plot(t_spikes, self.v[neuron, id_t_spikes], color="teal")
        ax.set_ylim(bottom=-0.045)
        return ax

    def plot_full_sim(self, ax, cycle):
        self.plot_mean_spikes(ax=ax[0], cycle=cycle)

        smooth_exp_mu = gaussian_filter1d(
                self._mean_silent[self.ids_t_full],
                sigma=3,
        )

        # ax[0].plot(self.t_full, self._mu_f_cycle[self.ids_t_full], "--")
        # ax[0].plot(self.t_full, smooth_exp_mu)
        durations = [
            0.1*pd.Series(find_peaks(self._mu_f_cycle[self.ids_t_full])[0]).diff()[1:],
            0.1*pd.Series(find_peaks(smooth_exp_mu)[0]).diff()[1:],
        ]
        ax[1].boxplot(
            durations,
            vert=False,
            patch_artist=True,
            labels=(
                f"Simulation \n mean = {np.mean(durations[0]): .2f}, std={np.std(durations[0]): .2f}",
                f"Experimental \n mean = {np.mean(durations[1]): .2f}, std={np.std(durations[1]): .2f}",

            )
        )
        ax[1].set_xlabel("Cycle period (ms)")
        ax[1].set_xlim(left=25, right=52)
        return ax

    def plot_one_neuron(self, ax, cycle):

        start_time = self._global_spikes.at[cycle, "start_spikes"] - self.t[150]
        end_cycle = self._global_spikes.at[cycle, "next_cycle"] - self.t[200]
        id_t_total = self.time_id([start_time, end_cycle])
        cell = self._spike_neurons[cycle][1]
        ax.plot(self.t[id_t_total], self.v[cell, id_t_total], label="Neuron potential")
        ax.axhline(self.v_thr, color="salmon", label="$V_{thr}=-0.004V$")
        ax.axhline(self.v_res, color="teal", label="$V_{res}=-0.007V$")
        ax.legend()
        ax.set_ylim(bottom=self.v_res*1.1, top=self.v_thr*0.9)
        ax.set_xlabel("Time (s)")
        ax.set_ylabel("Potential (V)")
        return ax

    def plot_two_neurons(self, ax, cycle):
        start_time = self._global_spikes.at[cycle[0], "start_spikes"]
        end_cycle = self._global_spikes.at[cycle[-1], "next_cycle"]
        id_t_total = self.time_id([start_time, end_cycle])
        cell1 = self._spike_neurons[cycle[-1]][1]
        cell2 = self._spike_neurons[cycle[-1]-1][1]
        # ax.plot(self.t[id_t_total], self.v[cell1, id_t_total], color="salmon", label="Neuron No. 1")
        ax.plot(self.t[id_t_total], self.v[cell2, id_t_total], color="tab:blue", label="Neuron Potential")
        ax.axhline(self.v_thr, color="salmon")
        ax.axhline(self.v_res, color="teal")
        ax.legend()
        ax.set_ylim(bottom=self.v_res*1.1, top=self.v_thr*0.9)
        ax.set_xlabel("Time (s)")
        ax.set_ylabel("Potential (V)")
        return ax

    def plot_mean_exp(self, ax, cycles, sim=True):
        cycles = np.arange(cycles[0], cycles[-1]+1)
        for cycle in cycles:
            start_time = self._global_spikes.at[cycle, "start_spikes"]
            end_spikes = self._global_spikes.at[cycle, "end_spikes"] - self.t[1]
            end_cycle = self._global_spikes.at[cycle, "next_cycle"]
            ax.axvspan(start_time, end_spikes, color="lightcoral", alpha=0.2)
            id_t_total = self.time_id([start_time, end_cycle])
            id_t_silent = self.time_id([end_spikes, end_cycle])
            plotted = False
            for neuron in self._silent_neurons[cycle]:
                sns.scatterplot(
                    x=self.t[id_t_total],
                    y=self._v[neuron, id_t_total],
                    color="navy",
                    s=2,
                    alpha=0.15,
                    markers=False,
                    ax=ax,
                )
                plotted = True
            plotted = False
            for neuron in self._spike_neurons[cycle]:
                sns.scatterplot(
                    x=self.t[id_t_silent],
                    y=self._v[neuron, id_t_silent],
                    color="salmon",
                    s=1,
                    alpha=0.15,
                    markers=False,
                    ax=ax,
                )

        start_time = self._global_spikes.at[cycles[0], "start_spikes"]
        end_spikes = self._global_spikes.at[cycles[-1], "next_cycle"] - self.t[1]
        id_t_total = self.time_id([start_time, end_spikes])
        if sim:
            ax.plot(self.t[id_t_total], self._mean_silent[id_t_total],"--", color="navy", label="Silent neurons")
            ax.plot(self.t[id_t_total], self._mean_spike[id_t_total], color="salmon", label="Spiking neurons")
        ax.set_ylim(bottom=-0.065)
        ax.set_ylabel("Potential (V)")
        ax.legend(loc="lower right")
        return ax
    def plot_raster_exp(self, ax, cycles):
        ax[0] = self.plot_mean_exp(ax=ax[0], cycles=cycles)
        ax[1] = self.plot_activity_pred(ax=ax[1], cycle=cycles, sim=False)
        ax[2] = self.plot_g_R_spikes(ax=ax[2], cycle=cycles, sim=False)
        ax[3] = self.plot_rasterplot(ax=ax[3], cycles=cycles)
        return ax
    def plot_raster_2(self, ax, cycles):
        ax[0] = self.plot_activity_pred(ax=ax[0], cycle=cycles, sim=False)
        ax[1] = self.plot_two_neurons(ax=ax[1], cycle=cycles)
        ax[2] = self.plot_rasterplot(ax=ax[2], cycles=cycles)
        return ax

    def plot_exp_dynamics(self, ax, cycles):
        ax[0] = self.plot_mean_exp(ax=ax[0], cycles=cycles)
        ax[1] = self.plot_activity_pred(ax=ax[1], cycle=cycles, sim=False)
        ax[2] = self.plot_g_R_spikes(ax=ax[2], cycle=cycles, sim=False)
        return ax


    def plot(self, figsize, cycle, plotter, path=None, nrows=1, ncols=1, gridspec_kw=None, dpi=600):

        print(f"Plotting {path.split('/')[-1]}")
        sns.set()
        fig, ax = plt.subplots(
            figsize=figsize
        )
        if nrows > 1:
            fig, ax = plt.subplots(
                ncols,
                nrows,
                figsize=figsize,
                gridspec_kw=gridspec_kw,
            )
            ax[0] = plotter(
                ax=ax[0],
                cycle=cycle[0]
            )
            ax[1] = plotter(
                ax=ax[1],
                cycle=cycle,
            )
        elif ncols > 1:
            fig, ax = plt.subplots(
                ncols,
                nrows,
                figsize=figsize,
                gridspec_kw=gridspec_kw,
                sharex=False,
            )
            ax = plotter(
                ax=ax,
                cycle=cycle,
            )
        else:
            ax = plotter(
                ax=ax,
                cycle=cycle
            )
        plt.tight_layout
        if path is not None:
            plt.savefig(path, dpi=dpi)
        else:
            plt.show()
        plt.close()

