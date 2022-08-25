import brian2 as b2
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import scipy as sp

import pickle
import json
import os
import datetime


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
        list_neuron_models = [
            "noisy_LIF",
        ]
        self._neuron_model = self.__config.get("neuron_model")
        if self._neuron_model not in list_neuron_models:
            raise Exception(f"Selected neuron model does not match any valid model. "
                            f"\nValid models: {list_neuron_models}")

        if self._neuron_model == "noisy_LIF":
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
        self._var_sim_g_sim = np.empty(total_length)
        self._var_sim_g_exp = np.empty(total_length)
        self._g_sim_R_sim = np.empty(total_length)
        self._g_sim_R_exp = np.empty(total_length)
        self._R_sim_f_sim = np.empty(total_length)
        self._R_sim_f_exp = np.empty(total_length)
        self._f_sim = np.empty(total_length)

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
        self._compute_non_spiking_moments()

    def save_simulation(self):
        with open(self.__sim_results_path, "w") as file:
            pickle.dump(self, file)

    def simulator_to_analyzer(self):
        """
        Changes class into NeuralAnalyzer and initializes it
        :return:
        """
        self.__class__ = NeuralAnalyzer
        self._init_NeuralAnalyzer()

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
        cycles = self._global_spikes.index[1:-2]
        for cycle in cycles:
            self.simulate_cycle(cycle)

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

        self._g_sim_R_sim[ids_t_silent] = g
        self._R_sim_f_sim[ids_t_silent] = R
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
                self._mean_non_spike[id_t_0],
                self._std_non_spike[id_t_0]**2,
            ),
            method='RK23',
            t_eval=t,
        )

        solver_g_exp = solve_ivp(
            fun=d_silent_dt_g_exp,
            t_span=(t[0], t[-1]),
            y0=(
                self._mean_non_spike[id_t_0],
                self._std_non_spike[id_t_0] ** 2,
            ),
            method='RK23',
            t_eval=t,
        )

        self._mu_sim_g_sim[ids_t_silent] = solver_g_sim.y[0]
        self._var_sim_g_sim[ids_t_silent] = solver_g_sim.y[1]
        self._mu_sim_g_exp[ids_t_silent] = solver_g_exp.y[0]
        self._var_sim_g_exp[ids_t_silent] = solver_g_exp.y[1]



    def simulate_spiking_phase(self, ids_t_spiking):
        id_t_start_spikes = ids_t_spiking[0]
        t_spiking = list(self.t[ids_t_spiking])
        d_spiking_dt_g_sim = self.d_spiking_dt_generator(g_exp=False)
        d_spiking_dt_g_exp = self.d_spiking_dt_generator(g_exp=True)

        y0_spiking_g_sim = [
            self._mean_non_spike[id_t_start_spikes],
            self._std_non_spike[id_t_start_spikes] ** 2,
            self.g_t[id_t_start_spikes],
            self.R_t[id_t_start_spikes],
        ]

        solver_g_sim = solve_ivp(
            fun=d_spiking_dt_g_sim,
            t_span=(t_spiking[0], t_spiking[-1]),
            y0=y0_spiking_g_sim,
            method='RK23',
            # Use the same time steps as the brian simulation
            t_eval=t_spiking
        )

        self._mu_sim_g_sim[ids_t_spiking] = solver_g_sim.y[0]
        self._var_sim_g_sim[ids_t_spiking] = solver_g_sim.y[1]
        self._g_sim_R_sim[ids_t_spiking] = solver_g_sim.y[2]
        self._R_sim_f_sim[ids_t_spiking] = solver_g_sim.y[3]
        a = self.a(solver_g_sim.y[2])
        b = self.b(solver_g_sim.y[2])
        self._f_sim[ids_t_spiking] = self.get_activity_time_range(b, a, solver_g_sim.y[0], np.sqrt(solver_g_sim.y[1]))

        y0_spiking_g_exp = [
            self._mean_non_spike[id_t_start_spikes],
            self._std_non_spike[id_t_start_spikes] ** 2,
        ]
        solver_g_exp = solve_ivp(
            fun=d_spiking_dt_g_exp,
            t_span=(t_spiking[0], t_spiking[-1]),
            y0=y0_spiking_g_exp,
            method='RK23',
            # Use the same time steps as the brian simulation
            t_eval=t_spiking
        )

        self._mu_sim_g_exp[ids_t_spiking] = solver_g_exp.y[0]
        self._var_sim_g_exp[ids_t_spiking] = solver_g_exp.y[1]


    def simulate_cycle(self, cycle):
        t_start_spikes = self._global_spikes.loc[cycle, 'start_spikes']
        t_end_spikes = self._global_spikes.loc[cycle, 'end_spikes']
        t_end_cycle = self._global_spikes.loc[cycle, 'next_cycle']

        id_t_start_spikes = self.time_id(t_start_spikes)
        id_t_end_spikes = self.time_id(t_end_spikes)
        id_t_end_cycle = self.time_id(t_end_cycle)

        id_t_spikes = np.arange(id_t_start_spikes, id_t_end_spikes)
        id_t_silent = np.arange(id_t_end_spikes, id_t_end_cycle)

        self.simulate_silent_phase(id_t_silent)
        self.simulate_spiking_phase(id_t_spikes)

    def d_spiking_dt_generator(self, g_exp):
        """
        Generates a function d_spiking_dt that returns the derivative of mu in the spiking stage
        """
        if not g_exp:
            def d_spiking_dt(t, y):
                mu = y[0]
                var = y[1]
                g = y[2]
                R = y[3]
                a = self._g_L + g * self._w
                b = self._g_L * self._v_L + self._I_dc + g * self._v_I * self._w

                f = self.N*self.get_activity(b, a, mu, np.sqrt(var))
                #f = self.N*self._global_rate.at[self.time_id(t), "rate"]
                return [
                    -a * mu + b,  # d_mu/dt
                    2*var*(-a) + self._sigma**2,  # d_var/dt
                    (R - g)/self._tau_D,  # dg/dt
                    f - R/self._tau_R,  # dR/dt
                ]
        else:
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
        return d_spiking_dt


    def get_activity_time_range(self, b_t, a_t, mu_t, std_t):
        return np.array([self.get_activity(b, a, mu, std) for (a, b, mu, std) in zip(a_t, b_t, mu_t, std_t)])

    def get_activity(self, b, a, mu, std):
        """
        Gets the activity f(t) depending on mu and std
        """
        density_thr = np.exp(-0.5*((self._v_thr-mu)/std)**2)/(std*np.sqrt(2*np.pi))
        return density_thr * self.mean_dot_v(a, b, mu) #/ density_thr

    def mean_dot_v(self, a, b, mu):
        def p_dot_V_dot_V(dot_V):
            correction = 75
            return dot_V*np.exp(-0.5*((dot_V-b+a*self._v_thr)/(self._sigma*correction))**2)/(np.sqrt(2*np.pi)*self._sigma*correction)

        return (quad(p_dot_V_dot_V, 0, np.inf)[0])

    def _compute_non_spiking_moments(self):
        """
        Computes the mean and std in time of the non spiking population at each cycle
        :return:
        """
        self._mean_non_spike = self._mean_v.copy()
        self._std_non_spike = self._std_v.copy()
        self._mean_spike = self._mean_v.copy()
        self._spike_neurons = {cycle: [] for cycle in self._global_spikes.index[1:-2]}
        self._silent_neurons = {cycle: [] for cycle in self._global_spikes.index[1:-2]}
        for cycle in self._global_spikes.index[1:-2]:
            # Select non spiking neurons of the cycle
            spike_neurons = self._spikes[self._spikes["cycle"] == cycle].index.get_level_values(0)
            all_neurons = set(range(self.N))
            non_spike_neurons = list(all_neurons - set(spike_neurons))

            # Select time indices of the cycle
            t0_id = self.time_id(self._global_spikes.at[cycle, "start_spikes"])
            t_end_id = self.time_id(self._global_spikes.at[cycle, "next_cycle"]) - 1

            # Compute mean and std
            self._mean_non_spike[t0_id:t_end_id] = self._v[non_spike_neurons, t0_id:t_end_id].mean(axis=0)
            self._std_non_spike[t0_id:t_end_id] = self._v[non_spike_neurons, t0_id:t_end_id].std(axis=0)

            self._mean_spike[t0_id:t_end_id] = self._v[spike_neurons[0], t0_id:t_end_id]

            self._spike_neurons[cycle] = spike_neurons
            self._silent_neurons[cycle] = non_spike_neurons


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
    def __init__(self, cycle=4, filename="config.json"):
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
        ax.plot(cycle_time, 5*smooth_activity, color="darkorange", linewidth=2,  label="Smoothed activity")

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

        xgrid = np.linspace(0, self.duration, self.__time_steps)
        ygrid = np.linspace(self.v_res, self.v_thr, self._voltage_resolution)
        # vmap = ax.pcolormesh(
        #     xgrid,
        #     ygrid,
        #     self.__v_density ** 0.8,
        #     shading="auto",
        #     cmap="BuPu",
        #     alpha=0.25,
        # )

        # Reduced Model
        color_sim = "salmon"
        color_exp = "tab:blue"
        mean_width = 1
        var_width = 0.5
        ax.plot(self.t, self._mean_non_spike, linewidth=mean_width, color=color_exp, label="Reduced model in the silent stage")
        ax.plot(t_silent, self._mu_sim_g_sim[id_t_silent], linewidth=mean_width, color=color_sim, label="Reduced model in the spiking stage")
        # ax.plot(self.t, self._std_non_spike + self._mean_non_spike, linewidth=var_width,  color=color_exp)
        # ax.plot(self.t, -self._var_sim ** 0.5 + self._mu_sim, linewidth=var_width, color=color_sim)
        # ax.plot(self.t, -self._std_non_spike + self._mean_non_spike, linewidth=var_width,  color=color_exp)
        # ax.plot(self.t, self._var_sim ** 0.5 + self._mu_sim, linewidth=var_width, color=color_sim)

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
        ax.set_ylabel("Voltage (V)")
        ax.legend(loc=8)
        return ax

    def plot_mean_silent(self, cycle=None, g_exp=False, t0:float=0, t_end:float=None, ax=None, fig=None):
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
        mean_width = 1.5
        mu_exp = self._mean_non_spike[id_t_silent]
        mu_sim_g_sim = self._mu_sim_g_sim[id_t_silent]
        mu_sim_g_exp = self._mu_sim_g_exp[id_t_silent]

        r2_mu_sim_g_exp = r2_score(mu_exp, mu_sim_g_exp)
        r2_mu_sim_g_sim = r2_score(mu_exp, mu_sim_g_sim)

        ax.plot(self.t, self._mean_non_spike, linewidth=mean_width, color="tab:blue", label="$\mu_{exp}$ (experimental)")
        ax.plot(t_silent, self._mu_sim_g_sim[id_t_silent], linewidth=mean_width, color="salmon", label="$\mu_{sim}(g_{model}) (r^2$"+ f" score = {r2_mu_sim_g_exp: .3f})")
        ax.plot(t_silent, self._mu_sim_g_exp[id_t_silent], linewidth=mean_width, color="tab:green", label="$\mu_{sim}(g_{exp}) (r^2$"+ f" score = {r2_mu_sim_g_sim: .3f})")
        ax.axvspan(t0, t0_silent, alpha=0.2, color="lightcoral")

        for neuron in self._silent_neurons[cycle]:
            sns.scatterplot(
                x=self.t,
                y=self._v[neuron, :],
                color="navy",
                s=1,
                alpha=0.15,
                markers=False,
                ax=ax
            )

        ax.set_xlim(t0, t_end)
        ax.set_ylim(-0.06, -0.04)

        ax.set_xlabel("Time (s)")
        ax.set_ylabel("Voltage (V)")
        ax.legend(loc=9)
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
        sns.set()
        if ax is None:
            fig, ax = plt.subplots()

        t0 = self._global_spikes.at[cycle, "start_spikes"]
        t_end_spikes = self.get_global_spikes().at[cycle, "end_spikes"] - self.t[1]
        t_end = t_end_spikes + 0.4*(t_end_spikes - t0)

        id_t_spikes = self.time_id([t0, t_end_spikes])
        id_t_total = self.time_id([t0 - self.t[10], t_end])

        t_spikes = self.t[id_t_spikes]
        t_total = self.t[id_t_total]

        # Reduced Model
        mean_width = 1.5
        mu_exp = self._mean_non_spike[id_t_spikes]
        mu_sim_g_sim = self._mu_sim_g_sim[id_t_spikes]
        mu_sim_g_exp = self._mu_sim_g_exp[id_t_spikes]

        r2_mu_sim_g_sim = r2_score(mu_exp, mu_sim_g_sim)

        ax.plot(t_total, self._mean_non_spike[id_t_total], linewidth=mean_width, color="tab:blue", label="$\mu_{exp}$ (experimental)")
        ax.plot(t_spikes, self._mu_sim_g_sim[id_t_spikes], linewidth=mean_width, color="salmon", label="$\mu_{sim}(g_{model}) (r^2$"+ f" score = {r2_mu_sim_g_sim: .3f})")
        ax.axvspan(t0, t_end_spikes, alpha=0.2, color="lightcoral")

        for neuron in self._silent_neurons[cycle][:50]:
            sns.scatterplot(
                x=t_total,
                y=self._v[neuron, id_t_total],
                color="navy",
                alpha=0.15,
                s=12,
                markers=False,
                ax=ax
            )

        ax.set_ylim(-0.045, -0.04)

        ax.set_xlabel("Time (s)")
        ax.set_ylabel("Voltage (V)")
        ax.legend(loc=1)
        return ax


    def plot_mean_inf(self, cycle=None, zoom=False, t0:float=0, t_end_silent:float=None, ax=None, fig=None):
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
        mu_exp = self._mean_non_spike[id_t_silent]
        mu_inf = self._mu_inf[id_t_silent]

        diff = np.abs((mu_exp[-1] - mu_inf[-1])/mu_exp[-1])

        ax.plot(self.t, self._mean_non_spike, linewidth=mean_width, color="tab:blue", label="$\mu_{exp}$ ")
        ax.plot(t_silent, mu_inf, linewidth=mean_width, color="salmon", label="$\mu_{\infty}$")
        # ax.plot(t_silent, self._mu_sim_g_exp[id_t_silent], linewidth=mean_width, color="tab:green", label="$\mu_{sim}(g_{sim})$ (r^2"+ f" score = {r2_mu_sim_g_sim: .3f})")

        ax.axvspan(t0, t0_silent, alpha=0.2, color="lightcoral")
        ax.axvline(self.t[id_t_end_silent-1], alpha=0.5, color="navy", linewidth=2, label="$t_0^{spk}=$"+f"{t_end_silent}, Relative error" + f" {diff: .3e})")
        ax.axvspan(self.t[id_t_end_silent-1], t_end, alpha=0.2, color="lightcoral")
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

        if zoom:
            ax.set_xlim(t_end-0.01, t_end)
            ax.set_ylim(-0.05, -0.04)
        else:
            ax.set_xlim(t0, t_end)
            ax.set_ylim(-0.06, -0.03)

        ax.set_xlabel("Time (s)")
        ax.set_ylabel("Voltage (V)")
        ax.legend(loc=9)
        return ax

    def plot_mean_inf_zoom(self, cycle=None, ax=None):
        return self.plot_mean_inf(cycle=cycle, zoom=True, ax=ax)

    def plot_std_silent(self, cycle=None, g_exp=False, t0: float = 0, t_end: float = None, ax=None, fig=None):
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
        t_end = self._global_spikes.at[cycle, "next_cycle"] - self.t[1]
        t0_silent = self.get_global_spikes().at[cycle, "end_spikes"]

        id_t_silent = self.time_id([t0_silent, t_end])
        t_silent = self.t[id_t_silent]

        id_t_total = self.time_id([t0, t_end])
        t_total = self.t[id_t_total]

        # Reduced Model
        color_sim = "salmon"
        color_exp = "tab:blue"
        mean_width = 1.5
        std_exp = self._std_non_spike[id_t_silent]
        std_sim_g_sim = self._var_sim_g_sim[id_t_silent] ** 0.5
        std_sim_g_exp = self._var_sim_g_exp[id_t_silent] ** 0.5

        r2_std_sim_g_exp = r2_score(std_exp, std_sim_g_exp)
        r2_std_sim_g_sim = r2_score(std_exp, std_sim_g_sim)

        ax.plot(t_total, self._std_non_spike[id_t_total], linewidth=mean_width, color="tab:blue",
                label="$\epsilon_{exp}$ (experimental)")
        ax.plot(t_silent, self._var_sim_g_sim[id_t_silent] ** 0.5, linewidth=mean_width, color="salmon",
                label="$\epsilon_{sim}(g_{model}$) ($r^2$" + f" score = {r2_std_sim_g_exp: .3f})")
        ax.plot(t_silent, self._var_sim_g_exp[id_t_silent] ** 0.5, linewidth=mean_width, color="tab:green",
                label="$\epsilon_{sim}(g_{exp})$ ($r^2$" + f" score = {r2_std_sim_g_sim: .3f})")
        ax.axvspan(t0, t0_silent, alpha=0.2, color="lightcoral")

        ax.set_xlabel("Time (s)")
        ax.set_ylabel("Voltage (V)")
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
        std_exp = self._std_non_spike[id_t_spikes]
        std_sim_g_sim = self._var_sim_g_sim[id_t_spikes] ** 0.5
        std_sim_g_exp = self._var_sim_g_exp[id_t_spikes] ** 0.5

        r2_std_sim_g_exp = r2_score(std_exp, std_sim_g_exp)
        r2_std_sim_g_sim = r2_score(std_exp, std_sim_g_sim)

        ax.plot(t_total, self._std_non_spike[id_t_total], linewidth=mean_width, color="tab:blue",
                label="$\epsilon_{exp}$ (experimental)")
        ax.plot(t_spikes, std_sim_g_sim, linewidth=mean_width, color="salmon",
                label="$\epsilon_{sim}(g_{model}$) ($r^2$" + f" score = {r2_std_sim_g_exp: .3f})")
        ax.plot(t_spikes, std_sim_g_exp, linewidth=mean_width, color="tab:green",
                label="$\epsilon_{sim}(g_{exp})$ ($r^2$" + f" score = {r2_std_sim_g_sim: .3f})")
        ax.axvspan(t0, t_end_spikes, alpha=0.2, color="lightcoral")

        ax.set_xlabel("Time (s)")
        ax.set_ylabel("Voltage (V)")
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

        r2_g = r2_score(self.g_t[id_t_silent], self._g_sim_R_sim[id_t_silent])

        r2_R = r2_score(self.R_t[id_t_silent], self._R_sim_f_sim[id_t_silent])

        ax.plot(self.t, self.g_t, linewidth=mean_width, color=color_exp, label="$g_{exp}$ (experimental)")
        ax.plot(t_silent, self._g_sim_R_sim[id_t_silent], linewidth=mean_width, color=color_sim, label="$g_{sim}$ (model) ($r^2$ score =" + f"{r2_g: .5f})")

        ax.plot(self.t, self.R_t, linewidth=mean_width, color="tab:green", label="$R_{exp}$ (experimental)")
        ax.plot(t_silent, self._R_sim_f_sim[id_t_silent], linewidth=mean_width, color=color_sim, label="$R_{sim}$ (model) ($r^2$ score =" + f"{r2_R: .5f})")

        ax.axvspan(t0, t0_silent, alpha=0.2, color="lightcoral")
        ax.set_xlim(t0, t_end)
        ax.set_ylim(0, 1.05*max(self._R_sim_f_sim[id_t_silent]))

        ax.set_xlabel("Time (s)")
        ax.set_ylabel("Conductancy ($S/cm^2$)")
        ax.legend(loc=1)
        return ax


    def plot_g_R_spikes(self, cycle, ax=None):
        if ax is None:
            fig, ax = plt.subplots()

        start_time = self._global_spikes.at[cycle, "start_spikes"]
        end_spikes = self._global_spikes.at[cycle, "end_spikes"] - self.t[1]
        end_time = end_spikes + 0.5* (end_spikes - start_time)
        id_t_spikes = self.time_id([start_time, end_spikes])
        id_t_total = self.time_id([start_time - self.t[15], end_time])
        total_time = self.t[id_t_total]
        silent_time = self.t[id_t_spikes]

        exp_activity = self._global_rate.loc[id_t_total, "rate"]

        ax.axvspan(start_time, end_spikes, color="lightcoral", alpha=0.2)

        # sns.lineplot(x=cycle_time, y=exp_activity, color="tab:blue", linewidth=0.5, label="$f_{exp}(t)$")
        # sns.lineplot(x=cycle_time, y=self._f_sim[id_t_total], color="salmon", label="$f_{sim}(t)$")
        # ax.set_xlabel("Time (s)")
        # ax.set_ylabel("Conductivity ($\cdot \Omega ^{-1}/cm^{2}$)")
        # ax.legend()


        # Reduced Model
        color_sim = "salmon"
        color_exp = "tab:blue"
        mean_width = 1

        r2_g = r2_score(self.g_t[id_t_spikes], self._g_sim_R_sim[id_t_spikes])
        r2_R = r2_score(self.R_t[id_t_spikes], self._R_sim_f_sim[id_t_spikes])

        ax.plot(total_time, self.g_t[id_t_total], linewidth=mean_width, color=color_exp, label="$g_{exp}$ (experimental)")
        ax.plot(silent_time, self._g_sim_R_sim[id_t_spikes], linewidth=mean_width, color=color_sim, label="$g_{sim}$ (model) ($r^2$ score =" + f"{r2_g: .2f})")

        ax.plot(total_time, self.R_t[id_t_total], linewidth=mean_width, color="tab:green", label="$R_{exp}$ (experimental)")
        ax.plot(silent_time, self._R_sim_f_sim[id_t_spikes], linewidth=mean_width, color=color_sim, label="$R_{sim}$ (model) ($r^2$ score =" + f"{r2_R: .2f})")

        ax.legend(loc="best")

    def plot_activity_pred(self, cycle, ax=None):
        if ax is None:
            fig, ax = plt.subplots()

        start_time = self._global_spikes.at[cycle, "start_spikes"]
        end_spikes = self._global_spikes.at[cycle, "end_spikes"]
        end_time = end_spikes + 0.5*(end_spikes - start_time)
        id_t = self.time_id([start_time - self.t[15], end_time])
        id_t = id_t[:-1]

        exp_activity = self._global_rate.loc[id_t, "rate"]
        cycle_time = self.t[id_t]
        ax.axvspan(start_time, end_spikes, color="lightcoral", alpha=0.2)
        sns.lineplot(x=cycle_time, y=exp_activity, color="tab:blue", linewidth=0.5, label="$f_{exp}(t)$")
        sns.lineplot(x=cycle_time, y=self._f_sim[id_t], color="salmon", label="$f_{sim}(t)$")
        ax.set_xlabel("Time (s)")
        ax.set_ylabel("Activity ($Hz \cdot \Omega ^{-1}/cm^{2}$)")
        ax.legend()
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
            y=self._mean_non_spike,
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
        ax.set_ylabel("Voltage (V)")
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
        ax.set_ylabel("Voltage (V)")
        ax.legend(loc="upper right")
        return ax

    def plot_rasterplot(self, ax=None):
        if ax is None:
            ax = plt.gca()

        ax.set_xlabel("Time (s)")
        ax.set_ylabel("Neuron Cell")
        xgrid = np.linspace(0, self.duration, self.time_steps)
        ygrid = np.array(range(self.N_neurons))
        ax.pcolormesh(xgrid, ygrid, self.voltages, shading="auto")
        return ax

    def plot_qq(self, cycle=4, ax=None):
        if ax is None:
            fig, ax = plt.subplots()
        time_spikes = self._global_spikes.at[cycle, "end_spikes"]
        voltages = self._v[self._silent_neurons[cycle], self.time_id(time_spikes)]
        stat, pvalue = shapiro(voltages)
        qqplot(voltages, fit=True, ax=ax, line='s', label=f"Shapiro p-value = {pvalue: .3e}")
        ax.legend(loc="best")
        return ax

    def plot_voltage_dist_t_0_silence(self, ax=None, cycle=4):
        if ax is None:
            fig, ax = plt.subplots()

        t_0_silent = self._global_spikes.at[cycle, "end_spikes"]
        neurons = self._silent_neurons[cycle]
        voltages = self._v[neurons, self.time_id(t_0_silent)]
        mu = self._mean_non_spike[self.time_id(t_0_silent)]
        std = self._std_non_spike[self.time_id(t_0_silent)]
        exp_tag = " ($\mu = $" + f"{mu: .4f}, " + "$\epsilon = $" + f"{std: .2e})"


        bins = ax.hist(voltages, density=1, bins=100, label="Voltage distribution" + exp_tag)
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
        gauss_tag = " ($\mu = $" + f"{mu_gauss: .4f}, " + "$\epsilon = $" + f"{std_gauss: .2e})"

        sns.lineplot(x=x, y=gaussian_curve_emp, color="salmon", linewidth=1.5, label="Gaussian fit" + gauss_tag, alpha=1)

        ax.set_ylim(bottom=-5)
        ax.set_ylabel("Probability density")
        ax.set_xlabel("Voltage (V)")

        return ax

    def plot_voltage_dist_spiking(self, ax=None, cycle=4):
        if ax is None:
            fig, ax = plt.subplots()

        time_spikes = self._global_spikes.at[cycle, "end_spikes"]
        voltages = self._v[:, self.time_id(time_spikes)]
        bins = ax.hist(voltages, density=1, bins=70, label="Voltage distribution")
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


        sns.lineplot(x=x, y=gaussian_curve_emp, color="darkorange", linewidth=3, label="Gaussian fit", alpha=0.2)

        ax.set_ylim(bottom=-10)
        ax.set_ylabel("Probability density")
        ax.set_xlabel("Voltage (V)")

        return ax

    def plot_attractor(self, ax=None):
        if ax is None:
            fig, ax = plt.subplots()

        V = self._mean_v[1000:]
        g = self._g_i.mean(axis=0)[1000:]
        f = self._global_rate["rate_smooth"][1000:]

        ax.plot3D(V, g, f, alpha=0.5)
        plt.tight_layout()
        ax.set_xlabel("V")
        ax.set_ylabel("g")
        ax.set_zlabel("f")

        ax.set_yticklabels([])
        ax.set_xticklabels([])
        ax.set_zticklabels([])

        return ax

    def plot(self, figsize, cycle, plotter, path=None, dpi=600):
        print(f"Plotting file in {path}")
        sns.set()
        fig, ax = plt.subplots(figsize=figsize)
        ax = plotter(ax=ax, cycle=cycle)
        plt.tight_layout
        if path is not None:
            plt.savefig(path, dpi=dpi)
        else:
            plt.show()
        plt.close()