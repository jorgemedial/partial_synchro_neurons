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
from scipy.integrate import solve_ivp


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
        self._g = None
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
            method='euler',
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
        self._g = np.array(monitor.g)
        self._R = np.array(monitor.R)
        self._t = pd.Series(np.array(monitor.t))
        self._mean_v = np.mean(self.v, axis=0)
        self._std_v = np.std(self.v, axis=0)
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
            global_spikes.loc[cycle, ['end_spikes']] = self._spikes["time"].loc[cycle_index].max()

            # Get mean and std of the start time
            try:
                start_time_id = self.time_id(start_time)
                global_spikes.loc[cycle, ['mu']] = self.mean[start_time_id - 1]
                global_spikes.loc[cycle, ['var']] = self.std[start_time_id - 1] ** 2
            except:
                global_spikes.loc[cycle, ['mu']] = np.nan
                global_spikes.loc[cycle, ['var']] = np.nan

            # Theoretical onset of spiking
            onset_time_id = self.time_id(global_spikes["start_cycle"].iat[cycle])
            time_id_end = self.time_id(global_spikes["peak_time"].iat[cycle])
            found = False
            p_th = 0
            while (onset_time_id < time_id_end) & (not found):
                mu = self._mean_v[onset_time_id]
                std = self._std_v[onset_time_id]
                Z_th = (mu - self.v_thr) / (np.sqrt(2) * std)
                p_th = self.N * (1 + sp.special.erf(Z_th))
                if p_th > 1:
                    found = True
                onset_time_id += 1

            onset_time = self.t[onset_time_id]
            global_spikes.at[cycle, "pred_start_spikes"] = onset_time
            global_spikes.at[cycle, "pred_p_th"] = p_th

        # Redefine next_cycle at the next start_spikes
        global_spikes.loc[:, "next_cycle"] = global_spikes["start_spikes"].shift(-1)

        # Reorder dataframe columns
        global_spikes = global_spikes[
            [
                "start_cycle",
                "start_spikes",
                "pred_start_spikes",
                "peak_time",
                "end_spikes",
                "next_cycle",
                "%_spiking_neurons",
                "mu",
                "var",
                "pred_p_th",
                "pred_start_spikes",
            ]
        ]
        return global_spikes

    def time_id(self, time):
        """
        Gets the index of the element in "self.time" that has the same value as "time".
        If such element does not exist, the immediate smaller is returned.
        :param time:
        :return:
        """
        if hasattr(time, '__iter__'):
            time_ids = np.vectorize(self.time_id)
            return time_ids(time)

        else:
            time = float(time)
            time_index = 0
            while self.t.iat[time_index] < time:
                time_index += 1
            return time_index

    def simulate_reduced_model(self, cycle=4):
        self._t_start_spikes = self._global_spikes.loc[cycle, 'start_spikes']
        self._t_peak_time = self._global_spikes.loc[cycle, "peak_time"]
        self._t_end_spikes = self._global_spikes.loc[cycle, 'end_spikes']
        self._t_end_cycle = self._global_spikes.loc[cycle, 'next_cycle']

        self.id_t_start_spikes = self.time_id(self._t_start_spikes)
        self.id_t_peak_time = self.time_id(self._t_start_spikes)
        self.id_t_end_spikes = self.time_id(self._t_end_spikes)
        self.id_t_end_cycle = self.time_id(self._t_end_cycle)

        self._t_spikes = list(self.t[self.id_t_start_spikes:self.id_t_end_spikes])
        self._t_silent = list(self.t[self.id_t_end_spikes:self.id_t_end_cycle])

        S_t = np.mean(self.g, axis=0)
        G_t = S_t
        G_t_start_spikes = G_t[self.id_t_start_spikes]
        G_t_end_spikes = G_t[self.id_t_end_spikes]
        R_t = np.mean(self._R, axis=0)

        d_mu2_dt = self.d_mu_dt_generator(self.g_L, self._v_L, self._I_dc, G_t, self._v_I, self._w)
        d_var_dt = self.d_var_dt_generator(G_t, self._g_L, self._sigma, self._tau, self._w)
        d_mu_spiking_dt = self.d_mu_spiking_dt_generator2(G_t[self.id_t_start_spikes])
        d_mu_dt = self.d_mu_dt_generator2(self.g_L, self._v_L, self._I_dc, self._v_I, self._w, self._tau_R, self._tau_D)

        mu_start_spikes = self._mean_non_spike[self.id_t_start_spikes]
        var_start_spikes = self._std_non_spike[self.id_t_start_spikes] ** 2
        g_b_start_spikes = 0
        R_b_start_spikes = 0
        y0_spiking = [
            mu_start_spikes,
            var_start_spikes,
            g_b_start_spikes,
            R_b_start_spikes,
        ]
        self._mu_spikes = solve_ivp(
            fun=d_mu_spiking_dt,
            t_span=(self._t_spikes[0], self._t_spikes[-1]),
            y0=y0_spiking,
            method='DOP853',
            # Use the same time steps as the brian simulation
            t_eval=self._t_spikes
        )

        mu_silent = self._mean_non_spike[self.id_t_end_spikes]
        var_silent = self._std_non_spike[self.id_t_end_spikes] ** 2
        g_silent = G_t[self.id_t_end_spikes]
        R_silent = R_t[self.id_t_end_spikes]
        y0 = [mu_silent, var_silent, g_silent, R_silent]

        self._mu_silent = solve_ivp(
            fun=d_mu_dt,
            t_span=(self._t_silent[0], self._t_silent[-1]),
            y0=y0,
            method='RK23',
            # Use the same time steps as the brian simulation
            t_eval=self._t_silent
        )


    def d_mu_dt_generator(self, g_L, v_L, I_dc, G_t, v_I, w):
        """
        Generates a function d_mu_dt that returns the derivative of mu in the silent stage
        """

        def d_mu_dt(t, mu):
            g_syn = w * G_t[self.time_id(t)]
            a = g_L + g_syn
            b = g_L * v_L + I_dc + g_syn * v_I
            return float(-a * mu + b)

        return d_mu_dt

    def d_mu_dt_generator2(self, g_L, v_L, I_dc, v_I, w, tau_R, tau_D):
        """
        Generates a function d_mu_dt that returns the derivative of mu in the silent stage
        """
        sigma = self._sigma
        tau = self._tau
        def d_mu_dt(t, y):
            """
            :param t:
            :param y: y[0] = mu, y[1] = g, y[2] = R
            :return:
            """
            mu = y[0]
            var = y[1]
            g = y[2]
            R = y[3]

            a = g_L + w * g
            b = g_L * v_L + I_dc + w * g * v_I
            return [-a * mu + b, -2 * a * var + sigma ** 2 / tau,  (R - g)/tau_D, -R/tau_R]

        return d_mu_dt

    def d_mu_spiking_dt_generator(self, g_L, v_L, I_dc, G_t, v_I, w, id_t_start_spikes):
        """
        Generates a function d_mu_dt that returns the derivative of mu in the spiking stage
        """

        G_t0 = w * G_t[id_t_start_spikes]
        a = g_L + G_t0
        b = g_L * v_L + I_dc + G_t0 * v_I

        def d_mu_dt(t, mu):
            g_b_syn = w * G_t[self.time_id(t)] - G_t0
            return float(-a * mu + b + g_b_syn * (v_I - mu))
        return d_mu_dt

    def d_mu_spiking_dt_generator2(self, G_t0):
        """
        Generates a function d_mu_dt that returns the derivative of mu in the spiking stage
        """

        a = self._g_L + G_t0 * self._w
        self.a = a
        b = self._g_L * self._v_L + self._I_dc + G_t0 * self._v_I * self._w
        self.b = b

        def d_mu_dt(t, y):
            mu = y[0]
            var = y[1]
            g = y[2]
            R = y[3]

            f = self._global_rate.at[self.time_id(t), "rate"]
            return [
                -a * mu + b + self._N * g * (self._v_I - mu) * self._w,  # d_mu/dt
                2*var*(-a + self._N * self._w * g),  # d_var/dt
                (R - g)/self._tau_D,  # dg/dt
                f - R/self._tau_R,  # dR/dt
            ]

        return d_mu_dt

    def d_var_dt_generator(self, G_t, g_L, sigma, tau, w):
        # Generates d(var)/dt
        def d_var_dt(t, var):
            g_syn = w * G_t[self.time_id(t)]
            a = g_L + g_syn
            return float(-2 * a * var ** 2 + sigma ** 2 / tau)
        return d_var_dt

    def get_activity_time_range(self, cycle=4):
        id_time_spiking = self.time_id(self._mu_spikes.t)
        mu_t = self._mean_non_spike[id_time_spiking]
        std_t = self._std_non_spike[id_time_spiking]
        g_t = self._g.mean(axis=0)[id_time_spiking]
        a_t = self._g_L + self._w*g_t
        b_t = self.g_L*self._v_L + self._I_dc + self._w*g_t*self._v_I
        return np.array([self.get_activity(b, a, mu, std) for (a, b, mu, std) in zip(a_t, b_t, mu_t, std_t)])

    def get_activity(self, b, a, mu, std):
        """
        Gets the activity f(t) depending on mu and std
        """
        density_thr = np.exp(-0.5*((self._v_thr-mu)/std)**2)/(std*np.sqrt(2*np.pi))
        dV_dt = b - a * mu + self._sigma*self._tau**(-0.5)/2
        return density_thr * dV_dt


    def _compute_non_spiking_moments(self):
        """
        Computes the mean and std in time of the non spiking population at each cycle
        :return:
        """
        self._mean_non_spike = self._mean_v.copy()
        self._std_non_spike = self._std_v.copy()
        self._mean_spike = self._mean_v.copy()
        self._spike_neurons = {cycle: [] for cycle in self._global_spikes.index[1:-2]}
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
    def g(self):
        return self._g

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
        super().simulate_reduced_model(cycle)
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
        ax.plot(cycle_time, 5*smooth_activity, color="darkorange", linewidth=3,  label="Smoothed activity")

        ax.set_xlabel("Time (s)")
        ax.legend()
        ax.set_yscale("linear")
        return fig, ax


    def plot_density(self, t0:float=0, t_end:float=None, ax=None, fig=None):
        """

        :param ax:  Axes
                    The axes to draw to
        :param t0:  Start time
                    Time where the x-axis starts from
        :param t_end:   End time
                        Time where the x-axis ends
        :return:
        """
        if ax is None:
            fig, ax = plt.subplots()

        if t_end is None:
            t_end = self.duration

        xgrid = np.linspace(0, self.duration, self.__time_steps)
        ygrid = np.linspace(self.v_res, self.v_thr, self._voltage_resolution)
        vmap = ax.pcolormesh(
            xgrid,
            ygrid,
            self.__v_density ** 0.8,
            shading="auto",
            cmap="BuPu"
        )
        #
        # ax.plot(xgrid, self._mean_non_spike, label="Mean", color="orange")
        # ax.plot(xgrid, self._mean_non_spike + self._std_non_spike, label="Upper bound", color="salmon")
        # ax.plot(xgrid, self._mean_non_spike - self._std_non_spike, label="Lower bound", color="salmon")

        # Reduced Model
        ax.plot(self._mu_silent.t, self._mu_silent.y[0], linewidth=3, color="navy", label="Reduced model in the silent stage")
        ax.plot(self._mu_spikes.t, self._mu_spikes.y[0], linewidth=3, color="orange", label="Reduced model in the spiking stage")
        ax.plot(self._mu_silent.t, self._mu_silent.y[1] ** 0.5 + self._mu_silent.y[0], linewidth=0.7,  color="navy")
        ax.plot(self._mu_silent.t, -self._mu_silent.y[1] ** 0.5 + self._mu_silent.y[0], linewidth=0.7, color="navy")

        for neuron in self._spike_neurons[4]:
            sns.scatterplot(
                x=xgrid,
                y=self._v[neuron, :],
                color="navy",
                s=1,
                alpha=0.2,
                markers=False,
                ax=ax,
            )

        ax.vlines(
            x=self._t_end_spikes,
            ymin=self.v_res,
            ymax=self.v_thr,
            linewidth=1,
            color="powderblue"
        )  # Vertical Line that marks the beginning of the silent period

        ax.set_xlim(t0-0.003, t_end)
        ax.set_ylim(self.v_res+0.015, self.v_thr)

        ax.set_xlabel("Time (s)")
        ax.set_ylabel("Voltage (V)")
        ax.legend(loc=8)
        return ax

    def plot_activity_pred(self, cycle=4, ax=None):
        if ax is None:
            fig, ax = plt.subplots()

        start_time = self._global_spikes.at[cycle, "start_spikes"]
        end_time = self._global_spikes.at[cycle, "next_cycle"]

        id_start_time = self.time_id(start_time)
        id_end_time = self.time_id(end_time)

        exp_activity = self._global_rate.loc[id_start_time:id_end_time, "rate"]
        smooth_activity = self._global_rate.loc[id_start_time:id_end_time, "rate_smooth"]

        cycle_time = np.linspace(start_time, end_time, len(exp_activity))
        ax.plot(cycle_time, smooth_activity, color="darkorange", linewidth=3,  label="Smoothed activity")
        ax.plot(cycle_time, exp_activity, color="navy", linewidth=0.8, label="Activity")
        ax.plot(self._mu_spikes.t, self.get_activity_time_range())

        ax.set_xlabel("Time (s)")
        ax.legend()


    def plot_all_neurons(self, t0: float = 0, t_end: float = None, ax=None, fig=None):
        if ax is None:
            fig, ax = plt.subplots()

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

    def plot_one_neuron(self, t0: float = 0, t_end: float = None, ax=None, fig=None):
        if ax is None:
            fig, ax = plt.subplots()

        if t_end is None:
            t_end = self.duration

        xgrid = np.linspace(0, self.duration, self.__time_steps)

        ax.set_xlim(t0, t_end)
        ax.set_ylim(self.v_res, self.v_thr+0.01)

        sns.lineplot(
            x=xgrid,
            y=self._v[20],
            ax=ax,
            label="Voltage of a single neuron",
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

    def plot_voltage_dist_spiking(self, ax=None, cycle=4):
        if ax is None:
            fig, ax = plt.subplots()

        time_spikes = self._global_spikes.at[4, "end_spikes"]
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
        g = self._g.mean(axis=0)[1000:]
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
