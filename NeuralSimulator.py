import brian2 as b2
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
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
        self.__N = self.__sim_constants.get("N_neurons")
        self._v_L = self.__sim_constants.get("v_L")
        self.__v_I = self.__sim_constants.get("v_I")
        self.__v_thr = self.__sim_constants.get("v_thr")
        self.__v_res = self.__sim_constants.get("v_res")
        self.__duration = self.__sim_constants.get("duration")
        self.__sigma = self.__sim_constants.get("sigma")
        self.__I_dc = self.__sim_constants.get("I_dc")
        self.__g_L = self.__sim_constants.get("g_L")
        self.__tau = self.__sim_constants.get("tau")
        self.__tau_D = self.__sim_constants.get("tau_D")
        self.__tau_R = self.__sim_constants.get("tau_R")
        self.__w = self.__sim_constants.get("weights")
        self.__initial_g = self.__sim_constants.get("initial_g")

        # The initial voltage of the neurons is uniformly distributed between the reset potential and the threshold
        self.__initial_v = np.random.uniform(low=self.__v_res, high=self.__v_thr, size=self.__N)

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
        if self.__N_recorded > self.__N:
            self.__N_recorded = self.__N
        rng = np.random.default_rng()
        self.__recorded_neurons_id = rng.choice(np.arange(self.__N), size=self.__N_recorded)

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
            N = self.__N
            v_L = self._v_L * b2.volt
            v_I = self.__v_I * b2.volt
            v_thr = self.__v_thr * b2.volt
            v_res = self.__v_res * b2.volt
            duration = self.__duration * b2.second
            sigma = self.__sigma * b2.volt
            I_dc = self.__I_dc * b2.volt * b2.hertz
            g_L = self.__g_L * b2.hertz
            tau = self.__tau * b2.second
            tau_D = self.__tau_D * b2.second
            tau_R = self.__tau_R * b2.second
            w = self.__w
            self.__neuron_model_eqs = '''
                dv/dt = g_L*(v_L-v) + I_dc + w*g*(v_I - v) + sigma*xi*tau**(-0.5) : volt
                dg/dt = (R-g)/tau_D : hertz
                dR/dt = -R/tau_R : hertz
            '''

        # Neuron group
        G = b2.NeuronGroup(
            N=self.__N,
            model=self.__neuron_model_eqs,
            threshold='v > v_thr',
            reset='v = v_res',
            method='euler',
        )
        G.v = self.__initial_v * b2.volt
        G.g = self.__initial_g * b2.hertz

        # Synapses
        S = b2.Synapses(G, G, model='''delta : hertz # synaptic weight''', on_pre='R+=delta')
        S.connect(condition='True')
        S.delta = 1 * b2.hertz

        # Preparing monitors and network
        spike_monitor = b2.SpikeMonitor(G)
        monitor = b2.StateMonitor(G, variables=self.__recorded_variables, record=True)
        rate_monitor = b2.PopulationRateMonitor(G)
        net = b2.Network(b2.collect())  # automatically include G and S
        net.add(monitor)  # manually add the monitors

        # Simulation
        print("Running simulation")
        start_time = datetime.datetime.now()
        net.run(self.__duration * b2.second)
        elapsed_time = datetime.datetime.now() - start_time
        print(f"Elapsed time: {elapsed_time.total_seconds()}s")

        self._v = np.array(monitor.v)
        self._g = np.array(monitor.g)
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
            sigma=25,
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
        global_spikes = global_spikes[[
            "start_cycle",
            "start_spikes",
            "pred_start_spikes",
            "peak_time",
            "end_spikes",
            "next_cycle",
            "%_spiking_neurons",
            "mu",
            "var",
            "pred_p_th"]
        ]
        return global_spikes

    def time_id(self, time):
        """
        Gets the index of the element in "self.time" that has the same value as "time".
        If such element does not exist, the immediate smaller is returned.
        :param time:
        :return:
        """
        time = float(time)

        try:
            return self.t[self.t == time].index[0]
        except IndexError:
            return self.t[self.t < time].index[-1]

    def simulate_reduced_model(self, cycle=4):
        self._t_start_spikes = self._global_spikes.loc[cycle, 'start_spikes']
        self._t_end_spikes = self._global_spikes.loc[cycle, 'end_spikes']
        self._t_end_cycle = self._global_spikes.loc[cycle, 'next_cycle']

        id_t_start_spikes = self.time_id(self._t_start_spikes)
        id_t_end_spikes = self.time_id(self._t_end_spikes)
        id_t_end_cycle = self.time_id(self._t_end_cycle)
        self._t_spikes = list(self.t[id_t_start_spikes: id_t_end_spikes])
        self._t_silent = list(self.t[id_t_end_spikes: id_t_end_cycle])

        S_t = np.mean(self.g, axis=0)
        G_t = S_t
        G_t_start_spikes = G_t[id_t_start_spikes]
        G_t_end_spikes = G_t[id_t_end_spikes]

        d_mu_dt = self.d_mu_dt_generator(self.g_L, self._v_L, self.__I_dc, G_t, self.__v_I, self.__w)
        d_var_dt = self.d_var_dt_generator(G_t, self.__g_L, self.__sigma, self.__tau, self.__w)
        d_mu_spiking_dt = self.d_mu_spiking_dt_generator(self.g_L, self._v_L, self.__I_dc, G_t, self.__v_I, self.__w,
                                                         id_t_start_spikes)

        mu_start_spikes = self._mean_non_spike[id_t_start_spikes]
        self._mu_spikes = solve_ivp(
            fun=d_mu_dt,
            t_span=(self._t_start_spikes, self._t_end_spikes),
            y0=[mu_start_spikes],
            method='DOP853',
            # Use the same time steps as the brian simulation
            t_eval=self._t_spikes
        )

        mu_end_spikes = self._mean_non_spike[id_t_end_spikes]
        self._mu_silent = solve_ivp(
            fun=d_mu_dt,
            t_span=(self._t_end_spikes, self._t_end_cycle),
            y0=[mu_end_spikes],
            method='DOP853',
            # Use the same time steps as the brian simulation
            t_eval=np.linspace(self._t_end_spikes, self._t_end_cycle, 1000)
        )

        var_end_spikes = self._std_non_spike[id_t_end_spikes]
        self._var_silent = solve_ivp(
            fun=d_var_dt,
            t_span=(self._t_end_spikes, self._t_end_cycle),
            y0=[var_end_spikes],
            method='DOP853',
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

    def d_mu_spiking_dt_generator(self, g_L, v_L, I_dc, G_t, v_I, w, id_t_start_spikes):
        """
        Generates a function d_mu_dt that returns the derivate of mu in the spiking stage
        """

        G_t0 = w * G_t[id_t_start_spikes]
        a = g_L + G_t0
        b = g_L * v_L + I_dc + G_t0 * v_I

        def d_mu_dt(t, mu):
            g_b_syn = w * (G_t[self.time_id(t)] - G_t0)
            return float(-a * mu + b + g_b_syn * (v_I - mu))

        return d_mu_dt

    def d_var_dt_generator(self, G_t, g_L, sigma, tau, w):
        # Generates d(var)/dt
        def d_var_dt(t, var):
            g_syn = w * G_t[self.time_id(t)]
            a = g_L + g_syn
            return float(-2 * a * var ** 2 + sigma ** 2 / tau)

        return d_var_dt

    def _compute_non_spiking_moments(self):
        """
        Computes the mean and std in time of the non spiking population at each cycle
        :return:
        """
        self._mean_non_spike = self._mean_v.copy()
        self._std_non_spike = self._std_v.copy()
        for cycle in self._global_spikes.index[1:-2]:
            # Select non spiking neurons of the cycle
            spike_neurons = set(self._spikes[self._spikes["cycle"] == cycle].index.get_level_values(0))
            all_neurons = set(range(self.N))
            non_spike_neurons = list(all_neurons - spike_neurons)

            # Select time indices of the cycle
            t0_id = self.time_id(self._global_spikes.at[cycle, "start_spikes"])
            t_end_id = self.time_id(self._global_spikes.at[cycle, "next_cycle"]) - 1

            # Compute mean and std
            self._mean_non_spike[t0_id:t_end_id] = self._v[non_spike_neurons, t0_id:t_end_id].mean(axis=0)
            self._std_non_spike[t0_id:t_end_id] = self._v[non_spike_neurons, t0_id:t_end_id].std(axis=0)

    @property
    def N(self) -> int:
        return self.__N

    @property
    def duration(self):
        return self.__duration

    @property
    def v_res(self):
        return self.__v_res

    @property
    def v_thr(self):
        return self.__v_thr

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
        return self.__initial_v

    @property
    def interspike_times(self):
        return self._interspike_times

    @property
    def g_L(self):
        return self.__g_L


class NeuralAnalyzer(NeuralSimulator):
    def __init__(self, cycle, filename="config.json"):
        super().__init__(filename)
        super().simulate()
        super().simulate_reduced_model(cycle)
        self._init_NeuralAnalyzer()

    def _init_NeuralAnalyzer(self):
        self.__time_steps = self.v.shape[1]
        self.__voltage_resolution = 200
        self.__v_density = self.compute_voltage_distribution()
        self.mean = np.mean(self.v, axis=0)
        self.std = np.std(self.v, axis=0)

    def compute_voltage_distribution(self):
        """
        Computes the voltage distribution over time and stores it
        :return:
        """
        v_density = np.empty((self.__voltage_resolution, self.__time_steps))

        for t in range(self.__time_steps):
            v_density[:, t] = np.histogram(
                self.v[:, t],
                range=(self.v_res, self.v_thr),
                bins=self.__voltage_resolution,
            )[0]
        return v_density

    def plot_density(self, t0: float = 0, t_end: float = None, ax=None, fig=None):
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
        ygrid = np.linspace(self.v_res, self.v_thr, self.__voltage_resolution)
        vmap = ax.pcolormesh(
            xgrid,
            ygrid,
            np.sqrt(self.__v_density),
            shading="auto",
        )
        fig.colorbar(vmap)
        ax.plot(xgrid, self._mean_non_spike, label="Mean", color="orange")
        ax.plot(xgrid, self._mean_non_spike + self._std_non_spike, label="Upper bound", color="salmon")
        ax.plot(xgrid, self._mean_non_spike - self._std_non_spike, label="Lower bound", color="salmon")

        # Reduced Model
        ax.plot(self._mu_silent.t, self._mu_silent.y[0], color="green", label="Reduced model mu silent stage")
        ax.plot(self._mu_spikes.t, self._mu_spikes.y[0], color="purple", label="Reduced model mu spiking stage")

        ax.vlines(x=self._t_end_spikes, ymin=self.v_res,
                  ymax=self.v_thr)  # Vertical Line that marks the beginning of the silent period
        ax.set_xlim(t0, t_end)
        ax.set_ylim(self.v_res, self.v_thr)

        ax.set_xlabel("Time (ms)")
        ax.set_ylabel("Voltage (mV)")
        ax.legend(loc=8)
        return ax

    def plot_rasterplot(self, ax=None):
        if ax is None:
            ax = plt.gca()

        ax.set_xlabel("Time (ms)")
        ax.set_ylabel("Neuron Cell")
        xgrid = np.linspace(0, self.duration, self.time_steps)
        ygrid = np.array(range(self.N_neurons))
        ax.pcolormesh(xgrid, ygrid, self.voltages, shading="auto")
        return ax
