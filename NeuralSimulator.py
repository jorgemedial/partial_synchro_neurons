import pickle
import datetime

import brian2 as b2
import numpy as np
import pandas as pd

from scipy.signal import find_peaks
from scipy.ndimage import gaussian_filter1d

from NeuralConstants import NeuralConstants


class NeuralSimulator(NeuralConstants):
    """
    This class simulates the neural network and stores the output
    """
    def __init__(self, neural_constants: dict):
        super().__init__(neural_constants)
        self._init_simulation()

    def _init_simulation(self):
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
        self.t = pd.Series(np.array(monitor.t))

        spikes = spike_monitor.values('t')
        spikes = pd.DataFrame.from_dict(spikes, orient='index').stack()
        self._spikes = spikes.map(lambda time: time * b2.hertz).to_frame(name="time")
        self._spikes["cycle"] = 0
        self._interspike_times = self._spikes["time"].groupby(level=0).diff()
        self._global_rate = self.get_global_rate(rate_monitor)

        self._global_spikes = self.get_global_spikes()
        self.cycles = self._global_spikes.index[2:-2]
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

    def time_id(self, time):
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


    def _compute_non_spiking_moments(self):
        """
        Computes the mean and std in time of the non spiking population at each cycle
        :return:
        """
        self._mean_silent = self._mean_v.copy()  # hacer en función del tiempo
        self._std_silent = self._std_v.copy()
        self._std_spike = self._std_v.copy()
        self._mean_spike = self._mean_v.copy()
        self._spike_neurons = {cycle: [] for cycle in self.cycles}
        self._silent_neurons = {cycle: [] for cycle in self.cycles}
        for cycle in self.cycles:
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

