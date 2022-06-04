import brian2 as b2
import numpy as np
import pandas as pd
import pickle
import json
import os
import datetime

class NeuralSimulator:
    """
    This class simulates the neural network and stores the output
    """

    def __init__(self):
        self.__dirname = os.path.dirname(__file__)
        self.__configurate()

    def __configurate(self):
        """
        Reads config.json file
        """
        self.__config_path = os.path.join(self.__dirname, "config/config.json")
        self.__config = json.load(open(self.__config_path, "r"))
        self.__load_constants()
        self.__load_recording_configuration()

    def __load_constants(self):
        """
        Computes neural constants (i.e. potentials, conductivities) from the data in config.json
        """
        self.__sim_constants = self.__config.get("simulation_constants")
        self.__N = self.__sim_constants.get("N_neurons")
        self.__v_L = self.__sim_constants.get("v_L") * b2.mV
        self.__v_I = self.__sim_constants.get("v_I") * b2.mV
        self.__v_thr = self.__sim_constants.get("v_thr") * b2.mV
        self.__v_res = self.__sim_constants.get("v_res") * b2.mV
        self.__duration = self.__sim_constants.get("duration") * b2.ms
        self.__sigma = self.__sim_constants.get("sigma") * b2.mV
        self.__I_dc = self.__sim_constants.get("I_dc") * b2.hertz * b2.mV
        self.__g_L = self.__sim_constants.get("g_L") * b2.hertz
        self.__tau = self.__sim_constants.get("tau") * b2.ms
        self.__tau_D = self.__sim_constants.get("tau_D") * b2.ms
        self.__tau_R = self.__sim_constants.get("tau_R") * b2.ms
        self.__w = self.__sim_constants.get("weights")
        self.__initial_g = self.__sim_constants.get("initial_g") / b2.ms
        self.__v_resolution = self.__sim_constants.get("voltage_resolution")

        # The initial voltage of the neurons is uniformly distributed between the reset potential and the reset threshold
        self.__initial_v = np.random.uniform(low=self.__v_res, high=self.__v_thr, size=self.__N) * b2.mV

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
        Simulates the system and stores the recorded data
        """
        list_neuron_models = [
            "noisy_LIF",
        ]
        self.__neuron_model = self.__config.get("neuron_model")
        if self.__neuron_model not in list_neuron_models:
            raise Exception(f"Selected neuron model does not match any model in {list_neuron_models}")

        if self.__neuron_model == "noisy_LIF":
            b2.start_scope()
            N = self.__N
            v_L = self.__v_L
            v_I = self.__v_I
            v_thr = self.__v_thr
            v_res = self.__v_res
            duration = self.duration
            sigma = self.__sigma
            I_dc = self.__I_dc
            g_L = self.__g_L
            tau = self.__tau
            tau_D = self.__tau_D
            tau_R = self.__tau_R
            self.__initial_g
            w = self.__w = self.__sim_constants.get("weights")
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
            G.v = self.__initial_v
            G.g = self.__initial_g

            # Synapses
            S = b2.Synapses(G, G, model='''delta : hertz # synaptic weight''', on_pre='R+=delta')
            S.connect(condition='True')
            S.delta = 1 * b2.hertz

            # Preparing monitors and network
            spike_monitor = b2.SpikeMonitor(G)
            monitor = b2.StateMonitor(G, variables=['v', 'g'], record=self.__recorded_neurons_id)
            mon_syn = b2.StateMonitor(S, variables=True, record=False)
            rate_monitor = b2.PopulationRateMonitor(G)
            net = b2.Network(b2.collect())  # automatically include G and S
            net.add(monitor)  # manually add the monitors

            # Simulation
            print("Running simulation")
            start_time = datetime.datetime.now()
            net.run(self.duration)
            elapsed_time = datetime.datetime.now() - start_time
            print(f"Elapsed time: {elapsed_time.total_seconds()}s")

            self.__v = np.array(monitor.v)
            self.__g = np.array(monitor.g)
            self.__t = pd.Series(monitor.t)
            spikes = spike_monitor.values('t')
            self.__spikes = pd.DataFrame.from_dict(spikes, orient='index').stack()

    @property
    def N(self) -> int:
        return self.__N

    @property
    def duration(self) -> b2.ms:
        return self.__duration

    @property
    def v_res(self) -> b2.mV:
        return self.__v_res

    @property
    def v_thr(self) -> b2.mV:
        return self.__v_thr

    @property
    def v(self):
        return self.__v

    @property
    def g(self):
        return self.__g

    @property
    def t(self):
        return self.__t

    @property
    def spikes(self):
        return self.__spikes





