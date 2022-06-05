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

    def __init__(self, filename="config.json"):
        self.__v = None
        self.__neuron_model = None
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
        self.__v_L = self.__sim_constants.get("v_L")
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
        self.__neuron_model = self.__config.get("neuron_model")
        if self.__neuron_model not in list_neuron_models:
            raise Exception(f"Selected neuron model does not match any valid model. "
                            f"\nValid models: {list_neuron_models}")

        if self.__neuron_model == "noisy_LIF":
            b2.start_scope()
            N = self.__N
            v_L = self.__v_L * b2.mV
            v_I = self.__v_I * b2.mV
            v_thr = self.__v_thr * b2.mV
            v_res = self.__v_res * b2.mV
            duration = self.__duration * b2.ms
            sigma = self.__sigma * b2.mV
            I_dc = self.__I_dc * b2.mV * b2.hertz
            g_L = self.__g_L * b2.hertz
            tau = self.__tau * b2.ms
            tau_D = self.__tau_D * b2.ms
            tau_R = self.__tau_R * b2.ms
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
        G.v = self.__initial_v * b2.mV
        G.g = self.__initial_g * b2.hertz

        # Synapses
        S = b2.Synapses(G, G, model='''delta : hertz # synaptic weight''', on_pre='R+=delta')
        S.connect(condition='True')
        S.delta = 1 * b2.hertz

        # Preparing monitors and network
        spike_monitor = b2.SpikeMonitor(G)
        monitor = b2.StateMonitor(G, variables=self.__recorded_variables, record=self.__recorded_neurons_id)
        mon_syn = b2.StateMonitor(S, variables=True, record=False)
        rate_monitor = b2.PopulationRateMonitor(G)
        net = b2.Network(b2.collect())  # automatically include G and S
        net.add(monitor)  # manually add the monitors

        # Simulation
        print("Running simulation")
        start_time = datetime.datetime.now()
        net.run(self.__duration * b2.ms)
        elapsed_time = datetime.datetime.now() - start_time
        print(f"Elapsed time: {elapsed_time.total_seconds()}s")

        self.__v = np.array(monitor.v)
        self.__g = np.array(monitor.g)
        self.__t = pd.Series(np.array(monitor.t))
        spikes = spike_monitor.values('t')
        spikes = pd.DataFrame.from_dict(spikes, orient='index').stack()
        self.__spikes = spikes.map(lambda time: time*b2.hertz)

    def save_simulation(self):
        with open(self.__sim_results_path, "w") as file:
            pickle.dump(self, file)

    def simulator_to_analyzer(self):
        """
        Method that changes class into NeuralAnalyzer and initializes it
        :return:
        """
        self.__class__ = NeuralAnalyzer
        self._init_NeuralAnalyzer()

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

    @property
    def output_path(self):
        return self.__sim_results_path

    @property
    def initial_v(self):
        return self.__initial_v




class NeuralAnalyzer(NeuralSimulator):
    def __init__(self, filename):
        super().__init__(filename)
        self._init_NeuralAnalyzer()

    def _init_NeuralAnalyzer(self):
        self.__time_steps = self.v.shape[1]

