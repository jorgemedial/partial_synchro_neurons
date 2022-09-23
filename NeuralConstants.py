import numpy as np


class NeuralConstants:
    def __init__(self, neural_constants):
        pass

    def _load_constants(self, neural_constants):
        """
        Reads neural constants (i.e. potentials, conductivities) from neural_constants.json
        """
        self._neural_constants = neural_constants
        self._N = self._neural_constants.get("N_neurons")
        self._v_L = self._neural_constants.get("v_L")
        self._v_I = self._neural_constants.get("v_I")
        self._v_thr = self._neural_constants.get("v_thr")
        self._v_res = self._neural_constants.get("v_res")
        self._duration = self._neural_constants.get("duration")
        self._sigma = self._neural_constants.get("sigma")
        self._I_dc = self._neural_constants.get("I_dc")
        self._g_L = self._neural_constants.get("g_L")
        self._tau = self._neural_constants.get("tau")
        self._tau_D = self._neural_constants.get("tau_D")
        self._tau_R = self._neural_constants.get("tau_R")
        self._w = self._neural_constants.get("weights")
        self._correction = self._neural_constants.get("correction")
        self._initial_g = self._neural_constants.get("initial_g")

        # The initial voltage of the neurons is uniformly distributed between the reset potential and the threshold
        self._initial_v = np.random.uniform(low=self._v_res, high=self._v_thr, size=self._N)
