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

from ReducedModel import ReducedModel
from NeuralSimulator import NeuralSimulator


class ModelComparison:
    def __init__(self):
        self.constants = NeuralConstants(filename="config.json")
        self.neural_model = NeuralSimulator(filename="config.json")

    def simulate_neural_model(self):
        self.neural_model.simulate()
        self.neural_model.simulate_reduced_model()

    def init_NeuralAnalyser(self):
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


class NeuralConstants:
    def __init__(self, filename):
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


if __name__ == '__main__':
    model_comparison = ModelComparison()
    model_comparison.simulate_neural_model()
    model_comparison.init_NeuralAnalyser()
