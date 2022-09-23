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
    def __init__(self, filename="neural_constants.json"):
        self._init_config(filename=filename)
        self.neural_model = NeuralSimulator(neural_constants=self.__config)


    def _init_config(self, filename):
        self.__config_filename = filename
        self.__dirname = os.path.dirname(__file__)
        self.__sim_results_path = os.path.join(self.__dirname, "simulation_results/", "results.P")
        self.__config_path = os.path.join(self.__dirname, "config/", self.__config_filename)
        self.__config = json.load(open(self.__config_path, "r"))

    def simulate_neural_model(self):
        self.neural_model._init_simulation()
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


if __name__ == '__main__':
    model_comparison = ModelComparison()
    model_comparison.simulate_neural_model()
