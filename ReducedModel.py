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

from NeuralSimulator import NeuralSimulator


class ReducedModel:
    def __init__(self, total_length):
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


    pass

