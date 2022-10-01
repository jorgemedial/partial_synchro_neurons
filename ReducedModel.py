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
from NeuralSimulator import NeuralSimulator
from NeuralConstants import NeuralConstants


class ReducedModel(NeuralConstants):
    def __init__(self, neural_sim: NeuralSimulator, neural_constants: dict):
        super().__init__(neural_constants)
        self.neural_sim = neural_sim

        total_length = len(self.neural_sim.t)
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

    def get_global_spikes(self):
        """

        :return:
        """
        peak_indices = find_peaks(self.neural_sim._global_rate["rate_smooth"])[0]
        global_spikes = pd.DataFrame({"peak_time": self.neural_sim._global_rate["time"].iloc[peak_indices]})

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
            cycle_index = self.neural_sim._spikes["time"].astype(float).between(
                global_spikes["next_cycle"].iat[cycle - 1],
                global_spikes["next_cycle"].iat[cycle],
            )

            global_spikes.loc[cycle, ["%_spiking_neurons"]] = 100 * cycle_index.sum() / self.N
            self.neural_sim._spikes.loc[cycle_index, "cycle"] = int(cycle)
            # global_spikes.loc[cycle, ["spiking_neurons"]] = self.spikes.loc[cycle_index].index.get_level_values(0))

            # Experimental onset of spiking. First spike
            start_time = self.neural_sim._spikes["time"].loc[cycle_index].min()
            global_spikes.loc[cycle, ['start_spikes']] = start_time

            # Experimental end of spiking. Last spike
            global_spikes.loc[cycle, ['end_spikes']] = self.neural_sim._spikes["time"].loc[cycle_index].max() + self.t[1]

            # Get mean and std of the start time
            try:
                start_time_id = self.neural_sim.time_id(start_time)
                global_spikes.loc[cycle, ['mu']] = self.neural_sim.mean[start_time_id - 1]
                global_spikes.loc[cycle, ['var']] = self.neural_sim.std[start_time_id - 1] ** 2
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
        onset_time_id = self.neural_sim.time_id(global_spikes["start_cycle"].iat[cycle])
        time_id_end = self.neural_sim.time_id(global_spikes["peak_time"].iat[cycle])
        found = False
        p_th = 0
        while (onset_time_id < time_id_end) & (not found):
            mu = self.neural_sim._mean_v[onset_time_id]
            std = self.neural_sim._std_v[onset_time_id]
            Z_th = (mu - self.v_thr) / (np.sqrt(2) * std)
            p_th = self.N * (1 + sp.special.erf(Z_th))/2
            if p_th > 1:
                found = True
            else:
                onset_time_id += 1

        onset_time = self.neural_sim.t[onset_time_id-1]
        global_spikes.at[cycle, "pred_start_spikes"] = onset_time
        global_spikes.at[cycle, "pred_p_th"] = p_th

    def theo_onset_inf(self, global_spikes, cycle):
        onset_time_id = self.neural_sim.time_id(global_spikes["start_cycle"].iat[cycle])
        time_id_end = self.neural_sim.time_id(global_spikes["peak_time"].iat[cycle])
        found = False
        p_th = 0
        R_0 = self.neural_sim.R_t[onset_time_id]
        g_0 = self.neural_sim.g_t[onset_time_id]
        t_0 = self.neural_sim.t[onset_time_id]
        while (onset_time_id < time_id_end) & (not found):
            t = self.neural_sim.t[onset_time_id]
            g = self.g_compute(t, t_0, R_0, g_0)
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

        onset_time = self.neural_sim.t[onset_time_id]
        global_spikes.at[cycle, "pred_start_spikes_inf"] = onset_time + 1/self.a(self.g_compute(t-1/self.a(g), t_0, R_0, g_0))
        global_spikes.at[cycle, "pred_p_th_inf"] = p_th

    def simulate_silent_phase(self, ids_t_silent):
        t = self.neural_sim.t[ids_t_silent].to_numpy()
        id_t_0 = ids_t_silent[0]
        t_0 = self.neural_sim.t[ids_t_silent[0]]

        g = self.g_compute(t, t_0, self.neural_sim.R_t[id_t_0], self.neural_sim.g_t[id_t_0])
        a = self.a(g)
        time_ids = np.vectorize(self.neural_sim.time_id)
        lag_t_id = time_ids(t - 1 / a)
        g_lag = self.neural_sim.g_t[lag_t_id]
        a_lag = self.a(g_lag)
        b_lag = self.b(g_lag)
        R = self.neural_sim.R_t[id_t_0] * np.exp(-(t - t_0) / self._tau_R)

        self._g_f_sim[ids_t_silent] = g
        self._R_f_sim[ids_t_silent] = R
        self._f_sim[ids_t_silent] = 0
        self._a_lag[ids_t_silent] = a_lag
        self._b_lag[ids_t_silent] = b_lag
        self._mu_inf[ids_t_silent] = b_lag / a_lag
        self._var_inf[ids_t_silent] = self._sigma ** 2 / (2 * a)

    def simulate_reduced_model(self):
        for cycle in self.neural_sim.cycles:
            self.simulate_cycle(cycle)
        self.simulate_spiking_phase()
        self.simulate_full_dynamics()

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
        t = self.neural_sim.t[ids_t_silent].to_numpy()
        id_t_0 = ids_t_silent[0]
        t_0 = self.neural_sim.t[ids_t_silent[0]]

        g = self.g_compute(t, t_0, self.neural_sim.R_t[id_t_0], self.neural_sim.g_t[id_t_0])
        a = self.a(g)
        time_ids = np.vectorize(self.neural_sim.time_id)
        lag_t_id = time_ids(t-1/a)
        g_lag = self.neural_sim.g_t[lag_t_id]
        a_lag = self.a(g_lag)
        b_lag = self.b(g_lag)
        R = self.neural_sim.R_t[id_t_0]*np.exp(-(t-t_0)/self._tau_R)

        self._g_f_sim[ids_t_silent] = g
        self._R_f_sim[ids_t_silent] = R
        self._f_sim[ids_t_silent] = 0
        self._a_lag[ids_t_silent] = a_lag
        self._b_lag[ids_t_silent] = b_lag
        self._mu_inf[ids_t_silent] = b_lag/a_lag
        self._var_inf[ids_t_silent] = self._sigma ** 2 / (2 * a)

        def d_silent_dt_g_sim(t, y):
            mu = y[0]
            var = y[1]
            g_t = self.g_compute(t, t_0, R_0=self.neural_sim.R_t[id_t_0], g_0=self.neural_sim.g_t[id_t_0])
            a_t = self.a(g_t)
            b_t = self.b(g_t)
            return [
                b_t - a_t*mu,
                -2*a_t*var + self._sigma**2
            ]

        def d_silent_dt_g_exp(t, y):
            mu = y[0]
            var = y[1]
            g_t = self.neural_sim.g_t[self.neural_sim.time_id(t)]
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
                self.neural_sim._mean_silent[id_t_0],
                self.neural_sim._std_silent[id_t_0] ** 2,
            ),
            method='RK23',
            t_eval=t,
        )

        solver_g_exp = solve_ivp(
            fun=d_silent_dt_g_exp,
            t_span=(t[0], t[-1]),
            y0=(
                self.neural_sim._mean_silent[id_t_0],
                self.neural_sim._std_silent[id_t_0] ** 2,
            ),
            method='RK23',
            t_eval=t,
        )

        solver_spiked = solve_ivp(
            fun=d_silent_dt_g_sim,
            t_span=(t[0], t[-1]),
            y0=(
                self.neural_sim._mean_spike[id_t_0],
                self.neural_sim._std_spike[id_t_0] ** 2,
            ),
            method='RK23',
            t_eval=t,
        )
        self._mu_sim_g_sim[ids_t_silent] = solver_g_sim.y[0]
        self._var_sim_g_sim[ids_t_silent] = solver_g_sim.y[1]
        self._mu_sim_g_exp[ids_t_silent] = solver_g_exp.y[0]
        self._var_sim_g_exp[ids_t_silent] = solver_g_exp.y[1]
        self._mu_sim_spiked[ids_t_silent] = solver_spiked.y[0]
        self._var_spiked[ids_t_silent] = solver_spiked.y[1]

    def simulate_full_dynamics(self):
        t_start_spikes = self.neural_sim._global_spikes.loc[self.neural_sim.cycles[0], 'start_spikes']
        t_end_cycle = self.neural_sim._global_spikes.loc[self.neural_sim.cycles[-1], 'next_cycle']

        self.ids_t_full = self.neural_sim.time_id([t_start_spikes, t_end_cycle])
        self.t_full = self.neural_sim.t[self.ids_t_full].to_numpy()
        d_spiking_dt_f_cycle = self.d_spiking_dt_generator(mode="full_sim")
        solver_f_cycle = solve_ivp(
            fun=d_spiking_dt_f_cycle,
            t_span=(self.t_full[0], self.t_full[-1]),
            y0=(
                self.neural_sim._mean_silent[self.ids_t_full[0]],
                self.neural_sim._std_silent[self.ids_t_full[0]] ** 2,
                self.neural_sim.g_t[self.ids_t_full[0]],
                self.neural_sim.R_t[self.ids_t_full[0]],
            ),
            method='RK23',
            # Use the same time steps as the brian simulation
            t_eval=self.t_full
        )
        self._mu_f_cycle[self.ids_t_full] = solver_f_cycle.y[0]
        self._var_f_cycle[self.ids_t_full] = solver_f_cycle.y[1]
        self._g_f_cycle[self.ids_t_full] = solver_f_cycle.y[2]
        self._R_f_cycle[self.ids_t_full] = solver_f_cycle.y[3]
        a = self.a(solver_f_cycle.y[2])
        b = self.b(solver_f_cycle.y[2])
        self._f_cycle[self.ids_t_full] = self.get_activity_time_range(b, a, solver_f_cycle.y[0],
                                                                      np.sqrt(solver_f_cycle.y[1]))

    def simulate_spiking_phase(self):
        t_start_spikes = self.neural_sim._global_spikes.loc[self.neural_sim.cycles[0], 'start_spikes']
        t_end_cycle = self.neural_sim._global_spikes.loc[self.neural_sim.cycles[-1], 'next_cycle']
        ids_t_spiking = self.neural_sim.time_id([t_start_spikes, t_end_cycle])
        id_t_start_spikes = ids_t_spiking[0]
        t_spiking = list(self.neural_sim.t[ids_t_spiking])
        d_spiking_dt_R_exp = self.d_spiking_dt_generator(mode="R_exp")
        d_spiking_dt_f_sim = self.d_spiking_dt_generator(mode="f_sim")

        # solver_f_cycle = solve_ivp(
        #     fun=d_spiking_dt_f_cycle,
        #     t_span=(t_spiking[0], t_spiking[-1]),
        #     y0=(
        #         self._mean_silent[id_t_start_spikes],
        #         self._std_silent[id_t_start_spikes] ** 2,
        #         self.g_t[id_t_start_spikes],
        #         self.R_t[id_t_start_spikes],
        #     ),
        #     method='RK23',
        #     # Use the same time steps as the brian simulation
        #     t_eval=t_spiking
        # )
        #
        # self._mu_f_cycle[ids_t_spiking] = solver_f_cycle.y[0]
        # self._var_f_cycle[ids_t_spiking] = solver_f_cycle.y[1]
        # self._g_f_cycle[ids_t_spiking] = solver_f_cycle.y[2]
        # self._R_f_cycle[ids_t_spiking] = solver_f_cycle.y[3]
        # a = self.a(solver_f_cycle.y[2])
        # b = self.b(solver_f_cycle.y[2])
        # self._f_cycle[ids_t_spiking] = self.get_activity_time_range(b, a, solver_f_cycle.y[0], np.sqrt(solver_f_cycle.y[1]))

        solver_f_sim = solve_ivp(
            fun=d_spiking_dt_f_sim,
            t_span=(t_spiking[0], t_spiking[-1]),
            y0=(
                self.neural_sim._mean_silent[id_t_start_spikes],
                self.neural_sim._std_silent[id_t_start_spikes] ** 2,
                self.neural_sim.g_t[id_t_start_spikes],
                self.neural_sim.R_t[id_t_start_spikes],
            ),
            method='RK23',
            # Use the same time steps as the brian simulation
            t_eval=t_spiking
        )

        self._mu_f_sim[ids_t_spiking] = solver_f_sim.y[0]
        self._var_f_sim[ids_t_spiking] = solver_f_sim.y[1]
        self._g_f_sim[ids_t_spiking] = solver_f_sim.y[2]
        self._R_f_sim[ids_t_spiking] = solver_f_sim.y[3]
        g_exp = self.neural_sim.g_t[ids_t_spiking]
        b_exp = self.b(g_exp)
        a_exp = self.a(g_exp)
        mu_exp = self.neural_sim._mean_silent[ids_t_spiking]
        std_exp = self.neural_sim._std_silent[ids_t_spiking]
        self._f_sim[ids_t_spiking] = self.get_activity_time_range(b_exp, a_exp, mu_exp, std_exp)

        solver_R_exp = solve_ivp(
            fun=d_spiking_dt_R_exp,
            t_span=(t_spiking[0], t_spiking[-1]),
            y0=(
                self.neural_sim._mean_silent[id_t_start_spikes],
                self.neural_sim._std_silent[id_t_start_spikes] ** 2,
                self.neural_sim.g_t[id_t_start_spikes],
            ),
            method='RK23',
            # Use the same time steps as the brian simulation
            t_eval=t_spiking
        )

        self._mu_R_exp[ids_t_spiking] = solver_R_exp.y[0]
        self._var_R_exp[ids_t_spiking] = solver_R_exp.y[1]
        self._g_R_exp[ids_t_spiking] = solver_R_exp.y[2]

    def simulate_cycle(self, cycle):
        t_start_spikes = self.neural_sim._global_spikes.loc[cycle, 'start_spikes']
        t_end_spikes = self.neural_sim._global_spikes.loc[cycle, 'end_spikes']
        t_end_cycle = self.neural_sim._global_spikes.loc[cycle, 'next_cycle']

        id_t_start_spikes = self.neural_sim.time_id(t_start_spikes)
        id_t_end_spikes = self.neural_sim.time_id(t_end_spikes)
        id_t_end_cycle = self.neural_sim.time_id(t_end_cycle)

        id_t_spikes = np.arange(id_t_start_spikes, id_t_end_cycle)
        id_t_silent = np.arange(id_t_end_spikes, id_t_end_cycle)

        self.simulate_silent_phase(id_t_silent)

    def d_spiking_dt_generator(self, mode):
        """
        Generates a function d_spiking_dt that returns the derivative of mu in the spiking stage
        """
        if mode == "f_sim":
            def d_spiking_dt(t, y):
                mu = y[0]
                var = y[1]
                g = y[2]
                R = y[3]
                a = self._g_L + g * self._w
                b = self._g_L * self._v_L + self._I_dc + g * self._v_I * self._w

                id_t = self.neural_sim.time_id(t)
                g_exp = self.neural_sim.g_t[id_t]
                b_exp = self.b(g_exp)
                a_exp = self.a(g_exp)
                mu_exp = self.neural_sim._mean_silent[id_t]
                std_exp = self.neural_sim._std_silent[id_t]
                f = self.N * self.get_activity(b_exp, a_exp, mu_exp, std_exp)
                return [
                    -a * mu + b,  # d_mu/dt
                    2 * var * (-a) + self._sigma ** 2,  # d_var/dt
                    (R - g) / self._tau_D,  # dg/dt
                    f - R / self._tau_R,  # dR/dt
                ]

        elif mode == "R_exp":
            def d_spiking_dt(t, y):
                mu = y[0]
                var = y[1]
                g = y[2]
                R = self.neural_sim.R_t[self.neural_sim.time_id(t)]
                a = self._g_L + g * self._w
                b = self._g_L * self._v_L + self._I_dc + g * self._v_I * self._w
                return [
                    -a * mu + b,  # d_mu/dt
                    2*var*(-a) + self._sigma**2,  # d_var/dt
                    (R - g)/self._tau_D,  # dg/dt
                ]
        elif mode == "g_exp":
            def d_spiking_dt(t, y):
                mu = y[0]
                var = y[1]
                g = self.neural_sim.g_t[self.neural_sim.time_id(t)]
                a = self._g_L + g * self._w
                b = self._g_L * self._v_L + self._I_dc + g * self._v_I * self._w
                return [
                    -a * mu + b,  # d_mu/dt
                    2 * var * (-a) + self._sigma ** 2,  # d_var/dt
                ]
        elif mode == "full_sim":
            def d_spiking_dt(t, y):
                mu = y[0]
                var = y[1]
                g = y[2]
                R = y[3]
                a = self._g_L + g * self._w
                b = self._g_L * self._v_L + self._I_dc + g * self._v_I * self._w
                f = self.N * self.get_activity(b, a, mu, var**0.5)

                return [
                    -a * mu + b,  # d_mu/dt
                    2 * var * (-a) + self._sigma ** 2,  # d_var/dt
                    (R - g) / self._tau_D,  # dg/dt
                    f - R / self._tau_R,  # dR/dt
                ]
        else:
            raise Exception("No valid mode")

        return d_spiking_dt

    def get_activity_time_range(self, b_t, a_t, mu_t, std_t):
        get_activity = np.vectorize(self.get_activity)
        return get_activity(b_t, a_t, mu_t, std_t)

    def get_activity(self, b, a, mu, std):
        """
        Gets the activity f(t) depending on mu and std
        """
        if mu > self.v_thr - 0.0025:
            density_thr = np.exp(-0.5*((self._v_thr-mu)/std)**2)/(std*np.sqrt(2*np.pi))
            act = density_thr * self.mean_dot_v(a, b, mu) #/ density_thr
        else:
            act = 0
        return act

    def mean_dot_v(self, a, b, mu):
        def p_dot_V_dot_V(dot_V):
            return dot_V * np.exp(-0.5 * ((dot_V-b+a*self._v_thr) / (self._sigma * self._correction)) ** 2) / (np.sqrt(2 * np.pi) * self._sigma * self._correction)

        return quad(p_dot_V_dot_V, 0, np.inf)[0]

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

    def g_L(self):
        return self._g_L




