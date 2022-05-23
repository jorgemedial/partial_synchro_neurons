import brian2 as b2
import matplotlib.pyplot as plt
import brian2tools as b2t
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
import pandas as pd
from scipy.signal import find_peaks
from scipy.ndimage import gaussian_filter1d
import scipy as sp


class VoltageMap:
    def __init__(self, voltages, time, spikes, duration, v_res, v_thr, v_resolution=500):
        self.voltages = voltages
        self.time = pd.Series(time)
        self.v_res = v_res
        self.v_thr = v_thr
        self.v_resolution = v_resolution
        self.duration = duration
        self.time_steps = voltages.shape[1]
        self.N_neurons = voltages.shape[0]
        self.density = np.empty((v_resolution, self.time_steps))
        self.density_computed = False
        self.spikes = pd.DataFrame.from_dict(spikes, orient="index").stack()
        self.interspike_times = self.spikes.groupby(level=0).diff()
        # Mean and variance
        self.mean = np.mean(self.voltages, axis=0)
        self.std = np.std(self.voltages, axis=0)

    def get_density(self):
        for t in range(self.time_steps):
            self.density[:, t] = np.histogram(
                self.voltages[:, t],
                range=(self.v_res, self.v_thr),
                bins=self.v_resolution,
            )[0]
        self.density_computed = True

    def plot_density(self, t0=0 * b2.second, t_end=None):
        if t_end is None:
            t_end = self.duration

        if not self.density_computed:
            self.get_density()

        fig, ax = plt.subplots(figsize=(20, 8))
        ax.set_xlabel("Time (ms)")
        ax.set_ylabel("Voltage (mV)")
        xgrid = np.linspace(0, self.duration, self.time_steps)
        ygrid = np.linspace(self.v_res, self.v_thr, self.v_resolution)
        v_map = ax.pcolormesh(
            xgrid,
            ygrid,
            np.sqrt(self.density),
            shading="auto",
        )
        fig.colorbar(v_map)
        plt.plot(xgrid, self.mean, label="Mean", color="orange")
        plt.plot(xgrid, self.mean + self.std, label="Upper bound", color="salmon")
        plt.plot(xgrid, self.mean - self.std, label="Lower bound", color="salmon")
        plt.xlim(t0, t_end)
        plt.ylim(self.v_res, self.v_thr)
        plt.legend()
        return v_map

    def plot_rasterplot(self):
        if not self.density_computed:
            self.get_density()

        fig, ax = plt.subplots(figsize=(20, 8))
        ax.set_xlabel("Time (ms)")
        ax.set_ylabel("Neuron Cell")
        xgrid = np.linspace(0, self.duration, self.time_steps)
        ygrid = np.array(range(self.N_neurons))
        v_map = ax.pcolormesh(xgrid, ygrid, self.voltages, shading="auto")
        fig.colorbar(v_map)
        return v_map

    def time_id(self, time):
        time = float(time)

        try:
            return self.time[self.time == time].index[0]
        except:
            return self.time[self.time < time].index[-1]


def get_global_spikes(rate_mon, duration, v_map):
    # Store monitor data
    global_rate = pd.DataFrame(
        {"rate": rate_mon.rate / b2.hertz, "time": rate_mon.t / b2.second},
    )
    # Smooth spike rate to find the maxima
    global_rate["rate_smooth"] = gaussian_filter1d(
        input=global_rate["rate"],
        sigma=25,
    )
    peak_indices = find_peaks(global_rate["rate_smooth"])[0]
    global_spikes = pd.DataFrame({"peak_time": global_rate["time"].iloc[peak_indices]})

    global_spikes = pd.concat(
        [
            pd.DataFrame({global_spikes.columns[0]: [0]}),
            global_spikes,
            pd.DataFrame({global_spikes.columns[0]: [duration] / b2.second}),
        ],
        ignore_index=True,
    )

    global_spikes["next_cycle"] = global_spikes["peak_time"].rolling(2).mean().shift(-1)
    global_spikes["next_cycle"].iat[-1] = global_spikes["peak_time"].iat[-1]
    global_spikes["start_cycle"] = global_spikes["next_cycle"].shift(1)

    for cycle in global_spikes.index[1:-1]:
        cycle_index = v_map.spikes.astype(float).between(
            global_spikes["next_cycle"].iat[cycle - 1],
            global_spikes["next_cycle"].iat[cycle],
        )

        global_spikes.loc[cycle, ["%_spiking_neurons"]] = 100*cycle_index.sum() / v_map.N_neurons
        
        start_time = v_map.spikes.loc[cycle_index].min() # Experimental onset of spiking. First spike
        global_spikes.loc[cycle, ['start_spikes']] = start_time
        try:
            start_time_id = v_map.time_id(start_time)
            global_spikes.loc[cycle, ['mu']] = v_map.mean[start_time_id - 1]
            global_spikes.loc[cycle, ['var']] = v_map.std[start_time_id - 1]**2
        except:
            global_spikes.loc[cycle, ['mu']] = np.nan
            global_spikes.loc[cycle, ['var']] = np.nan
        global_spikes.loc[cycle, ['end_spikes']] = v_map.spikes.loc[cycle_index].max()

       
        # Theoretical onset of spiking

        time_id = v_map.time_id(global_spikes["start_cycle"].iat[cycle])
        time_id_end = v_map.time_id(global_spikes["peak_time"].iat[cycle])
        found = False
        p_th = 0
        while (time_id < time_id_end) & (not found):
            p_th2 = p_th
            mu = v_map.mean[time_id]
            std = v_map.std[time_id]
            Z_th = (mu-v_map.v_thr)/(np.sqrt(2)*std)
            p_th = v_map.N_neurons*(1 + sp.special.erf(Z_th))
            if p_th > 1:
                found = True

            time_id+= 1
        onset_time = v_map.time[time_id]
        global_spikes.at[cycle, "pred_start_spikes"] = onset_time
        global_spikes.at[cycle, "pred_p_th"] = p_th2
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

    return global_spikes, global_rate

    # Store monitor data
    global_rate = pd.DataFrame(
        {"rate": rate_mon.rate / b2.hertz, "time": rate_mon.t / b2.second},
    )
    # Smooth spike rate to find the maxima
    global_rate["rate_smooth"] = gaussian_filter1d(
        input=global_rate["rate"],
        sigma=25,
    )
    peak_indices = find_peaks(global_rate["rate_smooth"])[0]
    global_spikes = pd.DataFrame({"peak_time": global_rate["time"].iloc[peak_indices]})

    global_spikes = pd.concat(
        [
            pd.DataFrame({global_spikes.columns[0]: [0]}),
            global_spikes,
            pd.DataFrame({global_spikes.columns[0]: [duration] / b2.second}),
        ],
        ignore_index=True,
    )

    global_spikes["next_cycle"] = global_spikes["peak_time"].rolling(2).mean().shift(-1)
    global_spikes["next_cycle"].iat[-1] = global_spikes["peak_time"].iat[-1]
    global_spikes["start_cycle"] = global_spikes["next_cycle"].shift(1)

    for cycle in global_spikes.index[1:-1]:
        cycle_index = v_map.spikes.astype(float).between(
            global_spikes["next_cycle"].iat[cycle - 1],
            global_spikes["next_cycle"].iat[cycle],
        )

        global_spikes.loc[cycle, ["%_spiking_neurons"]] = 100*cycle_index.sum() / v_map.N_neurons
        
        start_time = v_map.spikes.loc[cycle_index].min() # Experimental onset of spiking. First spike
        global_spikes.loc[cycle, ['start_spikes']] = start_time
        try:
            start_time_id = v_map.time_id(start_time)
            global_spikes.loc[cycle, ['mu']] = v_map.mean[start_time_id - 1]
            global_spikes.loc[cycle, ['var']] = v_map.std[start_time_id - 1]**2
        except:
            global_spikes.loc[cycle, ['mu']] = np.nan
            global_spikes.loc[cycle, ['var']] = np.nan
        global_spikes.loc[cycle, ['end_spikes']] = v_map.spikes.loc[cycle_index].max()

        # Theoretical onset of spiking. 
        # We need to find at which t we have such mu and sigma such that P[V>V_th]=1
        # First, obtain mu and sigma
        pre_spike_index = v_map.time.astype(float).between(
            global_spikes["start_cycle"].iat[cycle],
            global_spikes["peak_time"].iat[cycle],
        )
        mu = v_map.mean[pre_spike_index]
        std = v_map.std[pre_spike_index]

        # Then, compute P[V>V_th]
        p_vth = v_map.N_neurons*(1 + sp.special.erf((mu-v_map.v_thr)/(np.sqrt(2)*std)))

        #Finally, ontain the t at which P[V>V_th]=1
        onset_index = np.argmin(np.abs(p_vth - 1))
        onset_time = v_map.time[onset_index]

        global_spikes["pred_start_spikes"] = onset_time
        global_spikes = global_spikes[[
            "start_cycle",
            "start_spikes",
            "pred_start_spikes",
            "peak_time",
            "end_spikes",
            "next_cycle",
            "%_spiking_neurons",
            "mu",
            "var",]
        ]

    return global_spikes, global_rat