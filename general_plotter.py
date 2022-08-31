from pathlib import Path
import pickle
from NeuralSimulator import NeuralAnalyzer
import seaborn as sns
import matplotlib.pyplot as plt

if __name__ == '__main__':

    dump = False
    if dump:
        neural_sim: NeuralAnalyzer = NeuralAnalyzer()

        with open("/home/jorge/PycharmProjects/partial_synchro_neurons/simulation_results/results_full_alpha_60.P", "wb") as f:
            pickle.dump(neural_sim, f)
        print("Finished")
    else:
        with open("/home/jorge/PycharmProjects/partial_synchro_neurons/simulation_results/results_full_alpha_55.P", "rb") as f:
            neural_sim = pickle.load(f)
    print("Neuralsim loaded")
    plotters = [
        # (neural_sim.plot_mean_silent, "mean_silent"),
        # (neural_sim.plot_mean_spikes, "mean_spikes"),
        # (neural_sim.plot_mean_zoom, "mean_zoom"),
        # (neural_sim.plot_std_silent, "std_silent"),
        # (neural_sim.plot_std_spikes, "std_spikes"),
        # (neural_sim.plot_std_zoom, "std_zoom"),
        # (neural_sim.plot_g_R_silent,"g_R_silent"),
        # (neural_sim.plot_g_R_spikes, "g_R_spikes"),
        # (neural_sim.plot_activity_pred, "activity"),
        # (neural_sim.plot_voltage_dist_spiking, "methodology/voltage_dist_spiking"),
        # (neural_sim.plot_voltage_dist_onset, "voltage_dist_t_0_silence"),
        # (neural_sim.plot_qq,"qq"),
        # (neural_sim.plot_sample_lag_dynamics, "extension/lag_dynamics"),
        # (neural_sim.plot_onset_times, "onset_times"),
        # (neural_sim.plot_numeric_noise, "methodology/numeric_noise"),
        # (neural_sim.plot_full_sim, "full_sim"),
        (neural_sim.plot_raster_exp, "methodology/raster_exp"),
        (neural_sim.plot_attractor, "extension/attractor"),
    ]

    figures_path = Path("/home/jorge/PycharmProjects/partial_synchro_neurons/figures")
    cycle = [3, 7]
    for plotter, figure_name in plotters:
        path = str(figures_path / (figure_name + ".png"))
        print(f"Plotting {figure_name}")
        sns.set()
        double_hor_plot = []
        double_vert_plot = ["full_sim"]
        ncols = 1
        nrows = 1
        figsize = (10, 5)
        gridspec_kw = None
        if figure_name in double_hor_plot:
            figsize = (10, 5)
            nrows = 2
            gridspec_kw = {'width_ratios': [2, 5]}
            fig, ax = plt.subplots(
                ncols,
                nrows,
                figsize=figsize,
                gridspec_kw=gridspec_kw,
            )
            ax[0] = plotter(
                ax=ax[0],
                cycle=cycle[0]
            )
            ax[1] = plotter(
                ax=ax[1],
                cycle=cycle,
            )
        elif figure_name in double_vert_plot:
            figsize = (10, 5)
            ncols = 2
            gridspec_kw = {'height_ratios': [5, 2]}
            fig, ax = plt.subplots(
                ncols,
                nrows,
                figsize=figsize,
                gridspec_kw=gridspec_kw,
                sharex=False,
            )
            ax = plotter(
                ax=ax,
                cycle=cycle,
            )
        elif figure_name is "extension/lag_dynamics":
            figsize = (10, 5)
            ncols = 2
            gridspec_kw = {'height_ratios': [1, 1]}
            fig, ax = plt.subplots(
                ncols,
                nrows,
                figsize=figsize,
                gridspec_kw=gridspec_kw,
                sharex=True,
            )
            ax[0] = plotter(
                ax=ax[0],
                lagged=True,
            )
            ax[1] = plotter(
                ax=ax[1],
                lagged=False,
            )
        elif figure_name is "methodology/raster_exp":
            figsize = (10, 10)
            ncols = 4
            gridspec_kw = {'height_ratios': [2, 2, 2, 2]}
            fig, ax = plt.subplots(
                ncols,
                nrows,
                figsize=figsize,
                gridspec_kw=gridspec_kw,
                sharex=True,
            )
            ax = plotter(
                ax=ax,
                cycles=[3,5],
            )
        else:
            fig, ax = plt.subplots(
                figsize=figsize
            )
            ax = plotter(
                ax=ax,
                cycle=cycle
            )
        plt.tight_layout()

        if path is not None:
            plt.savefig(path, dpi=300)
        else:
            plt.show()
        plt.close()

    print("Finished")

