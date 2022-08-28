from pathlib import Path
import pickle
from NeuralSimulator import NeuralAnalyzer

if __name__ == '__main__':

    dump = False
    if dump:
        neural_sim: NeuralAnalyzer = NeuralAnalyzer()
        with open("/home/jorge/PycharmProjects/partial_synchro_neurons/simulation_results/results.P", "wb") as f:
            pickle.dump(neural_sim, f)
        print("Finished")
    else:
        with open("/home/jorge/PycharmProjects/partial_synchro_neurons/simulation_results/results.P", "rb") as f:
            neural_sim = pickle.load(f)
    print("Neuralsim loaded")
    plotters = [
        (neural_sim.plot_mean_silent, "mean_silent"),
        # (neural_sim.plot_mean_spikes, "mean_spikes"),
        # (neural_sim.plot_mean_inf, "mean_inf"),
        # (neural_sim.plot_mean_inf_zoom, "mean_inf_zoom"),
        # (neural_sim.plot_std_silent, "std_silent"),
        # (neural_sim.plot_std_spikes, "std_spikes"),
        # (neural_sim.plot_std_inf, "std_inf"),
        # (neural_sim.plot_g_R_silent,"g_R_silent"),
        # (neural_sim.plot_g_R_spikes, "g_R_spikes"),
        # (neural_sim.plot_activity_pred, "activity"),
        # (neural_sim.plot_voltage_dist_spiking, "methodology/voltage_dist_spiking"),
        # (neural_sim.plot_voltage_dist_onset, "voltage_dist_t_0_silence"),
        # (neural_sim.plot_qq,"qq"),
        # (neural_sim.plot_sample_lag_dynamics, "extension/lag_dynamics"),
    ]

    figures_path = Path("/home/jorge/PycharmProjects/partial_synchro_neurons/figures")

    for plotter, figure_name in plotters:
        double_hor_plot = []
        ncols = 1
        nrows = 1
        figsize = (10, 5)
        if figure_name in double_hor_plot:
            figsize = (10, 8)
            ncols = 2

        neural_sim.plot(
            figsize=figsize,
            cycle=6,
            plotter=plotter,
            path=str(figures_path / (figure_name + ".png")),
            dpi=300,
            ncols=ncols,
            nrows=nrows,
        )

    print("Finished")

