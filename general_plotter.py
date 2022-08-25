from pathlib import Path
import pickle

if __name__ == '__main__':
    with open("/home/jorge/PycharmProjects/partial_synchro_neurons/simulation_results/results.P", "rb") as f:
        neural_sim = pickle.load(f)
    plotters = [
        (neural_sim.plot_mean_silent, "mean_silent"),
        (neural_sim.plot_mean_spikes, "mean_spikes"),
        (neural_sim.plot_mean_inf, "mean_inf"),
        (neural_sim.plot_mean_inf_zoom, "mean_inf_zoom"),
        (neural_sim.plot_std_silent, "std_silent"),
        (neural_sim.plot_std_spikes, "std_spikes"),
        (neural_sim.plot_g_R_silent,"g_R_silent"),
        (neural_sim.plot_g_R_spikes,"g_R_spikes"),
        (neural_sim.plot_activity_pred,"activity"),
        (neural_sim.plot_voltage_dist_t_0_silence,"voltage_dist_t_0_silence"),
        (neural_sim.plot_qq,"qq"),
]

    figures_path = Path("/home/jorge/PycharmProjects/partial_synchro_neurons/figures")
    subpaths = [
        "mean_silent",
        "mean_spikes",
        "mean_inf",
        "mean_inf_zoom",
        "std_silent",
        "std_spikes",
        "g_R_silent",
        "g_R_spikes",
        "activity",
        "voltage_dist_t_0_silence",
        "qq",
    ]

    for plotter, figure_name in plotters:
        neural_sim.plot(
            figsize=(10, 5),
            cycle=6,
            plotter=plotter,
            path=str(figures_path / (figure_name + ".png")),
            dpi=300,
        )

    print("Finished")

