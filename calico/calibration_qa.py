import numpy as np
import pyuvdata
from calico import cost_function_calculations
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
import multiprocessing
import sys


def calculate_per_antenna_cost(caldata_obj):
    """
    Calculate the contribution of each antenna to the cost function. The cost
    function used is the standard "sky-based" per frequency, per polarization
    cost function evaluated in cost_function_calculations.cost_skycal.

    Parameters
    ----------
    caldata_obj : CalData

    Returns
    -------
    per_ant_cost_normalized : array of float
        Shape (Nants, N_feed_pols). Encodes the contribution to the cost from
        each antenna and feed, normalized by the number of unflagged baselines.
    """

    per_ant_cost = np.zeros((caldata_obj.Nants, caldata_obj.N_feed_pols), dtype=float)
    per_ant_baselines = np.zeros(
        (caldata_obj.Nants, caldata_obj.N_feed_pols), dtype=int
    )

    for ant_ind in range(caldata_obj.Nants):
        bl_inds = np.where(
            np.logical_or(
                caldata_obj.ant1_inds != ant_ind,
                caldata_obj.ant2_inds != ant_ind,
            )
        )[0]
        if len(bl_inds) == 0:
            per_ant_cost[ant_ind, :] = 0
        else:
            use_model_vis = caldata_obj.model_visibilities[:, bl_inds, :, :]
            use_data_vis = caldata_obj.data_visibilities[:, bl_inds, :, :]
            use_weights = caldata_obj.visibility_weights[:, bl_inds, :, :]
            use_ant1_inds = caldata_obj.ant1_inds[bl_inds]
            use_ant2_inds = caldata_obj.ant2_inds[bl_inds]
            per_ant_baselines[ant_ind, :] = len(bl_inds)
            for pol_ind in range(caldata_obj.N_feed_pols):
                vis_pol_ind = np.where(
                    caldata_obj.vis_polarization_array
                    == caldata_obj.feed_polarization_array[pol_ind]
                )[0][0]
                if caldata_obj.gains_multiply_model:
                    per_ant_cost[ant_ind, pol_ind] = (
                        cost_function_calculations.cost_skycal(
                            caldata_obj.gains[:, :, [pol_ind]],
                            use_model_vis[:, :, :, [vis_pol_ind]],
                            use_data_vis[:, :, :, [vis_pol_ind]],
                            use_weights[:, :, :, [vis_pol_ind]],
                            use_ant1_inds,
                            use_ant2_inds,
                            caldata_obj.lambda_val,
                        )
                    )
                else:
                    per_ant_cost[ant_ind, pol_ind] = (
                        cost_function_calculations.cost_skycal(
                            caldata_obj.gains[:, :, [pol_ind]],
                            use_data_vis[:, :, :, [vis_pol_ind]],
                            use_model_vis[:, :, :, [vis_pol_ind]],
                            use_weights[:, :, :, [vis_pol_ind]],
                            use_ant1_inds,
                            use_ant2_inds,
                            caldata_obj.lambda_val,
                        )
                    )

    per_ant_cost_normalized = np.abs(per_ant_cost / per_ant_baselines)  # Normalize
    return per_ant_cost_normalized


def plot_per_ant_cost(per_ant_cost, antenna_names, plot_output_dir, plot_prefix=""):
    """
    Plot the per-antenna cost.

    Parameters
    ----------
    per_ant_cost : array of float
        Shape (Nants, N_feed_pols). Encodes the contribution to the cost from
        each antenna and feed, normalized by the number of unflagged baselines.
    antenna_names : array of str
        Shape (Nants,). Ordering matches the ordering of the per_ant_cost.
    plot_output_dir : str
        Path to the directory where the plots will be saved.
    plot_prefix : str
        Optional string to be appended to the start of the file names.
    """

    # Format antenna names
    sort_inds = np.argsort(antenna_names)
    ant_names_sorted = antenna_names[sort_inds]
    per_ant_cost_sorted = per_ant_cost[sort_inds, :]
    ant_nums = np.array([int(name[3:]) for name in ant_names_sorted])

    # Parse strings
    use_plot_prefix = plot_prefix
    if len(plot_prefix) > 0:
        if not use_plot_prefix.endswith("_"):
            use_plot_prefix = f"{use_plot_prefix}_"
    use_plot_output_dir = plot_output_dir
    if plot_output_dir.endswith("/"):
        use_plot_output_dir = use_plot_output_dir[:-1]

    # Plot style parameters
    colors = ["tab:blue", "tab:orange"]
    linewidth = 0
    markersize = 0.5
    legend_elements = [
        Line2D(
            [0],
            [0],
            color=colors[0],
            marker="o",
            lw=linewidth,
            markersize=markersize,
            label="X",
        ),
        Line2D(
            [0],
            [0],
            color=colors[1],
            marker="o",
            lw=linewidth,
            markersize=markersize,
            label="Y",
        ),
    ]

    fig, ax = plt.subplots()
    for pol_ind in range(np.shape(per_ant_cost_sorted)[1]):
        ax.plot(
            ant_nums,
            per_ant_cost_sorted[:, pol_ind],
            "-o",
            linewidth=linewidth,
            markersize=markersize,
            label=(["X", "Y"])[pol_ind],
            color=colors[pol_ind],
        )
    ax.set_ylim([0, np.nanmax(per_ant_cost_sorted[np.isfinite(per_ant_cost_sorted)])])
    ax.set_xlim([0, np.max(ant_nums)])
    ax.set_xlabel("Antenna Name")
    ax.set_ylabel("Per Antenna Cost Contribution")
    plt.legend(
        handles=legend_elements,
        frameon=False,
    )
    plt.tight_layout()
    plt.savefig(
        f"{use_plot_output_dir}/{use_plot_prefix}per_ant_cost.png",
        dpi=600,
    )
    plt.close()


def plot_gains(
    cal, plot_output_dir, plot_prefix="", plot_reciprocal=False, ymin=0, ymax=None
):
    """
    Plot gain values. Creates two set of plots for each the gain amplitudes and
    phases. Each figure contains 12 panel, each corresponding to one antenna.
    The feed polarizations are overplotted in each panel.

    Parameters
    ----------
    cal : UVCal object or str
        pyuvdata UVCal object or path to a calfits file.
    plot_output_dir : str
        Path to the directory where the plots will be saved.
    plot_prefix : str
        Optional string to be appended to the start of the file names.
    plot_reciprocal : bool
        Plot 1/gains.
    ymin : float
        Minimum of the gain amplitude y-axis. Default 0.
    ymax : float
        Maximum of the gain amplitude y-axis. Default is the maximum gain amplitude.
    """

    # Read data
    if isinstance(cal, str):
        cal_obj = pyuvdata.UVCal()
        cal_obj.read_calfits(cal)
        cal = cal_obj

    # Parse strings
    use_plot_prefix = plot_prefix
    if len(plot_prefix) > 0:
        if not use_plot_prefix.endswith("_"):
            use_plot_prefix = f"{use_plot_prefix}_"
    use_plot_output_dir = plot_output_dir
    if plot_output_dir.endswith("/"):
        use_plot_output_dir = use_plot_output_dir[:-1]

    # Plot style parameters
    colors = ["tab:blue", "tab:orange"]
    linewidth = 0.2
    markersize = 0.5
    legend_elements = [
        Line2D(
            [0],
            [0],
            color=colors[0],
            marker="o",
            lw=linewidth,
            markersize=markersize,
            label="X",
        ),
        Line2D(
            [0],
            [0],
            color=colors[1],
            marker="o",
            lw=linewidth,
            markersize=markersize,
            label="Y",
        ),
    ]

    ant_names = np.sort(cal.antenna_names)
    freq_axis_mhz = cal.freq_array.flatten() / 1e6

    # Apply flags
    cal.gain_array[np.where(cal.flag_array)] = np.nan + 1j * np.nan

    if cal.gain_array.ndim == 5:
        cal.gain_array = cal.gain_array[:, 0, :, :, :]

    if plot_reciprocal:
        cal.gain_array = 1.0 / cal.gain_array

    # Plot amplitudes
    if ymax is None:
        ymax = np.nanmean(np.abs(cal.gain_array)) + 3 * np.nanstd(
            np.abs(cal.gain_array)
        )
    y_range = [ymin, ymax]
    x_range = [np.min(freq_axis_mhz), np.max(freq_axis_mhz)]
    subplot_ind = 0
    plot_ind = 1
    for name in ant_names:
        if subplot_ind == 0:
            fig, ax = plt.subplots(
                nrows=3, ncols=4, figsize=(10, 8), sharex=True, sharey=True
            )
        ant_ind = np.where(np.array(cal.antenna_names) == name)[0][0]
        if np.isnan(np.nanmean(cal.gain_array[ant_ind, :, 0, :])):  # All gains flagged
            ax.flat[subplot_ind].text(
                np.mean(x_range),
                np.mean(y_range),
                "ALL FLAGGED",
                horizontalalignment="center",
                verticalalignment="center",
                color="red",
            )
        for pol_ind in range(cal.Njones):
            ax.flat[subplot_ind].plot(
                freq_axis_mhz,
                np.abs(cal.gain_array[ant_ind, :, 0, pol_ind]),
                "-o",
                linewidth=linewidth,
                markersize=markersize,
                label=(["X", "Y"])[pol_ind],
                color=colors[pol_ind],
            )
        ax.flat[subplot_ind].set_ylim(y_range)
        ax.flat[subplot_ind].set_xlim(x_range)
        ax.flat[subplot_ind].set_title(name)
        subplot_ind += 1
        if subplot_ind == len(ax.flat) or name == ant_names[-1]:
            fig.supxlabel("Frequency (MHz)")
            fig.supylabel("Gain Amplitude")
            plt.legend(
                handles=legend_elements,
                bbox_to_anchor=(1.04, 0),
                loc="lower left",
                frameon=False,
            )
            plt.tight_layout()
            plt.savefig(
                f"{use_plot_output_dir}/{use_plot_prefix}gain_amp_{plot_ind:02d}.png",
                dpi=600,
            )
            plt.close()
            subplot_ind = 0
            plot_ind += 1

    # Plot phases
    subplot_ind = 0
    plot_ind = 1
    for name in ant_names:
        if subplot_ind == 0:
            fig, ax = plt.subplots(
                nrows=3, ncols=4, figsize=(10, 8), sharex=True, sharey=True
            )
        ant_ind = np.where(np.array(cal.antenna_names) == name)[0][0]
        if np.isnan(np.nanmean(cal.gain_array[ant_ind, :, 0, :])):  # All gains flagged
            ax.flat[subplot_ind].text(
                np.mean(x_range),
                0,
                "ALL FLAGGED",
                horizontalalignment="center",
                verticalalignment="center",
                color="red",
            )
        for pol_ind in range(cal.Njones):
            ax.flat[subplot_ind].plot(
                freq_axis_mhz,
                np.angle(cal.gain_array[ant_ind, :, 0, pol_ind]),
                "-o",
                linewidth=linewidth,
                markersize=markersize,
                label=(["X", "Y"])[pol_ind],
                color=colors[pol_ind],
            )
        ax.flat[subplot_ind].set_ylim([-np.pi, np.pi])
        ax.flat[subplot_ind].set_xlim(x_range)
        ax.flat[subplot_ind].set_title(name)
        subplot_ind += 1
        if subplot_ind == len(ax.flat) or name == ant_names[-1]:
            fig.supxlabel("Frequency (MHz)")
            fig.supylabel("Gain Phase (rad.)")
            plt.legend(
                handles=legend_elements,
                bbox_to_anchor=(1.04, 0),
                loc="lower left",
                frameon=False,
            )
            plt.tight_layout()
            plt.savefig(
                f"{use_plot_output_dir}/{use_plot_prefix}gain_phase_{plot_ind:02d}.png",
                dpi=600,
            )
            plt.close()
            subplot_ind = 0
            plot_ind += 1
