import numpy as np
import pyuvdata
from calico import cost_function_calculations
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
import multiprocessing


def calculate_per_antenna_cost(
    caldata_obj,
    parallel=False,
    max_processes=40,
    pool=None,
):
    """
    Calculate the contribution of each antenna to the cost function. The cost
    function used is the standard "sky-based" per frequency, per polarization
    cost function evaluated in cost_function_calculations.cost_skycal.

    Parameters
    ----------
    caldata_obj : CalData
    parallel : bool
        Set to True to parallelize with multiprocessing. Default False.
    max_processes : int
        Maximum number of multithreaded processes to use. Applicable only if
        parallel is True and pool is None. If None, uses the multiprocessing
        default. Default 40.
    pool : multiprocessing.pool.Pool or None
        Pool for multiprocessing. If None and parallel=True, a new pool will be
        created for this process and then terminated.

    Returns
    -------
    per_ant_cost_normalized : array of float
        Shape (Nants, N_feed_pols). Encodes the contribution to the cost from
        each antenna and feed, normalized by the number of unflagged baselines.
    """

    if pool is not None:
        parallel = True

    per_ant_cost = np.zeros((caldata_obj.Nants, caldata_obj.N_feed_pols), dtype=float)
    per_ant_baselines = np.zeros(
        (caldata_obj.Nants, caldata_obj.N_feed_pols), dtype=int
    )

    if parallel:
        args_list = []
        caldata_list = caldata_obj.expand_in_frequency()

    for pol_ind in range(caldata_obj.N_feed_pols):

        total_visibilities = np.count_nonzero(
            caldata_obj.visibility_weights[:, :, :, pol_ind]
        )

        # Get total cost
        if parallel:
            for freq_ind in range(caldata_obj.Nfreqs):
                use_gains = caldata_list[freq_ind].gains[:, 0, pol_ind]
                use_gains[np.where(~np.isfinite(use_gains))[0]] = (
                    0.0 + 1j * 0.0
                )  # Remove nans
                if caldata_list[freq_ind].gains_multiply_model:
                    args = (
                        use_gains,
                        caldata_list[freq_ind].data_visibilities[:, :, 0, pol_ind],
                        caldata_list[freq_ind].model_visibilities[:, :, 0, pol_ind],
                        caldata_list[freq_ind].visibility_weights[:, :, 0, pol_ind],
                        caldata_list[freq_ind].ant1_inds,
                        caldata_list[freq_ind].ant2_inds,
                        0.0,
                    )
                else:
                    args = (
                        use_gains,
                        caldata_list[freq_ind].model_visibilities[:, :, 0, pol_ind],
                        caldata_list[freq_ind].data_visibilities[:, :, 0, pol_ind],
                        caldata_list[freq_ind].visibility_weights[:, :, 0, pol_ind],
                        caldata_list[freq_ind].ant1_inds,
                        caldata_list[freq_ind].ant2_inds,
                        0.0,
                    )
                args_list.append(args)
        else:
            total_cost = 0.0
            use_gains = caldata_obj.gains[:, :, pol_ind]
            nan_gains = np.where(~np.isfinite(use_gains))
            use_gains[nan_gains[0], nan_gains[1]] = 0.0 + 1j * 0.0  # Remove nans
            for freq_ind in range(caldata_obj.Nfreqs):
                if caldata_obj.gains_multiply_model:
                    total_cost += cost_function_calculations.cost_skycal(
                        use_gains[:, freq_ind],
                        caldata_obj.data_visibilities[:, :, freq_ind, pol_ind],
                        caldata_obj.model_visibilities[:, :, freq_ind, pol_ind],
                        caldata_obj.visibility_weights[:, :, freq_ind, pol_ind],
                        caldata_obj.ant1_inds,
                        caldata_obj.ant2_inds,
                        0.0,
                    )
                else:
                    total_cost += cost_function_calculations.cost_skycal(
                        use_gains[:, freq_ind],
                        caldata_obj.model_visibilities[:, :, freq_ind, pol_ind],
                        caldata_obj.data_visibilities[:, :, freq_ind, pol_ind],
                        caldata_obj.visibility_weights[:, :, freq_ind, pol_ind],
                        caldata_obj.ant1_inds,
                        caldata_obj.ant2_inds,
                        0.0,
                    )

        # Get per ant cost
        for ant_ind in range(caldata_obj.Nants):
            bl_inds = np.where(
                np.logical_or(
                    caldata_obj.ant1_inds == ant_ind,
                    caldata_obj.ant2_inds == ant_ind,
                )
            )[0]
            ant_excluded_weights = np.copy(
                caldata_obj.visibility_weights[:, :, :, pol_ind]
            )
            ant_excluded_weights[:, bl_inds, :] = (
                0  # Exclude antenna by setting all associated weights to zero
            )
            per_ant_baselines[ant_ind, pol_ind] = total_visibilities - np.count_nonzero(
                ant_excluded_weights
            )
            if parallel:
                for freq_ind in range(caldata_obj.Nfreqs):
                    use_gains = caldata_list[freq_ind].gains[:, 0, pol_ind]
                    use_gains[np.where(~np.isfinite(use_gains))[0]] = (
                        0.0 + 1j * 0.0
                    )  # Remove nans
                    if caldata_list[freq_ind].gains_multiply_model:
                        args = (
                            use_gains,
                            caldata_list[freq_ind].data_visibilities[:, :, 0, pol_ind],
                            caldata_list[freq_ind].model_visibilities[:, :, 0, pol_ind],
                            ant_excluded_weights[:, :, freq_ind],
                            caldata_list[freq_ind].ant1_inds,
                            caldata_list[freq_ind].ant2_inds,
                            0.0,
                        )
                    else:
                        args = (
                            use_gains,
                            caldata_list[freq_ind].model_visibilities[:, :, 0, pol_ind],
                            caldata_list[freq_ind].data_visibilities[:, :, 0, pol_ind],
                            ant_excluded_weights[:, :, freq_ind],
                            caldata_list[freq_ind].ant1_inds,
                            caldata_list[freq_ind].ant2_inds,
                            0.0,
                        )
                    args_list.append(args)
            else:
                per_ant_cost[ant_ind, pol_ind] = total_cost
                use_gains = caldata_obj.gains[:, :, pol_ind]
                nan_gains = np.where(~np.isfinite(use_gains))
                use_gains[nan_gains[0], nan_gains[1]] = 0.0 + 1j * 0.0  # Remove nans
                for freq_ind in range(caldata_obj.Nfreqs):
                    if caldata_obj.gains_multiply_model:
                        per_ant_cost[
                            ant_ind, pol_ind
                        ] -= cost_function_calculations.cost_skycal(
                            use_gains[:, freq_ind],
                            caldata_obj.data_visibilities[:, :, freq_ind, pol_ind],
                            caldata_obj.model_visibilities[:, :, freq_ind, pol_ind],
                            ant_excluded_weights[:, :, freq_ind],
                            caldata_obj.ant1_inds,
                            caldata_obj.ant2_inds,
                            0.0,
                        )
                    else:
                        per_ant_cost[
                            ant_ind, pol_ind
                        ] -= cost_function_calculations.cost_skycal(
                            use_gains[:, freq_ind],
                            caldata_obj.model_visibilities[:, :, freq_ind, pol_ind],
                            caldata_obj.data_visibilities[:, :, freq_ind, pol_ind],
                            ant_excluded_weights[:, :, freq_ind],
                            caldata_obj.ant1_inds,
                            caldata_obj.ant2_inds,
                            0.0,
                        )

    # Run parallelized jobs
    if parallel:
        if pool is None:
            if max_processes is None:
                use_pool = multiprocessing.Pool()
            else:
                use_pool = multiprocessing.Pool(processes=max_processes)
        else:
            use_pool = pool
        result = use_pool.starmap(
            cost_function_calculations.cost_skycal,
            args_list,
        )
        cost_values = np.zeros(
            (caldata_obj.N_feed_pols, caldata_obj.Nants + 1, caldata_obj.Nfreqs),
            dtype=float,
        )
        if pool is None:
            use_pool.terminate()

        list_ind = 0
        for pol_ind in range(caldata_obj.N_feed_pols):
            for ant_ind in range(caldata_obj.Nants + 1):
                for freq_ind in range(caldata_obj.Nfreqs):
                    cost_values[pol_ind, ant_ind, freq_ind] = result[list_ind]
                    list_ind += 1

        total_cost = np.sum(cost_values[:, 0, :], axis=1)
        per_ant_cost = np.transpose(
            total_cost[:, np.newaxis] - np.sum(cost_values[:, 1:, :], axis=2)
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
