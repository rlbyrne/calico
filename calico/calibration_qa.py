import numpy as np
import pyuvdata
from calico import cost_function_calculations
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
import multiprocessing
import sys
from numpy.typing import NDArray


def calculate_per_antenna_cost(caldata_obj) -> NDArray[np.floating]:
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


def plot_per_ant_cost(
    per_ant_cost: NDArray[np.floating],
    antenna_names: NDArray[str],
    plot_output_dir: str,
    plot_prefix: str = "",
) -> None:
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


def get_cal_data(cal: pyuvdata.UVCal, zero_mean_phase: bool = False) -> pyuvdata.UVCal:
    """
    Read and format calibration data.

    Parameters
    ----------
    cal : UVCal object, str, or list
        pyuvdata UVCal object, path to a .calfits file, or path to a CASA .bcal file.
        Alternatively, list containing UVCal objects or paths. If a list is provided,
        the elements will be concatenated across frequency.
    zero_mean_phase : bool
        If True, forces the mean phase of the gains to be zero. Default False.

    Returns
    -------
    cal : UVCal object
    """

    if type(cal) is str:
        cal_obj = pyuvdata.UVCal()
        if cal.endswith("calfits"):
            cal_obj.read_calfits(cal)
        elif cal.endswith("bcal"):
            cal_obj.read_ms_cal(cal)
        cal = cal_obj
    elif type(cal) is list:
        for subband_ind, cal_subband in enumerate(cal):
            if type(cal_subband) is str:
                new_cal = pyuvdata.UVCal()
                if cal_subband.endswith("calfits"):
                    new_cal.read_calfits(cal_subband)
                elif cal_subband.endswith("bcal"):
                    new_cal.read_ms_cal(cal_subband)
            else:
                new_cal = cal_subband
            if subband_ind == 0:
                cal_concatenated = new_cal
            else:
                if (
                    np.max(np.abs(cal_concatenated.time_array - new_cal.time_array))
                    != 0
                ):  # Force time arrays to be the same
                    if (
                        np.max(np.abs(cal_concatenated.time_array - new_cal.time_array))
                        > 1e-5
                    ):
                        print("ERROR: time_array values are not close.")
                    elif (
                        np.max(np.abs(cal_concatenated.lst_array - new_cal.lst_array))
                        > 1e-5
                    ):
                        print("ERROR: lst_array values are not close.")
                    else:
                        new_cal.time_array = cal_concatenated.time_array
                        new_cal.lst_array = cal_concatenated.lst_array
                cal_concatenated.fast_concat(new_cal, "freq", inplace=True)
        cal = cal_concatenated

    if zero_mean_phase:
        mean_phase = np.nanmean(np.angle(cal.gain_array), axis=(0, 2))
        cal.gain_array *= np.exp(-1j * mean_phase[np.newaxis, :, np.newaxis, :])

    return cal


def plot_gains(
    cal: pyuvdata.UVCal,
    cal2: pyuvdata.UVCal = None,
    plot_output_dir: str | None = None,
    cal_name: str = "",
    plot_reciprocal: bool = False,
    ymin: float | None = 0,
    ymax: float | None = None,
    zero_mean_phase: bool = False,
    savefig: bool = True,
) -> None:
    """
    Plot gain values. Creates two set of plots for each the gain amplitudes and
    phases. Each figure contains 12 panel, each corresponding to one antenna.
    The feed polarizations are overplotted in each panel.

    Parameters
    ----------
    cal : UVCal object, str, or list
        pyuvdata UVCal object, path to a .calfits file, or path to a CASA .bcal file.
        Alternatively, list containing UVCal objects or paths. If a list is provided,
        the elements will be concatenated across frequency.
    cal2 : None or UVCal object, str, or list
        Default None. Set to overplot two calibration solutions. pyuvdata UVCal object,
        path to a .calfits file, or path to a CASA .bcal file. Alternatively, list
        containing UVCal objects or paths. If a list is provided, the elements will
        be concatenated across frequency.
    plot_output_dir : str
        Path to the directory where the plots will be saved.
    cal_name : str or list of str
        Optional string to be appended to the start of the file names. If two calibration
        solutions are overplotted, a list can be provided corresponding to each solution.
    plot_reciprocal : bool
        Plot 1/gains.
    ymin : float
        Minimum of the gain amplitude y-axis. Default 0.
    ymax : float
        Maximum of the gain amplitude y-axis. Default is the maximum gain amplitude.
    zero_mean_phase : bool
        If True, forces the mean phase of the gains to be zero. This helps compare
        calibration results generated with different reference antennas. Default False.
    savefig : bool
        If True, save figures as png.
    """

    cal = get_cal_data(cal, zero_mean_phase=zero_mean_phase)
    if cal2 is not None:
        cal2 = get_cal_data(cal2, zero_mean_phase=zero_mean_phase)

    # Parse strings
    if isinstance(cal_name, list):
        if len(cal_name) == 1:
            plot_prefix = cal_name[0]
        else:
            plot_prefix = f"{cal_name[0]}_vs_{cal_name[1]}_"
    else:
        plot_prefix = cal_name
    if len(plot_prefix) > 0:
        if not plot_prefix.endswith("_"):
            plot_prefix = f"{plot_prefix}_"
    use_plot_output_dir = plot_output_dir
    if plot_output_dir is None:
        use_plot_output_dir = ""
    if use_plot_output_dir.endswith("/"):
        use_plot_output_dir = use_plot_output_dir[:-1]

    # Plot style parameters
    colors_x = ["blue", "lightskyblue"]
    colors_y = ["orangered", "lightsalmon"]
    linewidth = 0.2
    markersize = 0.5
    if cal2 is None:
        legend_elements = [
            Line2D(
                [0],
                [0],
                color=colors_x[0],
                marker="o",
                lw=linewidth,
                markersize=markersize,
                label="X",
            ),
            Line2D(
                [0],
                [0],
                color=colors_y[0],
                marker="o",
                lw=linewidth,
                markersize=markersize,
                label="Y",
            ),
        ]
    else:
        if isinstance(cal_name, str):
            cal_name = [cal_name, ""]
        legend_elements = [
            Line2D(
                [0],
                [0],
                color=colors_x[0],
                marker="o",
                lw=linewidth,
                markersize=markersize,
                label=f"X, {cal_name[0]}",
            ),
            Line2D(
                [0],
                [0],
                color=colors_y[0],
                marker="o",
                lw=linewidth,
                markersize=markersize,
                label=f"Y, {cal_name[0]}",
            ),
            Line2D(
                [0],
                [0],
                color=colors_x[1],
                marker="o",
                lw=linewidth,
                markersize=markersize,
                label=f"X, {cal_name[1]}",
            ),
            Line2D(
                [0],
                [0],
                color=colors_y[1],
                marker="o",
                lw=linewidth,
                markersize=markersize,
                label=f"Y, {cal_name[1]}",
            ),
        ]

    ant_nums = cal.ant_array
    if cal2 is not None:
        ant_nums = np.intersect1d(ant_nums, cal2.ant_array)

    # Sort by antenna name
    ant_names = np.array(
        [
            cal.telescope.antenna_names[
                np.where(cal.telescope.antenna_numbers == ant_num)[0][0]
            ]
            for ant_num in ant_nums
        ]
    )
    sort_inds = ant_names.argsort()
    ant_nums = ant_nums[sort_inds]

    freq_axis_mhz = cal.freq_array.flatten() / 1e6

    # Apply flags
    cal.gain_array[np.where(cal.flag_array)] = np.nan + 1j * np.nan
    if cal2 is not None:
        cal2.gain_array[np.where(cal2.flag_array)] = np.nan + 1j * np.nan

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
    for ant_num in ant_nums:
        if subplot_ind == 0:
            fig, ax = plt.subplots(
                nrows=3, ncols=4, figsize=(10, 8), sharex=True, sharey=True
            )
        ant_ind = np.where(cal.ant_array == ant_num)[0][0]
        ant_name = cal.telescope.antenna_names[
            np.where(cal.telescope.antenna_numbers == ant_num)[0][0]
        ]
        if cal2 is not None:
            ant_ind2 = np.where(cal2.ant_array == ant_num)[0][0]
        all_flagged = np.isnan(np.nanmean(cal.gain_array[ant_ind, :, 0, :]))
        if all_flagged and (cal2 is not None):
            if not np.isnan(np.nanmean(cal2.gain_array[ant_ind2, :, 0, :])):
                all_flagged = False
        if all_flagged:  # All gains flagged
            ax.flat[subplot_ind].text(
                np.mean(x_range),
                np.mean(y_range),
                "ALL FLAGGED",
                horizontalalignment="center",
                verticalalignment="center",
                color="red",
            )
        for pol_ind in range(cal.Njones):
            use_colors = [colors_x, colors_y][pol_ind]
            ax.flat[subplot_ind].plot(
                freq_axis_mhz,
                np.abs(cal.gain_array[ant_ind, :, 0, pol_ind]),
                "-o",
                linewidth=linewidth,
                markersize=markersize,
                color=use_colors[0],
            )
        if cal2 is not None:
            for pol_ind in range(cal2.Njones):
                use_colors = [colors_x, colors_y][pol_ind]
                ax.flat[subplot_ind].plot(
                    freq_axis_mhz,
                    np.abs(cal2.gain_array[ant_ind2, :, 0, pol_ind]),
                    "-o",
                    linewidth=linewidth,
                    markersize=markersize,
                    color=use_colors[1],
                )
        ax.flat[subplot_ind].set_ylim(y_range)
        ax.flat[subplot_ind].set_xlim(x_range)
        ax.flat[subplot_ind].set_title(ant_name)
        subplot_ind += 1
        if subplot_ind == len(ax.flat) or ant_num == ant_nums[-1]:
            fig.supxlabel("Frequency (MHz)")
            fig.supylabel("Gain Amplitude")
            plt.legend(
                handles=legend_elements,
                bbox_to_anchor=(1.04, 0),
                loc="lower left",
                frameon=False,
            )
            plt.tight_layout()
            if savefig:
                plt.savefig(
                    f"{use_plot_output_dir}/{plot_prefix}gain_amp_{plot_ind:02d}.png",
                    dpi=600,
                )
            else:
                plt.show()
            plt.close()
            subplot_ind = 0
            plot_ind += 1

    # Plot phases
    subplot_ind = 0
    plot_ind = 1
    for ant_num in ant_nums:
        if subplot_ind == 0:
            fig, ax = plt.subplots(
                nrows=3, ncols=4, figsize=(10, 8), sharex=True, sharey=True
            )
        ant_ind = np.where(cal.ant_array == ant_num)[0][0]
        ant_name = cal.telescope.antenna_names[
            np.where(cal.telescope.antenna_numbers == ant_num)[0][0]
        ]
        if cal2 is not None:
            ant_ind2 = np.where(cal2.ant_array == ant_num)[0][0]
        all_flagged = np.isnan(np.nanmean(cal.gain_array[ant_ind, :, 0, :]))
        if all_flagged and (cal2 is not None):
            if not np.isnan(np.nanmean(cal2.gain_array[ant_ind2, :, 0, :])):
                all_flagged = False
        if all_flagged:  # All gains flagged
            ax.flat[subplot_ind].text(
                np.mean(x_range),
                0,
                "ALL FLAGGED",
                horizontalalignment="center",
                verticalalignment="center",
                color="red",
            )
        for pol_ind in range(cal.Njones):
            use_colors = [colors_x, colors_y][pol_ind]
            ax.flat[subplot_ind].plot(
                freq_axis_mhz,
                np.angle(cal.gain_array[ant_ind, :, 0, pol_ind]),
                "-o",
                linewidth=linewidth,
                markersize=markersize,
                color=use_colors[0],
            )
        if cal2 is not None:
            for pol_ind in range(cal2.Njones):
                use_colors = [colors_x, colors_y][pol_ind]
                ax.flat[subplot_ind].plot(
                    freq_axis_mhz,
                    np.angle(cal2.gain_array[ant_ind2, :, 0, pol_ind]),
                    "-o",
                    linewidth=linewidth,
                    markersize=markersize,
                    color=use_colors[1],
                )
        ax.flat[subplot_ind].set_ylim([-np.pi, np.pi])
        ax.flat[subplot_ind].set_xlim(x_range)
        ax.flat[subplot_ind].set_title(ant_name)
        subplot_ind += 1
        if subplot_ind == len(ax.flat) or ant_num == ant_nums[-1]:
            fig.supxlabel("Frequency (MHz)")
            fig.supylabel("Gain Phase (rad.)")
            plt.legend(
                handles=legend_elements,
                bbox_to_anchor=(1.04, 0),
                loc="lower left",
                frameon=False,
            )
            plt.tight_layout()
            if savefig:
                plt.savefig(
                    f"{use_plot_output_dir}/{plot_prefix}gain_phase_{plot_ind:02d}.png",
                    dpi=600,
                )
            else:
                plt.show()
            plt.close()
            subplot_ind = 0
            plot_ind += 1
