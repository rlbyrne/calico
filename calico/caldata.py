import numpy as np
import sys
import pyuvdata
from pyuvdata import UVCal
from astropy.units import Quantity
import scipy
import copy
from calico import calibration_qa, calibration_optimization
import multiprocessing
from numpy.typing import NDArray


def _run_skycal_single_freq(args):
    """
    Wrapper for calibration_optimization.run_skycal_optimization_per_pol_single_freq that makes
    the function compatible with multiprocessing.
    """
    (
        caldata_obj,
        xtol,
        maxiter,
        freq_ind,
        verbose,
        get_crosspol_phase,
        crosspol_phase_strategy,
    ) = args
    gains_fit = calibration_optimization.run_skycal_optimization_per_pol_single_freq(
        caldata_obj,
        xtol,
        maxiter,
        freq_ind=freq_ind,
        verbose=verbose,
        get_crosspol_phase=get_crosspol_phase,
        crosspol_phase_strategy=crosspol_phase_strategy,
    )
    return freq_ind, gains_fit


class CalData:
    """
    Container for all data and parameters needed for calibration.

    Attributes
    -------
    gains : array of complex
        If n_directions=1, shape (Nants, Nfreqs, N_feed_pols,). Otherwise shape
        (Nants, Nfreqs, N_feed_pols, n_directions,).
    abscal_params : array of float
        Shape (3, Nfreqs, N_feed_pols). abscal_params[0, :, :] are the overall amplitudes,
        abscal_params[1, :, :] are the x-phase gradients in units 1/m, and abscal_params[2, :, :]
        are the y-phase gradients in units 1/m.
    Nants : int
        Number of antennas.
    Nbls : int
        Number of baselines.
    Ntimes : int
        Number of time intervals.
    Nfreqs : int
        Number of frequency channels.
    N_feed_pols : int
        Number of gain polarizations.
    N_vis_pols : int
        Number of visibility polarizations.
    n_directions : int
        Number of calibration directions.
    feed_polarization_array : array of int
        Shape (N_feed_pols). Array of polarization integers. Indicates the
        ordering of the polarization axis of the gains. X is -5 and Y is -6.
    vis_polarization_array : array of int
        Shape (N_vis_pols,). Array of polarization integers. Indicates the
        ordering of the polarization axis of the model_visibilities,
        data_visibilities, and visibility_weights. XX is -5, YY is -6, XY is -7,
        and YX is -8.
    model_visibilities : array of complex
        If n_directions=1, shape (Ntimes, Nbls, Nfreqs, N_vis_pols,). Otherwise shape
        (Ntimes, Nbls, Nfreqs, N_vis_pols, n_directions,).
    data_visibilities : array of complex
        Shape (Ntimes, Nbls, Nfreqs, N_vis_pols,).
    visibility_weights : array of float
        Shape (Ntimes, Nbls, Nfreqs, N_vis_pols,).
    dwcal_inv_covariance : array of complex
        Matrix defining frequency-frequency covariances used in delay-weighted
        calibration. Needed only if delay weighting is used in calibration.
        If dwcal_memory_save_mode is False, dwcal_inv_covariance has shape
        (Ntimes, Nbls, Nfreqs, Nfreqs, N_vis_pols,). If dwcal_memory_save_mode
        is True, dwcal_inv_covariance has shape (Ntimes, Nbls, Nfreqs, N_vis_pols,).
        Alternatively, if the time or polarization axes have length 1,
        dw_inv_covariance is assumed to be identical across time steps or polarization.
    dwcal_memory_save_mode : bool
        Defines the format of dwcal_inv_covariance. If True, dwcal_inv_covariance
        is assumed to be Toeplitz and is stored in a more compact form.
    ant1_inds : array of int
        Shape (Nbls,).
    ant2_inds : array of int
        Shape (Nbls,).
    gains_multiply_model : bool
        If True, measurement equation is defined as v_ij ≈ g_i g_j^* m_ij. If False,
        measurement equation is defined as g_i g_j^* v_ij ≈ m_ij.
    antenna_names : array of str
        Shape (Nants,). Ordering matches the ordering of the gains attribute.
    antenna_numbers : array of int
        Shape (Nants,). Ordering matches the ordering of the gains attribute.
    antenna_positions : array of float
        Shape (Nants, 3,). Units meters, ITRF frame, relative to telescope location.
    antenna_distances : array of float
        Shape (Nants,). Units meters. Planar distance of each antenna from the array center.
        Ignores up-down distances.
    uv_array : array of float
        Shape (Nbls, 2,). Baseline positions in the UV plane, units meters.
    channel_width : float
        Width of frequency channels in Hz.
    freq_array : array of float
        Shape (Nfreqs,). Units Hz.
    integration_time : float
        Length of integration in seconds.
    time : float
        Time of observation in Julian Date.
    telescope : pyuvdata.Telescope
        Object containing the telescope metadata.
    lst : str
        Local sidereal time (LST), in radians.
    lambda_val : float
        Weight of the phase regularization term; must be positive or zero.
    ddcal_max_source_offset_deg : float or None
        Allowable distance that a source is allowed to drift in direction-dependent
        calibration.
    ddcal_source_offset_taper_deg : float or None
        Taper on the source drift regularization for direction-dependent calibration.
    """

    def __init__(self):
        self.gains = None
        self.abscal_params = None
        self.Nants = 0
        self.Nbls = 0
        self.Ntimes = 0
        self.Nfreqs = 0
        self.N_feed_pols = 0
        self.N_vis_pols = 0
        self.n_directions = 1
        self.feed_polarization_array = None
        self.vis_polarization_array = None
        self.model_visibilities = None
        self.data_visibilities = None
        self.visibility_weights = None
        self.dwcal_inv_covariance = None
        self.dwcal_memory_save_mode = None
        self.ant1_inds = None
        self.ant2_inds = None
        self.gains_multiply_model = None
        self.antenna_names = None
        self.antenna_numbers = None
        self.antenna_positions = None
        self.antenna_distances = None
        self.uv_array = None
        self.channel_width = None
        self.freq_array = None
        self.integration_time = None
        self.time = None
        self.telescope = None
        self.lst = None
        self.lambda_val = None
        self.ddcal_max_source_offset_deg = None
        self.ddcal_source_offset_taper_deg = None

    def copy(self):
        return copy.deepcopy(self)

    def set_gains_from_calfile(self, calfile: str) -> None:
        """
        Use a pyuvdata-formatted calfits file to set gains.

        Parameters
        ----------
        calfile : str
            Path to a pyuvdata-formatted calfits file or a CASA-formatted .bcal file.
        """

        uvcal = pyuvdata.UVCal()
        if calfile.endswith(".calfits"):
            uvcal.read_calfits(calfile)
        elif calfile.endswith(".bcal"):
            uvcal.read_ms_cal(calfile)
        else:
            print(f"ERROR: Unknown file extension for file {calfile}. Exiting.")
            sys.exit(1)
        uvcal.select(frequencies=self.freq_array, antenna_names=self.antenna_names)
        if self.feed_polarization_array is None:
            self.feed_polarization_array = uvcal.jones_array
        else:
            uvcal.select(jones=self.feed_polarization_array)
        uvcal.reorder_freqs(channel_order="freq")
        uvcal.reorder_jones()
        uvcal.gain_array[np.where(uvcal.flag_array)] = np.nan + 1j * np.nan
        use_gains = np.nanmean(uvcal.gain_array, axis=2)  # Average over times

        # Make antenna ordering match
        cal_ant_names = np.array(
            [
                uvcal.telescope.antenna_names[
                    np.where(uvcal.telescope.antenna_numbers == ant_num)
                ]
                for ant_num in uvcal.ant_array
            ]
        )
        cal_ant_inds = np.array(
            [list(cal_ant_names).index(name) for name in self.antenna_names]
        )

        if self.gains_multiply_model:
            if uvcal.gain_convention != "divide":
                use_gains = 1 / use_gains
        else:
            if uvcal.gain_convention != "multiply":
                use_gains = 1 / use_gains

        self.gains = use_gains[cal_ant_inds, :]

    def load_data(
        self,
        data: pyuvdata.UVData,
        model: pyuvdata.UVData | None = None,
        model_list: list[pyuvdata.UVData] | None = None,
        gain_init_calfile: str | None = None,
        gain_init_to_vis_ratio: bool = True,
        gains_multiply_model: bool = False,
        gain_init_stddev: float = 0.0,
        check_vis_ordering: bool = True,
        N_feed_pols: int | None = None,
        feed_polarization_array: NDArray[int] | None = None,
        min_cal_baseline_m: float | None = None,
        max_cal_baseline_m: float | None = None,
        min_cal_baseline_lambda: float | None = None,
        max_cal_baseline_lambda: float | None = None,
        lambda_val: float = 0.0,
        ddcal_max_source_offset_deg: float | None = None,
        ddcal_source_offset_taper_deg: float | None = None,
        time_match_tol: float = 1e-5,
        freq_match_tol: float = 1e-5,
    ) -> None:
        """
        Format CalData object with parameters from data and model UVData
        objects.

        Parameters
        ----------
        data : pyuvdata UVData object
            Data to be calibrated.
        model : pyuvdata UVData object or None
            Model visibilities to be used in calibration. Must have the same
            parameters at data. May be None if model_list is provided.
        model_list : list of pyuvdata UVData objects or None
            List of model visibilities to be used for direction-dependent calibration.
            Must have the same parameters at data. May be None if model is provided.
        gain_init_calfile : str or None
            Default None. If not None, provides a path to a pyuvdata-formatted
            calfits file containing gains values for calibration initialization.
        gain_init_to_vis_ratio : bool
            Used only if gain_init_calfile is None. If True, initializes gains
            to the median ratio between the amplitudes of the model and data
            visibilities. If False, the gains are initialized to 1. Default
            True.
        gains_multiply_model : bool
            If True, measurement equation is defined as v_ij ≈ g_i g_j^* m_ij. If
            False, measurement equation is defined as g_i g_j^* v_ij ≈ m_ij. This
            parameter affects how calibration is performed, and whether the data is
            multiplied or divided by the gains when calibration solutions are applied.
            Default False.
        gain_init_stddev : float
            Default 0.0. Standard deviation of a random complex Gaussian
            perturbation to the initial gains.
        check_vis_ordering : bool
            Default True. If False, the ordering of the data and model visibilities are
            assumed to be identical. This can cause errors if used incorrectly.
        N_feed_pols : int
            Default min(2, N_vis_pols). Number of feed polarizations, equal to
            the number of gain values to be calculated per antenna.
        feed_polarization_array : array of int or None
            Feed polarizations to calibrate. Shape (N_feed_pols,). Options are
            -5 for X or -6 for Y. Default None. If None, feed_polarization_array
            is set to ([-5, -6])[:N_feed_pols].
        min_cal_baseline_m : float or None
            Minimum baseline length, in meters, to use in calibration. If both
            min_cal_baseline_m and min_cal_baseline_lambda are None, arbitrarily
            short baselines are used. Default None.
        max_cal_baseline_m : float or None
            Maximum baseline length, in meters, to use in calibration. If both
            max_cal_baseline_m and max_cal_baseline_lambda are None, arbitrarily
            long baselines are used. Default None.
        min_cal_baseline_lambda : float or None
            Minimum baseline length, in wavelengths, to use in calibration. If
            both min_cal_baseline_m and min_cal_baseline_lambda are None,
            arbitrarily short baselines are used. Default None.
        max_cal_baseline_lambda : float or None
            Maximum baseline length, in wavelengths, to use in calibration. If
            both max_cal_baseline_m and max_cal_baseline_lambda are None,
            arbitrarily long baselines are used. Default None.
        lambda_val : float
            Weight of the phase regularization term; must be positive or zero.
            Default 0.
        ddcal_max_source_offset_deg : float or None
            Allowable source offset for direction-dependent calibration, in degrees,
        ddcal_source_offset_taper_deg : float or None
            Taper on the source offset regularization for direction-dependent calibration.
        time_match_tol : float
            Tolerance threshold for time agreement between data and model. Units
            Julian Date. Default 1e-5. Used only if check_vis_ordering is True.
        freq_match_tol : float
            Tolerance threshold for frequency agreement between data and model. Units
            Julian Date. Default 1e-5. Used only if check_vis_ordering is True.
        """

        if model_list is not None:
            self.n_directions = len(model_list)
        else:
            self.n_directions = 1
            model_list = [model]

        # Autocorrelations are not currently supported
        data.select(ant_str="cross")
        for model in model_list:
            model.select(ant_str="cross")

        if check_vis_ordering:
            # Ensure polarizations match
            for model in model_list:
                if model.Npols > data.Npols:
                    model.select(polarizations=data.polarization_array)

                # Ensure times match
                if (
                    np.max(
                        np.abs(
                            np.sort(list(set(data.time_array)))
                            - np.sort(list(set(model.time_array)))
                        )
                    )
                    > time_match_tol
                ):
                    print("ERROR: Data and model times do not match. Exiting.")
                    sys.exit(1)

                # Ensure frequencies match
                if (
                    np.max(np.abs(np.sort(data.freq_array) - np.sort(model.freq_array)))
                    > freq_match_tol
                ):
                    print("ERROR: Data and model frequencies do not match. Exiting.")
                    sys.exit(1)

        # Downselect baselines
        if (
            (min_cal_baseline_m is not None)
            or (max_cal_baseline_m is not None)
            or (min_cal_baseline_lambda is not None)
            or (max_cal_baseline_lambda is not None)
        ):
            if min_cal_baseline_m is None:
                min_cal_baseline_m = 0.0
            if max_cal_baseline_m is None:
                max_cal_baseline_m = np.inf
            if min_cal_baseline_lambda is None:
                min_cal_baseline_lambda = 0.0
            if max_cal_baseline_lambda is None:
                max_cal_baseline_lambda = np.inf

            max_cal_baseline_m = np.min(
                [
                    max_cal_baseline_lambda * 3e8 / np.min(data.freq_array),
                    max_cal_baseline_m,
                ]
            )
            min_cal_baseline_m = np.max(
                [
                    min_cal_baseline_lambda * 3e8 / np.max(data.freq_array),
                    min_cal_baseline_m,
                ]
            )

            data_baseline_lengths_m = np.sqrt(np.sum(data.uvw_array**2.0, axis=1))
            data_use_baselines = np.where(
                (data_baseline_lengths_m >= min_cal_baseline_m)
                & (data_baseline_lengths_m <= max_cal_baseline_m)
            )
            data.select(blt_inds=data_use_baselines)

            for model in model_list:
                model_baseline_lengths_m = np.sqrt(np.sum(model.uvw_array**2.0, axis=1))
                model_use_baselines = np.where(
                    (model_baseline_lengths_m >= min_cal_baseline_m)
                    & (model_baseline_lengths_m <= max_cal_baseline_m)
                )
                model.select(blt_inds=model_use_baselines)

        if check_vis_ordering:  # Ensure baselines match
            data.conjugate_bls()
            data.reorder_blts()
            for model in model_list:
                model.conjugate_bls()
                model.reorder_blts()

            n_blts_list = [model.Nblts for model in model_list]
            n_blts_list.append(data.Nblts)
            if len(set(n_blts_list)) > 1:
                select_baselines = True
            else:
                select_baselines = False
                for model in model_list:
                    if (np.max(np.abs(data.ant_1_array - model.ant_1_array)) > 0) or (
                        np.max(np.abs(data.ant_2_array - model.ant_2_array)) > 0
                    ):
                        select_baselines = True
                        break

            if select_baselines:
                baselines = [
                    list(set(zip(model.ant_1_array, model.ant_2_array)))
                    for model in model_list
                ]
                baselines.append(list(set(zip(data.ant_1_array, data.ant_2_array))))
                use_baselines = set(baselines[0]).intersection(*baselines[1:])
                if len(use_baselines) < data.Nbls:
                    print(
                        f"WARNING: Model does not contain all baselines. Downselecting from {data.Nbls} to {len(use_baselines)}."
                    )
                data.select(bls=use_baselines)
                for model in model_list:
                    model.select(bls=use_baselines)

        self.Nants = data.Nants_data
        self.Nbls = data.Nbls
        self.Ntimes = data.Ntimes
        self.Nfreqs = data.Nfreqs
        self.N_vis_pols = data.Npols

        # Format visibilities
        self.data_visibilities = np.zeros(
            (
                self.Ntimes,
                self.Nbls,
                self.Nfreqs,
                self.N_vis_pols,
            ),
            dtype=complex,
        )
        if self.n_directions == 1:
            self.model_visibilities = np.zeros(
                (
                    self.Ntimes,
                    self.Nbls,
                    self.Nfreqs,
                    self.N_vis_pols,
                ),
                dtype=complex,
            )
        else:
            self.model_visibilities = np.zeros(
                (
                    self.Ntimes,
                    self.Nbls,
                    self.Nfreqs,
                    self.N_vis_pols,
                    self.n_directions,
                ),
                dtype=complex,
            )
        flag_array = np.zeros(
            (self.Ntimes, self.Nbls, self.Nfreqs, self.N_vis_pols), dtype=bool
        )

        for time_ind, time_val in enumerate(np.unique(data.time_array)):
            data_copy = data.select(times=time_val, inplace=False)
            data_copy.reorder_blts()
            data_copy.reorder_pols(order="AIPS")
            data_copy.reorder_freqs(channel_order="freq")
            if time_ind == 0:
                metadata_reference = data_copy.copy(metadata_only=True)
            self.data_visibilities[time_ind, :, :, :] = np.reshape(
                data_copy.data_array,
                (data_copy.Nblts, data_copy.Nfreqs, data_copy.Npols),
            )
            flag_array[time_ind, :, :, :] = np.reshape(
                data_copy.flag_array,
                (data_copy.Nblts, data_copy.Nfreqs, data_copy.Npols),
            )

            for model_ind, model in enumerate(model_list):
                model_times = list(set(model.time_array))
                model_copy = model.select(
                    times=model_times[
                        np.where(
                            np.abs(model_times - time_val)
                            == np.min(np.abs(model_times - time_val))
                        )[0][
                            0
                        ]  # Account for times that are close but not exactly equal
                    ],
                    inplace=False,
                )
                model_copy.reorder_blts()
                model_copy.reorder_pols(order="AIPS")
                model_copy.reorder_freqs(channel_order="freq")

                if self.n_directions == 1:
                    self.model_visibilities[time_ind, :, :, :] = np.reshape(
                        model_copy.data_array,
                        (model_copy.Nblts, model_copy.Nfreqs, model_copy.Npols),
                    )
                else:
                    self.model_visibilities[time_ind, :, :, :, model_ind] = np.reshape(
                        model_copy.data_array,
                        (model_copy.Nblts, model_copy.Nfreqs, model_copy.Npols),
                    )

                # Update flag_array if the model contains flags
                flag_array[time_ind, :, :, :] = np.max(
                    np.stack(
                        [
                            np.reshape(
                                model_copy.flag_array,
                                (model_copy.Nblts, model_copy.Nfreqs, model_copy.Npols),
                            ),
                            flag_array[time_ind, :, :, :],
                        ]
                    ),
                    axis=0,
                )

        # Free memory
        data.__init__()
        for model in model_list:
            model.__init__()
        data_copy = model_copy = None

        # Grab other metadata from uvfits
        self.channel_width = np.mean(metadata_reference.channel_width)
        self.freq_array = np.reshape(metadata_reference.freq_array, (self.Nfreqs))
        self.integration_time = np.mean(metadata_reference.integration_time)
        self.time = np.mean(metadata_reference.time_array)
        self.telescope = metadata_reference.telescope
        self.lst = np.mean(metadata_reference.lst_array)

        if (min_cal_baseline_lambda is not None) or (
            max_cal_baseline_lambda is not None
        ):
            baseline_lengths_m = np.sqrt(
                np.sum(metadata_reference.uvw_array**2.0, axis=1)
            )
            baseline_lengths_lambda = (
                baseline_lengths_m[:, np.newaxis]
                * np.reshape(
                    metadata_reference.freq_array, (1, metadata_reference.Nfreqs)
                )
                / 3e8
            )
            flag_array[
                :,
                np.where(
                    (baseline_lengths_lambda < min_cal_baseline_lambda)
                    & (baseline_lengths_lambda > max_cal_baseline_lambda)
                ),
                :,
            ] = True

        # Define antenna to baseline mapping
        self.ant1_inds = np.zeros(self.Nbls, dtype=int)
        self.ant2_inds = np.zeros(self.Nbls, dtype=int)
        self.antenna_numbers = np.unique(
            [metadata_reference.ant_1_array, metadata_reference.ant_2_array]
        )
        for baseline in range(metadata_reference.Nbls):
            self.ant1_inds[baseline] = np.where(
                self.antenna_numbers == metadata_reference.ant_1_array[baseline]
            )[0]
            self.ant2_inds[baseline] = np.where(
                self.antenna_numbers == metadata_reference.ant_2_array[baseline]
            )[0]

        # Get ordered list of antenna names
        self.antenna_names = np.array(
            [
                np.array(metadata_reference.telescope.antenna_names)[
                    np.where(metadata_reference.telescope.antenna_numbers == ant_num)[
                        0
                    ][0]
                ]
                for ant_num in self.antenna_numbers
            ]
        )
        self.antenna_positions = np.array(
            [
                np.array(metadata_reference.telescope.antenna_positions)[
                    np.where(metadata_reference.telescope.antenna_numbers == ant_num)[
                        0
                    ][0],
                    :,
                ]
                for ant_num in self.antenna_numbers
            ]
        )

        # Get UV locations
        antpos_ecef = self.antenna_positions + Quantity(
            metadata_reference.telescope.location.geocentric
        ).to_value(
            "m"
        )  # Get antennas positions in ECEF
        antenna_positions_topocentric = pyuvdata.utils.ENU_from_ECEF(
            antpos_ecef, center_loc=metadata_reference.telescope.location
        )  # Convert to topocentric (East, North, Up or ENU) coords.
        self.antenna_distances = np.sqrt(
            antenna_positions_topocentric[:, 0] ** 2.0
            + antenna_positions_topocentric[:, 1] ** 2.0
        )

        uvw_array = (
            antenna_positions_topocentric[self.ant1_inds, :]
            - antenna_positions_topocentric[self.ant2_inds, :]
        )
        self.uv_array = uvw_array[:, :2]

        # Get polarization ordering
        self.vis_polarization_array = np.array(metadata_reference.polarization_array)

        if N_feed_pols is None:
            self.N_feed_pols = np.min([2, self.N_vis_pols])
        else:
            self.N_feed_pols = N_feed_pols

        if feed_polarization_array is None:
            self.feed_polarization_array = np.array([], dtype=int)
            if (
                (-5 in self.vis_polarization_array)
                or (-7 in self.vis_polarization_array)
                or (-8 in self.vis_polarization_array)
            ):
                self.feed_polarization_array = np.append(
                    self.feed_polarization_array, -5
                )
            if (
                (-6 in self.vis_polarization_array)
                or (-7 in self.vis_polarization_array)
                or (-8 in self.vis_polarization_array)
            ):
                self.feed_polarization_array = np.append(
                    self.feed_polarization_array, -6
                )
            self.feed_polarization_array = self.feed_polarization_array[
                : self.N_feed_pols
            ]
        else:
            self.feed_polarization_array = feed_polarization_array

        # Initialize gains
        self.gains_multiply_model = gains_multiply_model
        if gain_init_calfile is None:
            if self.n_directions == 1:
                self.gains = np.ones(
                    (
                        self.Nants,
                        self.Nfreqs,
                        self.N_feed_pols,
                    ),
                    dtype=complex,
                )
            else:
                self.gains = np.ones(
                    (
                        self.Nants,
                        self.Nfreqs,
                        self.N_feed_pols,
                        self.n_directions,
                    ),
                    dtype=complex,
                )
            if gain_init_to_vis_ratio:  # Use mean ratio of visibility amplitudes
                if self.n_directions == 1:
                    vis_amp_ratio = np.abs(self.model_visibilities) / np.abs(
                        self.data_visibilities
                    )
                    vis_amp_ratio[np.where(self.data_visibilities == 0.0)] = np.nan
                    for feed_pol_ind, feed_pol in enumerate(
                        self.feed_polarization_array
                    ):
                        vis_pol_ind = np.where(self.vis_polarization_array == feed_pol)[
                            0
                        ][0]
                        if self.gains_multiply_model:
                            self.gains[:, :, feed_pol_ind] = np.nanmedian(
                                1 / np.sqrt(vis_amp_ratio[:, :, :, vis_pol_ind])
                            )
                        else:
                            self.gains[:, :, feed_pol_ind] = np.nanmedian(
                                np.sqrt(vis_amp_ratio[:, :, :, vis_pol_ind])
                            )
                else:
                    for direction_ind in range(self.n_directions):
                        vis_amp_ratio = np.abs(
                            self.model_visibilities[:, :, :, :, direction_ind]
                        ) / np.abs(self.data_visibilities)
                        vis_amp_ratio[np.where(self.data_visibilities == 0.0)] = np.nan
                        for feed_pol_ind, feed_pol in enumerate(
                            self.feed_polarization_array
                        ):
                            vis_pol_ind = np.where(
                                self.vis_polarization_array == feed_pol
                            )[0][0]
                            if self.gains_multiply_model:
                                self.gains[:, :, feed_pol_ind, direction_ind] = (
                                    np.nanmedian(
                                        1 / np.sqrt(vis_amp_ratio[:, :, :, vis_pol_ind])
                                    )
                                )
                            else:
                                self.gains[:, :, feed_pol_ind, direction_ind] = (
                                    np.nanmedian(
                                        np.sqrt(vis_amp_ratio[:, :, :, vis_pol_ind])
                                    )
                                )
        else:  # Initialize from file
            self.set_gains_from_calfile(gain_init_calfile)
            # Capture nan-ed gains as flags
            for feed_pol_ind, feed_pol in enumerate(self.feed_polarization_array):
                nan_gains = np.where(~np.isfinite(self.gains[:, :, feed_pol_ind]))
                if len(nan_gains[0]) > 0:
                    if feed_pol == -5:
                        flag_pols = np.where(
                            (metadata_reference.polarization_array == -5)
                            | (metadata_reference.polarization_array == -7)
                            | (metadata_reference.polarization_array == -8)
                        )[0]
                    elif feed_pol == -6:
                        flag_pols = np.where(
                            (metadata_reference.polarization_array == -6)
                            | (metadata_reference.polarization_array == -7)
                            | (metadata_reference.polarization_array == -8)
                        )[0]
                    for flag_ind in range(len(nan_gains[0])):
                        flag_bls = np.unique(
                            np.concatenate(
                                (
                                    np.where(self.ant1_inds == nan_gains[0][flag_ind])[
                                        0
                                    ],
                                    np.where(self.ant2_inds == nan_gains[0][flag_ind])[
                                        0
                                    ],
                                )
                            )
                        )
                        flag_freq = nan_gains[1][flag_ind]
                        for flag_pol in flag_pols:
                            flag_array[
                                :,
                                flag_bls,
                                flag_freq,
                                flag_pol,
                            ] = True
                    # self.gains[nan_gains[0], nan_gains[1], feed_pol_ind] = (
                    #    0.0 + 0.0*1j  # Nans in the gains produce matrix multiplication errors, set to zero
                    # )

        # Free memory
        metadata_reference = None

        # Random perturbation of initial gains
        if gain_init_stddev != 0.0:
            self.gains += np.random.normal(
                0.0,
                gain_init_stddev,
                size=np.shape(self.gains),
            ) + 1.0j * np.random.normal(
                0.0,
                gain_init_stddev,
                size=np.shape(self.gains),
            )

        # Initialize abscal parameters
        self.abscal_params = np.zeros((3, self.Nfreqs, self.N_feed_pols), dtype=float)
        self.abscal_params[0, :, :] = 1.0

        # Define visibility weights
        self.visibility_weights = np.ones(
            (
                self.Ntimes,
                self.Nbls,
                self.Nfreqs,
                self.N_vis_pols,
            ),
            dtype=float,
        )
        if np.max(flag_array):  # Apply flagging
            self.visibility_weights[np.where(flag_array)] = 0.0

        # Regularization terms
        self.lambda_val = lambda_val
        self.ddcal_max_source_offset_deg = ddcal_max_source_offset_deg
        self.ddcal_source_offset_taper_deg = ddcal_source_offset_taper_deg

    def convert_to_uvcal(self) -> UVCal:
        """
        Generate a pyuvdata UVCal object.

        Returns
        -------
        uvcal : pyuvdata UVCal object
        """

        uvcal = pyuvdata.UVCal()
        uvcal.Nants = self.Nants
        uvcal.Nants_data = self.Nants
        uvcal.Nants_telescope = self.Nants
        uvcal.Nfreqs = self.Nfreqs
        uvcal.Njones = self.N_feed_pols
        uvcal.Nspws = 1
        uvcal.Ntimes = 1
        uvcal.antenna_names = self.antenna_names
        uvcal.ant_array = self.antenna_numbers
        uvcal.antenna_numbers = self.antenna_numbers
        uvcal.antenna_positions = self.antenna_positions
        uvcal.cal_style = "sky"
        uvcal.cal_type = "gain"
        uvcal.channel_width = np.full((self.Nfreqs), self.channel_width)
        uvcal.freq_array = self.freq_array
        if self.gains_multiply_model:
            uvcal.gain_convention = "divide"
        else:
            uvcal.gain_convention = "multiply"
        uvcal.history = "calibrated with calico"
        uvcal.integration_time = np.array([self.integration_time])
        uvcal.jones_array = self.feed_polarization_array
        uvcal.spw_array = np.array([0])
        uvcal.telescope = self.telescope
        uvcal.lst_array = np.array([self.lst])
        uvcal.time_array = np.array([self.time])
        uvcal.x_orientation = "east"
        uvcal.gain_array = self.gains[:, :, np.newaxis, :]
        uvcal.ref_antenna_name = "none"
        uvcal.sky_catalog = ""
        uvcal.wide_band = False
        uvcal.flex_spw_id_array = np.zeros(self.Nfreqs, dtype=int)

        # Get flags from nan-ed gains and zeroed weights
        uvcal.flag_array = (np.isnan(self.gains))[:, :, np.newaxis, :]

        # Get flags from visibility_weights
        antenna_weights = np.zeros(
            (self.Nants, self.Nfreqs, self.N_feed_pols), dtype=float
        )
        for ant_ind in range(self.Nants):
            for pol_ind in range(self.N_feed_pols):
                if self.feed_polarization_array[pol_ind] == -5:
                    use_vis_pol_inds_ant1 = np.where(
                        (self.vis_polarization_array == -5)
                        | (self.vis_polarization_array == -7)
                    )[0]
                    use_vis_pol_inds_ant2 = np.where(
                        (self.vis_polarization_array == -5)
                        | (self.vis_polarization_array == -8)
                    )[0]
                elif self.feed_polarization_array[pol_ind] == -6:
                    use_vis_pol_inds_ant1 = np.where(
                        (self.vis_polarization_array == -6)
                        | (self.vis_polarization_array == -8)
                    )[0]
                    use_vis_pol_inds_ant2 = np.where(
                        (self.vis_polarization_array == -6)
                        | (self.vis_polarization_array == -7)
                    )[0]
                ant1_antenna_weights = np.zeros((self.Nfreqs))
                ant2_antenna_weights = np.zeros((self.Nfreqs))
                for vis_pol_ind in use_vis_pol_inds_ant1:
                    ant1_antenna_weights += np.sum(
                        self.visibility_weights[
                            :, np.where(self.ant1_inds == ant_ind)[0], :, vis_pol_ind
                        ],
                        axis=(0, 1),
                    )
                for vis_pol_ind in use_vis_pol_inds_ant2:
                    ant2_antenna_weights += np.sum(
                        self.visibility_weights[
                            :, np.where(self.ant2_inds == ant_ind)[0], :, vis_pol_ind
                        ],
                        axis=(0, 1),
                    )
                antenna_weights[ant_ind, :, pol_ind] = (
                    ant1_antenna_weights + ant2_antenna_weights
                )
        uvcal.flag_array[np.where(antenna_weights[:, :, np.newaxis, :] == 0)] = True

        try:
            uvcal.check()
        except:
            print("ERROR: UVCal check failed.")

        return uvcal

    def sky_based_calibration(
        self,
        xtol: float = 1e-5,
        maxiter: int = 200,
        get_crosspol_phase: bool = True,
        crosspol_phase_strategy: str = "crosspol model",
        parallel: bool = False,
        n_workers: int | None = 40,
        verbose: bool = False,
    ) -> None:
        """
        Run calibration per polarization. Updates the gains attribute with calibrated values.
        Here the XX and YY visibilities are calibrated individually and the cross-polarization
        phase is applied from the XY and YX visibilities after the fact. Option to parallelize
        calibration across frequency.

        Parameters
        ----------
        xtol : float
            Accuracy tolerance for optimizer. Default 1e-5.
        maxiter : int
            Maximum number of iterations for the optimizer. Default 200.
        get_crosspol_phase : bool
            If True, crosspol phase is calculated. Default True.
        crosspol_phase_strategy : str
            Options are "crosspol model" or "pseudo Stokes V". Used only if
            get_crosspol_phase is True. If "crosspol model", contrains the crosspol
            phase using the crosspol model visibilities. If "pseudo Stokes V", constrains
            crosspol phase by minimizing pseudo Stokes V. Default "crosspol model".
        parallel : bool
            Set to True to parallelize across frequency with multiprocessing. Default False.
        n_workers : int
            Maximum number of multithreaded processes to use. Applicable only if
            parallel is True. If None, uses the multiprocessing default. Default 40.
        verbose : bool
            Set to True to print optimization outputs. Default False.

        Raises
        ------
        ValueError
            If inputs are invalid (n_directions>1, non-positive xtol or maxiter, invalid
            crosspol_phase_strategy, or n_workers < 1).
        """

        if self.n_directions > 1:
            raise ValueError(
                "sky_based_calibration does not support multiple directions. "
                "Use direction_dependent_calibration instead."
            )
        if xtol <= 0:
            raise ValueError(f"xtol must be positive, got {xtol}.")
        if maxiter <= 0:
            raise ValueError(f"maxiter must be a positive integer, got {maxiter}.")
        if get_crosspol_phase and crosspol_phase_strategy not in [
            "crosspol model",
            "pseudo Stokes V",
        ]:
            raise ValueError(
                f"crosspol_phase_strategy must be one of "
                f"['crosspol model', 'pseudo Stokes V'], got {crosspol_phase_strategy!r}."
            )
        if parallel and n_workers is not None and n_workers < 1:
            raise ValueError(
                f"n_workers must be a positive integer or None, got {n_workers}."
            )
        if self.Nfreqs < 1:
            raise ValueError(f"self.Nfreqs must be positive, got {self.Nfreqs}.")

        if np.max(self.visibility_weights) == 0.0:
            warnings.warn(
                "All data flagged; setting all gains to NaN and skipping calibration.",
                stacklevel=2,
            )
            self.gains[:, :, :] = np.nan + 1j * np.nan
            return

        if parallel:
            n_workers = min(self.Nfreqs, n_workers)
            ctx = multiprocessing.get_context("fork")
            task_args = [
                (
                    self,
                    xtol,
                    maxiter,
                    freq_ind,
                    verbose,
                    get_crosspol_phase,
                    crosspol_phase_strategy,
                )
                for freq_ind in range(self.Nfreqs)
            ]
            with ctx.Pool(processes=n_workers, maxtasksperchild=10) as pool:
                for freq_ind, gains_fit in pool.imap_unordered(
                    _run_skycal_single_freq,
                    task_args,
                ):
                    self.gains[:, [freq_ind], :] = gains_fit[:, np.newaxis, :]
        else:
            for freq_ind in range(self.Nfreqs):
                gains_fit = calibration_optimization.run_skycal_optimization_per_pol_single_freq(
                    self,
                    xtol,
                    maxiter,
                    freq_ind=freq_ind,
                    verbose=verbose,
                    get_crosspol_phase=get_crosspol_phase,
                    crosspol_phase_strategy=crosspol_phase_strategy,
                )
                self.gains[:, [freq_ind], :] = gains_fit[:, np.newaxis, :]

    def direction_dependent_calibration(
        self,
        xtol: float = 1e-5,
        maxiter: int = 200,
        verbose: bool = False,
    ) -> None:
        """
        Run direction-dependent calibration for each polarization and frequency. Updates the
        gains attribute with calibrated values.

        Parameters
        ----------
        xtol : float
            Accuracy tolerance for optimizer. Default 1e-5.
        maxiter : int
            Maximum number of iterations for the optimizer. Default 200.
        verbose : bool
            Set to True to print optimization outputs. Default False.
        """

        if not self.gains_multiply_model:
            print(
                "ERROR: gains_multiply_model is False. Direction-dependent calibration requires that gains_multiply_model=True."
            )
            sys.exit(1)

        for pol_ind in range(self.N_feed_pols):
            for freq_ind in range(self.Nfreqs):
                gains_fit = calibration_optimization.run_ddcal_optimization(
                    self,
                    xtol,
                    maxiter,
                    freq_ind=freq_ind,
                    pol_ind=pol_ind,
                    verbose=verbose,
                )
                self.gains[:, freq_ind, pol_ind, :] = gains_fit

    def delay_weighted_calibration(
        self, xtol: float = 1e-5, maxiter: int = 200, verbose: bool = False
    ) -> None:
        """
        Run delay-weighted calibration (DWCal). Updates attribute gains with calibrated values.

        Parameters
        ----------
        xtol : float
            Accuracy tolerance for optimizer. Default 1e-5.
        maxiter : int
            Maximum number of iterations for the optimizer. Default 200.
        verbose : bool
            Set to True to print optimization outputs. Default False.
        """

        if self.gains_multiply_model:
            print(
                "ERROR: gains_multiply_model is True. Delay-weighted calibration requires that gains_multiply_model=False."
            )
            sys.exit(1)

        for feed_pol_ind in range(self.N_feed_pols):
            self.gains[:, :, feed_pol_ind] = (
                calibration_optimization.run_dwcal_optimization_per_pol(
                    self,
                    xtol,
                    maxiter,
                    feed_pol_ind,
                    verbose=verbose,
                )
            )

    def convert_to_dwcal_memory_save_mode(self, check=True):

        if not self.dwcal_memory_save_mode:
            if check:
                check_passed = True

                # Check if Hermitian
                if not np.allclose(
                    self.dwcal_inv_covariance,
                    np.conj(
                        np.transpose(self.dwcal_inv_covariance, axes=(0, 1, 3, 2, 4))
                    ),
                ):
                    print("ERROR: dwcal_inv_covariance is not Hermitian.")
                    sys.exit(1)

                # Check if Toeplitz
                for ind1 in range(1, self.Nfreqs):
                    for ind2 in range(1, self.Nfreqs):
                        if not np.allclose(
                            self.dwcal_inv_covariance[:, :, ind1, ind2, :],
                            self.dwcal_inv_covariance[:, :, ind1 - 1, ind2 - 1, :],
                        ):
                            check_passed = False
                            break
                    if not check_passed:
                        break
                if not check_passed:
                    print("ERROR: dwcal_inv_covariance is not Toeplitz.")
                    sys.exit(1)

            self.dwcal_memory_save_mode = True
            self.dwcal_inv_covariance = self.dwcal_inv_covariance[:, :, :, 0, :]

    def convert_from_dwcal_memory_save_mode(self):

        if self.dwcal_memory_save_mode:
            self.dwcal_memory_save_mode = False
            dwcal_inv_covariance_new = np.zeros(
                (
                    self.Ntimes,
                    self.Nbls,
                    self.Nfreqs,
                    self.Nfreqs,
                    self.N_vis_pols,
                ),
                dtype=complex,
            )
            for time_ind in range(self.Ntimes):
                for bl_ind in range(self.Nbls):
                    for pol_ind in range(self.N_vis_pols):
                        dwcal_inv_covariance_new[time_ind, bl_ind, :, :, pol_ind] = (
                            scipy.linalg.toeplitz(
                                self.dwcal_inv_covariance[time_ind, bl_ind, :, pol_ind]
                            )
                        )
            self.dwcal_inv_covariance = dwcal_inv_covariance_new

    def abscal(
        self, xtol: float = 1e-5, maxiter: int = 200, verbose: bool = False
    ) -> None:
        """
        Run absolute calibration ("abscal"). Updates the abscal_params attribute with calibrated values.

        Parameters
        ----------
        xtol : float
            Accuracy tolerance for optimizer. Default 1e-5.
        maxiter : int
            Maximum number of iterations for the optimizer. Default 200.
        verbose : bool
            Set to True to print optimization outputs. Default False.
        """

        for feed_pol_ind in range(self.N_feed_pols):
            for freq_ind in range(self.Nfreqs):
                abscal_params = (
                    calibration_optimization.run_abscal_optimization_single_freq(
                        self,
                        xtol,
                        maxiter,
                        freq_ind=freq_ind,
                        feed_pol_ind=feed_pol_ind,
                        verbose=verbose,
                    )
                )
                self.abscal_params[:, freq_ind, feed_pol_ind] = abscal_params

    def dw_abscal(
        self, xtol: float = 1e-5, maxiter: int = 200, verbose: bool = False
    ) -> None:
        """
        Run absolute calibration ("abscal") with delay weighting. Updates the
        abscal_params attribute with calibrated values.

        Parameters
        ----------
        xtol : float
            Accuracy tolerance for optimizer. Default 1e-5.
        maxiter : int
            Maximum number of iterations for the optimizer. Default 200.
        verbose : bool
            Set to True to print optimization outputs. Default False.
        """

        for feed_pol_ind in range(self.N_feed_pols):
            self.abscal_params[:, :, feed_pol_ind] = (
                calibration_optimization.run_dw_abscal_optimization(
                    self,
                    xtol,
                    maxiter,
                    feed_pol_ind=feed_pol_ind,
                    verbose=verbose,
                )
            )

    def flag_antennas_from_per_ant_cost(
        self,
        flagging_threshold: float = 2.5,
        return_antenna_flag_list: bool = False,
        verbose: bool = True,
    ) -> list | None:
        """
        Flags antennas based on the per-antenna cost function. Updates
        visibility_weights according to the flags. The cost function used is the
        standard "sky-based" per frequency, per polarization cost function evaluated
        in cost_function_calculations.cost_function_single_pol.

        Parameters
        ----------
        self : CalData
        flagging_threshold : float
            Flagging threshold. Per antenna cost values equal to flagging_threshold
            times the mean value will be flagged. Default 2.5.
        return_antenna_flag_list : bool
            If True, returns list of flagged antennas.
        verbose : bool

        Returns
        -------
        flag_antenna_list : list of str or None
            If return_antenna_flag_list is True, returns a list of flagged antenna names.
        """

        # TODO: Allow this function to be run in parallel

        per_ant_cost = calibration_qa.calculate_per_antenna_cost(self)

        where_finite = np.isfinite(per_ant_cost)
        if np.sum(where_finite) > 0:
            mean_per_ant_cost = np.mean(per_ant_cost[where_finite])
            flag_antenna_list = []
            for pol_ind in range(self.N_feed_pols):
                flag_antenna_inds = np.where(
                    np.logical_or(
                        per_ant_cost[:, pol_ind]
                        > flagging_threshold * mean_per_ant_cost,
                        ~np.isfinite(per_ant_cost[:, pol_ind]),
                    )
                )[0]
                flag_antenna_list.append(self.antenna_names[flag_antenna_inds])

                for ant_ind in flag_antenna_inds:
                    bl_inds_1 = np.where(self.ant1_inds == ant_ind)[0]
                    bl_inds_2 = np.where(self.ant2_inds == ant_ind)[0]
                    if self.feed_polarization_array[pol_ind] == -5:
                        if -5 in self.vis_polarization_array:
                            vis_pol_ind = np.where(self.vis_polarization_array == -5)[0]
                            self.visibility_weights[:, bl_inds_1, :, vis_pol_ind] = 0
                            self.visibility_weights[:, bl_inds_2, :, vis_pol_ind] = 0
                        if -7 in self.vis_polarization_array:
                            vis_pol_ind = np.where(self.vis_polarization_array == -7)[0]
                            self.visibility_weights[:, bl_inds_1, :, vis_pol_ind] = 0
                        if -8 in self.vis_polarization_array:
                            vis_pol_ind = np.where(self.vis_polarization_array == -8)[0]
                            self.visibility_weights[:, bl_inds_2, :, vis_pol_ind] = 0
                    elif self.feed_polarization_array[pol_ind] == -6:
                        if -6 in self.vis_polarization_array:
                            vis_pol_ind = np.where(self.vis_polarization_array == -6)[0]
                            self.visibility_weights[:, bl_inds_1, :, vis_pol_ind] = 0
                            self.visibility_weights[:, bl_inds_2, :, vis_pol_ind] = 0
                        if -7 in self.vis_polarization_array:
                            vis_pol_ind = np.where(self.vis_polarization_array == -7)[0]
                            self.visibility_weights[:, bl_inds_2, :, vis_pol_ind] = 0
                        if -8 in self.vis_polarization_array:
                            vis_pol_ind = np.where(self.vis_polarization_array == -8)[0]
                            self.visibility_weights[:, bl_inds_1, :, vis_pol_ind] = 0

        else:  # Flag everything
            flag_antenna_list = []
            for pol_ind in range(self.N_feed_pols):
                flag_antenna_list.append(self.antenna_names)
            self.visibility_weights[:, :, :, :] = 0

        if verbose:
            print("Completed antenna flagging based on per-antenna cost function.")
            print(f"Flagged antennas: {flag_antenna_list}")
            sys.stdout.flush()

        if return_antenna_flag_list:
            return flag_antenna_list

    def get_dwcal_weights_from_delay_spectra(
        self,
        delay_spectrum_variance: NDArray[np.floating],
        bl_length_bin_edges: NDArray[np.floating],
        delay_axis: NDArray[np.floating],
        oversample_factor: int = 128,
    ) -> None:
        """
        This function calculates the matrix that captures delay weighting (or frequency
        covariance). The input is an array of expected variances as a function of baseline
        length and delay.

        Parameters
        ----------
        delay_spectrum_variance : array of float
            Array containing the expected variance as a function of baseline length and delay.
            Shape (Nbins, Ndelays,).
        bl_length_bin_edges : array of float
            Defines the baseline length axis of delay_spectrum_variance. Values correspond to
            limits of each baseline length bin. Shape (Nbins+1,).
        delay_axis : array of float
            Defines the delay axis of delay_spectrum_variance. Shape (Ndelays,).
        oversample_factor : int
            Factor by which to oversample the delay axis. Setting > 1 reduces Fourier aliasing
            effects. Default 128.
        """

        bl_lengths = np.sqrt(np.sum(self.uv_array**2.0, axis=1))
        delay_array_use = np.fft.fftfreq(
            self.Nfreqs * int(oversample_factor), d=self.channel_width
        )
        dwcal_variance_use = np.zeros(
            (
                self.Nbls,
                self.Nfreqs * int(oversample_factor),
            ),
            dtype=float,
        )
        for bl_ind, bl_length in enumerate(bl_lengths):
            bin_ind = np.max(np.where(bl_length_bin_edges <= bl_length)[0])
            if (bin_ind == len(bl_length_bin_edges) - 1) or (
                not bl_length_bin_edges[bin_ind + 1] > bl_length
            ):
                print(
                    f"WARNING: Baseline length range does not cover baseline of length {bl_length} m. Skipping."
                )
                continue
            dwcal_variance_use[bl_ind, :] = np.interp(
                delay_array_use, delay_axis, delay_spectrum_variance[bin_ind, :]
            )

        freq_weighting = np.fft.ifft(1.0 / dwcal_variance_use, axis=1)
        freq_weighting = freq_weighting[
            :, : self.Nfreqs
        ]  # Truncate frequency axis to remove oversampling
        weight_mat = np.zeros((self.Nbls, self.Nfreqs, self.Nfreqs), dtype=complex)
        for freq_ind1 in range(self.Nfreqs):
            for freq_ind2 in range(self.Nfreqs):
                if freq_ind1 < freq_ind2:
                    weight_mat[:, freq_ind1, freq_ind2] = np.conj(
                        freq_weighting[:, np.abs(freq_ind1 - freq_ind2)]
                    )
                else:
                    weight_mat[:, freq_ind1, freq_ind2] = freq_weighting[
                        :, np.abs(freq_ind1 - freq_ind2)
                    ]

        # Use the same matrix for all times and polarizations
        # These are included as variables so that time- and polarization-dependence
        # can be built in later if needed
        use_Ntimes = 1
        use_N_vis_pols = 1

        if self.dwcal_memory_save_mode:
            weight_mat = np.repeat(
                np.repeat(weight_mat[np.newaxis, :, :, np.newaxis], use_Ntimes, axis=0),
                use_N_vis_pols,
                axis=3,
            )
        else:
            weight_mat = np.repeat(
                np.repeat(
                    weight_mat[np.newaxis, :, :, :, np.newaxis], use_Ntimes, axis=0
                ),
                use_N_vis_pols,
                axis=4,
            )

        # Deal with nan-ed values
        if self.dwcal_memory_save_mode:
            nan_weight_indices = np.where(~np.isfinite(np.sum(weight_mat, axis=2)))
        else:
            nan_weight_indices = np.where(~np.isfinite(np.sum(weight_mat, axis=(2, 3))))
        if len(nan_weight_indices[0]) > 0:
            print(
                "WARNING: nan values encountered in DWCal inverse convariance matrix. Updating weights."
            )
            for freq_ind in range(self.Nfreqs):
                self.visibility_weights[:, :, freq_ind, :][nan_weight_indices] = 0
            weight_mat[np.where(~np.isfinite(weight_mat))] = (
                0.0 + 1j * 0.0
            )  # Remove nan values to prevent issues later on

        # Fix normalization
        use_visibility_weights = self.visibility_weights
        if self.Ntimes > use_Ntimes:
            use_visibility_weights = np.mean(use_visibility_weights, axis=0)[
                np.newaxis, :, :, :
            ]
        if self.N_vis_pols > use_N_vis_pols:
            use_visibility_weights = np.mean(use_visibility_weights, axis=3)[
                :, :, :, np.newaxis
            ]
        for time_ind in range(use_Ntimes):
            for vis_pol_ind in range(use_N_vis_pols):
                normalization_numerator = np.sum(
                    use_visibility_weights[time_ind, :, :, vis_pol_ind]
                )
                if self.dwcal_memory_save_mode:
                    normalization_denominator = np.real(
                        np.sum(
                            use_visibility_weights[time_ind, :, :, vis_pol_ind]
                            * weight_mat[time_ind, :, :, vis_pol_ind]
                        )
                    )
                else:
                    normalization_denominator = np.real(
                        np.sum(
                            np.trace(
                                np.sqrt(
                                    use_visibility_weights[
                                        time_ind, :, :, np.newaxis, vis_pol_ind
                                    ]
                                )
                                * np.sqrt(
                                    use_visibility_weights[
                                        time_ind, :, np.newaxis, :, vis_pol_ind
                                    ]
                                )
                                * weight_mat[time_ind, :, :, :, vis_pol_ind],
                                axis1=1,
                                axis2=2,
                            )
                        )
                    )
                normalization_factor = (
                    normalization_numerator / normalization_denominator
                )
                if self.dwcal_memory_save_mode:
                    weight_mat[time_ind, :, :, vis_pol_ind] *= normalization_factor
                else:
                    weight_mat[time_ind, :, :, :, vis_pol_ind] *= normalization_factor

        self.dwcal_inv_covariance = weight_mat
