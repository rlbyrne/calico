import numpy as np
from numpy.typing import NDArray
import sys
import time
import pyuvdata
import multiprocessing
from calico import caldata
from pyuvdata import UVData, UVCal


def sky_based_calibration_wrapper(
    data: str | UVData,
    model: str | UVData,
    data_use_column: str = "DATA",
    model_use_column: str = "MODEL_DATA",
    gain_init_calfile: str | None = None,
    gains_multiply_model: bool = False,
    gain_init_to_vis_ratio: bool = True,
    gain_init_stddev: float = 0.0,
    N_feed_pols: int | None = None,
    feed_polarization_array: NDArray[int] | None = None,
    min_cal_baseline_m: float | None = None,
    max_cal_baseline_m: float | None = None,
    min_cal_baseline_lambda: float | None = None,
    max_cal_baseline_lambda: float | None = None,
    lambda_val: float = 0.0,
    xtol: float = 1e-5,
    maxiter: int = 200,
    get_crosspol_phase: bool = True,
    crosspol_phase_strategy: str = "crosspol model",
    antenna_flagging_iterations: int = 1,
    antenna_flagging_threshold: float = 2.5,
    parallel: bool = True,
    max_processes: int | None = 40,
    verbose: bool = False,
    log_file_path: str | None = None,
) -> UVCal:
    """
    Top-level wrapper for running sky-based calibration per polarization. This is the
    simplest sky-based calibration approach. Function creates a CalData object,
    updates the gains attribute, and returns a pyuvdata UVCal object containing
    the calibration solutions. Here the XX and YY visibilities are calibrated
    individually and the cross-polarization phase is applied from the XY and YX
    visibilities after the fact. Option to parallelize calibration across frequency.

    Parameters
    ----------
    data : str or UVData
        Path to the pyuvdata-readable file containing the data visibilities
        or a pyuvdata UVData object.
    model : str or UVData
        Path to the pyuvdata-readable file containing the model visibilities
        or a pyuvdata UVData object.
    data_use_column : str, default="DATA"
        Column in an ms file to use for the data visibilities. Used only if
        data_file_path points to an ms file.
    model_use_column : str, default="MODEL_DATA"
        Column in an ms file to use for the model visibilities. Used only if
        data_file_path points to an ms file.
    gain_init_calfile : str, optional, default=None
        If not None, provides a path to a pyuvdata-formatted calfits file
        containing gains values for calibration initialization.
    gain_init_to_vis_ratio : bool, default=True
        Used only if gain_init_calfile is None. If True, initializes gains
        to the median ratio between the amplitudes of the model and data
        visibilities. If False, the gains are initialized to 1.
    gains_multiply_model : bool, default=False
        If True, measurement equation is defined as v_ij ≈ g_i g_j^* m_ij. If
        False, measurement equation is defined as g_i g_j^* v_ij ≈ m_ij.
    gain_init_stddev : float, default=0.0
        Standard deviation of a random complex Gaussian perturbation to the
        initial gains.
    N_feed_pols : int, default=min(2, N_vis_pols)
        Number of feed polarizations, equal to the number of gain values to be
        calculated per antenna.
    feed_polarization_array : array of int, optional
        Feed polarizations to calibrate. Shape (N_feed_pols,). Options are
        -5 for X or -6 for Y. Default None. If None, feed_polarization_array
        is set to ([-5, -6])[:N_feed_pols].
    min_cal_baseline_m : float, optional, default=None
        Minimum baseline length, in meters, to use in calibration. If both
        min_cal_baseline_m and min_cal_baseline_lambda are None, arbitrarily
        short baselines are used.
    max_cal_baseline_m : float, optional, default=None
        Maximum baseline length, in meters, to use in calibration. If both
        max_cal_baseline_m and max_cal_baseline_lambda are None, arbitrarily
        long baselines are used.
    min_cal_baseline_lambda : float, optional, default=None
        Minimum baseline length, in wavelengths, to use in calibration. If
        both min_cal_baseline_m and min_cal_baseline_lambda are None,
        arbitrarily short baselines are used.
    max_cal_baseline_lambda : float, optional, default=None
        Maximum baseline length, in wavelengths, to use in calibration. If
        both max_cal_baseline_m and max_cal_baseline_lambda are None,
        arbitrarily long baselines are used.
    lambda_val : float, default=0.0
        Weight of the phase regularization term; must be positive or 0.
    xtol : float, default=1e-5
        Accuracy tolerance for optimizer.
    maxiter : int, default=200
        Maximum number of iterations for the optimizer.
    get_crosspol_phase : bool, default=True
        If True, crosspol phase is calculated.
    crosspol_phase_strategy : str, default="crosspol model"
        Options are "crosspol model" or "pseudo Stokes V". Used only if
        get_crosspol_phase is True. If "crosspol model", contrains the crosspol
        phase using the crosspol model visibilities. If "pseudo Stokes V", constrains
        crosspol phase by minimizing pseudo Stokes V.
    antenna_flagging_iterations : int, default=1
        If >0, pre-calibrate and flag antennas based on the residual per-antenna cost.
    antenna_flagging_threshold : float, default=2.5
        Used only if antenna_flagging_iterations>0. Per antenna cost values equal to
        flagging_threshold times the mean value will be flagged.
    parallel : bool
        Set to True to parallelize across frequency with multiprocessing.
        Default True if Nfreqs > 1.
    max_processes : int, optional, default=40
        Maximum number of multithreaded processes to use. Applicable only if
        parallel is True. If None, uses the multiprocessing default.
    verbose : bool, default=False
        Set to True to print optimization outputs.
    log_file_path : str, optional, default=None
        Path to the log file.

    Returns
    -------
    UVCal
        A UVCal object containing the calibrated complex antenna gains.
    """

    if log_file_path is not None:
        stdout_orig = sys.stdout
        stderr_orig = sys.stderr
        sys.stdout = sys.stderr = log_file_new = open(log_file_path, "w")

    start_time = time.time()

    if parallel:  # Start multiprocessing pool
        if max_processes is None:
            pool = multiprocessing.Pool()
        else:
            pool = multiprocessing.Pool(processes=max_processes)
    else:
        pool = None

    if verbose:
        data_read_start_time = time.time()

    print_data_read_time = False
    if isinstance(data, str):  # Read data
        data_file_path = data
        data = pyuvdata.UVData()
        if data_file_path.endswith(".ms"):
            data.read_ms(
                data_file_path,
                data_column=data_use_column,
                ignore_single_chan=False,
            )
        elif data_file_path.endswith(".uvfits"):
            data.read_uvfits(data_file_path)
        else:
            data.read(data_file_path)
        print_data_read_time = True
    if isinstance(model, str):  # Read model
        model_file_path = model
        model = pyuvdata.UVData()
        if model_file_path.endswith(".ms"):
            model.read_ms(
                model_file_path,
                data_column=model_use_column,
                ignore_single_chan=False,
            )
        elif model_file_path.endswith(".uvfits"):
            model.read_uvfits(model_file_path)
        else:
            model.read(model_file_path)
        print_data_read_time = True

    # Ensure data and model are phased the same
    data.phase_to_time(np.mean(data.time_array))
    model.phase_to_time(np.mean(data.time_array))

    if verbose:
        if print_data_read_time:
            print(
                f"Done. Data read time {(time.time() - data_read_start_time)/60.} minutes."
            )
        print("Formatting data...")
        sys.stdout.flush()
        data_format_start_time = time.time()

    caldata_obj = caldata.CalData()
    caldata_obj.load_data(
        data=data,
        model=model,
        gain_init_calfile=gain_init_calfile,
        gain_init_to_vis_ratio=gain_init_to_vis_ratio,
        gains_multiply_model=gains_multiply_model,
        gain_init_stddev=gain_init_stddev,
        N_feed_pols=N_feed_pols,
        feed_polarization_array=feed_polarization_array,
        min_cal_baseline_m=min_cal_baseline_m,
        max_cal_baseline_m=max_cal_baseline_m,
        min_cal_baseline_lambda=min_cal_baseline_lambda,
        max_cal_baseline_lambda=max_cal_baseline_lambda,
        lambda_val=lambda_val,
    )

    if caldata_obj.Nfreqs < 2:  # Do not parallelize
        parallel = False
        if pool is not None:
            pool.terminate()
        pool = None

    if verbose:
        print(
            f"Done. Data formatting time {(time.time() - data_format_start_time)/60.} minutes."
        )
        print("Running calibration optimization...")
        sys.stdout.flush()
        optimization_start_time = time.time()

    for ant_flag_iter in range(antenna_flagging_iterations):
        caldata_obj.sky_based_calibration(
            xtol=xtol * 10,  # Higher tolerance for antenna flagging
            maxiter=int(maxiter / 2),  # Lower maxiter for antenna flagging
            get_crosspol_phase=False,  # No crosspol phase needed for antenna flagging
            parallel=parallel,
            verbose=verbose,
            pool=pool,
        )
        if verbose:
            print(
                f"Optimization time: {caldata_obj.Nfreqs} frequency channels in {(time.time() - optimization_start_time)/60.} minutes."
            )
            print(
                f"Initial calibration optimization done. Antenna flagging iteration {ant_flag_iter+1} of {antenna_flagging_iterations}."
            )
            sys.stdout.flush()
        caldata_obj.flag_antennas_from_per_ant_cost(
            flagging_threshold=antenna_flagging_threshold,
            verbose=verbose,
        )

    caldata_obj.sky_based_calibration(
        xtol=xtol,
        maxiter=maxiter,
        get_crosspol_phase=get_crosspol_phase,
        crosspol_phase_strategy=crosspol_phase_strategy,
        parallel=parallel,
        verbose=verbose,
        pool=pool,
    )
    if verbose:
        print(
            f"Done. Optimization time: {caldata_obj.Nfreqs} frequency channels in {(time.time() - optimization_start_time)/60.} minutes"
        )
        sys.stdout.flush()

    if parallel:
        pool.terminate()

    # Convert to UVCal object
    uvcal = caldata_obj.convert_to_uvcal()

    if verbose:
        print(f"Total processing time {(time.time() - start_time)/60.} minutes.")
        sys.stdout.flush()

    if log_file_path is not None:
        sys.stdout = stdout_orig
        sys.stderr = stderr_orig
        log_file_new.close()

    return uvcal


def abscal_wrapper(
    data: str | UVData,
    model: str | UVData,
    data_use_column: str = "DATA",
    model_use_column: str = "MODEL_DATA",
    N_feed_pols: int | None = None,
    feed_polarization_array: NDArray[int] | None = None,
    gains_multiply_model: bool = False,
    min_cal_baseline_m: float | None = None,
    max_cal_baseline_m: float | None = None,
    min_cal_baseline_lambda: float | None = None,
    max_cal_baseline_lambda: float | None = None,
    xtol: float = 1e-4,
    maxiter: int = 100,
    verbose: bool = False,
    log_file_path: str | None = None,
) -> NDArray[np.floating]:
    """
    Top-level wrapper for running absolute calibration ("abscal").

    Parameters
    ----------
    data : str or UVData
        Path to the pyuvdata-readable file containing the relatively calibrated
        data visibilities or a pyuvdata UVData object.
    model : str or UVData
        Path to the pyuvdata-readable file containing the model visibilities
        or a pyuvdata UVData object.
    data_use_column : str, default="DATA"
        Column in an ms file to use for the data visibilities. Used only if
        data_file_path points to an ms file.
    model_use_column : str, default="MODEL_DATA"
        Column in an ms file to use for the model visibilities. Used only if
        data_file_path points to an ms file.
    N_feed_pols : int, default=min(2, N_vis_pols)
        Number of feed polarizations, equal to the number of gain values to be
        calculated per antenna.
    feed_polarization_array : array of int, optional
        Feed polarizations to calibrate. Shape (N_feed_pols,). Options are
        -5 for X or -6 for Y. Default None. If None, feed_polarization_array
        is set to ([-5, -6])[:N_feed_pols].
    gains_multiply_model : bool, default=False
        If True, the abscal parameters multiply the model visibilities.
    min_cal_baseline_m : float, optional, default=None
        Minimum baseline length, in meters, to use in calibration. If both
        min_cal_baseline_m and min_cal_baseline_lambda are None, arbitrarily
        short baselines are used.
    max_cal_baseline_m : float, optional, default=None
        Maximum baseline length, in meters, to use in calibration. If both
        max_cal_baseline_m and max_cal_baseline_lambda are None, arbitrarily
        long baselines are used.
    min_cal_baseline_lambda : float, optional, default=None
        Minimum baseline length, in wavelengths, to use in calibration. If
        both min_cal_baseline_m and min_cal_baseline_lambda are None,
        arbitrarily short baselines are used.
    max_cal_baseline_lambda : float, optional, default=None
        Maximum baseline length, in wavelengths, to use in calibration. If
        both max_cal_baseline_m and max_cal_baseline_lambda are None,
        arbitrarily long baselines are used. Default None.
    xtol : float, default=1e-4
        Accuracy tolerance for optimizer.
    maxiter : int, default=100
        Maximum number of iterations for the optimizer.
    verbose : bool, default=False
        Set to True to print optimization outputs.
    log_file_path : str, optional, default=None
        Path to the log file.

    Returns
    -------
    ndarray of float, shape (3, Nfreqs, N_feed_pols)
        Array of solved abscal parameters. abscal_params[0, :, :] are the overall amplitudes,
        abscal_params[1, :, :] are the x-phase gradients in units 1/m, and abscal_params[2, :, :]
        are the y-phase gradients in units 1/m.
    """

    if log_file_path is not None:
        stdout_orig = sys.stdout
        stderr_orig = sys.stderr
        sys.stdout = sys.stderr = log_file_new = open(log_file_path, "w")

    start_time = time.time()

    data_read_start_time = time.time()
    print_data_read_time = False
    if isinstance(data, str):  # Read data
        if verbose:
            print("Reading data...")
            sys.stdout.flush()
        print_data_read_time = True
        data_file_path = np.copy(data)
        data = pyuvdata.UVData()
        if data_file_path.endswith(".ms"):
            data.read_ms(data_file_path, data_column=data_use_column)
        else:
            data.read(data_file_path)
    if isinstance(model, str):  # Read model
        if verbose:
            print("Reading model...")
            sys.stdout.flush()
        print_data_read_time = True
        model_file_path = np.copy(model)
        model = pyuvdata.UVData()
        if model_file_path.endswith(".ms"):
            model.read_ms(model_file_path, data_column=model_use_column)
        else:
            model.read(model_file_path)

    if verbose and print_data_read_time:
        print(
            f"Done. Data read time {(time.time() - data_read_start_time)/60.} minutes."
        )
        sys.stdout.flush()
    if verbose:
        print("Formatting data...")
        sys.stdout.flush()
        data_format_start_time = time.time()

    caldata_obj = caldata.CalData()
    caldata_obj.load_data(
        data,
        model,
        N_feed_pols=N_feed_pols,
        feed_polarization_array=feed_polarization_array,
        gains_multiply_model=gains_multiply_model,
        min_cal_baseline_m=min_cal_baseline_m,
        max_cal_baseline_m=max_cal_baseline_m,
        min_cal_baseline_lambda=min_cal_baseline_lambda,
        max_cal_baseline_lambda=max_cal_baseline_lambda,
    )

    if verbose:
        print(
            f"Done. Data formatting time {(time.time() - data_format_start_time)/60.} minutes."
        )
        print("Running calibration optimization...")
        sys.stdout.flush()

    optimization_start_time = time.time()

    caldata_obj.abscal(xtol=xtol, maxiter=maxiter, verbose=verbose)

    if verbose:
        print(
            f"Done. Optimization time: {caldata_obj.Nfreqs} frequency channels in {(time.time() - optimization_start_time)/60.} minutes"
        )
        print(f"Total processing time {(time.time() - start_time)/60.} minutes.")
        sys.stdout.flush()

    if log_file_path is not None:
        sys.stdout = stdout_orig
        sys.stderr = stderr_orig
        log_file_new.close()

    return caldata_obj.abscal_params


def dw_absolute_calibration(
    data: str | UVData,
    model: str | UVData,
    delay_spectrum_variance: NDArray[np.floating],
    bl_length_bin_edges: NDArray[np.floating],
    delay_axis: NDArray[np.floating],
    data_use_column: str = "DATA",
    model_use_column: str = "MODEL_DATA",
    initial_abscal_params: NDArray[np.floating] | None = None,
    gains_multiply_model: bool = False,
    N_feed_pols: int | None = None,
    feed_polarization_array: NDArray[int] | None = None,
    min_cal_baseline_m: float | None = None,
    max_cal_baseline_m: float | None = None,
    min_cal_baseline_lambda: float | None = None,
    max_cal_baseline_lambda: float | None = None,
    xtol: float = 1e-6,
    maxiter: int = 100,
    verbose: bool = False,
    log_file_path: str | None = None,
) -> NDArray[np.floating]:
    """
    Top-level wrapper for running absolute calibration ("abscal") with delay weighting.

    Parameters
    ----------
    data : str or UVData
        Path to the pyuvdata-readable file containing the relatively calibrated
        data visibilities or a pyuvdata UVData object.
    model : str or UVData
        Path to the pyuvdata-readable file containing the model visibilities
        or a pyuvdata UVData object.
    delay_spectrum_variance : array of float, shape (Nbins, Ndelays)
        Array containing the expected variance as a function of baseline length and delay.
    bl_length_bin_edges : array of float, shape (Nbins+1,)
        Defines the baseline length axis of delay_spectrum_variance. Values correspond to
        limits of each baseline length bin.
    delay_axis : array of float, shape (Ndelays,)
        Defines the delay axis of delay_spectrum_variance.
    data_use_column : str, default="DATA"
        Column in an ms file to use for the data visibilities. Used only if
        data_file_path points to an ms file.
    model_use_column : str, default="MODEL_DATA"
        Column in an ms file to use for the model visibilities. Used only if
        data_file_path points to an ms file.
    initial_abscal_params : array of float, shape (3, Nfreqs, N_feed_pols)
        Parameters to initialize with. abscal_params[0, :, :] are the overall amplitudes,
        abscal_params[1, :, :] are the x-phase gradients in units 1/m, and
        abscal_params[2, :, :] are the y-phase gradients in units 1/m. Currently the
        frequency and polarization axes must match those in the data (this should be fixed).
    gains_multiply_model : bool, default=False
        If True, the abscal parameters multiply the model visibilities.
    N_feed_pols : int, default=min(2, N_vis_pols)
        Number of feed polarizations, equal to the number of gain values to be calculated
        per antenna.
    feed_polarization_array : array of int, optional, shape (N_feed_pols,), default=None
        Feed polarizations to calibrate. Options are -5 for X or -6 for Y.
        If None, feed_polarization_array is set to ([-5, -6])[:N_feed_pols].
    min_cal_baseline_m : float, optional, default=None
        Minimum baseline length, in meters, to use in calibration. If both
        min_cal_baseline_m and min_cal_baseline_lambda are None, arbitrarily
        short baselines are used.
    max_cal_baseline_m : float, optional, default=None
        Maximum baseline length, in meters, to use in calibration. If both
        max_cal_baseline_m and max_cal_baseline_lambda are None, arbitrarily
        long baselines are used.
    min_cal_baseline_lambda : float, optional, default=None
        Minimum baseline length, in wavelengths, to use in calibration. If
        both min_cal_baseline_m and min_cal_baseline_lambda are None,
        arbitrarily short baselines are used.
    max_cal_baseline_lambda : float, optional, default=None
        Maximum baseline length, in wavelengths, to use in calibration. If
        both max_cal_baseline_m and max_cal_baseline_lambda are None,
        arbitrarily long baselines are used.
    xtol : float, default=1e-6
        Accuracy tolerance for optimizer.
    maxiter : int, default=100
        Maximum number of iterations for the optimizer.
    verbose : bool, default=False
        Set to True to print optimization outputs. Default False.
    log_file_path : str, optional, default=None
        Path to the log file.

    Returns
    -------
    ndarray of float, shape (3, Nfreqs, N_feed_pols)
        abscal_params[0, :, :] are the overall amplitudes, abscal_params[1, :, :] are the
        x-phase gradients in units 1/m, and abscal_params[2, :, :] are the y-phase gradients
        in units 1/m.
    """

    if log_file_path is not None:
        stdout_orig = sys.stdout
        stderr_orig = sys.stderr
        sys.stdout = sys.stderr = log_file_new = open(log_file_path, "w")

    start_time = time.time()

    if verbose:
        print("Reading data...")
        sys.stdout.flush()
        data_read_start_time = time.time()

    print_data_read_time = False
    if isinstance(data, str):  # Read data
        print_data_read_time = True
        data_file_path = np.copy(data)
        data = pyuvdata.UVData()
        if data_file_path.endswith(".ms"):
            data.read_ms(data_file_path, data_column=data_use_column)
        else:
            data.read(data_file_path)
    if isinstance(model, str):  # Read model
        print_data_read_time = True
        model_file_path = np.copy(model)
        model = pyuvdata.UVData()
        if model_file_path.endswith(".ms"):
            model.read_ms(model_file_path, data_column=model_use_column)
        else:
            model.read(model_file_path)

    if verbose and print_data_read_time:
        print(
            f"Done. Data read time {(time.time() - data_read_start_time)/60.} minutes."
        )
        sys.stdout.flush()
    if verbose:
        print("Formatting data...")
        sys.stdout.flush()
        data_format_start_time = time.time()

    caldata_obj = caldata.CalData()
    caldata_obj.load_data(
        data,
        model,
        N_feed_pols=N_feed_pols,
        feed_polarization_array=feed_polarization_array,
        gains_multiply_model=gains_multiply_model,
        min_cal_baseline_m=min_cal_baseline_m,
        max_cal_baseline_m=max_cal_baseline_m,
        min_cal_baseline_lambda=min_cal_baseline_lambda,
        max_cal_baseline_lambda=max_cal_baseline_lambda,
    )

    if initial_abscal_params is not None:
        caldata_obj.abscal_params = initial_abscal_params

    if verbose:
        print(
            f"Done. Data formatting time {(time.time() - data_format_start_time)/60.} minutes."
        )
        print("Calculating delay weighting matrix...")
        sys.stdout.flush()

    caldata_obj.get_dwcal_weights_from_delay_spectra(
        delay_spectrum_variance,
        bl_length_bin_edges,
        delay_axis,
    )

    if verbose:
        print(
            f"Done. Time calculating delay weighting matrix {(time.time() - data_format_start_time)/60.} minutes."
        )
        print("Running calibration optimization...")
        sys.stdout.flush()
        optimization_start_time = time.time()

    caldata_obj.dw_abscal(xtol=xtol, maxiter=maxiter, verbose=verbose)

    if verbose:
        print(
            f"Done. Optimization time: {caldata_obj.Nfreqs} frequency channels in {(time.time() - optimization_start_time)/60.} minutes"
        )
        print(f"Total processing time {(time.time() - start_time)/60.} minutes.")
        sys.stdout.flush()

    if log_file_path is not None:
        sys.stdout = stdout_orig
        sys.stderr = stderr_orig
        log_file_new.close()

    return caldata_obj.abscal_params


def apply_abscal(
    uvdata: UVData,
    abscal_params: NDArray[np.floating],
    feed_polarization_array: NDArray[np.int_],
    gains_multiply_model: bool = False,
    inplace: bool = False,
) -> UVData | None:
    """
    Apply absolute calibration solutions to data.

    Parameters
    ----------
    uvdata : pyuvdata UVData object
        pyuvdata UVData object containing the data.
    abscal_params : array of float, shape (3, Nfreqs, N_feed_pols)
        abscal_params[0, :, :] are the overall amplitudes, abscal_params[1, :, :] are the
        x-phase gradients in units 1/m, and abscal_params[2, :, :] are the y-phase gradients
        in units 1/m.
    feed_polarization_array : array of int, shape (N_feed_pols,)
        Array of polarization integers. Indicates the ordering of the polarization axis of
        the gains. X is -5 and Y is -6.
    gains_multiply_model : bool, default=False
        If True, the data is divided by the abscal term. If False, data is multiplied by the abscal
        term.
    inplace : bool, default=False
        If True, updates uvdata. If False, returns a new UVData object.

    Returns
    -------
    UVData or None
        Calibrated UVData object if `inplace=False`, otherwise None.
    """

    if not inplace:
        uvdata_new = uvdata.copy()

    # Get antenna locations
    # Create gains expand matrices
    gains_exp_mat_1 = np.zeros(
        (uvdata.Nblts, len(uvdata.telescope.antenna_numbers)), dtype=int
    )
    gains_exp_mat_2 = np.zeros(
        (uvdata.Nblts, len(uvdata.telescope.antenna_numbers)), dtype=int
    )
    for baseline in range(uvdata.Nblts):
        gains_exp_mat_1[
            baseline,
            np.where(uvdata.telescope.antenna_numbers == uvdata.ant_1_array[baseline]),
        ] = 1
        gains_exp_mat_2[
            baseline,
            np.where(uvdata.telescope.antenna_numbers == uvdata.ant_2_array[baseline]),
        ] = 1
    antpos_enu = uvdata.telescope.get_enu_antpos()
    antpos_en = antpos_enu[:, :2]
    ant1_positions = np.matmul(gains_exp_mat_1, antpos_en)
    ant2_positions = np.matmul(gains_exp_mat_2, antpos_en)

    for vis_pol_ind, vis_pol in enumerate(uvdata.polarization_array):
        if vis_pol == -5:
            pol1 = pol2 = np.where(feed_polarization_array == -5)[0][0]
        elif vis_pol == -6:
            pol1 = pol2 = np.where(feed_polarization_array == -6)[0][0]
        elif vis_pol == -7:
            pol1 = np.where(feed_polarization_array == -5)[0][0]
            pol2 = np.where(feed_polarization_array == -6)[0][0]
        elif vis_pol == -8:
            pol1 = np.where(feed_polarization_array == -6)[0][0]
            pol2 = np.where(feed_polarization_array == -5)[0][0]
        else:
            print(f"ERROR: Polarization {vis_pol} not recognized.")
            sys.exit(1)

        amp_term = (
            abscal_params[0, :, pol1] * abscal_params[0, :, pol2]
        )  # Shape (Nfreqs,)
        phase_correction = np.exp(
            1j
            * (
                abscal_params[1, np.newaxis, :, pol1] * ant1_positions[:, np.newaxis, 0]
                - abscal_params[1, np.newaxis, :, pol2]
                * ant2_positions[:, np.newaxis, 0]
                + abscal_params[2, np.newaxis, :, pol1]
                * ant1_positions[:, np.newaxis, 1]
                - abscal_params[2, np.newaxis, :, pol2]
                * ant2_positions[:, np.newaxis, 1]
            )
        )  # Shape (Nbls, Nfreqs,)

        if inplace:
            if gains_multiply_model:
                uvdata.data_array[:, :, vis_pol_ind] /= (
                    amp_term[np.newaxis, :] * phase_correction
                )
            else:
                uvdata.data_array[:, :, vis_pol_ind] *= (
                    amp_term[np.newaxis, :] * phase_correction
                )
        else:
            if gains_multiply_model:
                uvdata_new.data_array[:, :, vis_pol_ind] /= (
                    amp_term[np.newaxis, :] * phase_correction
                )
            else:
                uvdata_new.data_array[:, :, vis_pol_ind] *= (
                    amp_term[np.newaxis, :] * phase_correction
                )

    if not inplace:
        return uvdata_new
