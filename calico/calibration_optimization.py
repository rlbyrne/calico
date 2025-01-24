import numpy as np
import sys
import scipy
import scipy.optimize
import time
from calico import cost_function_calculations

"""
    Takes 3D array of 2D Hessian parts and flattens them into one 2D Hessian
    to return to scipy. For now, expected shape has third dimension of three 
    parts: real-real, real-imag, imag-imag. This is for gains only. For the 
    future, general case with model params there will be additional pieces 
    for those -- also RR, RI, and II -- as well as cross terms for u and g 
    that will constitute additional expected elements in that dimension of
    the array, handled similarly. This algorithm might update in the future.
"""
def flatten_hessian(
    hess_arrays,
    Nants_unflagged,
):
    hess_flattened = np.full(
        (2 * Nants_unflagged, 2 * Nants_unflagged), np.nan, dtype=float
    )
    for ant_ind_1 in range(Nants_unflagged):
        for ant_ind_2 in range(Nants_unflagged):
            hess_flattened[2 * ant_ind_1, 2 * ant_ind_2] = hess_arrays[0][
                ant_ind_1, ant_ind_2
            ]
            hess_flattened[2 * ant_ind_1 + 1, 2 * ant_ind_2] = hess_arrays[1][
                ant_ind_1, ant_ind_2
            ]
            hess_flattened[2 * ant_ind_1, 2 * ant_ind_2 + 1] = np.conj(
                hess_arrays[1][ant_ind_2, ant_ind_1]
            )
            hess_flattened[2 * ant_ind_1 + 1, 2 * ant_ind_2 + 1] = hess_arrays[2][
                ant_ind_1, ant_ind_2
            ]
    return hess_flattened

def cost_skycal_wrapper(
    gains_flattened,
    caldata_obj,
    ant_inds,
):
    """
    Wrapper for function cost_skycal. Reformats the input gains to be compatible
    with the scipy.optimize.minimize function.

    Parameters
    ----------
    gains_flattened : array of float
        Array of gain values. Even indices correspond to the real components of the
        gains and odd indices correspond to the imaginary components. Shape
        (2*Nants_unflagged,).
    caldata_obj : CalData
    ant_inds : array of int
        Indices of unflagged antennas to be calibrated. Shape (Nants_unflagged,).
    freq_ind : int
        Frequency channel index.
    vis_pol_ind : int
        Index of the visibility polarization.

    Returns
    -------
    cost : float
        Value of the cost function.
    """

    gains_reshaped = np.reshape(gains_flattened, (len(ant_inds), 2))
    gains_reshaped = gains_reshaped[:, 0] + 1.0j * gains_reshaped[:, 1]
    gains = np.ones((caldata_obj.Nants), dtype=complex)
    gains[ant_inds] = gains_reshaped
    if caldata_obj.gains_multiply_model:
        cost = cost_function_calculations.cost_skycal(
            gains,
            caldata_obj.data_vis_reshaped,
            caldata_obj.model_vis_reshaped,
            caldata_obj.vis_weights_reshaped,
            caldata_obj.ant1_inds,
            caldata_obj.ant2_inds,
            caldata_obj.lambda_val,
        )
    else:
        cost = cost_function_calculations.cost_skycal(
            gains,
            caldata_obj.model_vis_reshaped,
            caldata_obj.data_vis_reshaped,
            caldata_obj.vis_weights_reshaped,
            caldata_obj.ant1_inds,
            caldata_obj.ant2_inds,
            caldata_obj.lambda_val,
        )
    return cost


def jacobian_skycal_wrapper(
    gains_flattened,
    caldata_obj,
    ant_inds,
):
    """
    Wrapper for function jacobian_skycal. Reformats the input gains and
    output Jacobian to be compatible with the scipy.optimize.minimize function.

    Parameters
    ----------
    gains_flattened : array of float
        Array of gain values. Even indices correspond to the real components of the
        gains and odd indices correspond to the imaginary components. Shape
        (2*Nants_unflagged,).
    caldata_obj : CalData
    ant_inds : array of int
        Indices of unflagged antennas to be calibrated. Shape (Nants_unflagged,).
    freq_ind : int
        Frequency channel index.
    vis_pol_ind : int
        Index of the visibility polarization.

    Returns
    -------
    jac_flattened : array of float
        Jacobian of the cost function, shape (2*Nants_unflagged,).
        Even indices correspond to the derivatives with respect to the real part
        of the gains and odd indices correspond to derivatives with respect to
        the imaginary part of the gains.
    """

    gains_reshaped = np.reshape(gains_flattened, (len(ant_inds), 2))
    gains_reshaped = gains_reshaped[:, 0] + 1.0j * gains_reshaped[:, 1]
    gains = np.ones((caldata_obj.Nants), dtype=complex)
    gains[ant_inds] = gains_reshaped
    if caldata_obj.gains_multiply_model:
        jac = cost_function_calculations.jacobian_skycal(
            gains,
            caldata_obj.data_vis_reshaped,
            caldata_obj.model_vis_reshaped,
            caldata_obj.vis_weights_reshaped,
            caldata_obj.ant1_inds,
            caldata_obj.ant2_inds,
            caldata_obj.lambda_val,
        )
    else:
        jac = cost_function_calculations.jacobian_skycal(
            gains,
            caldata_obj.model_vis_reshaped,
            caldata_obj.data_vis_reshaped,
            caldata_obj.vis_weights_reshaped,
            caldata_obj.ant1_inds,
            caldata_obj.ant2_inds,
            caldata_obj.lambda_val,
        )
    jac_flattened = np.stack(
        (np.real(jac[ant_inds]), np.imag(jac[ant_inds])), axis=1
    ).flatten()
    return jac_flattened


def hessian_skycal_wrapper(
    gains_flattened,
    caldata_obj,
    ant_inds,
):
    """
    Wrapper for function hessian_skycal. Reformats the input gains and
    output Hessian to be compatible with the scipy.optimize.minimize function.

    Parameters
    ----------
    gains_flattened : array of float
        Array of gain values. Even indices correspond to the real components of the
        gains and odd indices correspond to the imaginary components. Shape
        (2*Nants_unflagged,).
    caldata_obj : CalData
    ant_inds : array of int
        Indices of unflagged antennas to be calibrated. Shape (Nants_unflagged,).
    freq_ind : int
        Frequency channel index.
    vis_pol_ind : int
        Index of the visibility polarization.

    Returns
    -------
    hess_flattened : array of float
        Hessian of the cost function, shape (2*Nants_unflagged, 2*Nants_unflagged,).
    """

    Nants_unflagged = len(ant_inds)
    gains_reshaped = np.reshape(gains_flattened, (Nants_unflagged, 2))
    gains_reshaped = gains_reshaped[:, 0] + 1.0j * gains_reshaped[:, 1]
    gains = np.ones((caldata_obj.Nants), dtype=complex)
    gains[ant_inds] = gains_reshaped
    if caldata_obj.gains_multiply_model:
        (
            hess_real_real,
            hess_real_imag,
            hess_imag_imag,
        ) = cost_function_calculations.hessian_skycal(
            gains,
            caldata_obj.Nants,
            caldata_obj.Nbls,
            caldata_obj.data_vis_reshaped,
            caldata_obj.model_vis_reshaped,
            caldata_obj.vis_weights_reshaped,
            caldata_obj.ant1_inds,
            caldata_obj.ant2_inds,
            caldata_obj.lambda_val,
        )
    else:
        (
            hess_real_real,
            hess_real_imag,
            hess_imag_imag,
        ) = cost_function_calculations.hessian_skycal(
            gains,
            caldata_obj.Nants,
            caldata_obj.Nbls,
            caldata_obj.model_vis_reshaped,
            caldata_obj.data_vis_reshaped,
            caldata_obj.vis_weights_reshaped,
            caldata_obj.ant1_inds,
            caldata_obj.ant2_inds,
            caldata_obj.lambda_val,
        )

    return flatten_hessian(
        [
            hess_real_real,
            hess_real_imag,
            hess_imag_imag,
        ],
        Nants_unflagged,
    )


def cost_abscal_wrapper(abscal_parameters, caldata_obj):
    """
    Wrapper for function cost_function_abs_cal.

    Parameters
    ----------
    abscal_parameters : array of float
        Shape (3,).
    caldata_obj : CalData

    Returns
    -------
    cost : float
        Value of the cost function.
    """

    if caldata_obj.gains_multiply_model:
        cost = cost_function_calculations.cost_function_abs_cal(
            abscal_parameters[0],
            abscal_parameters[1:],
            caldata_obj.data_visibilities[:, :, 0, 0],
            caldata_obj.model_visibilities[:, :, 0, 0],
            caldata_obj.uv_array,
            caldata_obj.visibility_weights[:, :, 0, 0],
        )
    else:
        cost = cost_function_calculations.cost_function_abs_cal(
            abscal_parameters[0],
            abscal_parameters[1:],
            caldata_obj.model_visibilities[:, :, 0, 0],
            caldata_obj.data_visibilities[:, :, 0, 0],
            caldata_obj.uv_array,
            caldata_obj.visibility_weights[:, :, 0, 0],
        )
    return cost


def jacobian_abscal_wrapper(abscal_parameters, caldata_obj):
    """
    Wrapper for function jacobian_abs_cal.

    Parameters
    ----------
    abscal_parameters : array of float
        Shape (3,).
    caldata_obj : CalData

    Returns
    -------
    jac : array of float
        Shape (3,).
    """

    jac = np.zeros((3,), dtype=float)
    if caldata_obj.gains_multiply_model:
        amp_jac, phase_jac = cost_function_calculations.jacobian_abs_cal(
            abscal_parameters[0],
            abscal_parameters[1:],
            caldata_obj.data_visibilities[:, :, 0, 0],
            caldata_obj.model_visibilities[:, :, 0, 0],
            caldata_obj.uv_array,
            caldata_obj.visibility_weights[:, :, 0, 0],
        )
    else:
        amp_jac, phase_jac = cost_function_calculations.jacobian_abs_cal(
            abscal_parameters[0],
            abscal_parameters[1:],
            caldata_obj.model_visibilities[:, :, 0, 0],
            caldata_obj.data_visibilities[:, :, 0, 0],
            caldata_obj.uv_array,
            caldata_obj.visibility_weights[:, :, 0, 0],
        )
    jac[0] = amp_jac
    jac[1:] = phase_jac
    return jac


def hessian_abscal_wrapper(abscal_parameters, caldata_obj):
    """
    Wrapper for function hess_abs_cal.

    Parameters
    ----------
    abscal_parameters : array of float
        Shape (3,).
    caldata_obj : CalData

    Returns
    -------
    hess : array of float
        Shape (3, 3,).
    """

    hess = np.zeros((3, 3), dtype=float)
    if caldata_obj.gains_multiply_model:
        (
            hess_amp_amp,
            hess_amp_phasex,
            hess_amp_phasey,
            hess_phasex_phasex,
            hess_phasey_phasey,
            hess_phasex_phasey,
        ) = cost_function_calculations.hess_abs_cal(
            abscal_parameters[0],
            abscal_parameters[1:],
            caldata_obj.data_visibilities[:, :, 0, 0],
            caldata_obj.model_visibilities[:, :, 0, 0],
            caldata_obj.uv_array,
            caldata_obj.visibility_weights[:, :, 0, 0],
        )
    else:
        (
            hess_amp_amp,
            hess_amp_phasex,
            hess_amp_phasey,
            hess_phasex_phasex,
            hess_phasey_phasey,
            hess_phasex_phasey,
        ) = cost_function_calculations.hess_abs_cal(
            abscal_parameters[0],
            abscal_parameters[1:],
            caldata_obj.model_visibilities[:, :, 0, 0],
            caldata_obj.data_visibilities[:, :, 0, 0],
            caldata_obj.uv_array,
            caldata_obj.visibility_weights[:, :, 0, 0],
        )
    hess[0, 0] = hess_amp_amp
    hess[0, 1] = hess[1, 0] = hess_amp_phasex
    hess[0, 2] = hess[2, 0] = hess_amp_phasey
    hess[1, 1] = hess_phasex_phasex
    hess[2, 2] = hess_phasey_phasey
    hess[1, 2] = hess[2, 1] = hess_phasex_phasey
    return hess


def cost_dw_abscal_wrapper(
    abscal_parameters_flattened, unflagged_freq_inds, caldata_obj
):
    """
    Wrapper for function cost_function_dw_abscal.

    Parameters
    ----------
    abscal_parameters_flattened : array of float
        Abscal parameters, flattened across the frequency axis. Shape (3 * Nfreqs_unflagged,).
    unflagged_freq_inds : array of int
        Array of indices of frequency channels that are not fully flagged. Shape (Nfreqs_unflagged,).
    caldata_obj : CalData

    Returns
    -------
    cost : float
        Value of the cost function.
    """

    abscal_parameters = np.zeros((3, caldata_obj.Nfreqs))
    abscal_parameters[:, unflagged_freq_inds] = np.reshape(
        abscal_parameters_flattened, (3, len(unflagged_freq_inds))
    )
    if caldata_obj.gains_multiply_model:
        visibility_values_1 = caldata_obj.data_visibilities[:, :, :, 0]
        visibility_values_2 = caldata_obj.model_visibilities[:, :, :, 0]
    else:
        visibility_values_1 = caldata_obj.model_visibilities[:, :, :, 0]
        visibility_values_2 = caldata_obj.data_visibilities[:, :, :, 0]
    if caldata_obj.dwcal_memory_save_mode:
        cost = cost_function_calculations.cost_function_dw_abscal_toeplitz(
            abscal_parameters[0, :],
            abscal_parameters[1:, :],
            visibility_values_1,
            visibility_values_2,
            caldata_obj.uv_array,
            caldata_obj.visibility_weights[:, :, :, 0],
            caldata_obj.dwcal_inv_covariance[:, :, :, 0],
        )
    else:
        cost = cost_function_calculations.cost_function_dw_abscal(
            abscal_parameters[0, :],
            abscal_parameters[1:, :],
            visibility_values_1,
            visibility_values_2,
            caldata_obj.uv_array,
            caldata_obj.visibility_weights[:, :, :, 0],
            caldata_obj.dwcal_inv_covariance[:, :, :, :, 0],
        )
    return cost


def jacobian_dw_abscal_wrapper(
    abscal_parameters_flattened, unflagged_freq_inds, caldata_obj
):
    """
    Wrapper for function jacobian_dw_abscal.

    Parameters
    ----------
    abscal_parameters_flattened : array of float
        Abscal parameters, flattened across the frequency axis. Shape (3 * Nfreqs_unflagged,).
    unflagged_freq_inds : array of int
        Array of indices of frequency channels that are not fully flagged. Shape (Nfreqs_unflagged,).
    caldata_obj : CalData

    Returns
    -------
    jac_flattened : array of float
        Flattened array of derivatives of the cost function with respect to the abscal
        parameters. Shape (3 * Nfreqs,).
    """

    abscal_parameters = np.zeros((3, caldata_obj.Nfreqs))
    abscal_parameters[:, unflagged_freq_inds] = np.reshape(
        abscal_parameters_flattened, (3, len(unflagged_freq_inds))
    )
    if caldata_obj.gains_multiply_model:
        visibility_values_1 = caldata_obj.data_visibilities[:, :, :, 0]
        visibility_values_2 = caldata_obj.model_visibilities[:, :, :, 0]
    else:
        visibility_values_1 = caldata_obj.model_visibilities[:, :, :, 0]
        visibility_values_2 = caldata_obj.data_visibilities[:, :, :, 0]
    if caldata_obj.dwcal_memory_save_mode:
        amp_jac, phase_jac = cost_function_calculations.jacobian_dw_abscal_toeplitz(
            abscal_parameters[0, :],
            abscal_parameters[1:, :],
            visibility_values_1,
            visibility_values_2,
            caldata_obj.uv_array,
            caldata_obj.visibility_weights[:, :, :, 0],
            caldata_obj.dwcal_inv_covariance[:, :, :, 0],
        )
    else:
        amp_jac, phase_jac = cost_function_calculations.jacobian_dw_abscal(
            abscal_parameters[0, :],
            abscal_parameters[1:, :],
            visibility_values_1,
            visibility_values_2,
            caldata_obj.uv_array,
            caldata_obj.visibility_weights[:, :, :, 0],
            caldata_obj.dwcal_inv_covariance[:, :, :, :, 0],
        )
    jac_array = np.zeros((3, caldata_obj.Nfreqs), dtype=float)
    jac_array[0, :] = amp_jac
    jac_array[1:, :] = phase_jac
    jac_array = np.take(jac_array, unflagged_freq_inds, axis=1)
    return jac_array.flatten()


def hessian_dw_abscal_wrapper(
    abscal_parameters_flattened, unflagged_freq_inds, caldata_obj
):
    """
    Wrapper for function hess_dw_abscal.

    Parameters
    ----------
    abscal_parameters_flattened : array of float
        Abscal parameters, flattened across the frequency axis. Shape (3 * Nfreqs_unflagged,).
    unflagged_freq_inds : array of int
        Array of indices of frequency channels that are not fully flagged. Shape (Nfreqs_unflagged,).
    caldata_obj : CalData

    Returns
    -------
    hess : array of float
        Array of second derivatives of the cost function with respect to the abscal
        parameters. Shape (3 * Nfreqs, 3 * Nfreqs,).
    """

    abscal_parameters = np.zeros((3, caldata_obj.Nfreqs))
    abscal_parameters[:, unflagged_freq_inds] = np.reshape(
        abscal_parameters_flattened, (3, len(unflagged_freq_inds))
    )
    if caldata_obj.gains_multiply_model:
        visibility_values_1 = caldata_obj.data_visibilities[:, :, :, 0]
        visibility_values_2 = caldata_obj.model_visibilities[:, :, :, 0]
    else:
        visibility_values_1 = caldata_obj.model_visibilities[:, :, :, 0]
        visibility_values_2 = caldata_obj.data_visibilities[:, :, :, 0]
    if caldata_obj.dwcal_memory_save_mode:
        (
            hess_amp_amp,
            hess_amp_phasex,
            hess_amp_phasey,
            hess_phasex_phasex,
            hess_phasey_phasey,
            hess_phasex_phasey,
        ) = cost_function_calculations.hess_dw_abscal_toeplitz(
            abscal_parameters[0, :],
            abscal_parameters[1:, :],
            visibility_values_1,
            visibility_values_2,
            caldata_obj.uv_array,
            caldata_obj.visibility_weights[:, :, :, 0],
            caldata_obj.dwcal_inv_covariance[:, :, :, 0],
        )
    else:
        (
            hess_amp_amp,
            hess_amp_phasex,
            hess_amp_phasey,
            hess_phasex_phasex,
            hess_phasey_phasey,
            hess_phasex_phasey,
        ) = cost_function_calculations.hess_dw_abscal(
            abscal_parameters[0, :],
            abscal_parameters[1:, :],
            visibility_values_1,
            visibility_values_2,
            caldata_obj.uv_array,
            caldata_obj.visibility_weights[:, :, :, 0],
            caldata_obj.dwcal_inv_covariance[:, :, :, :, 0],
        )
    hess = np.zeros((3, caldata_obj.Nfreqs, 3, caldata_obj.Nfreqs), dtype=float)
    hess[0, :, 0, :] = hess_amp_amp
    hess[0, :, 1, :] = hess_amp_phasex.T
    hess[1, :, 0, :] = hess_amp_phasex
    hess[0, :, 2, :] = hess_amp_phasey.T
    hess[2, :, 0, :] = hess_amp_phasey
    hess[1, :, 1, :] = hess_phasex_phasex
    hess[2, :, 2, :] = hess_phasey_phasey
    hess[1, :, 2, :] = hess_phasex_phasey.T
    hess[2, :, 1, :] = hess_phasex_phasey
    hess = np.take(
        np.take(hess, unflagged_freq_inds, axis=1), unflagged_freq_inds, axis=3
    )
    hess = np.reshape(
        hess, (3 * len(unflagged_freq_inds), 3 * len(unflagged_freq_inds))
    )
    return hess


def run_skycal_optimization_per_pol_single_freq(
    caldata_obj,
    xtol,
    maxiter,
    freq_ind=0,
    verbose=True,
    get_crosspol_phase=True,
    crosspol_phase_strategy="crosspol model",
):
    """
    Run calibration per polarization. Here the XX and YY visibilities are
    calibrated individually. If get_crosspol_phase is set, the cross-
    polarization phase is applied from the XY and YX visibilities after the
    fact.

    Parameters
    ----------
    caldata_obj : CalData
    xtol : float
        Accuracy tolerance for optimizer.
    maxiter : int
        Maximum number of iterations for the optimizer.
    freq_ind : int
        Frequency channel to process. Default 0.
    verbose : bool
        Set to True to print optimization outputs. Default True.
    get_crosspol_phase : bool
        Set to True to constrain the cross-polarizaton phase from the XY and YX
        visibilities. Default True.
    crosspol_phase_strategy : str
        Options are "crosspol model" or "pseudo Stokes V". Used only if
        get_crosspol_phase is True. If "crosspol model", contrains the crosspol
        phase using the crosspol model visibilities. If "pseudo Stokes V", constrains
        crosspol phase by minimizing pseudo Stokes V. Default "crosspol model".

    Returns
    -------
    gains_fit : array of complex
        Fit gain values. Shape (Nants, 1, N_feed_pols,).
    """

    print("***NEW CAL***")

    gains_fit = np.full(
        (caldata_obj.Nants, caldata_obj.N_feed_pols),
        np.nan + 1j * np.nan,
        dtype=complex,
    )
    if np.max(caldata_obj.visibility_weights[:, :, freq_ind, :]) == 0.0:
        print("ERROR: All data flagged.")
        gains_fit[:, :] = np.nan + 1j * np.nan
        return gains_fit

    for feed_pol_ind, feed_pol in enumerate(caldata_obj.feed_polarization_array):
        vis_pol_ind = np.where(caldata_obj.vis_polarization_array == feed_pol)[0]

        if (
            np.max(caldata_obj.visibility_weights[:, :, freq_ind, vis_pol_ind]) == 0.0
        ):  # All flagged
            gains_fit[:, feed_pol_ind] = np.nan + 1j * np.nan
        else:
            caldata_obj.set_ant_inds(freq_ind, feed_pol_ind)

            gains_init_flattened = caldata_obj.pack(freq_ind, feed_pol_ind)

            caldata_obj.reshape_data(freq_ind, vis_pol_ind)

            # Minimize the cost function
            start_optimize = time.time()
            result = scipy.optimize.minimize(
                cost_skycal_wrapper,
                gains_init_flattened,
                args=(caldata_obj, caldata_obj.ant_inds),
                method="Newton-CG",
                jac=jacobian_skycal_wrapper,
                hess=hessian_skycal_wrapper,
                options={"disp": verbose, "xtol": xtol, "maxiter": maxiter},
            )
            end_optimize = time.time()
            if verbose:
                print(result.message)
                print(
                    f"Optimization time: {(end_optimize - start_optimize)/60.} minutes"
                )
            sys.stdout.flush()
            gains_fit_single_pol = np.reshape(result.x, (len(caldata_obj.ant_inds), 2))
            gains_fit[caldata_obj.ant_inds, feed_pol_ind] = (
                gains_fit_single_pol[:, 0] + 1j * gains_fit_single_pol[:, 1]
            )

            # Ensure that the phase of the gains is mean-zero
            # This adds should be handled by the phase regularization term, but
            # this step removes any optimizer precision effects.
            avg_angle = np.arctan2(
                np.nanmean(np.sin(np.angle(gains_fit[:, feed_pol_ind]))),
                np.nanmean(np.cos(np.angle(gains_fit[:, feed_pol_ind]))),
            )
            gains_fit[:, feed_pol_ind] *= np.cos(avg_angle) - 1j * np.sin(avg_angle)

    # Constrain crosspol phase
    if (
        get_crosspol_phase
        and caldata_obj.N_feed_pols == 2
        and caldata_obj.N_vis_pols == 4
    ):
        if (
            caldata_obj.feed_polarization_array[0] == -5
            and caldata_obj.feed_polarization_array[1] == -6
        ):
            crosspol_polarizations = [-7, -8]
        elif (
            caldata_obj.feed_polarization_array[0] == -6
            and caldata_obj.feed_polarization_array[1] == -5
        ):
            crosspol_polarizations = [-8, -7]
        crosspol_indices = np.array(
            [
                np.where(caldata_obj.vis_polarization_array == pol)[0][0]
                for pol in crosspol_polarizations
            ]
        )
        if crosspol_phase_strategy.lower() == "pseudo stokes v":
            crosspol_phase = cost_function_calculations.set_crosspol_phase_pseudoV(
                gains_fit,
                caldata_obj.data_visibilities[:, :, freq_ind, crosspol_indices],
                caldata_obj.visibility_weights[:, :, freq_ind, crosspol_indices],
                caldata_obj.ant1_inds,
                caldata_obj.ant2_inds,
            )
        elif crosspol_phase_strategy.lower() == "crosspol model":
            crosspol_phase = cost_function_calculations.set_crosspol_phase(
                gains_fit,
                caldata_obj.model_visibilities[:, :, freq_ind, crosspol_indices],
                caldata_obj.data_visibilities[:, :, freq_ind, crosspol_indices],
                caldata_obj.visibility_weights[:, :, freq_ind, crosspol_indices],
                caldata_obj.ant1_inds,
                caldata_obj.ant2_inds,
            )
        else:
            print(
                "WARNING: Unknown crosspol_phase_strategy. Skipping fitting crosspol phase."
            )
            crosspol_phase = 0.0

        if caldata_obj.gains_multiply_model:
            gains_fit[:, 0] /= np.exp(-1j * crosspol_phase / 2)
            gains_fit[:, 1] /= np.exp(1j * crosspol_phase / 2)
        else:
            gains_fit[:, 0] *= np.exp(-1j * crosspol_phase / 2)
            gains_fit[:, 1] *= np.exp(1j * crosspol_phase / 2)

    return gains_fit


def run_abscal_optimization_single_freq(
    caldata_obj,
    xtol,
    maxiter,
    verbose=True,
):
    """
    Run absolute calibration ("abscal").

    Parameters
    ----------
    caldata_obj : CalData
    xtol : float
        Accuracy tolerance for optimizer.
    maxiter : int
        Maximum number of iterations for the optimizer.
    verbose : bool
        Set to True to print optimization outputs. Default True.

    Returns
    -------
    abscal_params : array of complex
        Fit abscal parameter values. Shape (3, 1, N_feed_pols,).
    """

    abscal_params = np.zeros((3, 1, caldata_obj.N_feed_pols), dtype=float)
    caldata_list = caldata_obj.expand_in_polarization()
    for feed_pol_ind, caldata_per_pol in enumerate(caldata_list):
        # Minimize the cost function
        start_optimize = time.time()
        result = scipy.optimize.minimize(
            cost_abscal_wrapper,
            caldata_per_pol.abscal_params[:, 0, 0],
            args=(caldata_per_pol),
            method="Newton-CG",
            jac=jacobian_abscal_wrapper,
            hess=hessian_abscal_wrapper,
            options={"disp": verbose, "xtol": xtol, "maxiter": maxiter},
        )
        abscal_params[:, 0, feed_pol_ind] = result.x
        end_optimize = time.time()
        if verbose:
            print(result.message)
            print(f"Optimization time: {(end_optimize - start_optimize)/60.} minutes")
        sys.stdout.flush()

    return abscal_params


def run_dw_abscal_optimization(
    caldata_obj,
    xtol,
    maxiter,
    verbose=True,
):
    """
    Run absolute calibration with delay weighting.

    Parameters
    ----------
    caldata_obj : CalData
    xtol : float
        Accuracy tolerance for optimizer.
    maxiter : int
        Maximum number of iterations for the optimizer.
    verbose : bool
        Set to True to print optimization outputs. Default True.

    Returns
    -------
    abscal_params : array of complex
        Fit abscal parameter values. Shape (3, Nfreqs, N_feed_pols,).
    """

    abscal_params = np.zeros_like(caldata_obj.abscal_params)
    caldata_list = caldata_obj.expand_in_polarization()
    for feed_pol_ind, caldata_per_pol in enumerate(caldata_list):
        unflagged_freq_inds = np.where(
            np.sum(caldata_per_pol.visibility_weights, axis=(0, 1, 3)) > 0
        )[0]
        if len(unflagged_freq_inds) == 0:
            print(f"ERROR: Data all flagged.")
            sys.stdout.flush()
            continue
        abscal_params_flattened = caldata_per_pol.abscal_params[
            :, unflagged_freq_inds, 0
        ].flatten()
        # Minimize the cost function
        start_optimize = time.time()
        result = scipy.optimize.minimize(
            cost_dw_abscal_wrapper,
            abscal_params_flattened,
            args=(unflagged_freq_inds, caldata_per_pol),
            method="Newton-CG",
            jac=jacobian_dw_abscal_wrapper,
            hess=hessian_dw_abscal_wrapper,
            options={"disp": verbose, "xtol": xtol, "maxiter": maxiter},
        )
        abscal_params[:, unflagged_freq_inds, feed_pol_ind] = np.reshape(
            result.x, (3, len(unflagged_freq_inds))
        )
        if verbose:
            print(result.message)
            print(f"Optimization time: {(time.time() - start_optimize)/60.} minutes")
        sys.stdout.flush()

    return abscal_params
