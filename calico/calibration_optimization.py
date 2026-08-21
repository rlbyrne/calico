import numpy as np
import sys
import scipy
import scipy.optimize
import time
from calico import cost_function_calculations
from numpy.typing import NDArray
import jax
import jax.numpy as jnp


def cost_skycal_wrapper(
    gains_flattened: NDArray[np.floating],
    caldata_obj,
    ant_inds: NDArray[int],
    freq_ind: int,
    vis_pol_ind: int,
) -> float:
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
            gains[:, np.newaxis, np.newaxis],
            np.reshape(
                caldata_obj.data_visibilities[:, :, freq_ind, vis_pol_ind],
                (caldata_obj.Ntimes, caldata_obj.Nbls, 1, 1),
            ),
            np.reshape(
                caldata_obj.model_visibilities[:, :, freq_ind, vis_pol_ind],
                (caldata_obj.Ntimes, caldata_obj.Nbls, 1, 1),
            ),
            np.reshape(
                caldata_obj.visibility_weights[:, :, freq_ind, vis_pol_ind],
                (caldata_obj.Ntimes, caldata_obj.Nbls, 1, 1),
            ),
            caldata_obj.ant1_inds,
            caldata_obj.ant2_inds,
            caldata_obj.lambda_val,
        )
    else:
        cost = cost_function_calculations.cost_skycal(
            gains[:, np.newaxis, np.newaxis],
            np.reshape(
                caldata_obj.model_visibilities[:, :, freq_ind, vis_pol_ind],
                (caldata_obj.Ntimes, caldata_obj.Nbls, 1, 1),
            ),
            np.reshape(
                caldata_obj.data_visibilities[:, :, freq_ind, vis_pol_ind],
                (caldata_obj.Ntimes, caldata_obj.Nbls, 1, 1),
            ),
            np.reshape(
                caldata_obj.visibility_weights[:, :, freq_ind, vis_pol_ind],
                (caldata_obj.Ntimes, caldata_obj.Nbls, 1, 1),
            ),
            caldata_obj.ant1_inds,
            caldata_obj.ant2_inds,
            caldata_obj.lambda_val,
        )
    return cost


def jacobian_skycal_wrapper(
    gains_flattened: NDArray[np.floating],
    caldata_obj,
    ant_inds: NDArray[int],
    freq_ind: int,
    vis_pol_ind: int,
) -> NDArray[np.floating]:
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
            gains[:, np.newaxis, np.newaxis],
            np.reshape(
                caldata_obj.data_visibilities[:, :, freq_ind, vis_pol_ind],
                (caldata_obj.Ntimes, caldata_obj.Nbls, 1, 1),
            ),
            np.reshape(
                caldata_obj.model_visibilities[:, :, freq_ind, vis_pol_ind],
                (caldata_obj.Ntimes, caldata_obj.Nbls, 1, 1),
            ),
            np.reshape(
                caldata_obj.visibility_weights[:, :, freq_ind, vis_pol_ind],
                (caldata_obj.Ntimes, caldata_obj.Nbls, 1, 1),
            ),
            caldata_obj.ant1_inds,
            caldata_obj.ant2_inds,
            caldata_obj.lambda_val,
        )
    else:
        jac = cost_function_calculations.jacobian_skycal(
            gains[:, np.newaxis, np.newaxis],
            np.reshape(
                caldata_obj.model_visibilities[:, :, freq_ind, vis_pol_ind],
                (caldata_obj.Ntimes, caldata_obj.Nbls, 1, 1),
            ),
            np.reshape(
                caldata_obj.data_visibilities[:, :, freq_ind, vis_pol_ind],
                (caldata_obj.Ntimes, caldata_obj.Nbls, 1, 1),
            ),
            np.reshape(
                caldata_obj.visibility_weights[:, :, freq_ind, vis_pol_ind],
                (caldata_obj.Ntimes, caldata_obj.Nbls, 1, 1),
            ),
            caldata_obj.ant1_inds,
            caldata_obj.ant2_inds,
            caldata_obj.lambda_val,
        )
    jac_flattened = np.stack(
        (np.real(jac[ant_inds, 0, 0]), np.imag(jac[ant_inds, 0, 0])), axis=1
    ).flatten()
    return jac_flattened


def hessian_skycal_wrapper(
    gains_flattened: NDArray[np.floating],
    caldata_obj,
    ant_inds: NDArray[int],
    freq_ind: int,
    vis_pol_ind: int,
) -> NDArray[np.floating]:
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
            gains[:, np.newaxis, np.newaxis],
            caldata_obj.Nants,
            caldata_obj.Nbls,
            np.reshape(
                caldata_obj.data_visibilities[:, :, freq_ind, vis_pol_ind],
                (caldata_obj.Ntimes, caldata_obj.Nbls, 1, 1),
            ),
            np.reshape(
                caldata_obj.model_visibilities[:, :, freq_ind, vis_pol_ind],
                (caldata_obj.Ntimes, caldata_obj.Nbls, 1, 1),
            ),
            np.reshape(
                caldata_obj.visibility_weights[:, :, freq_ind, vis_pol_ind],
                (caldata_obj.Ntimes, caldata_obj.Nbls, 1, 1),
            ),
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
            gains[:, np.newaxis, np.newaxis],
            caldata_obj.Nants,
            caldata_obj.Nbls,
            np.reshape(
                caldata_obj.model_visibilities[:, :, freq_ind, vis_pol_ind],
                (caldata_obj.Ntimes, caldata_obj.Nbls, 1, 1),
            ),
            np.reshape(
                caldata_obj.data_visibilities[:, :, freq_ind, vis_pol_ind],
                (caldata_obj.Ntimes, caldata_obj.Nbls, 1, 1),
            ),
            np.reshape(
                caldata_obj.visibility_weights[:, :, freq_ind, vis_pol_ind],
                (caldata_obj.Ntimes, caldata_obj.Nbls, 1, 1),
            ),
            caldata_obj.ant1_inds,
            caldata_obj.ant2_inds,
            caldata_obj.lambda_val,
        )
    hess_flattened = np.full(
        (2 * Nants_unflagged, 2 * Nants_unflagged), np.nan, dtype=float
    )
    for unflagged_ant_ind1, ant_ind_1 in enumerate(ant_inds):
        for unflagged_ant_ind2, ant_ind_2 in enumerate(ant_inds):
            hess_flattened[2 * unflagged_ant_ind1, 2 * unflagged_ant_ind2] = (
                hess_real_real[ant_ind_1, ant_ind_2, 0, 0]
            )
            hess_flattened[2 * unflagged_ant_ind1, 2 * unflagged_ant_ind2 + 1] = (
                hess_real_imag[ant_ind_1, ant_ind_2, 0, 0]
            )
            hess_flattened[2 * unflagged_ant_ind1 + 1, 2 * unflagged_ant_ind2] = (
                np.conj(
                    hess_real_imag[ant_ind_2, ant_ind_1, 0, 0]
                )  # I believe this should be real-valued, so does the conjugation make sense?
            )
            hess_flattened[2 * unflagged_ant_ind1 + 1, 2 * unflagged_ant_ind2 + 1] = (
                hess_imag_imag[ant_ind_1, ant_ind_2, 0, 0]
            )
    return hess_flattened


def cost_ddcal_wrapper(
    gains_flattened: NDArray[np.floating],
    caldata_obj,
    ant_inds: NDArray[int],
    freq_ind: int,
    vis_pol_ind: int,
) -> float:
    """
    Direction-dependent calibration cost function. Uses the function cost_ddcal.
    Reformats the input gains to be compatible with the scipy.optimize.minimize function.

    Parameters
    ----------
    gains_flattened : array of float
        Array of gain values. Even indices correspond to the real components of the
        gains and odd indices correspond to the imaginary components. Shape
        (2*Nants_unflagged*n_directions,).
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

    gains_reshaped = jnp.reshape(
        gains_flattened, (len(ant_inds), caldata_obj.n_directions, 2)
    )
    gains_reshaped = gains_reshaped[:, :, 0] + 1.0j * gains_reshaped[:, :, 1]
    gains = jnp.ones((caldata_obj.Nants, caldata_obj.n_directions), dtype=complex)
    gains = gains.at[jnp.ix_(ant_inds, jnp.arange(caldata_obj.n_directions))].set(
        gains_reshaped
    )

    if caldata_obj.ddcal_max_source_offset_deg is not None:
        use_antenna_distances = caldata_obj.antenna_distances
        use_freq_array = caldata_obj.freq_array[[freq_ind]]
    else:
        use_antenna_distances = None
        use_freq_array = None

    cost = cost_function_calculations.cost_ddcal(
        gains[:, jnp.newaxis, jnp.newaxis, :],
        jnp.reshape(
            caldata_obj.model_visibilities[:, :, freq_ind, vis_pol_ind, ...],
            (caldata_obj.Ntimes, caldata_obj.Nbls, 1, 1, caldata_obj.n_directions),
        ),
        jnp.reshape(
            caldata_obj.data_visibilities[:, :, freq_ind, vis_pol_ind],
            (caldata_obj.Ntimes, caldata_obj.Nbls, 1, 1),
        ),
        jnp.reshape(
            caldata_obj.visibility_weights[:, :, freq_ind, vis_pol_ind],
            (caldata_obj.Ntimes, caldata_obj.Nbls, 1, 1),
        ),
        caldata_obj.ant1_inds,
        caldata_obj.ant2_inds,
        caldata_obj.lambda_val,
        caldata_obj.ddcal_max_source_offset_deg,
        caldata_obj.ddcal_source_offset_taper_deg,
        use_antenna_distances,
        use_freq_array,
    )
    return cost


def cost_dwcal_wrapper(
    gains_flattened: NDArray[np.floating],
    caldata_obj,
    ant_inds: NDArray[int],
    freq_inds: NDArray[int],
    vis_pol_ind: int,
) -> float:
    """
    Wrapper for function cost_dwcal. Reformats the input gains to be compatible
    with the scipy.optimize.minimize function.

    Parameters
    ----------
    gains_flattened : array of float
        Array of gain values. Even indices correspond to the real components of the
        gains and odd indices correspond to the imaginary components. Shape
        (2*Nants_unflagged*Nfreqs_unflagged,).
    caldata_obj : CalData
    ant_inds : array of int
        Indices of unflagged antennas to be calibrated. Shape (Nants_unflagged,).
    freq_inds : int
        Indices of unflagged frequency channels to be calibrated. Shape (Nfreqs_unflagged,).
    vis_pol_ind : int
        Index of the visibility polarization.

    Returns
    -------
    cost : float
        Value of the cost function.
    """

    gains_reshaped = jnp.reshape(gains_flattened, (len(ant_inds), len(freq_inds), 2))
    gains_reshaped = gains_reshaped[:, :, [0]] + 1.0j * gains_reshaped[:, :, [1]]
    gains = jnp.ones((caldata_obj.Nants, caldata_obj.Nfreqs, 1), dtype=complex)
    gains = gains.at[jnp.ix_(ant_inds, freq_inds, jnp.array([0]))].set(gains_reshaped)

    if caldata_obj.dwcal_memory_save_mode:
        cost = cost_function_calculations.cost_dwcal_toeplitz(
            gains,
            caldata_obj.data_visibilities[:, :, :, [vis_pol_ind]],
            caldata_obj.model_visibilities[:, :, :, [vis_pol_ind]],
            caldata_obj.visibility_weights[:, :, :, [vis_pol_ind]],
            caldata_obj.dwcal_inv_covariance[:, :, :, [vis_pol_ind]],
            caldata_obj.ant1_inds,
            caldata_obj.ant2_inds,
            caldata_obj.lambda_val,
        )
    else:
        cost = cost_function_calculations.cost_dwcal(
            gains,
            caldata_obj.data_visibilities[:, :, :, [vis_pol_ind]],
            caldata_obj.model_visibilities[:, :, :, [vis_pol_ind]],
            caldata_obj.visibility_weights[:, :, :, [vis_pol_ind]],
            caldata_obj.dwcal_inv_covariance[:, :, :, :, [vis_pol_ind]],
            caldata_obj.ant1_inds,
            caldata_obj.ant2_inds,
            caldata_obj.lambda_val,
        )

    return cost


def cost_abscal_wrapper(
    abscal_parameters: NDArray[np.floating], caldata_obj, freq_ind, vis_pol_ind
) -> float:
    """
    Wrapper for function cost_function_abs_cal.

    Parameters
    ----------
    abscal_parameters : array of float
        Shape (3,).
    caldata_obj : CalData
    freq_ind : int
    vis_pol_ind : int

    Returns
    -------
    cost : float
        Value of the cost function.
    """

    if caldata_obj.gains_multiply_model:
        cost = cost_function_calculations.cost_function_abs_cal(
            abscal_parameters[0],
            abscal_parameters[1:],
            caldata_obj.data_visibilities[:, :, freq_ind, vis_pol_ind],
            caldata_obj.model_visibilities[:, :, freq_ind, vis_pol_ind],
            caldata_obj.uv_array,
            caldata_obj.visibility_weights[:, :, freq_ind, vis_pol_ind],
        )
    else:
        cost = cost_function_calculations.cost_function_abs_cal(
            abscal_parameters[0],
            abscal_parameters[1:],
            caldata_obj.model_visibilities[:, :, freq_ind, vis_pol_ind],
            caldata_obj.data_visibilities[:, :, freq_ind, vis_pol_ind],
            caldata_obj.uv_array,
            caldata_obj.visibility_weights[:, :, freq_ind, vis_pol_ind],
        )
    return cost


def jacobian_abscal_wrapper(
    abscal_parameters: NDArray[np.floating], caldata_obj, freq_ind, vis_pol_ind
) -> NDArray[np.floating]:
    """
    Wrapper for function jacobian_abs_cal.

    Parameters
    ----------
    abscal_parameters : array of float
        Shape (3,).
    caldata_obj : CalData
    freq_ind : int
    vis_pol_ind : int

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
            caldata_obj.data_visibilities[:, :, freq_ind, vis_pol_ind],
            caldata_obj.model_visibilities[:, :, freq_ind, vis_pol_ind],
            caldata_obj.uv_array,
            caldata_obj.visibility_weights[:, :, freq_ind, vis_pol_ind],
        )
    else:
        amp_jac, phase_jac = cost_function_calculations.jacobian_abs_cal(
            abscal_parameters[0],
            abscal_parameters[1:],
            caldata_obj.model_visibilities[:, :, freq_ind, vis_pol_ind],
            caldata_obj.data_visibilities[:, :, freq_ind, vis_pol_ind],
            caldata_obj.uv_array,
            caldata_obj.visibility_weights[:, :, freq_ind, vis_pol_ind],
        )
    jac[0] = amp_jac
    jac[1:] = phase_jac
    return jac


def hessian_abscal_wrapper(
    abscal_parameters: NDArray[np.floating], caldata_obj, freq_ind, vis_pol_ind
) -> NDArray[np.floating]:
    """
    Wrapper for function hess_abs_cal.

    Parameters
    ----------
    abscal_parameters : array of float
        Shape (3,).
    caldata_obj : CalData
    freq_ind : int
    vis_pol_ind : int

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
            caldata_obj.data_visibilities[:, :, freq_ind, vis_pol_ind],
            caldata_obj.model_visibilities[:, :, freq_ind, vis_pol_ind],
            caldata_obj.uv_array,
            caldata_obj.visibility_weights[:, :, freq_ind, vis_pol_ind],
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
            caldata_obj.model_visibilities[:, :, freq_ind, vis_pol_ind],
            caldata_obj.data_visibilities[:, :, freq_ind, vis_pol_ind],
            caldata_obj.uv_array,
            caldata_obj.visibility_weights[:, :, freq_ind, vis_pol_ind],
        )
    hess[0, 0] = hess_amp_amp
    hess[0, 1] = hess[1, 0] = hess_amp_phasex
    hess[0, 2] = hess[2, 0] = hess_amp_phasey
    hess[1, 1] = hess_phasex_phasex
    hess[2, 2] = hess_phasey_phasey
    hess[1, 2] = hess[2, 1] = hess_phasex_phasey
    return hess


def cost_dw_abscal_wrapper(
    abscal_parameters_flattened: NDArray[np.floating],
    unflagged_freq_inds: NDArray[int],
    caldata_obj,
    vis_pol_ind: int,
) -> float:
    """
    Wrapper for function cost_function_dw_abscal.

    Parameters
    ----------
    abscal_parameters_flattened : array of float
        Abscal parameters, flattened across the frequency axis. Shape (3 * Nfreqs_unflagged,).
    unflagged_freq_inds : array of int
        Array of indices of frequency channels that are not fully flagged. Shape (Nfreqs_unflagged,).
    caldata_obj : CalData
    vis_pol_ind : int

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
        visibility_values_1 = caldata_obj.data_visibilities[:, :, :, vis_pol_ind]
        visibility_values_2 = caldata_obj.model_visibilities[:, :, :, vis_pol_ind]
    else:
        visibility_values_1 = caldata_obj.model_visibilities[:, :, :, vis_pol_ind]
        visibility_values_2 = caldata_obj.data_visibilities[:, :, :, vis_pol_ind]
    if caldata_obj.dwcal_memory_save_mode:
        cost = cost_function_calculations.cost_function_dw_abscal_toeplitz(
            abscal_parameters[0, :],
            abscal_parameters[1:, :],
            visibility_values_1,
            visibility_values_2,
            caldata_obj.uv_array,
            caldata_obj.visibility_weights[:, :, :, vis_pol_ind],
            caldata_obj.dwcal_inv_covariance[:, :, :, vis_pol_ind],
        )
    else:
        cost = cost_function_calculations.cost_function_dw_abscal(
            abscal_parameters[0, :],
            abscal_parameters[1:, :],
            visibility_values_1,
            visibility_values_2,
            caldata_obj.uv_array,
            caldata_obj.visibility_weights[:, :, :, vis_pol_ind],
            caldata_obj.dwcal_inv_covariance[:, :, :, :, vis_pol_ind],
        )
    return cost


def jacobian_dw_abscal_wrapper(
    abscal_parameters_flattened: NDArray[np.floating],
    unflagged_freq_inds: NDArray[int],
    caldata_obj,
    vis_pol_ind: int,
) -> NDArray[np.floating]:
    """
    Wrapper for function jacobian_dw_abscal.

    Parameters
    ----------
    abscal_parameters_flattened : array of float
        Abscal parameters, flattened across the frequency axis. Shape (3 * Nfreqs_unflagged,).
    unflagged_freq_inds : array of int
        Array of indices of frequency channels that are not fully flagged. Shape (Nfreqs_unflagged,).
    caldata_obj : CalData
    vis_pol_ind : int

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
        visibility_values_1 = caldata_obj.data_visibilities[:, :, :, vis_pol_ind]
        visibility_values_2 = caldata_obj.model_visibilities[:, :, :, vis_pol_ind]
    else:
        visibility_values_1 = caldata_obj.model_visibilities[:, :, :, vis_pol_ind]
        visibility_values_2 = caldata_obj.data_visibilities[:, :, :, vis_pol_ind]
    if caldata_obj.dwcal_memory_save_mode:
        amp_jac, phase_jac = cost_function_calculations.jacobian_dw_abscal_toeplitz(
            abscal_parameters[0, :],
            abscal_parameters[1:, :],
            visibility_values_1,
            visibility_values_2,
            caldata_obj.uv_array,
            caldata_obj.visibility_weights[:, :, :, vis_pol_ind],
            caldata_obj.dwcal_inv_covariance[:, :, :, vis_pol_ind],
        )
    else:
        amp_jac, phase_jac = cost_function_calculations.jacobian_dw_abscal(
            abscal_parameters[0, :],
            abscal_parameters[1:, :],
            visibility_values_1,
            visibility_values_2,
            caldata_obj.uv_array,
            caldata_obj.visibility_weights[:, :, :, vis_pol_ind],
            caldata_obj.dwcal_inv_covariance[:, :, :, :, vis_pol_ind],
        )
    jac_array = np.zeros((3, caldata_obj.Nfreqs), dtype=float)
    jac_array[0, :] = amp_jac
    jac_array[1:, :] = phase_jac
    jac_array = np.take(jac_array, unflagged_freq_inds, axis=1)
    return jac_array.flatten()


def hessian_dw_abscal_wrapper(
    abscal_parameters_flattened: NDArray[np.floating],
    unflagged_freq_inds: NDArray[int],
    caldata_obj,
    vis_pol_ind: int,
) -> NDArray[np.floating]:
    """
    Wrapper for function hess_dw_abscal.

    Parameters
    ----------
    abscal_parameters_flattened : array of float
        Abscal parameters, flattened across the frequency axis. Shape (3 * Nfreqs_unflagged,).
    unflagged_freq_inds : array of int
        Array of indices of frequency channels that are not fully flagged. Shape (Nfreqs_unflagged,).
    caldata_obj : CalData
    vis_pol_ind : int

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
        visibility_values_1 = caldata_obj.data_visibilities[:, :, :, vis_pol_ind]
        visibility_values_2 = caldata_obj.model_visibilities[:, :, :, vis_pol_ind]
    else:
        visibility_values_1 = caldata_obj.model_visibilities[:, :, :, vis_pol_ind]
        visibility_values_2 = caldata_obj.data_visibilities[:, :, :, vis_pol_ind]
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
            caldata_obj.visibility_weights[:, :, :, vis_pol_ind],
            caldata_obj.dwcal_inv_covariance[:, :, :, vis_pol_ind],
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
            caldata_obj.visibility_weights[:, :, :, vis_pol_ind],
            caldata_obj.dwcal_inv_covariance[:, :, :, :, vis_pol_ind],
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
    freq_ind: int = 0,
    verbose: bool = True,
    get_crosspol_phase: bool = True,
    crosspol_phase_strategy: str = "crosspol model",
) -> NDArray[np.complexfloating]:
    """
    Run calibration per polarization. Here the XX and YY visibilities are
    calibrated individually. If get_crosspol_phase is set, the cross-
    polarization phase is applied from the XY and YX visibilities after the
    fact.

    Parameters
    ----------
    caldata_obj : CalData
    freq_ind : int
        Frequency channel to process. Default 0.

    Returns
    -------
    gains_fit : array of complex
        Fit gain values. Shape (Nants, 1, N_feed_pols,).
    """

    gains_fit = np.full(
        (caldata_obj.Nants, caldata_obj.N_feed_pols),
        np.nan + 1j * np.nan,
        dtype=complex,
    )
    if np.max(caldata_obj.visibility_weights[:, :, freq_ind, :]) == 0.0:
        if caldata_obj.verbose:
            print("WARNING: All data flagged.")
            sys.stdout.flush()
        gains_fit[:, :] = np.nan + 1j * np.nan
        return gains_fit

    for feed_pol_ind, feed_pol in enumerate(caldata_obj.feed_polarization_array):
        vis_pol_ind = np.where(caldata_obj.vis_polarization_array == feed_pol)[0]

        if (
            np.max(caldata_obj.visibility_weights[:, :, freq_ind, vis_pol_ind]) == 0.0
        ):  # All flagged
            gains_fit[:, feed_pol_ind] = np.nan + 1j * np.nan
        else:
            vis_weights_summed = np.sum(
                caldata_obj.visibility_weights[:, :, freq_ind, feed_pol_ind], axis=0
            )  # Sum over times
            weight_per_ant = np.bincount(
                caldata_obj.ant1_inds,
                weights=vis_weights_summed,
                minlength=caldata_obj.Nants,
            ) + np.bincount(
                caldata_obj.ant2_inds,
                weights=vis_weights_summed,
                minlength=caldata_obj.Nants,
            )
            ant_inds = np.where(weight_per_ant > 0.0)[0]

            gains_init_flattened = np.stack(
                (
                    np.real(caldata_obj.gains[ant_inds, freq_ind, feed_pol_ind]),
                    np.imag(caldata_obj.gains[ant_inds, freq_ind, feed_pol_ind]),
                ),
                axis=1,
            ).flatten()

            # Minimize the cost function
            start_optimize = time.time()
            result = scipy.optimize.minimize(
                cost_skycal_wrapper,
                gains_init_flattened,
                args=(caldata_obj, ant_inds, freq_ind, vis_pol_ind),
                method="Newton-CG",
                jac=jacobian_skycal_wrapper,
                hess=hessian_skycal_wrapper,
                options={
                    "disp": caldata_obj.verbose,
                    "xtol": caldata_obj.xtol,
                    "maxiter": caldata_obj.maxiter,
                },
            )
            end_optimize = time.time()
            if caldata_obj.verbose and not caldata_obj.parallel:
                print(result.message)
                print(
                    f"Freq. {freq_ind} Pol. {feed_pol_ind}, optimization time: {(end_optimize - start_optimize)/60.} minutes"
                )
                sys.stdout.flush()
            gains_fit_single_pol = np.reshape(result.x, (len(ant_inds), 2))
            gains_fit[ant_inds, feed_pol_ind] = (
                gains_fit_single_pol[:, 0] + 1j * gains_fit_single_pol[:, 1]
            )

            # Ensure that the phase of the gains is mean-zero
            # If lambda_val != 0, this should be handled by the phase regularization term, but
            # this step removes any optimizer precision effects.
            avg_angle = np.arctan2(
                np.nanmean(np.sin(np.angle(gains_fit[:, feed_pol_ind]))),
                np.nanmean(np.cos(np.angle(gains_fit[:, feed_pol_ind]))),
            )
            gains_fit[:, feed_pol_ind] *= np.cos(avg_angle) - 1j * np.sin(avg_angle)

    # Constrain crosspol phase
    if (
        caldata_obj.get_crosspol_phase
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
        if caldata_obj.crosspol_phase_strategy.lower() == "pseudo stokes v":
            crosspol_phase = cost_function_calculations.set_crosspol_phase_pseudoV(
                gains_fit,
                caldata_obj.data_visibilities[:, :, freq_ind, crosspol_indices],
                caldata_obj.visibility_weights[:, :, freq_ind, crosspol_indices],
                caldata_obj.ant1_inds,
                caldata_obj.ant2_inds,
            )
        elif caldata_obj.crosspol_phase_strategy.lower() == "crosspol model":
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


def run_skycal_optimization_per_pol_single_freq_parallel(args):
    """
    Wrapper for run_skycal_optimization_per_pol_single_freq that makes the function compatible with
    multiprocessing by unpacking a tuple or arguments.
    """
    (caldata_subset, freq_ind) = args
    start_optimize = time.time()
    gains_fit = run_skycal_optimization_per_pol_single_freq(
        caldata_subset,
        freq_ind=0,
    )
    end_optimize = time.time()
    if caldata_subset.verbose:
        print(
            f"Freq. {freq_ind}, optimization time: {(end_optimize - start_optimize)/60.} minutes"
        )
        sys.stdout.flush()
    return freq_ind, gains_fit


def run_ddcal_optimization(
    caldata_obj,
    freq_ind: int = 0,
    pol_ind: int = 0,
) -> NDArray[np.complexfloating]:
    """
    Run direction-dependent calibration per frequency and polarization.

    Parameters
    ----------
    caldata_obj : CalData
    freq_ind : int
        Frequency channel to process. Default 0.
    pol_ind : int
        Feed polarization index to process. Default 0.

    Returns
    -------
    gains_fit : array of complex
        Fit gain values. Shape (Nants, n_directions,).
    """

    gains_fit = np.full(
        (caldata_obj.Nants, caldata_obj.n_directions),
        np.nan + 1j * np.nan,
        dtype=complex,
    )

    vis_pol_ind = np.where(
        caldata_obj.vis_polarization_array
        == caldata_obj.feed_polarization_array[pol_ind]
    )[0]

    if (
        np.max(caldata_obj.visibility_weights[:, :, freq_ind, vis_pol_ind]) == 0.0
    ):  # All flagged
        if caldata_obj.verbose:
            print("WARNING: All data flagged.")
            sys.stdout.flush()
        gains_fit[...] = np.nan + 1j * np.nan
        return gains_fit

    vis_weights_summed = np.sum(
        caldata_obj.visibility_weights[:, :, freq_ind, pol_ind], axis=0
    )  # Sum over times
    weight_per_ant = np.bincount(
        caldata_obj.ant1_inds,
        weights=vis_weights_summed,
        minlength=caldata_obj.Nants,
    ) + np.bincount(
        caldata_obj.ant2_inds,
        weights=vis_weights_summed,
        minlength=caldata_obj.Nants,
    )
    ant_inds = np.where(weight_per_ant > 0.0)[0]

    if caldata_obj.n_directions == 1:
        gains_init_flattened = np.stack(
            (
                np.real(caldata_obj.gains[ant_inds, freq_ind, pol_ind, np.newaxis]),
                np.imag(caldata_obj.gains[ant_inds, freq_ind, pol_ind, np.newaxis]),
            ),
            axis=2,
        ).flatten()
    else:
        gains_init_flattened = np.stack(
            (
                np.real(caldata_obj.gains[ant_inds, freq_ind, pol_ind, :]),
                np.imag(caldata_obj.gains[ant_inds, freq_ind, pol_ind, :]),
            ),
            axis=2,
        ).flatten()

    # Minimize the cost function
    start_optimize = time.time()
    result = scipy.optimize.minimize(
        cost_ddcal_wrapper,
        gains_init_flattened,
        args=(caldata_obj, ant_inds, freq_ind, vis_pol_ind),
        method="Newton-CG",
        jac=jax.jacrev(cost_ddcal_wrapper),
        hess=jax.jacrev(jax.jacrev(cost_ddcal_wrapper)),
        options={
            "disp": caldata_obj.verbose,
            "xtol": caldata_obj.xtol,
            "maxiter": caldata_obj.maxiter,
        },
    )
    end_optimize = time.time()
    if caldata_obj.verbose and not caldata_obj.parallel:
        print(result.message)
        print(
            f"Freq. {freq_ind} Pol. {pol_ind}, optimization time: {(end_optimize - start_optimize)/60.} minutes"
        )
        sys.stdout.flush()
    gains_fit_single_pol = np.reshape(
        result.x, (len(ant_inds), caldata_obj.n_directions, 2)
    )
    gains_fit[ant_inds, :] = (
        gains_fit_single_pol[:, :, 0] + 1j * gains_fit_single_pol[:, :, 1]
    )

    # Ensure that the phase of the gains is mean-zero
    # If lambda_val != 0, this should be handled by the phase regularization term, but
    # this step removes any optimizer precision effects.
    for direction_ind in range(caldata_obj.n_directions):
        avg_angle = np.arctan2(
            np.nanmean(np.sin(np.angle(gains_fit[:, direction_ind]))),
            np.nanmean(np.cos(np.angle(gains_fit[:, direction_ind]))),
        )
        gains_fit[:, direction_ind] *= np.cos(avg_angle) - 1j * np.sin(avg_angle)

    return gains_fit


def run_ddcal_optimization_parallel(args):
    """
    Wrapper for run_ddcal_optimization that makes the function compatible with
    multiprocessing by unpacking a tuple or arguments.
    """
    (caldata_subset, freq_ind, pol_ind) = args
    start_optimize = time.time()
    gains_fit = run_ddcal_optimization(
        caldata_subset,
        freq_ind=0,
        pol_ind=0,
    )
    end_optimize = time.time()
    if caldata_subset.verbose:
        print(
            f"Freq. {freq_ind} Pol. {pol_ind}, optimization time: {(end_optimize - start_optimize)/60.} minutes"
        )
        sys.stdout.flush()
    return freq_ind, pol_ind, gains_fit


def run_dwcal_optimization_per_pol(
    caldata_obj,
    pol_ind: int = 0,
) -> NDArray[np.complexfloating]:
    """
    Run delay-weighted calibration for a single polarization. Uses automatic
    differentiation with jax to compute first and second derivatives for
    optimization with Newton's Method.

    Parameters
    ----------
    caldata_obj : CalData
    pol_ind : int
        Feed polarization index to process. Default 0.

    Returns
    -------
    gains_fit : array of complex
        Fit gain values. Shape (Nants, Nfreqs, 1,).
    """

    gains_fit = np.full(
        (caldata_obj.Nants, caldata_obj.Nfreqs),
        np.nan + 1j * np.nan,
        dtype=complex,
    )

    vis_pol_ind = np.where(
        caldata_obj.vis_polarization_array
        == caldata_obj.feed_polarization_array[pol_ind]
    )[0][0]
    unflagged_freq_inds = np.where(
        np.sum(caldata_obj.visibility_weights[:, :, :, vis_pol_ind], axis=(0, 1)) > 0
    )[0]
    if len(unflagged_freq_inds) == 0:
        print(f"ERROR: Data all flagged.")
        sys.stdout.flush()
        return gains_fit

    vis_weights_summed = np.sum(
        caldata_obj.visibility_weights[:, :, :, vis_pol_ind], axis=(0, 2)
    )  # Sum over times and frequencies
    weight_per_ant = np.bincount(
        caldata_obj.ant1_inds,
        weights=vis_weights_summed,
        minlength=caldata_obj.Nants,
    ) + np.bincount(
        caldata_obj.ant2_inds,
        weights=vis_weights_summed,
        minlength=caldata_obj.Nants,
    )
    ant_inds = np.where(weight_per_ant > 0.0)[0]

    gains_init_flattened = np.stack(
        (
            np.real(
                caldata_obj.gains[:, :, pol_ind][np.ix_(ant_inds, unflagged_freq_inds)]
            ),
            np.imag(
                caldata_obj.gains[:, :, pol_ind][np.ix_(ant_inds, unflagged_freq_inds)]
            ),
        ),
        axis=1,
    ).flatten()

    # Minimize the cost function
    start_optimize = time.time()
    result = scipy.optimize.minimize(
        cost_dwcal_wrapper,
        gains_init_flattened,
        args=(caldata_obj, ant_inds, unflagged_freq_inds, vis_pol_ind),
        method="Newton-CG",
        jac=jax.jacrev(cost_dwcal_wrapper),
        hess=jax.jacrev(jax.jacrev(cost_dwcal_wrapper)),
        options={
            "disp": caldata_obj.verbose,
            "xtol": caldata_obj.xtol,
            "maxiter": caldata_obj.maxiter,
        },
    )
    end_optimize = time.time()
    if caldata_obj.verbose:
        print(result.message)
        print(
            f"Pol. {pol_ind}, optimization time: {(end_optimize - start_optimize)/60.} minutes"
        )
        sys.stdout.flush()
    gains_fit_unflagged = np.reshape(
        result.x, (len(ant_inds), len(unflagged_freq_inds), 2)
    )
    gains_fit[np.ix_(ant_inds, unflagged_freq_inds)] = (
        gains_fit_unflagged[:, :, 0] + 1j * gains_fit_unflagged[:, :, 1]
    )
    return gains_fit


def run_abscal_optimization_single_freq(
    caldata_obj,
    freq_ind: int = 0,
    feed_pol_ind: int = 0,
) -> NDArray[np.complexfloating]:
    """
    Run absolute calibration ("abscal").

    Parameters
    ----------
    caldata_obj : CalData
    freq_ind : int
        Frequency channel to process. Default 0.
    feed_pol_ind : int
        Feed polarization index to process. Default 0.

    Returns
    -------
    abscal_params : array of complex
        Fit abscal parameter values. Shape (3, 1, N_feed_pols,).
    """

    start_optimize = time.time()
    vis_pol_ind = np.where(
        caldata_obj.vis_polarization_array
        == caldata_obj.feed_polarization_array[feed_pol_ind]
    )[0][0]
    result = scipy.optimize.minimize(
        cost_abscal_wrapper,
        caldata_obj.abscal_params[:, freq_ind, feed_pol_ind],
        args=(caldata_obj, freq_ind, vis_pol_ind),
        method="Newton-CG",
        jac=jacobian_abscal_wrapper,
        hess=hessian_abscal_wrapper,
        options={
            "disp": caldata_obj.verbose,
            "xtol": caldata_obj.xtol,
            "maxiter": caldata_obj.maxiter,
        },
    )
    abscal_params = result.x
    end_optimize = time.time()
    if caldata_obj.verbose:
        print(result.message)
        print(f"Optimization time: {(end_optimize - start_optimize)/60.} minutes")
    sys.stdout.flush()

    return abscal_params


def run_dw_abscal_optimization(
    caldata_obj,
    feed_pol_ind: int = 0,
) -> NDArray[np.complexfloating]:
    """
    Run absolute calibration with delay weighting.

    Parameters
    ----------
    caldata_obj : CalData
    feed_pol_ind : int
        Feed polarization index to process. Default 0.

    Returns
    -------
    abscal_params : array of complex
        Fit abscal parameter values. Shape (3, Nfreqs,).
    """

    vis_pol_ind = np.where(
        caldata_obj.vis_polarization_array
        == caldata_obj.feed_polarization_array[feed_pol_ind]
    )[0][0]
    abscal_params = np.zeros_like(caldata_obj.abscal_params[:, :, feed_pol_ind])

    unflagged_freq_inds = np.where(
        np.sum(caldata_obj.visibility_weights, axis=(0, 1, 3)) > 0
    )[0]
    if len(unflagged_freq_inds) == 0:
        print(f"ERROR: Data all flagged.")
        sys.stdout.flush()
        return
    abscal_params_flattened = caldata_obj.abscal_params[
        :, unflagged_freq_inds, feed_pol_ind
    ].flatten()
    # Minimize the cost function
    start_optimize = time.time()
    result = scipy.optimize.minimize(
        cost_dw_abscal_wrapper,
        abscal_params_flattened,
        args=(unflagged_freq_inds, caldata_obj, vis_pol_ind),
        method="Newton-CG",
        jac=jacobian_dw_abscal_wrapper,
        hess=hessian_dw_abscal_wrapper,
        options={
            "disp": caldata_obj.verbose,
            "xtol": caldata_obj.xtol,
            "maxiter": caldata_obj.maxiter,
        },
    )
    abscal_params[:, unflagged_freq_inds] = np.reshape(
        result.x, (3, len(unflagged_freq_inds))
    )
    if caldata_objverbose:
        print(result.message)
        print(f"Optimization time: {(time.time() - start_optimize)/60.} minutes")
    sys.stdout.flush()

    return abscal_params
