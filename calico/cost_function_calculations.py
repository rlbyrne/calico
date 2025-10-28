import numpy as np
import sys
from calico import utils
from numpy.typing import NDArray
from typing import Tuple


def cost_skycal(
    gains: NDArray[np.complexfloating],
    model_visibilities: NDArray[np.complexfloating],
    data_visibilities: NDArray[np.complexfloating],
    visibility_weights: NDArray[np.floating],
    ant1_inds: NDArray[np.integer],
    ant2_inds: NDArray[np.integer],
    lambda_val: float,
) -> float:
    """
    Calculate the cost function (chi-squared) value.

    Parameters
    ----------
    gains : array of complex
        Shape (Nants, Nfreqs, N_feed_pols).
    model_visibilities :  array of complex
        Shape (Ntimes, Nbls, Nfreqs, N_feed_pols). Cross-polarization visibilites are not
        supported; visibilities should include XX and YY only, ordered to correspond to the
        gain polarization convention.
    data_visibilities : array of complex
        Shape (Ntimes, Nbls, Nfreqs, N_feed_pols). Cross-polarization visibilites are not
        supported; visibilities should include XX and YY only, ordered to correspond to the
        gain polarization convention.
    visibility_weights : array of float
        Shape (Ntimes, Nbls, Nfreqs, N_feed_pols). Cross-polarization visibilites are not
        supported; visibilities should include XX and YY only, ordered to correspond to the
        gain polarization convention.
    ant1_inds : array of int
        Shape (Nbls,).
    ant2_inds : array of int
        Shape (Nbls,).
    lambda_val : float
        Weight of the phase regularization term; must be positive.

    Returns
    -------
    cost : float
        Value of the cost function.
    """

    gains_expanded = (gains[ant1_inds, :, :] * np.conj(gains[ant2_inds, :, :]))[
        np.newaxis, :, :, :
    ]
    res_vec = model_visibilities - gains_expanded * data_visibilities
    cost = np.sum(visibility_weights * np.abs(res_vec) ** 2)

    if lambda_val > 0:
        regularization_term = lambda_val * np.sum(np.angle(gains)) ** 2.0
        cost += regularization_term

    return cost


def jacobian_skycal(
    gains: NDArray[np.complexfloating],
    model_visibilities: NDArray[np.complexfloating],
    data_visibilities: NDArray[np.complexfloating],
    visibility_weights: NDArray[np.floating],
    ant1_inds: NDArray[np.integer],
    ant2_inds: NDArray[np.integer],
    lambda_val: float,
) -> NDArray[np.complexfloating]:
    """
    Calculate the Jacobian of the cost function.

    Parameters
    ----------
    gains : array of complex
        Shape (Nants, Nfreqs, N_feed_pols).
    model_visibilities :  array of complex
        Shape (Ntimes, Nbls, Nfreqs, N_feed_pols). Cross-polarization visibilites are not
        supported; visibilities should include XX and YY only, ordered to correspond to the
        gain polarization convention.
    data_visibilities : array of complex
        Shape (Ntimes, Nbls, Nfreqs, N_feed_pols). Cross-polarization visibilites are not
        supported; visibilities should include XX and YY only, ordered to correspond to the
        gain polarization convention.
    visibility_weights : array of float
        Shape (Ntimes, Nbls, Nfreqs, N_feed_pols). Cross-polarization visibilites are not
        supported; visibilities should include XX and YY only, ordered to correspond to the
        gain polarization convention.
    ant1_inds : array of int
        Shape (Nbls,).
    ant2_inds : array of int
        Shape (Nbls,).
    lambda_val : float
        Weight of the phase regularization term; must be positive.

    Returns
    -------
    jac : array of complex
        Jacobian of the chi-squared cost function, shape (Nants, Nfreqs, N_feed_pols).
        The real part corresponds to derivatives with respect to the real part of the
        gains; the imaginary part corresponds to derivatives with respect to the
        imaginary part of the gains.
    """

    # Convert gains to visibility space
    # Add time axis
    gains_expanded_1 = gains[np.newaxis, ant1_inds, :, :]
    gains_expanded_2 = gains[np.newaxis, ant2_inds, :, :]

    res_vec = (
        gains_expanded_1 * np.conj(gains_expanded_2) * data_visibilities
        - model_visibilities
    )
    term1 = np.sum(
        visibility_weights * gains_expanded_2 * np.conj(data_visibilities) * res_vec,
        axis=0,
    )
    term1 = utils.bincount_multidim(
        ant1_inds,
        weights=term1,
        minlength=np.max([np.max(ant1_inds), np.max(ant2_inds)]) + 1,
    )
    term2 = np.sum(
        visibility_weights * gains_expanded_1 * data_visibilities * np.conj(res_vec),
        axis=0,
    )
    term2 = utils.bincount_multidim(
        ant2_inds,
        weights=term2,
        minlength=np.max([np.max(ant1_inds), np.max(ant2_inds)]) + 1,
    )

    jac = 2 * (term1 + term2)

    if lambda_val > 0:
        regularization_term = (
            lambda_val * 1j * np.sum(np.angle(gains)) * gains / np.abs(gains) ** 2.0
        )
        jac += 2 * regularization_term

    return jac


def reformat_baselines_to_antenna_matrix(
    bl_array: NDArray,
    ant1_inds: NDArray[np.integer],
    ant2_inds: NDArray[np.integer],
    Nants: int,
    Nbls: int,
) -> NDArray:
    """
    Reformat an array indexed in baselines into a matrix with antenna indices.

    Parameters
    ----------
    bl_array : array of float or complex
        Shape (Nbls, ...,).
    ant1_inds : array of int
        Shape (Nbls,).
    ant2_inds : array of int
        Shape (Nbls,).
    Nants : int
        Number of antennas.
    Nbls : int
        Number of baselines.

    Returns
    -------
    antenna matrix : array of float or complex
        Shape (Nants, Nants, ...,). Same dtype as bl_array.
    """

    antenna_matrix = np.zeros_like(
        bl_array[0,],
        dtype=bl_array.dtype,
    )
    antenna_matrix = np.repeat(
        np.repeat(antenna_matrix[np.newaxis,], Nants, axis=0)[np.newaxis,],
        Nants,
        axis=0,
    )
    for bl_ind in range(Nbls):
        antenna_matrix[
            ant1_inds[bl_ind],
            ant2_inds[bl_ind],
        ] = bl_array[
            bl_ind,
        ]
    return antenna_matrix


def hessian_skycal(
    gains: NDArray[np.complexfloating],
    Nants: int,
    Nbls: int,
    model_visibilities: NDArray[np.complexfloating],
    data_visibilities: NDArray[np.complexfloating],
    visibility_weights: NDArray[np.floating],
    ant1_inds: NDArray[np.integer],
    ant2_inds: NDArray[np.integer],
    lambda_val: float,
) -> Tuple[NDArray[np.floating], NDArray[np.floating], NDArray[np.floating]]:
    """
    Calculate the Hessian of the cost function.

    Parameters
    ----------
    gains : array of complex
        Shape (Nants, Nfreqs, N_feed_pols).
    Nants : int
        Number of antennas.
    Nbls : int
        Number of baselines.
    model_visibilities :  array of complex
        Shape (Ntimes, Nbls, Nfreqs, N_feed_pols). Cross-polarization visibilites are not
        supported; visibilities should include XX and YY only, ordered to correspond to the
        gain polarization convention.
    data_visibilities : array of complex
        Shape (Ntimes, Nbls, Nfreqs, N_feed_pols). Cross-polarization visibilites are not
        supported; visibilities should include XX and YY only, ordered to correspond to the
        gain polarization convention.
    visibility_weights : array of float
        Shape (Ntimes, Nbls, Nfreqs, N_feed_pols). Cross-polarization visibilites are not
        supported; visibilities should include XX and YY only, ordered to correspond to the
        gain polarization convention.
    ant1_inds : array of int
        Shape (Nbls,).
    ant2_inds : array of int
        Shape (Nbls,).
    lambda_val : float
        Weight of the phase regularization term; must be positive.

    Returns
    -------
    hess_real_real : array of float
        Real-real derivative components of the Hessian of the cost function.
        Shape (Nants, Nants, Nfreqs, N_feed_pols,).
    hess_real_imag : array of float
        Real-imaginary derivative components of the Hessian of the cost
        function. Note that the transpose of this array gives the imaginary-real
        derivative components. Shape (Nants, Nants, Nfreqs, N_feed_pols,).
    hess_imag_imag : array of float
        Imaginary-imaginary derivative components of the Hessian of the cost
        function. Shape (Nants, Nants, Nfreqs, N_feed_pols,).
    """

    gains_expanded_1 = gains[ant1_inds, :, :]
    gains_expanded_2 = gains[ant2_inds, :, :]
    data_squared = np.sum(visibility_weights * np.abs(data_visibilities) ** 2.0, axis=0)
    data_times_model = np.sum(
        visibility_weights * model_visibilities * np.conj(data_visibilities), axis=0
    )

    # Calculate the antenna off-diagonal components
    hess_components = np.zeros(
        (Nbls, 4, np.shape(gains)[1], np.shape(gains)[2]), dtype=float
    )
    # Real-real Hessian component:
    hess_components[:, 0, :, :] = np.real(
        4 * np.real(gains_expanded_1) * np.real(gains_expanded_2) * data_squared
        - 2 * np.real(data_times_model)
    )
    # Real-imaginary Hessian component, term 1:
    hess_components[:, 1, :, :] = np.real(
        4 * np.real(gains_expanded_1) * np.imag(gains_expanded_2) * data_squared
        + 2 * np.imag(data_times_model)
    )
    # Real-imaginary Hessian component, term 2:
    hess_components[:, 2, :, :] = np.real(
        4 * np.imag(gains_expanded_1) * np.real(gains_expanded_2) * data_squared
        - 2 * np.imag(data_times_model)
    )
    # Imaginary-imaginary Hessian component:
    hess_components[:, 3, :, :] = np.real(
        4 * np.imag(gains_expanded_1) * np.imag(gains_expanded_2) * data_squared
        - 2 * np.real(data_times_model)
    )

    hess_components = reformat_baselines_to_antenna_matrix(
        hess_components,
        ant1_inds,
        ant2_inds,
        Nants,
        Nbls,
    )
    hess_real_real = hess_components[:, :, 0, :, :] + np.transpose(
        hess_components[:, :, 0, :, :], axes=(1, 0, 2, 3)
    )
    hess_real_imag = hess_components[:, :, 1, :, :] + np.transpose(
        hess_components[:, :, 2, :, :], axes=(1, 0, 2, 3)
    )
    hess_imag_imag = hess_components[:, :, 3, :, :] + np.transpose(
        hess_components[:, :, 3, :, :], axes=(1, 0, 2, 3)
    )

    # Calculate the antenna diagonals
    hess_diag = 2 * (
        utils.bincount_multidim(
            ant1_inds,
            weights=np.abs(gains_expanded_2) ** 2.0 * data_squared,
            minlength=Nants,
        )
        + utils.bincount_multidim(
            ant2_inds,
            weights=np.abs(gains_expanded_1) ** 2.0 * data_squared,
            minlength=Nants,
        )
    )
    hess_real_real[np.diag_indices(Nants)] = hess_diag
    hess_imag_imag[np.diag_indices(Nants)] = hess_diag
    hess_real_imag[np.diag_indices(Nants)] = 0.0

    if lambda_val > 0:  # Add regularization term
        gains_weighted = gains / np.abs(gains) ** 2.0
        arg_sum = np.sum(np.angle(gains))
        # Antenna off-diagonals
        hess_real_real += (
            2
            * lambda_val
            * np.imag(gains_weighted)[:, np.newaxis, :, :]
            * np.imag(gains_weighted)[np.newaxis, :, :, :]
        )
        hess_real_imag -= (
            2
            * lambda_val
            * np.imag(gains_weighted)[:, np.newaxis, :, :]
            * np.real(gains_weighted)[np.newaxis, :, :, :]
        )
        hess_imag_imag += (
            2
            * lambda_val
            * np.real(gains_weighted)[:, np.newaxis, :, :]
            * np.real(gains_weighted)[np.newaxis, :, :, :]
        )
        # Antenna diagonals
        hess_real_real[np.diag_indices(Nants)] += (
            4 * lambda_val * arg_sum * np.imag(gains_weighted) * np.real(gains_weighted)
        )
        hess_real_imag[np.diag_indices(Nants)] -= (
            2
            * lambda_val
            * arg_sum
            * (np.real(gains_weighted) ** 2.0 - np.imag(gains_weighted) ** 2.0)
        )
        hess_imag_imag[np.diag_indices(Nants)] -= (
            4 * lambda_val * arg_sum * np.imag(gains_weighted) * np.real(gains_weighted)
        )

    return hess_real_real, hess_real_imag, hess_imag_imag


def set_crosspol_phase(
    gains: NDArray[np.complexfloating],
    crosspol_model_visibilities: NDArray[np.complexfloating],
    crosspol_data_visibilities: NDArray[np.complexfloating],
    crosspol_visibility_weights: NDArray[np.floating],
    ant1_inds: NDArray[int],
    ant2_inds: NDArray[int],
) -> float:
    """
    Calculate the cross-polarization phase between the P and Q gains. This
    quantity is not constrained in typical per-polarization calibration but is
    required for polarized imaging. See Byrne et al. 2022 for details of the
    calculation.

    Parameters
    ----------
    gains : array of complex
        Shape (Nants, 2,). gains[:, 0] corresponds to the P-polarized gains and
        gains[:, 1] corresponds to the Q-polarized gains.
    crosspol_model_visibilities :  array of complex
        Shape (Ntimes, Nbls, 2,). Cross-polarized model visibilities.
        model_visilibities[:, :, 0] corresponds to the PQ-polarized visibilities
        and model_visilibities[:, :, 1] corresponds to the QP-polarized
        visibilities.
    crosspol_data_visibilities : array of complex
        Shape (Ntimes, Nbls, 2,). Cross-polarized data visibilities.
        model_visilibities[:, :, 0] corresponds to the PQ-polarized visibilities
        and model_visilibities[:, :, 1] corresponds to the QP-polarized
        visibilities.
    crosspol_visibility_weights : array of float
        Shape (Ntimes, Nbls, 2).
    ant1_inds : array of int
        Shape (Nbls,).
    ant2_inds : array of int
        Shape (Nbls,).

    Returns
    -------
    crosspol_phase : float
        Cross-polarization phase, in radians.
    """

    gains_expanded_1 = gains[np.newaxis, ant1_inds, :]
    gains_expanded_2 = gains[np.newaxis, ant2_inds, :]
    term1 = np.nansum(
        crosspol_visibility_weights[:, :, 0]
        * np.conj(crosspol_model_visibilities[:, :, 0])
        * gains_expanded_1[:, :, 0]
        * np.conj(gains_expanded_2[:, :, 1])
        * crosspol_data_visibilities[:, :, 0]
    )
    term2 = np.nansum(
        crosspol_visibility_weights[:, :, 1]
        * crosspol_model_visibilities[:, :, 1]
        * np.conj(gains_expanded_1[:, :, 1])
        * gains_expanded_2[:, :, 0]
        * np.conj(crosspol_data_visibilities[:, :, 1])
    )
    crosspol_phase = np.angle(term1 + term2)

    return crosspol_phase


def set_crosspol_phase_pseudoV(
    gains: NDArray[np.complexfloating],
    crosspol_data_visibilities: NDArray[np.complexfloating],
    crosspol_visibility_weights: NDArray[np.floating],
    ant1_inds: NDArray[int],
    ant2_inds: NDArray[int],
) -> float:
    """
    Calculate the cross-polarization phase between the P and Q gains. This
    quantity is not constrained in typical per-polarization calibration but is
    required for polarized imaging. See Byrne et al. 2022 for details of the
    calculation.

    Parameters
    ----------
    gains : array of complex
        Shape (Nants, 2,). gains[:, 0] corresponds to the P-polarized gains and
        gains[:, 1] corresponds to the Q-polarized gains.
    crosspol_data_visibilities : array of complex
        Shape (Ntimes, Nbls, 2,). Cross-polarized data visibilities.
        model_visilibities[:, :, 0] corresponds to the PQ-polarized visibilities
        and model_visilibities[:, :, 1] corresponds to the QP-polarized
        visibilities.
    crosspol_visibility_weights : array of float
        Shape (Ntimes, Nbls, 2).
    ant1_inds : array of int
        Shape (Nbls,).
    ant2_inds : array of int
        Shape (Nbls,).

    Returns
    -------
    crosspol_phase : float
        Cross-polarization phase, in radians.
    """

    gains_expanded_1 = gains[np.newaxis, ant1_inds, :]
    gains_expanded_2 = gains[np.newaxis, ant2_inds, :]
    crosspol_data_visibilities_calibrated = crosspol_data_visibilities
    crosspol_data_visibilities_calibrated[:, :, 0] *= gains_expanded_1[
        :, :, 0
    ] * np.conj(
        gains_expanded_2[:, :, 1]
    )  # Apply gains to PQ visibilities
    crosspol_data_visibilities_calibrated[:, :, 1] *= gains_expanded_1[
        :, :, 1
    ] * np.conj(
        gains_expanded_2[:, :, 0]
    )  # Apply gains to QP visibilities
    visibility_weights = np.nanmean(
        crosspol_visibility_weights, axis=2
    )  # Doesn't support different weights for PQ and QP
    sum_term = np.nansum(
        visibility_weights
        * crosspol_data_visibilities_calibrated[:, :, 0]
        * np.conj(crosspol_data_visibilities_calibrated[:, :, 1])
    )
    crosspol_phase = np.angle(sum_term) / 2
    return crosspol_phase


def cost_function_abs_cal(
    amp: float,
    phase_grad: NDArray[float],
    model_visibilities: NDArray[np.complexfloating],
    data_visibilities: NDArray[np.complexfloating],
    uv_array: NDArray[np.floating],
    visibility_weights: NDArray[np.floating],
) -> float:
    """
    Calculate the cost function (chi-squared) value for absolute calibration.

    Parameters
    ----------
    amp : float
        Overall visibility amplitude.
    phase_grad :  array of float
        Shape (2,). Phase gradient terms, in units of 1/m.
    model_visibilities : array of complex
        Shape (Ntimes, Nbls,).
    data_visibilities : array of complex
        Relatively calibrated data. Shape (Ntimes, Nbls,).
    uv_array : array of float
        Shape(Nbls, 2,)
    visibility_weights : array of float
        Shape (Ntimes, Nbls,).

    Returns
    -------
    cost : float
        Value of the cost function.
    """

    phase_term = np.sum(phase_grad[np.newaxis, :] * uv_array, axis=1)
    res_vec = (amp**2.0 * np.exp(1j * phase_term))[
        np.newaxis, :
    ] * data_visibilities - model_visibilities
    cost = np.sum(visibility_weights * np.abs(res_vec) ** 2)
    return cost


def jacobian_abs_cal(
    amp: float,
    phase_grad: NDArray[float],
    model_visibilities: NDArray[np.complexfloating],
    data_visibilities: NDArray[np.complexfloating],
    uv_array: NDArray[np.floating],
    visibility_weights: NDArray[np.floating],
) -> Tuple[float, NDArray[np.floating]]:
    """
    Calculate the Jacobian for absolute calibration.

    Parameters
    ----------
    amp : float
        Overall visibility amplitude.
    phase_grad :  array of float
        Shape (2,). Phase gradient terms, in units of 1/m.
    model_visibilities : array of complex
        Shape (Ntimes, Nbls,).
    data_visibilities : array of complex
        Relatively calibrated data. Shape (Ntimes, Nbls,).
    uv_array : array of float
        Shape(Nbls, 2,)
    visibility_weights : array of float
        Shape (Ntimes, Nbls,).

    Returns
    -------
    amp_jac : float
        Derivative of the cost with respect to the visibility amplitude term.
    phase_jac : array of float
        Derivatives of the cost with respect to the phase gradient terms. Shape (2,).
    """

    phase_term = np.sum(phase_grad[np.newaxis, :] * uv_array, axis=1)
    data_prod = (
        np.exp(1j * phase_term)[np.newaxis, :]
        * data_visibilities
        * np.conj(model_visibilities)
    )

    amp_jac = (
        4
        * amp
        * np.sum(
            visibility_weights
            * (amp**2.0 * np.abs(data_visibilities) ** 2.0 - np.real(data_prod))
        )
    )
    phase_jac = (
        2
        * amp**2.0
        * np.sum(
            visibility_weights[:, :, np.newaxis]
            * uv_array[np.newaxis, :, :]
            * np.imag(data_prod)[:, :, np.newaxis],
            axis=(0, 1),
        )
    )

    return amp_jac, phase_jac


def hess_abs_cal(
    amp: float,
    phase_grad: NDArray[float],
    model_visibilities: NDArray[np.complexfloating],
    data_visibilities: NDArray[np.complexfloating],
    uv_array: NDArray[np.floating],
    visibility_weights: NDArray[np.floating],
) -> Tuple[float, float, float, float, float, float]:
    """
    Calculate the Hessian for absolute calibration.

    Parameters
    ----------
    amp : float
        Overall visibility amplitude.
    phase_grad :  array of float
        Shape (2,). Phase gradient terms, in units of 1/m.
    model_visibilities : array of complex
        Shape (Ntimes, Nbls,).
    data_visibilities : array of complex
        Relatively calibrated data. Shape (Ntimes, Nbls,).
    uv_array : array of float
        Shape(Nbls, 2,)
    visibility_weights : array of float
        Shape (Ntimes, Nbls,).


    Returns
    -------
    hess_amp_amp : float
        Second derivative of the cost with respect to the amplitude term.
    hess_amp_phasex : float
        Second derivative of the cost with respect to the amplitude term and the phase gradient in x.
    hess_amp_phasey : float
        Second derivative of the cost with respect to the amplitude term and the phase gradient in y.
    hess_phasex_phasex : float
        Second derivative of the cost with respect to the phase gradient in x.
    hess_phasey_phasey : float
        Second derivative of the cost with respect to the phase gradient in x.
    hess_phasex_phasey : float
        Second derivative of the cost with respect to the phase gradient in x and y.
    """

    phase_term = np.sum(phase_grad[np.newaxis, :] * uv_array, axis=1)
    data_prod = (
        np.exp(1j * phase_term)[np.newaxis, :]
        * data_visibilities
        * np.conj(model_visibilities)
    )

    hess_amp_amp = np.sum(
        visibility_weights
        * (
            12.0 * amp**2.0 * np.abs(data_visibilities) ** 2.0
            - 4.0 * np.real(data_prod)
        )
    )

    hess_amp_phasex = (
        4.0
        * amp
        * np.sum(visibility_weights * uv_array[np.newaxis, :, 0] * np.imag(data_prod))
    )
    hess_amp_phasey = (
        4.0
        * amp
        * np.sum(visibility_weights * uv_array[np.newaxis, :, 1] * np.imag(data_prod))
    )

    hess_phasex_phasex = (
        2.0
        * amp**2.0
        * np.sum(
            visibility_weights * uv_array[np.newaxis, :, 0] ** 2.0 * np.real(data_prod)
        )
    )

    hess_phasey_phasey = (
        2.0
        * amp**2.0
        * np.sum(
            visibility_weights * uv_array[np.newaxis, :, 1] ** 2.0 * np.real(data_prod)
        )
    )

    hess_phasex_phasey = (
        2.0
        * amp**2.0
        * np.sum(
            visibility_weights
            * uv_array[np.newaxis, :, 0]
            * uv_array[np.newaxis, :, 1]
            * np.real(data_prod)
        )
    )

    return (
        hess_amp_amp,
        hess_amp_phasex,
        hess_amp_phasey,
        hess_phasex_phasex,
        hess_phasey_phasey,
        hess_phasex_phasey,
    )


def cost_function_dw_abscal(
    amp: float,
    phase_grad: NDArray[float],
    model_visibilities: NDArray[np.complexfloating],
    data_visibilities: NDArray[np.complexfloating],
    uv_array: NDArray[np.floating],
    visibility_weights: NDArray[np.floating],
    dwcal_inv_covariance: NDArray[np.complexfloating],
) -> float:
    """
    Calculate the cost function (chi-squared) value for absolute calibration
    with delay weighting.

    Parameters
    ----------
    amp : array of float
        Shape (Nfreqs,). Overall visibility amplitude.
    phase_grad :  array of float
        Shape (2, Nfreqs,). Phase gradient terms, in units of 1/m.
    model_visibilities : array of complex
        Shape (Ntimes, Nbls, Nfreqs,).
    data_visibilities : array of complex
        Relatively calibrated data. Shape (Ntimes, Nbls, Nfreqs,).
    uv_array : array of float
        Shape(Nbls, 2,)
    visibility_weights : array of float
        Shape (Ntimes, Nbls, Nfreqs,).
    dwcal_inv_covariance : array of complex
        Shape (Ntimes, Nbls, Nfreqs, Nfreqs,).

    Returns
    -------
    cost : float
        Value of the cost function.
    """

    phase_term = np.sum(
        phase_grad[np.newaxis, :, :] * uv_array[:, :, np.newaxis], axis=1
    )  # Shape (Nbls, Nfreqs,)
    res_vec = np.sqrt(visibility_weights) * (
        (amp[np.newaxis, :] ** 2.0 * np.exp(1j * phase_term))[np.newaxis, :, :]
        * data_visibilities
        - model_visibilities
    )  # Shape (Ntimes, Nbls, Nfreqs)
    cost = np.real(
        np.sum(
            dwcal_inv_covariance
            * np.conj(res_vec[:, :, :, np.newaxis])
            * res_vec[:, :, np.newaxis, :]
        )
    )
    print(f"DWAbscal cost: {cost}")
    sys.stdout.flush()
    return cost


def cost_function_dw_abscal_toeplitz(
    amp: float,
    phase_grad: NDArray[float],
    model_visibilities: NDArray[np.complexfloating],
    data_visibilities: NDArray[np.complexfloating],
    uv_array: NDArray[np.floating],
    visibility_weights: NDArray[np.floating],
    dwcal_inv_covariance: NDArray[np.complexfloating],
) -> float:
    """
    Calculate the cost function (chi-squared) value for absolute calibration
    with delay weighting.

    Parameters
    ----------
    amp : array of float
        Shape (Nfreqs,). Overall visibility amplitude.
    phase_grad :  array of float
        Shape (2, Nfreqs,). Phase gradient terms, in units of 1/m.
    model_visibilities : array of complex
        Shape (Ntimes, Nbls, Nfreqs,).
    data_visibilities : array of complex
        Relatively calibrated data. Shape (Ntimes, Nbls, Nfreqs,).
    uv_array : array of float
        Shape(Nbls, 2,)
    visibility_weights : array of float
        Shape (Ntimes, Nbls, Nfreqs,).
    dwcal_inv_covariance : array of complex
        Shape (Ntimes, Nbls, Nfreqs,).

    Returns
    -------
    cost : float
        Value of the cost function.
    """

    phase_term = np.sum(
        phase_grad[np.newaxis, :, :] * uv_array[:, :, np.newaxis], axis=1
    )  # Shape (Nbls, Nfreqs,)
    res_vec = np.sqrt(visibility_weights) * (
        (amp[np.newaxis, :] ** 2.0 * np.exp(1j * phase_term))[np.newaxis, :, :]
        * data_visibilities
        - model_visibilities
    )  # Shape (Ntimes, Nbls, Nfreqs)
    cost = np.real(
        np.sum(
            dwcal_inv_covariance
            * np.conj(res_vec[:, :, :, np.newaxis])
            * res_vec[:, :, np.newaxis, :]
        )
    )
    print(f"DWAbscal cost: {cost}")
    sys.stdout.flush()
    return cost


def jacobian_dw_abscal(
    amp: float,
    phase_grad: NDArray[float],
    model_visibilities: NDArray[np.complexfloating],
    data_visibilities: NDArray[np.complexfloating],
    uv_array: NDArray[np.floating],
    visibility_weights: NDArray[np.floating],
    dwcal_inv_covariance: NDArray[np.complexfloating],
) -> Tuple[NDArray[np.floating], NDArray[np.floating]]:
    """
    Calculate the Jacobian for absolute calibration with delay weighting.

    Parameters
    ----------
    amp : array of float
        Shape (Nfreqs,). Overall visibility amplitude.
    phase_grad :  array of float
        Shape (2, Nfreqs,). Phase gradient terms, in units of 1/m.
    model_visibilities : array of complex
        Shape (Ntimes, Nbls, Nfreqs,).
    data_visibilities : array of complex
        Relatively calibrated data. Shape (Ntimes, Nbls, Nfreqs,).
    uv_array : array of float
        Shape(Nbls, 2,)
    visibility_weights : array of float
        Shape (Ntimes, Nbls, Nfreqs,).
    dwcal_inv_covariance : array of complex
        Shape (Ntimes, Nbls, Nfreqs, Nfreqs,).

    Returns
    -------
    amp_jac : array of float
        Derivative of the cost with respect to the visibility amplitude terms. Shape (Nfreqs,).
    phase_jac : array of float
        Derivatives of the cost with respect to the phase gradient terms. Shape (2, Nfreqs,).

    """

    phase_term = np.sum(
        phase_grad[np.newaxis, :, :] * uv_array[:, :, np.newaxis], axis=1
    )  # Shape (Nbls, Nfreqs,)
    res_vec = np.sqrt(visibility_weights) * (
        (amp[np.newaxis, :] ** 2.0 * np.exp(1j * phase_term))[np.newaxis, :, :]
        * data_visibilities
        - model_visibilities
    )  # Shape (Ntimes, Nbls, Nfreqs,)
    derivative_term = (
        np.sqrt(visibility_weights)
        * np.exp(-1j * phase_term)[np.newaxis, :, :]
        * np.conj(data_visibilities)
    )
    amp_jac = (
        4
        * amp
        * np.real(
            np.sum(
                dwcal_inv_covariance
                * derivative_term[:, :, :, np.newaxis]
                * res_vec[:, :, np.newaxis, :],
                axis=(0, 1, 3),
            )
        )
    )
    phase_jac = (
        2
        * amp[:, np.newaxis] ** 2.0
        * np.real(
            np.sum(
                dwcal_inv_covariance[:, :, :, :, np.newaxis]
                * (-1j)
                * uv_array[np.newaxis, :, np.newaxis, np.newaxis, :]
                * derivative_term[:, :, :, np.newaxis, np.newaxis]
                * res_vec[:, :, np.newaxis, :, np.newaxis],
                axis=(0, 1, 3),
            )
        )
    ).T
    return amp_jac, phase_jac


def hess_dw_abscal(
    amp: float,
    phase_grad: NDArray[float],
    model_visibilities: NDArray[np.complexfloating],
    data_visibilities: NDArray[np.complexfloating],
    uv_array: NDArray[np.floating],
    visibility_weights: NDArray[np.floating],
    dwcal_inv_covariance: NDArray[np.complexfloating],
) -> Tuple[
    NDArray[np.floating],
    NDArray[np.floating],
    NDArray[np.floating],
    NDArray[np.floating],
    NDArray[np.floating],
    NDArray[np.floating],
]:
    """
    Calculate the Hessian for absolute calibration with delay weighting.

    Parameters
    ----------
    amp : array of float
        Shape (Nfreqs,). Overall visibility amplitude.
    phase_grad :  array of float
        Shape (2, Nfreqs,). Phase gradient terms, in units of 1/m.
    model_visibilities : array of complex
        Shape (Ntimes, Nbls, Nfreqs,).
    data_visibilities : array of complex
        Relatively calibrated data. Shape (Ntimes, Nbls, Nfreqs,).
    uv_array : array of float
        Shape(Nbls, 2,)
    visibility_weights : array of float
        Shape (Ntimes, Nbls, Nfreqs,).
    dwcal_inv_covariance : array of complex
        Shape (Ntimes, Nbls, Nfreqs, Nfreqs,).

    Returns
    hess_amp_amp : array of float
        Shape (Nfreqs, Nfreqs,). Second derivative of the cost with respect to the
        amplitude term.
    hess_amp_phasex : array of float
        Shape (Nfreqs, Nfreqs,). Second derivative of the cost with respect to the
        amplitude term and the phase gradient in x.
    hess_amp_phasey : array of float
        Shape (Nfreqs, Nfreqs,). Second derivative of the cost with respect to the
        amplitude term and the phase gradient in y.
    hess_phasex_phasex : array of float
        Shape (Nfreqs, Nfreqs,). Second derivative of the cost with respect to the
        phase gradient in x.
    hess_phasey_phasey : array of float
        Shape (Nfreqs, Nfreqs,). Second derivative of the cost with respect to the
        phase gradient in x.
    hess_phasex_phasey : array of float
        Shape (Nfreqs, Nfreqs,). Second derivative of the cost with respect to the
        phase gradient in x and y.
    -------

    """

    phase_term = np.sum(
        phase_grad[np.newaxis, :, :] * uv_array[:, :, np.newaxis], axis=1
    )  # Shape (Nbls, Nfreqs,)
    res_vec = np.sqrt(visibility_weights) * (
        (amp[np.newaxis, :] ** 2.0 * np.exp(1j * phase_term))[np.newaxis, :, :]
        * data_visibilities
        - model_visibilities
    )  # Shape (Ntimes, Nbls, Nfreqs,)
    derivative_term = (
        np.sqrt(visibility_weights)
        * np.exp(-1j * phase_term)[np.newaxis, :, :]
        * np.conj(data_visibilities)
    )  # Shape (Ntimes, Nbls, Nfreqs,)

    hess_amp_amp_diagonal_term = 4 * np.real(
        np.sum(
            dwcal_inv_covariance
            * derivative_term[:, :, :, np.newaxis]
            * res_vec[:, :, np.newaxis, :],
            axis=(0, 1, 3),
        )
    )
    hess_amp_amp = (
        8
        * amp[:, np.newaxis]
        * amp[np.newaxis, :]
        * np.real(
            np.sum(
                dwcal_inv_covariance
                * derivative_term[:, :, :, np.newaxis]
                * np.conj(derivative_term[:, :, np.newaxis, :]),
                axis=(0, 1),
            )
        )
    ) + np.diag(hess_amp_amp_diagonal_term)

    hess_amp_phase_diagonal_term = (
        4
        * amp[:, np.newaxis]
        * np.real(
            np.sum(
                dwcal_inv_covariance[:, :, :, :, np.newaxis]
                * (-1j)
                * uv_array[np.newaxis, :, np.newaxis, np.newaxis, :]
                * derivative_term[:, :, :, np.newaxis, np.newaxis]
                * res_vec[:, :, np.newaxis, :, np.newaxis],
                axis=(0, 1, 3),
            )
        )
    )
    hess_amp_phase = (
        4
        * amp[:, np.newaxis, np.newaxis] ** 2.0
        * amp[np.newaxis, :, np.newaxis]
        * np.real(
            np.sum(
                dwcal_inv_covariance[:, :, :, :, np.newaxis]
                * (-1j)
                * uv_array[np.newaxis, :, np.newaxis, np.newaxis, :]
                * derivative_term[:, :, np.newaxis, :, np.newaxis]
                * np.conj(derivative_term[:, :, :, np.newaxis, np.newaxis]),
                axis=(0, 1),
            )
        )
    )
    hess_amp_phasex = hess_amp_phase[:, :, 0] + np.diag(
        hess_amp_phase_diagonal_term[:, 0]
    )
    hess_amp_phasey = hess_amp_phase[:, :, 1] + np.diag(
        hess_amp_phase_diagonal_term[:, 1]
    )

    hess_phasex_phasex = (
        2
        * amp[:, np.newaxis] ** 2.0
        * amp[np.newaxis, :] ** 2.0
        * np.real(
            np.sum(
                dwcal_inv_covariance
                * uv_array[np.newaxis, :, np.newaxis, np.newaxis, 0] ** 2.0
                * derivative_term[:, :, np.newaxis, :]
                * np.conj(derivative_term[:, :, :, np.newaxis]),
                axis=(0, 1),
            )
        )
    )
    hess_phasey_phasey = (
        2
        * amp[:, np.newaxis] ** 2.0
        * amp[np.newaxis, :] ** 2.0
        * np.real(
            np.sum(
                dwcal_inv_covariance
                * uv_array[np.newaxis, :, np.newaxis, np.newaxis, 1] ** 2.0
                * derivative_term[:, :, np.newaxis, :]
                * np.conj(derivative_term[:, :, :, np.newaxis]),
                axis=(0, 1),
            )
        )
    )
    hess_phasex_phasey = (
        2
        * amp[:, np.newaxis] ** 2.0
        * amp[np.newaxis, :] ** 2.0
        * np.real(
            np.sum(
                dwcal_inv_covariance
                * uv_array[np.newaxis, :, np.newaxis, np.newaxis, 0]
                * uv_array[np.newaxis, :, np.newaxis, np.newaxis, 1]
                * derivative_term[:, :, np.newaxis, :]
                * np.conj(derivative_term[:, :, :, np.newaxis]),
                axis=(0, 1),
            )
        )
    )
    hess_phasex_phasex_diagonal_term = (
        -2
        * amp**2.0
        * np.real(
            np.sum(
                dwcal_inv_covariance
                * uv_array[np.newaxis, :, np.newaxis, np.newaxis, 0] ** 2.0
                * derivative_term[:, :, :, np.newaxis]
                * res_vec[:, :, np.newaxis, :],
                axis=(0, 1, 3),
            )
        )
    )
    hess_phasey_phasey_diagonal_term = (
        -2
        * amp**2.0
        * np.real(
            np.sum(
                dwcal_inv_covariance
                * uv_array[np.newaxis, :, np.newaxis, np.newaxis, 1] ** 2.0
                * derivative_term[:, :, :, np.newaxis]
                * res_vec[:, :, np.newaxis, :],
                axis=(0, 1, 3),
            )
        )
    )
    hess_phasex_phasey_diagonal_term = (
        -2
        * amp**2.0
        * np.real(
            np.sum(
                dwcal_inv_covariance
                * uv_array[np.newaxis, :, np.newaxis, np.newaxis, 0]
                * uv_array[np.newaxis, :, np.newaxis, np.newaxis, 1]
                * derivative_term[:, :, :, np.newaxis]
                * res_vec[:, :, np.newaxis, :],
                axis=(0, 1, 3),
            )
        )
    )
    hess_phasex_phasex += np.diag(hess_phasex_phasex_diagonal_term)
    hess_phasey_phasey += np.diag(hess_phasey_phasey_diagonal_term)
    hess_phasex_phasey += np.diag(hess_phasex_phasey_diagonal_term)

    return (
        hess_amp_amp,
        hess_amp_phasex,
        hess_amp_phasey,
        hess_phasex_phasex,
        hess_phasey_phasey,
        hess_phasex_phasey,
    )
