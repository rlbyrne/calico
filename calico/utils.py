import numpy as np
import jax.numpy as jnp


def bincount_multidim(x, weights=None, minlength=0, axis=0):
    """
    Extension of numpy.bincount that supports multidimensional data and complex values.

    Parameters
    ----------
    x : array of int
        Must be 1 dimension and contain only nonnegative ints.
    weights :  array of float or complex float, optional
        Length of axis specified by axis keyword must equal the length of x.
    minlength : int, optional
        See numpy.bincount documentation.
    axis : int
        Axis of weights over which bincount will be calculated. Default 0.

    Returns
    -------
    out : array of int, float, or complex float
        Dimensionality and dtype is the same as those of weights. Shape is the same
        for all axes except that of the axis keyword, which has length
        max(x) + 1.
    """

    if weights is None:
        return np.bincount(x, weights=weights, minlength=minlength)
    elif weights.ndim < 2 and np.min(np.isreal(weights)):
        return np.bincount(x, weights=np.real(weights), minlength=minlength).astype(
            weights.dtype
        )
    else:
        if np.min(np.isreal(weights)):
            use_weights_list = [np.real(weights)]
        else:
            use_weights_list = [np.real(weights), np.imag(weights)]
        out_list = []
        for use_weights in use_weights_list:  # Iterate over real and imaginary parts
            weights_transposed = np.rollaxis(use_weights, axis, start=0)
            output_shape = np.array(np.shape(weights_transposed))
            output_shape[0] = np.max([np.max(x) + 1, minlength])
            weights_transposed = weights_transposed.reshape(
                (
                    np.shape(weights_transposed)[0],
                    np.prod(np.shape(weights_transposed)[1:], dtype=int),
                )
            )
            out = np.zeros(
                (output_shape[0], np.shape(weights_transposed)[1]),
                dtype=weights.dtype,
            )
            for ind in range(np.shape(weights_transposed)[1]):
                out[:, ind] = np.bincount(
                    x, weights=weights_transposed[:, ind], minlength=minlength
                )
            out = out.reshape(output_shape)
            out = np.rollaxis(out, 0, start=axis + 1)
            out_list.append(out)
        if len(out_list) == 1:
            return out_list[0]
        else:
            return out_list[0] + 1j * out_list[1]


def tukey_taper(x, taper_start, taper_width):
    """
    Parameters
    ----------
    x : array of float
    taper_start :  float
        Value correponding to the start of the taper.
    taper_width : float
        Width of the taper.

    Returns
    -------
    out : array of float
        Same shape as x.
    """

    out = 0.5 * jnp.cos(2 * jnp.pi * (x - taper_start) / taper_width) + 0.5
    out = jnp.where(jnp.abs(x) < taper_start, 1.0, out)
    out = jnp.where(jnp.abs(x) > taper_start + taper_width, 0.0, out)
    return out


def gaussian_2d_taper(
    uv_array, stddev_primary, stddev_secondary, axis_angle, normalize=True
):
    """
    Parameters
    ----------
    uv_array : array of float
        Shape (Nbls, 2,)
    stddev_primary : array of float
        Standard deviation of the primary axis.
    stddev_secondary : array of float
        Standard deviation of the secondary axis.
    axis_angle : array of float
        Angle of the primary axis, in radians.
    normalize : bool
        If True, the average value is constrained to be 1.

    Returns
    -------
    out : array of float
        Shape (Nbls, ...)
    """

    n_dims = stddev_primary.ndim
    u = uv_array[:, 0].reshape((-1,) + (1,) * n_dims)  # (Nbls, 1, 1)
    v = uv_array[:, 1].reshape((-1,) + (1,) * n_dims)  # (Nbls, 1, 1)

    cos_a = jnp.cos(axis_angle[jnp.newaxis, ...])  # (1, Nfreqs, Npols)
    sin_a = jnp.sin(axis_angle[jnp.newaxis, ...])  # (1, Nfreqs, Npols)

    primary_axis_coords = u * cos_a + v * sin_a  # (Nbls, Nfreqs, Npols)
    secondary_axis_coords = u * sin_a + v * cos_a  # (Nbls, Nfreqs, Npols)

    gaussian_2d = jnp.exp(
        -(primary_axis_coords**2) / (2 * stddev_primary[jnp.newaxis, ...] ** 2)
        - secondary_axis_coords**2 / (2 * stddev_secondary[jnp.newaxis, ...] ** 2)
    )

    if normalize:
        normalization_factor = jnp.sum(gaussian_2d, axis=0) / jnp.shape(uv_array)[0]
        gaussian_2d /= normalization_factor[jnp.newaxis, ...]

    return gaussian_2d
