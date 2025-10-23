import numpy as np
from numba import njit


@njit
def bincount_multidim(x, weights, minlength=0, axis=0):
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

    if np.min(np.isreal(weights)):
        out = bincount_multidim_real(x, weights=weights, minlength=minlength, axis=axis)
    else:
        out_real = bincount_multidim_real(
            x, weights=np.real(weights), minlength=minlength, axis=axis
        )
        out_imag = bincount_multidim_real(
            x, weights=np.imag(weights), minlength=minlength, axis=axis
        )
        out = out_real + 1j * out_imag
    return out


@njit
def bincount_multidim_real(x, weights=None, minlength=0, axis=0):
    """
    Extension of numpy.bincount that supports multidimensional data and complex values.

    Parameters
    ----------
    x : array of int
        Must be 1 dimension and contain only nonnegative ints.
    weights :  array of float, optional
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
        return bincount_numba(x, np.ones(np.shape(x)), minlength=minlength)
    elif weights.ndim == 1 and np.min(np.isreal(weights)):
        return bincount_numba(x, np.real(weights), minlength=minlength).astype(
            weights.dtype
        )
    else:
        weights_transposed = rollaxis_numba(weights, axis, start=0)
        output_shape = np.array(np.shape(weights_transposed))
        output_shape[0] = max(np.max(x) + 1, minlength)
        weights_transposed = weights_transposed.reshape(
            (
                np.shape(weights_transposed)[0],
                np.prod(np.array(np.shape(weights_transposed)[1:])),
            )
        )
        out = np.zeros(
            (output_shape[0], np.shape(weights_transposed)[1]),
            dtype=weights.dtype,
        )
        for ind in range(np.shape(weights_transposed)[1]):
            out[:, ind] = bincount_numba(
                x, weights=weights_transposed[:, ind], minlength=minlength
            )
        out = out.reshape([use_int for use_int in output_shape])
        out = rollaxis_numba(out, 0, start=axis + 1)
        return out


@njit
def rollaxis_numba(a, axis, start=0):
    """
    Numba-compatible version of np.rollaxis.
    """
    ndim = a.ndim
    if axis < 0:
        axis += ndim
    if start < 0:
        start += ndim
    if not (0 <= axis < ndim and 0 <= start <= ndim):
        raise ValueError("axis or start out of range")

    if axis < start:
        for i in range(axis, start - 1):
            a = np.swapaxes(a, i, i + 1)
    elif axis > start:
        for i in range(axis, start, -1):
            a = np.swapaxes(a, i, i - 1)
    return np.ascontiguousarray(a)


@njit
def bincount_numba(x, weights, minlength=0):
    """
    Numba-compatible version of np.bincount.
    """

    outlength = max(np.amax(x) + 1, minlength)
    out = np.zeros(outlength, dtype=weights.dtype)
    for ind in range(outlength):
        use_inds = np.where(x == ind)[0]
        out[ind] = np.sum(weights[use_inds])
    return out
