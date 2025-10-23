import numpy as np
from numba import njit


@njit
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
        return bincount_numba(x, weights=weights, minlength=minlength)
    elif weights.ndim < 2 and np.min(np.isreal(weights)):
        return bincount_numba(x, weights=np.real(weights), minlength=minlength).astype(
            weights.dtype
        )
    else:
        if np.min(np.isreal(weights)):
            use_weights_list = [np.real(weights)]
        else:
            use_weights_list = [np.real(weights), np.imag(weights)]
        out_list = []
        for use_weights in use_weights_list:  # Iterate over real and imaginary parts
            weights_transposed = rollaxis_numba(use_weights, axis, start=0)
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
                out[:, ind] = bincount_numba(
                    x, weights=weights_transposed[:, ind], minlength=minlength
                )
            out = out.reshape(output_shape)
            out = rollaxis_numba(out, 0, start=axis + 1)
            out_list.append(out)
        if len(out_list) == 1:
            return out_list[0]
        else:
            return out_list[0] + 1j * out_list[1]
        
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
    return a

@njit
def bincount_numba(x, weights=None, minlength=0, axis=0):
    """
    Numba-compatible version of np.bincount.
    """
    if axis != 0:
        raise NotImplementedError("Only axis=0 supported")

    x = np.asarray(x)
    n = x.size

    # Handle empty input
    if n == 0:
        if weights is None:
            return np.zeros(minlength, dtype=np.float64)
        else:
            shape = (minlength,) + weights.shape[1:]
            return np.zeros(shape, dtype=weights.dtype)

    # Prepare weights array inside Numba
    if weights is None:
        dtype = np.float64
        weights_arr = np.ones(n, dtype=dtype)
        shape_weights = ()
    else:
        weights_arr = weights  # already a NumPy array, dtype consistent
        dtype = weights_arr.dtype
        shape_weights = weights_arr.shape[1:]

    # Find maximum index
    maxval = 0
    for i in range(n):
        xi = x[i]
        if xi < 0:
            raise ValueError("x contains negative values")
        if xi > maxval:
            maxval = xi

    n_bins = max(maxval + 1, minlength)

    # Initialize counts array
    counts = np.zeros((n_bins,) + shape_weights, dtype=dtype)

    # Accumulate
    for i in range(n):
        counts[x[i]] += weights_arr[i]

    return counts