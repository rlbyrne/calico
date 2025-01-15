import numpy as np


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
