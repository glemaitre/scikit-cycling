"""Methods to denoise the power signal provided from a ride."""

import numpy as np


def moving_average(X, win=30):
    """Apply an average filter to the data.

    Parameters
    ----------
    X : array-like, shape (n_samples, )
        Array containing the ride or a selection of a ride.

    win : interger, optional (default=30)
        Size of the sliding window.

    Returns
    -------
    avg : array-like (float)
        Return the denoised data mean-filter.

    """

    ret = np.cumsum(X)
    ret[win:] = ret[win:] - ret[:-win]

    return ret[win - 1:] / win
