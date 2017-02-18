"""Methods to denoise the power signal provided from a ride."""

import numpy as np


def outliers_rejection(X, method='threshold', thres=2500.):
    """Remove the outliers from the given ride.

    Parameters
    ----------
    X : array-like, shape (n_samples, )
        Array containing the power intensities for a ride.

    method : string, default 'threshold'
        String to specified which outliers detection method to use.

    thres : float, default 2500.
        The maximum power to consider in case the method 'threshold'
        is considered.

    Returns
    -------
    X : array-like, shape (n_samples, )
        Array containing the power intensities, outliers free.

    """
    if len(X.shape) != 1:
        raise ValueError('X should have 1 dimension. Got {}, instead'.format(
            len(X.shape)))

    # Detect the outliers
    if method == 'threshold':
        # Compute the mean value that we will use to make the replacement
        mean_ride = np.mean(X)
        X[X > thres] = mean_ride
        X[X < 0] = mean_ride
    else:
        raise ValueError('The outliers detection method is unknown.')

    return X


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

    ret = np.cumsum(X, dtype=np.float64)
    ret[win:] = ret[win:] - ret[:-win]

    return ret[win - 1:] / win
