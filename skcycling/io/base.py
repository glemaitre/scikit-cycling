"""Methods to load power data file."""

# Authors: Guillaume Lemaitre <g.lemaitre58@gmail.com>
#          Cedric Lemaitre
# License: BSD 3 clause

import numpy as np

from .fit import load_power_from_fit


def bikeread(filename):
    """Read power data file.

    Parameters
    ----------
    filename : str
        Path to the file to read.

    Returns
    -------
    data : DataFrame
        Power data and time data.

    Examples
    --------
    >>> from skcycling.datasets import load_fit
    >>> from skcycling.io import bikeread
    >>> activity = bikeread(load_fit()[0])
    >>> activity.head() # doctest : +NORMALIZE_WHITESPACE
                         power
    2014-05-07 12:26:22  256.0
    2014-05-07 12:26:23  185.0
    2014-05-07 12:26:24  343.0
    2014-05-07 12:26:25  344.0
    2014-05-07 12:26:26  389.0

    """
    df = load_power_from_fit(filename)

    # remove possible outliers by clipping the value
    df[df['power'] > 2500.] = np.nan

    # resample to have a precision of a second with additional linear
    # interpolation for missing value
    return df.resample('s').interpolate('linear')
