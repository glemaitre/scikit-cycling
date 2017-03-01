"""Ride probabilistic power-profile class."""
from __future__ import division, print_function

import numpy as np

from ..externals.joblib import Parallel, delayed
from ..utils import check_filename_fit
from ..utils import load_power_from_fit


def _rppp_parallel(X, idx_t_rpp, max_power, bins):
    """Function to compute the rpp in parallel.

    Parameters
    ----------
    X : ndarray, shape (n_samples)
        The power records.

    idx_t_rpp : int
        Index of the time to compute the probabilistic power-profile.

    max_power : float,
        The maximum power to consider while building the histogram for
        the current time

    Returns
    -------
    power : float
        Returns the best power for the given duration of the probabilistic
        power-profile.

    """
    # Slice the data such that we can compute efficiently the mean later
    t_crop = np.array([X[i:-idx_t_rpp + i:] for i in range(idx_t_rpp)])
    if t_crop.size > 0:
        t_crop_mean = np.mean(t_crop, axis=0)
        pdf, _ = np.histogram(t_crop_mean, bins=bins,
                              range=(0., max_power), density=True)
        return pdf
    else:
        return np.zeros(bins)


class RideProbabilisticPowerProfile(object):
    """Probabilistic Power Profile for a ride

    Contrary to the classic power-profile, the probabilistic power-profile
    tends to compute more information regarding each ride by computing the
    probability density function of the power instead of only the maximum.

    Parameters
    ----------
    max_duration_profile : int, optional (default=300)
        The maximum duration of the profile in minutes.

    cyclist_weight : float, optional (default=60.)
        The weight of the cyclist in kg.

    n_jobs : int, optional (default=1)
        Number of workers to use during parallel processing if possible.

    Attributes
    ----------
    filename_ : str
        The corresponding fit file attached to this power-profile.

    date_profile_ : date
        Date of the current power-profile.

    data_ : ndarray, shape (max_power, 60 * max_duration_profile)
        Array containing the probabilistic power-profile.

    data_norm_ : ndarray, shape (max_power, 60 * max_duration_profile)
        Array containing the probabilistic power-profile normalized by the
        weight.
    """

    def __init__(self, max_duration_profile=300, cyclist_weight=60., n_jobs=1):
        self.max_duration_profile = max_duration_profile
        self.cyclist_weight = cyclist_weight
        self.n_jobs = n_jobs

    def fit(self, filename):
        """Read and build the probabilistic power-profile from the fit file.

        Parameters
        ----------
        filename : str
            File name of the FIT file.

        Returns
        -------
        self : object
            Returns self.
        """
        self.filename_ = check_filename_fit(filename)

        ride_power, self.date_profile_ = load_power_from_fit(self.filename_)

        if len(ride_power.shape) != 1:
            raise ValueError('X should have 1 dimension. Got {},'
                             'instead'.format(len(ride_power.shape)))
        if self.max_duration_profile is None:
            raise ValueError('You need to specify the maximum duration that is'
                             ' required during the profile computation.')

        self.max_power_ = int(np.max(ride_power))

        pp = Parallel(n_jobs=self.n_jobs)(
            delayed(_rppp_parallel)(ride_power, idx_t_rpp, self.max_power_,
                                    self.max_power_ + 1)
            for idx_t_rpp in range(60 * self.max_duration_profile))
        # replace the nan by zeros
        self.data_ = np.nan_to_num(np.transpose(pp))

        if self.cyclist_weight is not None:
            self.data_norm_ = self.data_ / self.cyclist_weight
        else:
            self.data_norm_ = None

        return self
