"""Record probabilistic power-profile."""
from __future__ import division

import numpy as np

from scipy import sparse

from .base import BaseProbabilisticPowerProfile
from .ride_probabilistic_power_profile import RideProbabilisticPowerProfile
from ..utils.checker import check_tuple_date


class RecordProbabilisticPowerProfile(BaseProbabilisticPowerProfile):
    """Probabilistic Power Profile for a record

    Contrary to the classic power-profile, the probabilistic power-profile
    tends to compute more information regarding each ride by computing the
    probability density function of the power instead of only the maximum.

    Parameters
    ----------
    max_duration_profile : int, optional (default=300)
        The maximum duration of the profile in minutes.

    cyclist_weight : float, optional (default=60.)
        The weight of the cyclist in kg.

    Attributes
    ----------
    filename_ : str
        The corresponding fit file attached to this power-profile.

    date_profile_ : date
        Date of the current power-profile.

    data_ : csc_matrix, shape (max_power, 60 * max_duration_profile)
        Array containing the probabilistic power-profile.

    max_power_ : float,
        Correspond of the maximum power in the data.
    """

    def __init__(self, max_duration_profile=300, cyclist_weight=60.):
        self.max_duration_profile = max_duration_profile
        self.cyclist_weight = cyclist_weight

    def _validate_ride_ppp(self, ride_ppp):
        """Method to check the consistency of the ride probabilistic
        power-profile list.

        Parameters
        ----------
        ride_ppp : list of RideProbabilisticPowerProfile
            Normally a list of RideProbabilisticPowerProfile.

        Returns
        -------
        ride_ppp : list of RideProbabilisticPowerProfile
            Return the validated list of RideProbabilisticPowerProfile.

        """
        # Check that this is a list
        if isinstance(ride_ppp, list):
            # Check that each element are from the class
            # RideProbabilisticPowerProfile
            for rppp in ride_ppp:
                if not isinstance(rppp, RideProbabilisticPowerProfile):
                    raise ValueError('The object in the list need to be from'
                                     ' the type RideProbabilisticPowerProfile')
                # We need to check that each ride has been fitted
                if not hasattr(rppp, 'data_'):
                    raise ValueError('One of the ride never has been fitted.'
                                     ' Fit before to compute the record rpp.')
            # Create a list of all the max duration to check that they are
            # all equal
            max_duration = np.array(
                [rppp.max_duration_profile for rpp in ride_ppp])
            if self.max_duration_profile is None:
                raise ValueError('You need to specify the maximum duration for'
                                 ' the profile equal to the maximum duration'
                                 ' of each ride.')
            if not np.all(max_duration == self.max_duration_profile):
                raise ValueError('The maximum duration of the profile should'
                                 ' be the same for all the data.')
            return ride_ppp
        else:
            raise ValueError('The ride power-profile should be given as'
                             ' a list.')

    def fit(self, ride_ppp, date_profile=None):
        """Build the record power-profile from a list of ride power-profile.

        Parameters
        ----------
        ride_rpp: list of RideProbabilisticPowerProfile
            The list from which we will compute the record power-profile.

        date_profile: tuple of date, (start, finish)
            The starting and finishing date for which we have to compute the
            record power-profile.

        Returns
        -------
        self : object
            Returns self.

        """
        # Check that the ride power-profile list is ok
        ride_ppp = self._validate_ride_ppp(ride_ppp)

        # Check that the date provided are correct
        if date_profile is not None:
            date_profile = check_tuple_date(date_profile)

        # In the case that we want to compute the record power-profile from a
        # subset using the date, we need find the ride which are interesting
        if date_profile is not None:
            # Build a list of the date for each ride
            date_list = np.array([rppp.date_profile_ for rppp in ride_ppp])

            # Find the index which are inside the range of date
            idx_ride = np.flatnonzero(
                np.bitwise_and(date_list >= date_profile[0], date_list <=
                               date_profile[1]))

            # Store the date range
            self.date_profile_ = date_profile
        else:
            # Keep all the rides
            idx_ride = range(len(ride_ppp))

            # Store the date range
            date_list = np.array([rppp.date_profile_ for rppp in ride_ppp])
            self.date_profile_ = (np.ndarray.min(date_list),
                                  np.ndarray.max(date_list))

        # Find the maximum power for all the different rides
        self.max_power_ = max([ride_ppp[i].data_.shape[0] for i in idx_ride])

        # Resize each power-profile such that they can be sum-up
        # create a generator to modify the data
        data = (ride_ppp[i].data_.A for i in idx_ride)
        # pad the data
        data = map(lambda x: np.pad(x, ((0, self.max_power_ - x.shape[0]),
                                        (0, 0)), mode='constant'), data)
        # sum everything
        self.data_ = sparse.csc_matrix(reduce(lambda x, y: x + y, data))

        return self
