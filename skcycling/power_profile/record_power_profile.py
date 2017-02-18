"""Record Power-Profile class."""
from __future__ import division

import numpy as np

from .base_power_profile import BasePowerProfile
from .ride_power_profile import RidePowerProfile

from ..utils.checker import check_tuple_date


def maximal_mean_power(ride_pp):
    """Function to compute the maximal mean power for different time.

    Parameters
    ----------
    ride_pp : list of RidePowerProfile
        The list from the power-profile to consider.

    Returns
    -------
    mmp : ndarray, shape (max_duration_profile_, )
        The Maximal Mean Power for the different time
    """
    # Check that ride_pp is a list of RidePowerProfile
    if isinstance(ride_pp, list):
        # Check that each ride is the correct type
        for pp in ride_pp:
            if not isinstance(pp, RidePowerProfile):
                raise ValueError('The objects in the list need to be of'
                                 ' type RidePowerProfile.')
    else:
        raise ValueError('A list of RidePowerProfile needs to be passed'
                         ' as argument.')

    # From the list of RidePowerProfile, we need to extract the power
    # information
    profile = np.array([pp.data_ for pp in ride_pp])

    return np.max(profile, axis=0)


class RecordPowerProfile(BasePowerProfile):
    """Record power-profile.

    Parameters
    ----------
    max_duration_profile : int or None
        Integer representing the maximum duration in minutes to build the
        record power-profile model. It can be infered if the data are loaded
        from a npy file.

    cyclist_weight : float, default None
        Float in order to normalise the record power-profile depending
        of its weight. By default this is None in order to avoid
        using the data from normalized rpp without this data.

    Attributes
    ----------
    data_ : array-like, shape (60 * max_duration_profile_, )
        Array in which the record power-profile is stored.
        The units used is the second.

    data_norm_ : array-like, shape (60 * max_duration_rpp, )
        Array in which the weight-normalized record power-profile
        is stored. The units used is the seconds.

    max_duration_data_ : int
        The maximum duration of the record power-profile.

    cyclist_weight_ : float
        Cyclist weight.

    date_profile_ : tuple of date, shape (start, finish)
        Starting and finishing time to compute the record power-profile.

    """

    def __init__(self, max_duration_profile=300,
                 cyclist_weight=60.):
        # Call the constructor of the parent class
        super(RecordPowerProfile, self).__init__(
            max_duration_profile=max_duration_profile,
            cyclist_weight=cyclist_weight)

    def __str__(self):
        if hasattr(self, 'data_'):
            info = ("\n date: {}\n weight: {}\n duration profile: "
                    "{}\n data: {}".format(self.date_profile_,
                                           self.cyclist_weight,
                                           self.max_duration_profile,
                                           self.data_))
        else:
            info = ("\n weight: {}\n duration profile: {}".format(
                self.cyclist_weight,
                self.max_duration_profile))

        return info


    def _validate_ride_pp(self, ride_pp):
        """Method to check the consistency of the ride power-profile list.

        Parameters
        ----------
        ride_pp : list of RidePowerProfile
            Normally a list of RidePowerProfile.

        Returns
        -------
        ride_pp : list of RidePowerProfile
            Return the validated list of RidePowerProfile.

        """
        # Check that this is a list
        if isinstance(ride_pp, list):
            # Check that each element are from the class RidePowerProfile
            for rpp in ride_pp:
                if not isinstance(rpp, RidePowerProfile):
                    raise ValueError('The object in the list need to be from'
                                     ' the type RidePowerProfile')
                # We need to check that each ride has been fitted
                if (getattr(rpp, 'data_', None) is None or
                        rpp.max_duration_profile is None):
                    raise ValueError('One of the ride never has been fitted.'
                                     ' Fit before to compute the record rpp.')
            # Create a list of all the max duration to check that they are
            # all equal
            max_duration = np.array([rpp.max_duration_profile
                                     for rpp in ride_pp])
            if self.max_duration_profile is None:
                raise ValueError('You need to specify the maximum duration for'
                                 ' the profile equal to the maximum duration'
                                 ' of each ride.')
            if not np.all(max_duration == self.max_duration_profile):
                raise ValueError('The maximum duration of the profile should'
                                 ' be the same for all the data.')
            return ride_pp
        else:
            raise ValueError('The ride power-profile should be given as'
                             ' a list.')

    def fit(self, ride_pp, date_profile=None):
        """Build the record power-profile from a list of ride power-profile.

        Parameters
        ----------
        ride_rpp: list of RidePowerProfile
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
        ride_pp = self._validate_ride_pp(ride_pp)

        # Check that the date provided are correct
        if date_profile is not None:
            date_profile = check_tuple_date(date_profile)

        # In the case that we want to compute the record power-profile from a
        # subset using the date, we need find the ride which are interesting
        if date_profile is not None:
            # Build a list of the date for each ride
            date_list = np.array([rpp.date_profile_ for rpp in ride_pp])

            # Find the index which are inside the range of date
            idx_ride = np.flatnonzero(np.bitwise_and(
                date_list >= date_profile[0],
                date_list <= date_profile[1]))

            # Compute the record for these files only
            self.data_ = maximal_mean_power([ride_pp[i]
                                             for i in idx_ride])

            # Store the date range
            self.date_profile_ = date_profile
        else:
            # Compute the record for all the files
            self.data_ = maximal_mean_power(ride_pp)

            # Store the date range
            date_list = np.array([rpp.date_profile_ for rpp in ride_pp])
            self.date_profile_ = (np.ndarray.min(date_list),
                                  np.ndarray.max(date_list))

        # Compute the normalized rpp if we should
        if self.cyclist_weight is not None:
            self.data_norm_ = self.data_ / self.cyclist_weight
        else:
            self.data_norm_ = None

        return self
