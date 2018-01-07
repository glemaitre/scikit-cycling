"""Base classes for data management."""

import numpy as np
import pandas as pd

from .extraction import activity_power_profile
from .io import bikeread
from .utils import validate_filenames


class Rider(object):
    """User interface for a rider.

    User interface to easily add, remove, compute information related to power.

    Parameters
    ----------
    n_jobs : int, (default=1)
        The number of workers to use for the different processing.

    Attributes
    ----------
    power_profile_ : DataFrame
        DataFrame containing all information regarding the power-profile of a
        rider for each ride.

    """

    def __init__(self, n_jobs=1):
        self.n_jobs = n_jobs
        self.power_profile_ = None

    def add_activities(self, filenames):
        """Compute the power-profile for each activity and add it to the
        current power-profile.

        Parameters
        ----------
        filenames : str or list of str
            A string a list of string to the file to read. You can use
            wildcards to automatically check several files.

        Returns
        -------
        None

        Examples
        --------
        >>> from skcycling.datasets import load_fit
        >>> from skcycling.base import Rider
        >>> rider = Rider()
        >>> rider.add_activities(load_fit()[0])
        >>> rider.power_profile_.head() # doctest: +NORMALIZE_WHITESPACE
                  2014-05-07
        00:00:01  500.000000
        00:00:02  475.500000
        00:00:03  469.333333
        00:00:04  464.000000
        00:00:05  463.000000

        """
        filenames = validate_filenames(filenames)
        activities_pp = [activity_power_profile(bikeread(f),
                                                n_jobs=self.n_jobs)
                         for f in filenames]
        activities_pp = pd.concat(activities_pp, axis=1)

        if self.power_profile_ is not None:
            self.power_profile_ = pd.concat(
                [self.power_profile_, activities_pp], axis=1)
        else:
            self.power_profile_ = activities_pp

    def delete_activities(self, dates):
        """Delete the activities power-profile from some specific dates.

        Parameters
        ----------
        dates : list/tuple of datetime-like or str
            The dates of the activities to be removed. The format expected is:

            * datetime-like or str: a single activity will be deleted.
            * a list of datetime-like or str: each activity for which the date
              is contained in the list will be deleted.
            * a tuple of datetime-like or str ``(start_date, end_date)``: the
              activities for which the dates are included in the range will be
              deleted.

        Returns
        -------
        None

        Examples
        --------
        >>> from skcycling.datasets import load_rider
        >>> from skcycling import Rider
        >>> rider = Rider.from_csv(load_rider())
        >>> rider.delete_activities('07 May 2014')
        >>> print(rider) # doctest: +NORMALIZE_WHITESPACE
        RIDER INFORMATION:
         power-profile:
                  2014-05-11  2014-07-26
        00:00:01      717.00  750.000000
        00:00:02      717.00  741.000000
        00:00:03      590.00  731.666667
        00:00:04      552.25  719.500000
        00:00:05      552.60  712.200000

        """
        if isinstance(dates, tuple):
            if len(dates) != 2:
                raise ValueError("Wrong tuple format. Expecting a tuple of"
                                 " format (start_date, end_date). Got {!r}"
                                 " instead.".format(dates))
            mask_date = np.bitwise_and(self.power_profile_.columns >= dates[0],
                                       self.power_profile_.columns <= dates[1])
        elif isinstance(dates, list):
            mask_date = np.any([self.power_profile_.columns == d
                                for d in dates],
                               axis=0)
        else:
            mask_date = self.power_profile_.columns == dates

        mask_date = np.bitwise_not(mask_date)
        self.power_profile_ = self.power_profile_.loc[:, mask_date]

    @classmethod
    def from_csv(cls, filename, n_jobs=1):
        """Load rider information from a CSV file.

        Parameters
        ----------
        filename : str
            The path to the CSV file.

        n_jobs : int, (default=1)
            The number of workers to use for the different processing.

        Returns
        -------
        rider : skcycling.Rider
            The :class:`skcycling.Rider` instance.

        Examples
        --------
        >>> from skcycling.datasets import load_rider
        >>> from skcycling import Rider
        >>> rider = Rider.from_csv(load_rider())
        >>> print(rider) # doctest: +NORMALIZE_WHITESPACE
        RIDER INFORMATION:
         power-profile:
                  2014-05-07  2014-05-11  2014-07-26
        00:00:01  500.000000      717.00  750.000000
        00:00:02  475.500000      717.00  741.000000
        00:00:03  469.333333      590.00  731.666667
        00:00:04  464.000000      552.25  719.500000
        00:00:05  463.000000      552.60  712.200000

        """
        df = pd.read_csv(filename, index_col=0)
        df.columns = pd.to_datetime(df.columns)
        df.index = pd.to_timedelta(df.index)
        rider = cls(n_jobs=n_jobs)
        rider.power_profile_ = df
        return rider

    def to_csv(self, filename):
        """Drop the rider information into a CSV file.

        Parameters
        ----------
        filename : str
            The path to the CSV file.

        Returns
        -------
        None

        Examples
        --------
        >>> from skcycling.datasets import load_fit
        >>> from skcycling import Rider
        >>> rider = Rider(n_jobs=-1)
        >>> rider.add_activities(load_fit()[:1])
        >>> print(rider) # doctest: +NORMALIZE_WHITESPACE
        RIDER INFORMATION:
         power-profile:
                  2014-05-07
        00:00:01  500.000000
        00:00:02  475.500000
        00:00:03  469.333333
        00:00:04  464.000000
        00:00:05  463.000000

        """
        self.power_profile_.to_csv(filename)

    def __repr__(self):
        return 'RIDER INFORMATION:\n power-profile:\n {}'.format(
            self.power_profile_.head())
