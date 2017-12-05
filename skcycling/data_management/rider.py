""" Rider class.

This module contains the class to manage the data linked to a single rider.
"""

import os
import warnings
import json
import logging

from copy import deepcopy
from datetime import date

import numpy as np
import joblib

from ..power_profile import BasePowerProfile
from ..power_profile import RidePowerProfile
from ..power_profile import RecordPowerProfile
from ..utils.checker import check_tuple_date


def _date_handler(obj):
        """Handle the datetime.date when dropping in a JSON file."""
        if hasattr(obj, 'isoformat'):
            return obj.isoformat()
        else:
            raise TypeError


class Rider(object):
    """ Rider class to aggregate all the different power tools.

    This class should be the main class used to handle all the data.
    In this object, the information about the power-profile of each
    ride will be stored. Furthermore, a record power-profile can
    be updated by providing some starting and ending date.

    Parameters
    ----------
    cyclist_weight : float,
        Float in order to normalise the record power-profile depending
        of its weight.

    max_duration_profile : int, optional (default=300)
        Integer representing the maximum duration in minutes to build the
        record power-profile model. It can be inferred if the data are loaded
        from a pickle file.

    rides_pp : list of RidePowerProfile or None, optional (default=None)
        Initialize the list of RidePowerProfile

    Attributes
    ----------
    rides_pp_ : list of RidePowerProfile
        The list of the ride power-profile linked to a rider.

    record_pp_ : RecordPowerProfile
        The record power-profile of the rider.
    """

    def __init__(self, cyclist_weight, max_duration_profile=300,
                 rides_pp=None):
        """ Constructor. """
        self.cyclist_weight = cyclist_weight
        self.max_duration_profile = max_duration_profile
        self.rides_pp_ = self._validate_rides_pp(rides_pp)
        self.record_pp_ = RecordPowerProfile(
            max_duration_profile=self.max_duration_profile,
            cyclist_weight=self.cyclist_weight)
        self.logger = logging.getLogger(__name__)

    def _validate_rides_pp(self, rides_pp):
        """ Method to check the consistency of the ride power-profile list.

        Parameters
        ----------
        rides_pp : list of RidePowerProfile
            Normally a list of RidePowerProfile.

        Returns
        -------
        rides_pp : list of RidePowerProfile
            Return the validated list of RidePowerProfile.
        """
        # Check that this is a list
        if rides_pp is None:
            return []
        if isinstance(rides_pp, list):
            # Check that each element are from the class RidePowerProfile
            for rpp in rides_pp:
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
            max_duration = np.array(
                [rpp.max_duration_profile for rpp in rides_pp])
            if not np.all(max_duration == self.max_duration_profile):
                raise ValueError('The maximum duration of the profile should'
                                 ' be the same for all the data.')
            return deepcopy(rides_pp)
        else:
            raise ValueError('The ride power-profile should be given as'
                             ' a list.')

    @staticmethod
    def load_from_pickles(filename):
        """Function to load an object through pickles.

        Parameters
        ----------
        filename : str
            Filename to the pickle file. The extension should be `.p`.

        Returns
        -------
        bpp : object
            Returns Rider.
        """
        # Load the pickle
        bpp = joblib.load(filename)

        return bpp

    def save_to_pickles(self, filename):
        """Function to save an object RecordPowerProfile through pickles.

        Parameters
        ----------
        filename : str
            Filename to the pickle file. The extension should be `.p`.

        Returns
        -------
        None
        """
        # We need to check that the directory where the file will be exist
        dir_pickle = os.path.dirname(filename)
        if not os.path.exists(dir_pickle):
            os.makedirs(dir_pickle)
        # Create the pickle file
        joblib.dump(self, filename)

        return None

    def save_to_json(self, filename):
        """Function to save the object into a JSON format.

        Parameters
        ----------
        filename : str
            Filename to store the JSON file.

        Returns
        -------
        None
        """
        storage_dir = os.path.dirname(filename)
        if not os.path.exists(storage_dir):
            os.makedirs(storage_dir)

        object_dict = {}
        for key, value in self.__dict__.items():
            if key == 'logger':
                pass
            elif isinstance(value, list):
                # check that the first element is not a base record
                if isinstance(value[0], BasePowerProfile):
                    record = []
                    for elt in value:
                        record.append(elt._convert_to_json())
                    object_dict[key] = record
                else:
                    object_dict[key] = value
            elif isinstance(value, BasePowerProfile):
                object_dict[key] = value._convert_to_json()
            else:
                object_dict[key] = value

        with open(filename, "w") as f:
            json.dump(object_dict, f, default=_date_handler)

        return None

    def add_rides(self, location, overwrite=False, n_jobs=1):
        """Read files from a path, fit it, and add it to the rider profile.

        A list of files or file can be read from the location given by
        the user.

        Parameters
        ---------
        location : str or list of str,
            Correspond to the file name or to the path where several files
            have to be processed.

        overwrite : bool, optional (default=False)
            Overwrite the current ride power-profile list.

        Returns
        -------
        self : object
            Returns self.
        """
        if isinstance(location, list):
            # only select fit file
            filenames = [name for name in location if name.endswith('.fit')]
        elif not os.path.exists(location):
            raise ValueError('The path is not existing.')
        else:
            # decide if it is a file or a directory
            if os.path.isdir(location):
                filenames = sorted([
                    os.path.join(os.path.abspath(location), name)
                    for name in os.listdir(location) if name.endswith('.fit')
                ])
            else:
                filenames = [os.path.abspath(location)]

        # extract the file names
        rides_rpp = []
        for filename in filenames:
            # self.logger.info('Process the file {}'.format(filename))
            rpp = RidePowerProfile(
                max_duration_profile=self.max_duration_profile,
                cyclist_weight=self.cyclist_weight,
                n_jobs=n_jobs)
            rpp.fit(filename)
            rides_rpp.append(rpp)

        # Check if we have to overwrite the list
        if overwrite:
            self.rides_pp_ = rides_rpp
        else:
            self.rides_pp_ += rides_rpp

        return self

    def delete_ride(self, date_ride):
        """ Function to delete a specific ride from the list.

        Parameters
        ----------
        date_ride : date
            The date of the ride to remove.

        Returns
        -------
        self : object
            Returns self.
        """
        # Check the consistency of the date
        if not isinstance(date_ride, date):
            raise ValueError('The date should be a date object.')

        # From the list of ride, get the date of the ride
        rpp_date_rides = [rpp.date_profile_ for rpp in self.rides_pp_]

        # Find if there is any date corresponding to the one specified by
        # the user
        b_not_found = True
        for rpp_idx, rpp_date in enumerate(rpp_date_rides):
            if rpp_date == date_ride:
                del self.rides_pp_[rpp_idx]
                b_not_found = False
        if b_not_found:
            warnings.warn('No rides have been removed. No matching dates.')

        return self

    def compute_record_pp(self, date_start_finish=None):
        """ Function to compute the record power profile.

        Parameters
        ----------
        date_start_finish : tuple of date, shape (start, finish)
            Starting and finishing date to consider to compute the date.
            If None, the full range will be considered.

        Returns
        -------
        self : object
            Returns self.
        """
        # Check the consistency of the date tuple
        if date_start_finish is not None:
            date_start_finish = check_tuple_date(date_start_finish)

        self.record_pp_.fit(self.rides_pp_, date_profile=date_start_finish)

        return self

    def __getstate__(self):
        # remove logger before pickling
        state = self.__dict__.copy()
        # remove the logger
        state.pop('logger', None)
        return state

    def __setstate__(self, state):
        # add logger to object while unpickling
        state['logger'] = logging.getLogger(__name__)
        self.__dict__.update(state)
