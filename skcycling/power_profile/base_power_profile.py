"""Basic class for power profile."""

import os
import json
from abc import ABCMeta, abstractmethod

import numpy as np
import joblib

from scipy.interpolate import interp1d


class BasePowerProfile(object):
    """Basic class for power profile.

    Warning: This class should not be used directly. Use the derive classes
    instead.
    """
    __metaclass__ = ABCMeta

    @abstractmethod
    def __init__(self, max_duration_profile=None, cyclist_weight=None):
        """Constructor."""
        self.max_duration_profile = max_duration_profile
        self.cyclist_weight = cyclist_weight

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
            Returns BasePowerProfile.

        """
        # Load the pickle
        bpp = joblib.load(filename)

        return bpp

    def save_to_pickles(self, filename):
        """Function to save an object through pickles.

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

    def _convert_to_json(self):
        """Private function which convert the object to store it in JSON."""
        object_dict = {}
        for key, value in self.__dict__.items():
            if key == 'logger':
                pass
            elif isinstance(value, np.ndarray):
                # convert the array to list for the moment
                object_dict[key] = value.tolist()
            else:
                object_dict[key] = value

        return object_dict

    @abstractmethod
    def fit(self):
        """Method to compute the power profile."""
        raise NotImplementedError

    def resampling_rpp(self, ts, method_interp='linear', normalized=False):
        """Resampling the record power-profile

        Parameters
        ----------
        ts : array-like, shape (n_sample, )
            An array containaining the time landmark to sample.

        method_interp : string, default 'linear'
            Name of the method to interpolate the data.
            Specifies the kind of interpolation as a string ('linear',
            'nearest', 'zero', 'slinear', 'quadratic, 'cubic'
            where 'slinear', 'quadratic' and 'cubic' refer to a spline
            interpolation of first, second or third order) or as an integer
            specifying the order of the spline interpolator to use.

        normalized : bool, default False
            Return a weight-normalized rpp if True.

        Returns
        -------
        data : array-like, shape (n_samples, )
            Returns a resampled record power-profile.

        """

        # Shall used the rpp or weight-normalized rpp
        if normalized is True:
            # Check that the cyclist weight was provided
            if self.cyclist_weight is not None:
                data = self.data_norm_
            else:
                raise ValueError('You cannot get a normalized rpp if the'
                                 ' cyclist weight never has been given.')
        else:
            data = self.data_

        t = np.linspace(0, self.max_duration_profile, data.size)
        f = interp1d(t, data, kind=method_interp)

        return f(ts)
