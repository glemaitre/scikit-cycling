"""Basic class for power profile."""

import os
import pickle

from abc import ABCMeta, abstractmethod

import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns


class BaseProbabilisticPowerProfile(object):
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
        bpp = pickle.load(open(filename, 'rb'))

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
        pickle.dump(self, open(filename, 'wb'))

        return None

    @abstractmethod
    def fit(self):
        """Method to compute the power profile."""
        raise NotImplementedError

    def plot_heatmap(self, normalized=False, crop=None):
        """Plot the heatmap corresponding to the current ride.

        Parameters
        ----------
        normalized : bool, optional (default=False)
            If True, plot the power-profile normalized by the weight.

        crop : tuple of int or None, optional (default=None)
            A tuple can be given (min_power, max_power) to plot only
            this region of interest.
        """
        if not hasattr(self, 'data_'):
            raise ValueError('Fit the data before to plot them.')

        data = self.data_.A
        if crop is not None:
            data[:crop[0], :] = 0.
            data[crop[1]:, :] = 0.

        ax = sns.heatmap(data, cmap='viridis', xticklabels=60,
                         yticklabels=100)
        if normalized:
            if self.cyclist_weight is None:
                raise ValueError('You need to provide a weight to get a'
                                 ' normalized plot.')
            else:
                ax.set_yticklabels(np.arange(0, self.data_.shape[0], 100,
                                             dtype=float) /
                                   self.cyclist_weight)
        ax.invert_yaxis()
        ax.set_xticklabels(np.arange(0, self.data_.shape[1],
                                     60, dtype=int) // 60)
        ax.set_xlabel('Time in minutes')
        ax.set_ylabel('Power in watts')
        plt.tight_layout()
