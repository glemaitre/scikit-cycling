"""Methods to handle input/output files."""

# Authors: Guillaume Lemaitre <g.lemaitre58@gmail.com>
#          Cedric Lemaitre
# License: BSD 3 clause

import os
import pandas as pd
import six

from fitparse import FitFile


def check_filename_fit(filename):
    """Method to check if the filename corresponds to a fit file.

    Parameters
    ----------
    filename : str
        The fit file to check.

    Returns
    -------
    filename : str
        The checked filename.

    """

    # Check that filename is of string type
    if isinstance(filename, six.string_types):
        # Check that this is a fit file
        if filename.endswith('.fit'):
            # Check that the file is existing
            if os.path.isfile(filename):
                return filename
            else:
                raise ValueError('The file does not exist.')
        else:
            raise ValueError('The file is not a fit file.')
    else:
        raise ValueError('filename needs to be a string. Got {}'.format(
            type(filename)))


def load_power_from_fit(filename):
    """Method to open the power data from FIT file into a pandas dataframe.

    Parameters
    ----------
    filename : str,
        Path to the FIT file.

    Returns
    -------
    power_rec : ndarray, shape (n_samples)
        Power records of the ride.

    """
    filename = check_filename_fit(filename)
    activity = FitFile(filename)
    activity.parse()
    records = activity.get_messages(name='record')

    power, timestamp = zip(*[
        (rec.get_value('power'), rec.get_value('timestamp'))
        for rec in records])

    return pd.DataFrame({'power': power}, index=timestamp)
