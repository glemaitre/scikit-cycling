"""Methods to handle input/output files."""

import warnings
import numpy as np

from fitparse import FitFile

from .checker import check_filename_fit

from ..restoration.denoise import outliers_rejection


def load_power_from_fit(filename):
    """Method to open the power data from FIT file into a numpy array.

    Parameters
    ----------
    filename : str,
        Path to the FIT file.

    Returns
    -------
    power_rec : ndarray, shape (n_samples)
        Power records of the ride.

    date_rec : date
        Date associated with the ride.

    """
    # Check that the filename is existing
    filename = check_filename_fit(filename)

    # Create an object to open the activity
    activity = FitFile(filename)
    activity.parse()

    # Get only the power records
    records = list(activity.get_messages(name='record'))

    # Through an error if there is no data
    if not records:
        raise ValueError('There is no data to treat in that file.')

    # Extract the date from the first record
    # It should be more reliable than the device information
    date_rec = records[0].get_value('timestamp').date()

    # Append the different values inside a list which will be later
    # converted to numpy array
    power_rec = np.zeros((len(records), ))
    # Go through each record
    # In order to send multiple warnings
    warnings.simplefilter('always', UserWarning)
    warn_sample = 0
    for idx_rec, rec in enumerate(records):
        # Extract only the value regarding the power
        p = rec.get_value('power')
        if p is not None:
            power_rec[idx_rec] = float(p)
        else:
            # We put the value to 0 since that it will not influence
            # the computation of the RPP
            power_rec[idx_rec] = 0.
            # We keep track of the number of inconsitent data
            warn_sample += 1

    # Remove artefact if there is any
    power_rec = outliers_rejection(power_rec)

    # Through a warning if there is no power data found
    if len(records) == warn_sample:
        warnings.warn('This file does not contain any power data.'
                      ' Be aware.')

    return power_rec, date_rec
