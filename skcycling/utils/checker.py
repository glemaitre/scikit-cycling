"""Helper function to check data conformity."""

import os
import numpy as np

from datetime import date


def check_X(X):
    """Private function helper to check if X is of proper size

    Parameters
    ----------
    X : ndarray, shape (data, )
        Array to check the consistency.

    Returns
    -------
    X : ndarray, shape (data, )
        Array which is consistent.

    """

    # Check that X is a numpy vector
    if len(X.shape) is not 1:
        raise ValueError('The shape of X is not consistent.'
                         ' It should be a 1D numpy vector.')

    # Check that X is of type float
    if X.dtype != np.float64:
        X = X.astype(np.float64)

    return X


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
    if isinstance(filename, basestring):
        # Check that this is a fit file
        if filename.endswith('.fit'):
            # Check that the file is existing
            if os.path.isfile(filename):
                return filename
            else:
                raise ValueError('The file does not exist.')
        else:
            raise ValueError('The file is not an fit file.')
    else:
        raise ValueError('The filename needs to be a string.')


def check_tuple_date(date_tuple):
    """Function to check if the date tuple is consistent.

    Parameters
    ----------
    date_tuple : tuple of date, shape (start, finish)
        The tuple to check.

    Returns
    -------
    date_tuple : tuple of date, shape (start, finish)
        The validated tuple.

    """
    if isinstance(date_tuple, tuple) and len(date_tuple) == 2:
        # Check that the tuple is of write type
        if isinstance(date_tuple[0],
                      date) and isinstance(date_tuple[1],
                                               date):
            # Check that the first date is earlier than the second date
            if date_tuple[0] < date_tuple[1]:
                return date_tuple
            else:
                raise ValueError('The tuple need to be ordered'
                                 ' as (start, finish).')
        else:
            raise ValueError('Use the class `date` inside the tuple.')
    else:
        raise ValueError('The date are ordered a tuple of'
                         ' date (start, finsih).')
