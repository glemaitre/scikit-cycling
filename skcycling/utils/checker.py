"""Helper function to check data conformity."""

import os
from datetime import date

import six


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
        if isinstance(date_tuple[0], date) and isinstance(date_tuple[1], date):
            # Check that the first date is earlier than the second date
            if date_tuple[0] < date_tuple[1]:
                return date_tuple
            else:
                raise ValueError('The tuple need to be ordered'
                                 ' as (start, finish).')
        else:
            raise ValueError('Use the class `date` inside the tuple.')
    else:
        raise ValueError('date_tuple is an ordered tuple of'
                         ' date (start, finish). Got {}'.format(
                             type(date_tuple)))
