"""Function to extract gradient information about different features."""

# Authors: Guillaume Lemaitre <g.lemaitre58@gmail.com>
#          Cedric Lemaitre
# License: BSD 3 clause

from __future__ import division

from ..exceptions import MissingDataError


def acceleration(activity, periods=5, append=True):
    """Compute the acceleration (i.e. speed gradient).

    Parameters
    ----------
    activity : DataFrame
        The activity containing speed information.

    periods : int, default=5
        Periods to shift to compute the elevation gradient.

    append : bool, optional
        Whether to append the acceleration to the original activity (default)
        or to only return the acceleration as a Series.

    Returns
    -------
    data : DataFrame or Series
        The original activity with an additional column containing the
        acceleration or a single Series containing the acceleration.

    """
    if 'speed' not in activity.columns:
        raise MissingDataError('To compute the acceleration, speed data are '
                               'required. Got {} fields.'
                               .format(activity.columns))

    acceleration = activity['speed'].diff(periods=periods)

    if append:
        activity['acceleration'] = acceleration
        return activity
    else:
        return acceleration


def gradient_elevation(activity, periods=5, append=True):
    """Compute the elevation gradient.

    Parameters
    ----------
    activity : DataFrame
        The activity containing elevation and distance information.

    periods : int, default=5
        Periods to shift to compute the elevation gradient.

    append : bool, optional
        Whether to append the elevation gradient to the original activity
        (default) or to only return the elevation gradient as a Series.

    Returns
    -------
    data : DataFrame or Series
        The original activity with an additional column containing the
        elevation gradient or a single Series containing the elevation
        gradient.

    """
    if not {'elevation', 'distance'}.issubset(activity.columns):
        raise MissingDataError('To compute the elevation gradient, elevation '
                               'and distance data are required. Got {} fields.'
                               .format(activity.columns))

    diff_elevation = activity['elevation'].diff(periods=periods)
    diff_distance = activity['distance'].diff(periods=periods)
    gradient_elevation = diff_elevation / diff_distance

    if append:
        activity['gradient-elevation'] = gradient_elevation
        return activity
    else:
        return gradient_elevation
