"""Extraction of information based on the power-profile."""

# Authors: Guillaume Lemaitre <g.lemaitre58@gmail.com>
#          Cedric Lemaitre
# License: BSD 3 clause

from datetime import time, timedelta
from numbers import Integral

import pandas as pd
from joblib import Parallel, delayed

from ._power_profile import max_mean_power_interval


def _int2time(secs):
    """Convert secs into a time object.

    Parameters
    ----------
    secs : int
        Seconds.

    Returns
    -------
    time : time instance
        Time instance.

    """
    mins, secs = divmod(secs, 60)
    hours, mins = divmod(mins, 60)
    return time(hours, mins, secs)


def _time2int(dt):
    """Convert time object into secs.

    Parameters
    ----------
    dt : time instance
        Time instance.

    Returns
    -------
    secs : int
        Integer representing seconds.

    """
    return dt.hour * 3600 + dt.minute * 60 + dt.second


def activity_power_profile(activity, max_duration=None, n_jobs=1):
    """Compute the power profile for an activity.

    Parameters
    ----------
    activity : DataFrame
        A pandas DataFrame with a ``'power'`` column and the indices are the
        information about time. The activity can be read with
        :func:`skcycling.io.bikeread`.

    max_duration : int or time instance, optional
        The maximum duration for which the power-profile should be computed. By
        default, it will be computed for the duration of the activity. An
        integer represents seconds.

    n_jobs : int, (default=1)
        The number of workers to use.

    Returns
    -------
    power_profile : Series
        A pandas Series containing the power-profile.

    """
    if max_duration is None:
        max_duration = _int2time(activity.shape[0])
    elif isinstance(max_duration, Integral):
        max_duration = _int2time(max_duration)

    power_profile = Parallel(n_jobs=n_jobs)(
        delayed(max_mean_power_interval)(activity['power'].values, duration)
        for duration in range(1, _time2int(max_duration)))

    return pd.Series(
        power_profile,
        index=pd.timedelta_range("00:00:01",
                                 timedelta(seconds=_time2int(max_duration)-1),
                                 freq='s'),
        name=pd.Timestamp(activity.index[0].date()))
