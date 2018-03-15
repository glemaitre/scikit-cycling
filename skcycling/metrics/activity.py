""" Metrics to asses the performance of a cycling ride.

Functions named as ``*_score`` return a scalar value to maximize: the higher
the better.

Function named as ``*_error`` or ``*_loss`` return a scalar value to minimize:
the lower the better.
"""

# Authors: Guillaume Lemaitre <g.lemaitre58@gmail.com>
#          Cedric Lemaitre
# License: BSD 3 clause

from __future__ import division

import numpy as np

TS_SCALE_GRAPPE = dict([('I1', 2.), ('I2', 2.5), ('I3', 3.),
                        ('I4', 3.5), ('I5', 4.5), ('I6', 7.),
                        ('I7', 11.)])

ESIE_SCALE_GRAPPE = dict([('I1', (.3, .5)), ('I2', (.5, .6)),
                          ('I3', (.6, .75)), ('I4', (.75, .85)),
                          ('I5', (.85, 1.)), ('I6', (1., 1.80)),
                          ('I7', (1.8, 3.))])


def normalized_power_score(activity_power, mpa, window_width=30):
    """Compute the normalized power for a given ride.

    Parameters
    ----------
    activity_power : Series
        A Series containing the power data from an activity.

    mpa : float
        Maxixum Anaerobic Power.

    window_width : int, optional
        The width of the window used to smooth the power data before to compute
        the normalized power. The default width is 30 samples.

    Returns
    -------
    score : float
        Normalized power score.

    """

    smooth_activity = (activity_power.rolling(window_width, center=True)
                                     .mean().dropna())
    # removing value < I1-ESIE, i.e. 30 % MPA
    smooth_activity = smooth_activity[
        smooth_activity > ESIE_SCALE_GRAPPE['I1'][0] * mpa]

    return (smooth_activity ** 4).mean() ** (1 / 4)


def intensity_factor_ftp_score(activity_power, ftp):
    """Compute the intensity factor using the FTP.

    Parameters
    ----------
    activity_power : Series
        A Series containing the power data from an activity.

    ftp : float
        Functional Threshold Power.

    Returns
    -------
    score: float
        Intensity factor score.

    """

    return normalized_power_score(activity_power, ftp2mpa(ftp)) / ftp


def intensity_factor_mpa_score(activity_power, mpa):
    """Compute the intensity factor using the MPA.

    Parameters
    ----------
    activity_power : Series
        A Series containing the power data from an activity.

    mpa : float
        Maximum Anaerobic Power.

    Returns
    -------
    score: float
        Intensity factor.

    """

    return intensity_factor_ftp_score(activity_power, mpa2ftp(mpa))


def training_stress_ftp_score(activity_power, ftp):
    """Compute the training stress score using the FTP.

    Parameters
    ----------
    activity_power : Series
        A Series containing the power data from an activity.

    ftp : float
        Functional Threshold Power.

    Returns
    -------
    score: float
        Training stress score.

    """
    activity_power = activity_power.resample('1S').mean()
    if_score = intensity_factor_ftp_score(activity_power, ftp)
    return (activity_power.size * if_score ** 2) / 3600 * 100


def training_stress_mpa_score(activity_power, mpa):
    """Compute the training stress score.

    Parameters
    ----------
    activity_power : Series
        A Series containing the power data from an activity.

    mpa : float
        Maximum Anaerobic Power.

    Returns
    -------
    score: float
        Training stress score.

    """
    return training_stress_ftp_score(activity_power, mpa2ftp(mpa))


def mpa2ftp(mpa):
    """Convert the MPA to FTP.

    Parameters
    ----------
    mpa : float
        Maximum Anaerobic Power.

    Return:
    -------
    ftp : float
        Functioning Threhold Power.

    """
    return 0.76 * mpa


def ftp2mpa(ftp):
    """Convert the MPA to FTP.

    Parameters
    ----------
    ftp : float
        Functioning Threhold Power.

    Return:
    -------
    mpa : float
        Maximum Anaerobic Power.

    """
    return ftp / 0.76


def training_stress_mpa_grappe_score(activity_power, mpa):
    """Compute the training stress score using the MPA.

    Parameters
    ----------
    activity_power : Series
        A Series containing the power data from an activity.

    mpa : float
        Maximum Anaerobic Power.

    Returns
    -------
    tss_score: float
        Training stress score.

    """
    tss_score = 0.
    activity_power = activity_power.resample('1S').mean()
    for key in TS_SCALE_GRAPPE.keys():
        power_samples = activity_power[
            np.bitwise_and(activity_power >= ESIE_SCALE_GRAPPE[key][0] * mpa,
                           activity_power < ESIE_SCALE_GRAPPE[key][1] * mpa)]
        tss_score += power_samples.size / 60 * TS_SCALE_GRAPPE[key]
    return tss_score


def training_stress_ftp_grappe_score(activity_power, ftp):
    """Compute the training stress score using the FTP.

    Parameters
    ----------
    activity_power : Series
        A Series containing the power data from an activity.

    ftp : float
        Functional Threshold Power.

    Returns
    -------
    score: float
        Training stress score.

    """
    return training_stress_mpa_grappe_score(activity_power, ftp2mpa(ftp))
