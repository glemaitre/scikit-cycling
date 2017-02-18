""" Metrics to asses the power profile. """
from __future__ import division

import warnings
import numpy as np

from ..power_profile import BasePowerProfile

from ..utils.fit import log_linear_fitting
from ..utils.fit import log_linear_model


SAMPLING_WKO = np.array([0.016, 0.083, 0.5, 1, 3, 3.5, 4, 4.5, 5, 5.5,
                         6, 6.5, 7, 10, 20, 30, 45, 60, 120, 180, 240])


def aerobic_meta_model(profile, ts=None, normalized=False, method='lsq'):
    """Compute the aerobic metabolism model from the record power-profile.

    Parameters
    ----------
    profile : RidePowerProfile or RecordPowerProfile
        The profile that we want to use for computing the different
        statistics.

    ts : ndarray, shape (n_samples, ) or None
        Array containing the sample to take into account.
        If None, the sampling of the method of Pinot is applied, which is
        equivalent to the sampling from WKO+

    normalized : bool, default False
        Return a weight-normalized rpp if True.

    method : string, default 'lsq'
        Which type of tehcnic to use to make the fitting ('lsq', 'lm').

    Returns
    -------
    pma : float
        Maximum Aerobic Power.

    t_pma : int
        Time of the Maximum Aerobic Power in seconds.

    aei : float
        Aerobic Endurance Index.

    fit_info_pma_fitting : dict
        This is a dictionary with the information collected about the fitting
        related to the MAP. The attributes will be the following:

        - `slope`: slope of the linear fitting,
        - `intercept`: intercept of the linear fitting,
        - `std_err`: standard error of the fitting,
        - `coeff_det`: coefficient of determination.

    fit_info_aei_fitting : dict
        This is a dictionary with the information collected about the fitting
        related to the AEI. The attributes will be the following:

        - `slope`: slope of the linear fitting,
        - `intercept`: intercept of the linear fitting,
        - `std_err`: standard error of the fitting,
        - `coeff_det`: coefficient of determination.

    Notes
    -----
    The method implemented here follow the work presented in [1]_.

    References
    ----------
    .. [1] Pinot et al., "Determination of Maximal Aerobic Power
       on the Field in Cycling", Jounal of Science and Cycling, vol. 3(1),
       pp. 26-31, 2014.

    """
    # Check that the profile is inherating from the class BasePowerProfile
    if not issubclass(type(profile), BasePowerProfile):
        raise ValueError('The variable profile need to be of type'
                         ' RecordPowerProfile or RidePowerProfile.')

    # If ts is not provided we have to create a timeline
    if ts is None:
        # By default ts will be taken as in WKO+
        ts = SAMPLING_WKO.copy()

    if np.count_nonzero(ts > profile.max_duration_profile) > 0:
        # The values which are outside of the maximum duration need to
        # be removed
        ts = ts[np.nonzero(ts <= profile.max_duration_profile)]
        warnings.warn('Samples in `ts` have been removed since that they'
                      ' are not information inside the rpp.')

    # Compute the rpp
    rpp = profile.resampling_rpp(ts, normalized=normalized)

    # The zero values need to be avoided for the fitting
    # Keep the signal which is not zero
    ts = ts[np.nonzero(rpp)]
    rpp = rpp[np.nonzero(rpp)]

    # Find the MAP and the corresponding time
    # Only the time between 10 and 240 minutes is used for the regression
    ts_pma_reg = ts[np.nonzero(np.bitwise_and(ts >= 10, ts <= 240))]
    rpp_pma_reg = rpp[np.nonzero(np.bitwise_and(ts >= 10, ts <= 240))]

    # Apply the first log-linear fitting
    slope, intercept, std_err, coeff_det = log_linear_fitting(ts_pma_reg,
                                                              rpp_pma_reg,
                                                              method)

    # Store the value inside a dictionary
    fit_info_pma_fitting = {'slope': slope, 'intercept': intercept,
                            'std_err': std_err, 'coeff_det': coeff_det}

    # Find t_pma and pma
    # First record between 3 and 7 min in the confidence area
    ts_pma = ts[np.nonzero(np.bitwise_and(ts >= 3,
                                          ts <= 7))]
    rpp_pma = rpp[np.nonzero(np.bitwise_and(ts >= 3,
                                            ts <= 7))]

    # Compute the aerobic model found from the regression for
    # the range of interest
    aerobic_model = log_linear_model(ts_pma, slope, intercept)

    # Check the first value which entered in the confidence of 2 std
    if np.count_nonzero(np.abs(rpp_pma -
                               aerobic_model) < 2 * std_err) > 0:
        # Get the first value
        t_pma = ts_pma[np.flatnonzero(np.abs(rpp_pma -
                                             aerobic_model) < 2 *
                                      std_err)[0]]
        # Obtain the corresponding mpa
        pma = rpp_pma[np.flatnonzero(np.abs(rpp_pma -
                                            aerobic_model) < 2 *
                                     std_err)[0]]
    else:
        raise ValueError('There is no value entering in the confidence'
                         ' level between 3 and 7 minutes.')

    # Find aei
    # Get the rpp and ts between t_pma and 240 minutes
    ts_aei_reg = ts[np.nonzero(np.bitwise_and(ts >= t_pma,
                                              ts <= 240))]
    rpp_aei_reg = rpp[np.nonzero(np.bitwise_and(ts >= t_pma,
                                                ts <= 240))]
    # Express the rpp in term of percentage of PMA
    rpp_aei_reg = rpp_aei_reg / pma * 100

    # Apply a new regression with the aei value
    aei, intercept, std_err, coeff_det = log_linear_fitting(ts_aei_reg,
                                                            rpp_aei_reg,
                                                            method)

    # Store the value inside a dictionary
    fit_info_aei_fitting = {'slope': aei, 'intercept': intercept,
                            'std_err': std_err, 'coeff_det': coeff_det}

    return pma, t_pma, aei, fit_info_pma_fitting, fit_info_aei_fitting
