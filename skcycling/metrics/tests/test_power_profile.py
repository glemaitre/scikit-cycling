"""Test the metrics linked to the power profile."""

import os
import numpy as np

from numpy.testing import assert_raises
from numpy.testing import assert_almost_equal
from numpy.testing import assert_warns

from skcycling.power_profile import RidePowerProfile
from skcycling.metrics import aerobic_meta_model

DECIMAL_PRECISION = 3


def test_amm_wrong_profile():
    """ Test either if an error is raised when the class of the profile is
    incorrect. """
    assert_raises(ValueError, aerobic_meta_model, 1)


def test_amm_wrong_method():
    """ Test either if an error is raised when the method for fitting is
    not implemented. """
    # Create the path to read the npy file
    currdir = os.path.dirname(os.path.abspath(__file__))
    filename = os.path.join(currdir, 'data', 'ride_profile.p')

    my_ride_rpp = RidePowerProfile.load_from_pickles(filename)
    print(type(my_ride_rpp))
    # Apply an unknwon method
    assert_raises(NotImplementedError, aerobic_meta_model,
                  my_ride_rpp, method='None')


def test_amm_no_normalized():
    """ Test either if an error is raised when no normalized data
    where fitted. """
    # Create the path to read the npy file
    currdir = os.path.dirname(os.path.abspath(__file__))
    filename = os.path.join(currdir, 'data', 'ride_profile.p')

    my_ride_rpp = RidePowerProfile.load_from_pickles(filename)

    # Override the data such that no normalized data exist
    my_ride_rpp.data_norm_ = None
    my_ride_rpp.cyclist_weight = None

    # Compute for non normalized data
    assert_raises(ValueError, aerobic_meta_model, my_ride_rpp, normalized=True)


def test_amm_default_params():
    """ Test the aerobic modelling routine with default parameters. """
    # Create the path to read the npy file
    currdir = os.path.dirname(os.path.abspath(__file__))
    filename = os.path.join(currdir, 'data', 'ride_profile.p')

    my_ride_rpp = RidePowerProfile.load_from_pickles(filename)

    # Load a proper profile in order to make the experiments possible
    currdir = os.path.dirname(os.path.abspath(__file__))
    filename = os.path.join(currdir, 'data', 'profile.npy')
    data = np.load(filename)

    # Replace the value in the profile
    my_ride_rpp.data_ = data[:len(my_ride_rpp.data_)]
    my_ride_rpp.data_norm_ = my_ride_rpp.data_ / my_ride_rpp.cyclist_weight

    (pma, t_pma, aei,
     fit_info_pma_fitting,
     fit_info_aei_fitting) = aerobic_meta_model(my_ride_rpp)

    # Check the different value
    assert_almost_equal(pma, 453.37229888268155,
                        decimal=DECIMAL_PRECISION)
    assert_almost_equal(t_pma, 3.,
                        decimal=DECIMAL_PRECISION)
    assert_almost_equal(aei, -11.487957337374663,
                        decimal=DECIMAL_PRECISION)

def test_aerobic_meta_model_ts():
    """ Test the aerobic metabolism model using default parameters with
    a given ts. """
    # Create the path to read the npy file
    currdir = os.path.dirname(os.path.abspath(__file__))
    filename = os.path.join(currdir, 'data', 'ride_profile.p')

    my_ride_rpp = RidePowerProfile.load_from_pickles(filename)

    # Load a proper profile in order to make the experiments possible
    currdir = os.path.dirname(os.path.abspath(__file__))
    filename = os.path.join(currdir, 'data', 'profile.npy')
    data = np.load(filename)

    # Replace the value in the profile
    my_ride_rpp.data_ = data[:len(my_ride_rpp.data_)]
    my_ride_rpp.data_norm_ = my_ride_rpp.data_ / my_ride_rpp.cyclist_weight

    ts_reg = np.array([3., 4., 5., 6., 7., 10, 20, 30, 45, 60, 120, 180, 240])
    pma, t_pma, aei, fit_info_pma_fitting, fit_info_aei_fitting = aerobic_meta_model(my_ride_rpp, ts=ts_reg)

    # Check the different value
    assert_almost_equal(pma, 453.37229888268155,
                        decimal=DECIMAL_PRECISION)
    assert_almost_equal(t_pma, 3.,
                        decimal=DECIMAL_PRECISION)
    assert_almost_equal(aei, -11.497339815062894,
                        decimal=DECIMAL_PRECISION)


def test_aerobic_meta_model_weight():
    """ Test the aerobic metabolism model using default parameters with
    cyclist weight information. """
    # Create the path to read the npy file
    currdir = os.path.dirname(os.path.abspath(__file__))
    filename = os.path.join(currdir, 'data', 'ride_profile.p')

    my_ride_rpp = RidePowerProfile.load_from_pickles(filename)

    # Load a proper profile in order to make the experiments possible
    currdir = os.path.dirname(os.path.abspath(__file__))
    filename = os.path.join(currdir, 'data', 'profile.npy')
    data = np.load(filename)

    # Replace the value in the profile
    my_ride_rpp.data_ = data[:len(my_ride_rpp.data_)]
    my_ride_rpp.data_norm_ = my_ride_rpp.data_ / my_ride_rpp.cyclist_weight

    pma, t_pma, aei, fit_info_pma_fitting, fit_info_aei_fitting = aerobic_meta_model(my_ride_rpp, normalized=True)

    # Check the different value
    assert_almost_equal(pma, 7.5562049813780261,
                        decimal=DECIMAL_PRECISION)
    assert_almost_equal(t_pma, 3.,
                        decimal=DECIMAL_PRECISION)
    assert_almost_equal(aei, -11.487957337374663,
                        decimal=DECIMAL_PRECISION)


def test_aerobic_meta_model_lm():
    """ Test the aerobic metabolism model using default parameters and
    Levevenberg Marquardt. """
    # Create the path to read the npy file
    currdir = os.path.dirname(os.path.abspath(__file__))
    filename = os.path.join(currdir, 'data', 'ride_profile.p')

    my_ride_rpp = RidePowerProfile.load_from_pickles(filename)

    # Load a proper profile in order to make the experiments possible
    currdir = os.path.dirname(os.path.abspath(__file__))
    filename = os.path.join(currdir, 'data', 'profile.npy')
    data = np.load(filename)

    # Replace the value in the profile
    my_ride_rpp.data_ = data[:len(my_ride_rpp.data_)]
    my_ride_rpp.data_norm_ = my_ride_rpp.data_ / my_ride_rpp.cyclist_weight

    pma, t_pma, aei, fit_info_pma_fitting, fit_info_aei_fitting = aerobic_meta_model(my_ride_rpp, method='lm')

    # Check the different value
    assert_almost_equal(pma, 453.37229888268155,
                        decimal=DECIMAL_PRECISION)
    assert_almost_equal(t_pma, 3.,
                        decimal=DECIMAL_PRECISION)
    assert_almost_equal(aei, -11.487957337374663,
                        decimal=DECIMAL_PRECISION)


def test_aerobic_meta_model_no_conf():
    """ Test either if an error is raised when there is no data in the
    confidence level. """
    # Create the path to read the npy file
    currdir = os.path.dirname(os.path.abspath(__file__))
    filename = os.path.join(currdir, 'data', 'ride_profile.p')

    my_ride_rpp = RidePowerProfile.load_from_pickles(filename)
    assert_raises(ValueError, aerobic_meta_model, my_ride_rpp)


def test_amm_cropping_ts():
    """ Test either if a warning is raised when ts is cropped. """
        # Create the path to read the npy file
    currdir = os.path.dirname(os.path.abspath(__file__))
    filename = os.path.join(currdir, 'data', 'ride_profile.p')

    my_ride_rpp = RidePowerProfile.load_from_pickles(filename)

    # Load a proper profile in order to make the experiments possible
    currdir = os.path.dirname(os.path.abspath(__file__))
    filename = os.path.join(currdir, 'data', 'profile.npy')
    data = np.load(filename)

    # Replace the value in the profile
    my_ride_rpp.data_ = data[:len(my_ride_rpp.data_)]
    my_ride_rpp.data_norm_ = my_ride_rpp.data_ / my_ride_rpp.cyclist_weight

    ts_reg = np.array([3., 4., 5., 6., 7., 10, 20,
                       30, 45, 60, 120, 180, 240, 300])
    assert_warns(UserWarning, aerobic_meta_model, my_ride_rpp, ts=ts_reg)
    pma, t_pma, aei, fit_info_pma_fitting, fit_info_aei_fitting = aerobic_meta_model(my_ride_rpp, ts=ts_reg)

    # Check the different value
    assert_almost_equal(pma, 453.37229888268155,
                        decimal=DECIMAL_PRECISION)
    assert_almost_equal(t_pma, 3.,
                        decimal=DECIMAL_PRECISION)
    assert_almost_equal(aei, -11.497339815062894,
                        decimal=DECIMAL_PRECISION)
