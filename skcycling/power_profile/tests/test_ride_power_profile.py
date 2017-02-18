"""Test the Ride Power Profile class. """

import os
import numpy as np

from datetime import date

from numpy.testing import assert_array_equal
from numpy.testing import assert_equal
from numpy.testing import assert_raises

from skcycling.power_profile import RidePowerProfile
from skcycling.power_profile.ride_power_profile import _rpp_parallel
from skcycling.utils import load_power_from_fit


def test_ridepp_fit():
    """ Test the routine to compute the ride power-profile from a fit file. """
    # Create the path to read the npy file
    currdir = os.path.dirname(os.path.abspath(__file__))
    filename = os.path.join(currdir, 'data', 'fit_files',
                            '2014-05-07-14-26-22.fit')

    my_ride_rpp = RidePowerProfile(max_duration_profile=1)
    my_ride_rpp.fit(filename)

    data = np.load(os.path.join(currdir, 'data', 'profile_fit_1.npy'))
    assert_array_equal(my_ride_rpp.data_, data)
    assert_equal(my_ride_rpp.data_norm_, None)
    assert_equal(my_ride_rpp.cyclist_weight, 60)
    assert_equal(my_ride_rpp.max_duration_profile, 1)
    assert_equal(my_ride_rpp.date_profile_, date(2014, 05, 07))
    assert_equal(my_ride_rpp.filename_, filename)


def test_ridepp_fit_w_weight():
    """ Test the routine to compute the ride power-profile from a fit file with
    a given weight. """
    # Create the path to read the npy file
    currdir = os.path.dirname(os.path.abspath(__file__))
    filename = os.path.join(currdir, 'data', 'fit_files',
                            '2014-05-07-14-26-22.fit')

    my_ride_rpp = RidePowerProfile(max_duration_profile=1, cyclist_weight=60.)
    my_ride_rpp.fit(filename)

    data = np.load(os.path.join(currdir, 'data', 'profile_fit_1.npy'))
    assert_array_equal(my_ride_rpp.data_, data)
    data = np.load(os.path.join(currdir, 'data', 'profile_fit_weight_1.npy'))
    assert_array_equal(my_ride_rpp.data_norm_, data)
    assert_equal(my_ride_rpp.cyclist_weight, 60.)
    assert_equal(my_ride_rpp.max_duration_profile, 1)
    assert_equal(my_ride_rpp.date_profile_, date(2014, 05, 07))
    assert_equal(my_ride_rpp.filename_, filename)


def test_rpp_parallel():
    """ Test the best mean function. """
    # Create the path to read the npy file
    currdir = os.path.dirname(os.path.abspath(__file__))
    filename = os.path.join(currdir, 'data',
                            'fit_files', '2014-05-07-14-26-22.fit')

    # Load the fit file
    power_rec = load_power_from_fit(filename)

    # Compute the rpp for one specific value
    val = _rpp_parallel(power_rec, 1)
    assert_equal(val, 500.)


def test_ridepp_fit_no_duration():
    """ Test either if an error is raised if no duration is given. """
    currdir = os.path.dirname(os.path.abspath(__file__))
    filename = os.path.join(currdir, 'data', 'fit_files',
                            '2014-05-07-14-26-22.fit')

    my_ride_rpp = RidePowerProfile()
    assert_raises(ValueError, my_ride_rpp.fit, filename)
