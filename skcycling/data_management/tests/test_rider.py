""" Test the rider class. """

import os
import numpy as np
import unittest

from numpy.testing import assert_equal
from numpy.testing import assert_raises
from numpy.testing import assert_array_equal
from numpy.testing import assert_allclose
from numpy.testing import assert_warns

from datetime import date

from skcycling.datasets import load_toy
from skcycling.data_management import Rider
from skcycling.power_profile import RidePowerProfile

_dummy = _dummy = unittest.TestCase('__init__')
try:
    assert_raises_regex = _dummy.assertRaisesRegex
except AttributeError:
    # Python 2.7
    assert_raises_regex = _dummy.assertRaisesRegexp


def test_rider_init_empty():
    rider = Rider(cyclist_weight=60., max_duration_profile=1)
    assert_equal(rider.cyclist_weight, 60.)
    assert_equal(rider.max_duration_profile, 1)
    assert_equal(rider.rides_pp_, [])


def test_rider_init_rides():
    filename_list = load_toy()
    ride_pp_list = [
        RidePowerProfile(max_duration_profile=1)
        for i in range(len(filename_list))
    ]

    for ride, filename in zip(ride_pp_list, filename_list):
        ride.fit(filename)

    rider = Rider(
        cyclist_weight=60., max_duration_profile=1, rides_pp=ride_pp_list)
    assert_equal(rider.cyclist_weight, 60.)
    assert_equal(rider.max_duration_profile, 1)


def test_rider_init_rides_wrong_type():
    assert_raises(
        ValueError,
        Rider,
        cyclist_weight=60.,
        max_duration_profile=1,
        rides_pp=1)


def test_rider_init_rides_wrong_type_in_list():
    filename_list = load_toy()
    ride_pp_list = [
        RidePowerProfile(max_duration_profile=1)
        for i in range(len(filename_list))
    ]

    for ride, filename in zip(ride_pp_list, filename_list):
        ride.fit(filename)
    # Add something random
    ride_pp_list.append(1)

    assert_raises_regex(
        ValueError,
        "The object in the list need"
        " to be from the type RidePowerProfile",
        Rider,
        cyclist_weight=60.,
        max_duration_profile=1,
        rides_pp=ride_pp_list)


def test_rider_init_rides_not_init():
    filename_list = load_toy()
    ride_pp_list = [
        RidePowerProfile(max_duration_profile=1)
        for i in range(len(filename_list))
    ]

    assert_raises_regex(
        ValueError,
        "One of the ride never has been fitted."
        " Fit before to compute the record rpp.",
        Rider,
        cyclist_weight=60.,
        max_duration_profile=1,
        rides_pp=ride_pp_list)


def test_rider_init_rides_wrong_duration():
    filename_list = load_toy()
    ride_pp_list = [
        RidePowerProfile(max_duration_profile=2)
        for i in range(len(filename_list))
    ]

    for ride, filename in zip(ride_pp_list, filename_list):
        ride.fit(filename)

    assert_raises_regex(
        ValueError,
        "The maximum duration of the profile"
        " should be the same for all the data",
        Rider,
        cyclist_weight=60.,
        max_duration_profile=1,
        rides_pp=ride_pp_list)


def test_rider_add_rides_path_not_exist():
    rider = Rider(cyclist_weight=60., max_duration_profile=1)

    assert_raises_regex(ValueError, "The path is not existing.",
                        rider.add_rides, 'random')


def test_rider_add_rides_path():
    rider = Rider(cyclist_weight=60., max_duration_profile=1)

    path_fit = load_toy(returned_type='path')
    rider.add_rides(path_fit)

    filename_list = load_toy()
    ride_pp_list = [
        RidePowerProfile(max_duration_profile=1)
        for i in range(len(filename_list))
    ]

    for ride, filename in zip(ride_pp_list, filename_list):
        ride.fit(filename)

    # Initialize the rider
    rider2 = Rider(
        cyclist_weight=60., max_duration_profile=1, rides_pp=ride_pp_list)

    # Check the consistency between the two riders
    for r1, r2 in zip(rider.rides_pp_, rider2.rides_pp_):
        assert_array_equal(r1.data_, r2.data_)


def test_rider_add_rides_path_overwrite():
    filename_list = load_toy()[:2]
    ride_pp_list = [
        RidePowerProfile(max_duration_profile=1)
        for i in range(len(filename_list))
    ]
    for ride, filename in zip(ride_pp_list, filename_list):
        ride.fit(filename)
    rider = Rider(
        cyclist_weight=60., max_duration_profile=1, rides_pp=ride_pp_list)
    assert_equal(len(rider.rides_pp_), 2)

    path_fit = load_toy(returned_type='path')
    rider.add_rides(path_fit, overwrite=True)
    assert_equal(len(rider.rides_pp_), 3)


def test_rider_add_ride_fit():
    filename_list = load_toy()

    # rider and add up all files
    rider = Rider(cyclist_weight=60., max_duration_profile=1)
    for filename in filename_list:
        rider.add_rides(filename)

    ride_pp_list = [
        RidePowerProfile(max_duration_profile=1)
        for i in range(len(filename_list))
    ]
    for ride, filename in zip(ride_pp_list, filename_list):
        ride.fit(filename)
    rider2 = Rider(
        cyclist_weight=60., max_duration_profile=1, rides_pp=ride_pp_list)
    assert_array_equal(rider.rides_pp_[0].data_, rider2.rides_pp_[0].data_)


def test_rider_delete_ride_no_date():
    filename = load_toy()[0]
    rider = Rider(cyclist_weight=60., max_duration_profile=1)
    rider.add_rides(filename)
    assert_raises_regex(ValueError, "The date should be a date object",
                        rider.delete_ride, '2014, 05, 11')


def test_rider_delete_ride():
    filename = load_toy()[0]
    rider = Rider(cyclist_weight=60., max_duration_profile=1)
    rider.add_rides(filename)
    rider.delete_ride(date(2014, 5, 11))
    assert_equal(len(rider.rides_pp_), 0)


def test_rider_delete_ride_warning_nothing():
    filename = load_toy()[0]
    rider = Rider(cyclist_weight=60., max_duration_profile=1)
    rider.add_rides(filename)
    rider.delete_ride(date(2014, 5, 11))
    assert_warns(UserWarning, rider.delete_ride, date(2014, 5, 11))
    assert_equal(len(rider.rides_pp_), 0)


def test_compute_record_pp():
    path_fit = load_toy(returned_type='path')
    rider = Rider(cyclist_weight=60., max_duration_profile=1)
    rider.add_rides(path_fit)
    rider.compute_record_pp()
    record_pp = np.array([
        0., 750., 741., 731.66666667, 719.5, 712.2, 705., 694.85714286, 684.,
        669.22222222, 652.9, 634.36363636, 616.75, 601.84615385, 584.92857143,
        562.6, 540.3125, 520.64705882, 502.44444444, 480.57894737, 467.05,
        460.28571429, 452.45454545, 447.60869565, 442.625, 436.6, 433.76923077,
        430.07407407, 424.96428571, 422.03448276, 419.16666667, 415.5483871,
        412.65625, 408.87878788, 405.70588235, 403.37142857, 400.16666667,
        397.91891892, 395.57894737, 393.56410256, 392.15, 388.90243902,
        385.80952381, 384.11627907, 382.29545455, 380.64444444, 378.93478261,
        376.89361702, 375.89583333, 375.18367347, 373.24, 371.50980392, 369.25,
        367.64150943, 366.51851852, 365.47272727, 364.17857143, 362.87719298,
        361.70689655, 361.27118644
    ])
    assert_allclose(rider.record_pp_.data_, record_pp)


def test_compute_record_pp_date():
    path_fit = load_toy(returned_type='path')
    rider = Rider(cyclist_weight=60., max_duration_profile=1)
    rider.add_rides(path_fit)
    rider.compute_record_pp(date_start_finish=(date(2014, 5, 7),
                                               date(2014, 5, 11)))
    record_pp = np.array([
        0., 717., 717., 590., 552.25, 552.6, 551.83333333, 550.42857143, 547.,
        540.44444444, 539.8, 535.09090909, 529.75, 520.15384615, 509.85714286,
        502.13333333, 495.125, 489.82352941, 482.72222222, 474.78947368,
        467.05, 460.28571429, 452.45454545, 447.60869565, 442.625, 436.6,
        433.76923077, 430.07407407, 424.96428571, 422.03448276, 419.16666667,
        415.5483871, 412.65625, 408.87878788, 405.70588235, 403.37142857,
        400.16666667, 397.91891892, 395.57894737, 393.56410256, 392.15,
        388.90243902, 385.80952381, 384.11627907, 382.29545455, 380.64444444,
        378.93478261, 376.89361702, 375.89583333, 375.18367347, 373.24,
        371.50980392, 369.25, 367.64150943, 366.51851852, 365.47272727,
        364.17857143, 362.87719298, 361.70689655, 361.27118644
    ])
    assert_allclose(rider.record_pp_.data_, record_pp)
