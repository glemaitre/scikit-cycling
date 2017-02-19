"""Test the Ride Power Profile class. """

import os
import numpy as np

from datetime import date

from numpy.testing import assert_allclose
from numpy.testing import assert_equal
from numpy.testing import assert_raises

from skcycling.datasets import load_toy
from skcycling.power_profile import RidePowerProfile
from skcycling.power_profile.ride_power_profile import _rpp_parallel
from skcycling.utils import load_power_from_fit


def test_ridepp_fit():
    filename = load_toy()[0]
    ride_rpp = RidePowerProfile(max_duration_profile=1, cyclist_weight=None)
    ride_rpp.fit(filename)
    data = np.array([
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
    assert_allclose(ride_rpp.data_, data)
    assert_equal(ride_rpp.data_norm_, None)
    assert_equal(ride_rpp.cyclist_weight, None)
    assert_equal(ride_rpp.max_duration_profile, 1)
    assert_equal(ride_rpp.date_profile_, date(2014, 5, 11))
    assert_equal(ride_rpp.filename_, filename)


def test_ridepp_fit_w_weight():
    filename = load_toy()[0]
    ride_rpp = RidePowerProfile(max_duration_profile=1, cyclist_weight=60.)
    ride_rpp.fit(filename)
    data = np.array([
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
    assert_allclose(ride_rpp.data_, data)
    assert_allclose(ride_rpp.data_norm_, data / 60.)
    assert_allclose(ride_rpp.cyclist_weight, 60.)
    assert_equal(ride_rpp.max_duration_profile, 1)
    assert_equal(ride_rpp.date_profile_, date(2014, 5, 11))
    assert_equal(ride_rpp.filename_, filename)


def test_rpp_parallel():
    pattern = '2014-05-07-14-26-22.fit'
    filename_list = load_toy()
    for f in filename_list:
        if pattern in f:
            filename = f
    power_rec = load_power_from_fit(filename)
    val = _rpp_parallel(power_rec, 1)
    assert_allclose(val, 500.)
