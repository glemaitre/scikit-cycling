"""Test the Ride Power Profile class. """
import numpy as np

from datetime import date

from numpy.testing import assert_allclose
from numpy.testing import assert_equal

from skcycling.datasets import load_toy
from skcycling.power_profile import RidePowerProfile
from skcycling.power_profile.ride_power_profile import _rpp_parallel
from skcycling.utils import load_power_from_fit


def test_ridepp_fit():
    filename = load_toy()[0]
    ride_rpp = RidePowerProfile(max_duration_profile=1, cyclist_weight=None)
    ride_rpp.fit(filename)
    data = np.array([
        0., 500., 475.5, 469.33333333, 464., 463., 462.33333333, 461.71428571,
        455.875, 450.55555556, 447.3, 444.81818182, 442.08333333, 439.53846154,
        435.71428571, 432.06666667, 428.75, 424.35294118, 420.44444444,
        413.78947368, 409.9, 407.23809524, 402.5, 399.91304348, 396.45833333,
        394.76, 392.19230769, 388.62962963, 384.75, 380., 373.8, 367.70967742,
        362.96875, 357.90909091, 354.02941176, 349.68571429, 345.83333333,
        342.18918919, 338.36842105, 335.02564103, 331.375, 328.95121951,
        325.64285714, 322.37209302, 318.09090909, 315.15555556, 312.23913043,
        309.59574468, 307.08333333, 304.55102041, 301.9, 300.70588235, 300.5,
        299.90566038, 300.03703704, 298.92727273, 298.10714286, 297.56140351,
        296.48275862, 296.30508475
    ])
    assert_allclose(ride_rpp.data_, data)
    assert_equal(ride_rpp.data_norm_, None)
    assert_equal(ride_rpp.cyclist_weight, None)
    assert_equal(ride_rpp.max_duration_profile, 1)
    assert_equal(ride_rpp.date_profile_, date(2014, 5, 7))
    assert_equal(ride_rpp.filename_, filename)


def test_ridepp_fit_w_weight():
    filename = load_toy()[0]
    ride_rpp = RidePowerProfile(max_duration_profile=1, cyclist_weight=60.)
    ride_rpp.fit(filename)
    data = np.array([
        0., 500., 475.5, 469.33333333, 464., 463., 462.33333333, 461.71428571,
        455.875, 450.55555556, 447.3, 444.81818182, 442.08333333, 439.53846154,
        435.71428571, 432.06666667, 428.75, 424.35294118, 420.44444444,
        413.78947368, 409.9, 407.23809524, 402.5, 399.91304348, 396.45833333,
        394.76, 392.19230769, 388.62962963, 384.75, 380., 373.8, 367.70967742,
        362.96875, 357.90909091, 354.02941176, 349.68571429, 345.83333333,
        342.18918919, 338.36842105, 335.02564103, 331.375, 328.95121951,
        325.64285714, 322.37209302, 318.09090909, 315.15555556, 312.23913043,
        309.59574468, 307.08333333, 304.55102041, 301.9, 300.70588235, 300.5,
        299.90566038, 300.03703704, 298.92727273, 298.10714286, 297.56140351,
        296.48275862, 296.30508475
    ])
    assert_allclose(ride_rpp.data_, data)
    assert_allclose(ride_rpp.data_norm_, data / 60.)
    assert_allclose(ride_rpp.cyclist_weight, 60.)
    assert_equal(ride_rpp.max_duration_profile, 1)
    assert_equal(ride_rpp.date_profile_, date(2014, 5, 7))
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
