"""Test the Record Power Profile class. """
from itertools import product
from datetime import date

import numpy as np
from numpy.testing import assert_allclose
from numpy.testing import assert_equal
from numpy.testing import assert_raises

from skcycling.datasets import load_toy
from skcycling.power_profile import RidePowerProfile
from skcycling.power_profile import RecordPowerProfile
from skcycling.power_profile.record_power_profile import maximal_mean_power


def test_record_pp_fit():
    filename_list = load_toy()
    ride_pp_list = [
        RidePowerProfile(max_duration_profile=1)
        for i in range(len(filename_list))
    ]
    for ride, filename in zip(ride_pp_list, filename_list):
        ride.fit(filename)
    record_pp = RecordPowerProfile(max_duration_profile=1, cyclist_weight=None)
    record_pp.fit(ride_pp_list)

    data = np.array([
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

    assert_allclose(record_pp.data_, data)
    assert_equal(record_pp.data_norm_, None)
    assert_equal(record_pp.max_duration_profile, 1)
    assert_equal(record_pp.cyclist_weight, None)
    assert_equal(record_pp.date_profile_, (date(2014, 05, 07),
                                           date(2014, 07, 26)))


def test_record_pp_fit_w_weight():
    filename_list = load_toy()
    ride_pp_list = [
        RidePowerProfile(
            max_duration_profile=1, cyclist_weight=60.)
        for i in range(len(filename_list))
    ]
    for ride, filename in zip(ride_pp_list, filename_list):
        ride.fit(filename)
    record_pp = RecordPowerProfile(max_duration_profile=1, cyclist_weight=60.)
    record_pp.fit(ride_pp_list)

    data = np.array([
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

    assert_allclose(record_pp.data_, data)
    assert_allclose(record_pp.data_norm_, data / 60.)
    assert_equal(record_pp.max_duration_profile, 1)
    assert_allclose(record_pp.cyclist_weight, 60.)
    assert_equal(record_pp.date_profile_, (date(2014, 05, 07),
                                           date(2014, 07, 26)))


def test_record_pp_fit_date():
    filename_list = load_toy()
    ride_pp_list = [
        RidePowerProfile(
            max_duration_profile=1, cyclist_weight=60.)
        for i in range(len(filename_list))
    ]
    for ride, filename in zip(ride_pp_list, filename_list):
        ride.fit(filename)
    record_pp = RecordPowerProfile(max_duration_profile=1, cyclist_weight=60.)
    record_pp.fit(ride_pp_list,
                  date_profile=(date(2014, 05, 07), date(2014, 05, 11)))

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

    assert_allclose(record_pp.data_, data)
    assert_allclose(record_pp.data_norm_, data / 60.)
    assert_equal(record_pp.max_duration_profile, 1)
    assert_allclose(record_pp.cyclist_weight, 60.)
    assert_equal(record_pp.date_profile_, (date(2014, 05, 07),
                                           date(2014, 05, 11)))


def test_record_pp_fit_wrong_pp_type():
    record_pp = RecordPowerProfile(max_duration_profile=1, cyclist_weight=60.)
    assert_raises(ValueError, record_pp.fit, 1)


def test_record_pp_fit_wrong_pp_list_type():
    pattern = '2014-05-07-14-26-22.fit'
    filename_list = load_toy()
    for f in filename_list:
        if pattern in f:
            filename = [f]
    ride_pp_list = [
        RidePowerProfile(
            max_duration_profile=1, cyclist_weight=60.)
        for i in range(len(filename))
    ]
    ride_pp_list[0].fit(filename[0])
    ride_pp_list.insert(1, 1)
    record_pp = RecordPowerProfile(max_duration_profile=1, cyclist_weight=60.)
    assert_raises(ValueError, record_pp.fit, ride_pp_list)


def test_record_pp_fit_date_wrong_tuple():
    pattern = ['2014-05-07-14-26-22.fit', '2014-05-11-11-39-38.fit']
    filename_list = load_toy()
    filename = []
    for prod_pf in product(pattern, filename_list):
        if prod_pf[0] in prod_pf[1]:
            filename.append(prod_pf[1])
    ride_pp_list = [
        RidePowerProfile(
            max_duration_profile=1, cyclist_weight=60.)
        for i in range(len(filename))
    ]
    for ride, f in zip(ride_pp_list, filename):
        ride.fit(f)
    record_pp = RecordPowerProfile(max_duration_profile=1, cyclist_weight=60.)
    assert_raises(ValueError, record_pp.fit, ride_pp_list, date_profile=[1, 1])
    assert_raises(
        ValueError, record_pp.fit, ride_pp_list, date_profile=(1, 2, 3))
    assert_raises(ValueError, record_pp.fit, ride_pp_list, date_profile=(1, 2))
    assert_raises(
        ValueError,
        record_pp.fit,
        ride_pp_list,
        date_profile=(date(2014, 1, 2), date(2014, 1, 1)))


def test_record_pp_fit_forget_fitting():
    filename_list = load_toy()
    ride_pp_list = [
        RidePowerProfile(max_duration_profile=1)
        for i in range(len(filename_list))
    ]
    record_pp = RecordPowerProfile(max_duration_profile=1)
    assert_raises(ValueError, record_pp.fit, ride_pp_list)


def test_record_pp_fit_no_max_profile():
    filename_list = load_toy()
    ride_pp_list = [
        RidePowerProfile(max_duration_profile=1)
        for i in range(len(filename_list))
    ]
    for ride, filename in zip(ride_pp_list, filename_list):
        ride.fit(filename)
    record_pp = RecordPowerProfile(max_duration_profile=None)
    assert_raises(ValueError, record_pp.fit, ride_pp_list)


def test_record_pp_fit_different_max_duration():
    filename_list = load_toy()
    ride_pp_list = [
        RidePowerProfile(max_duration_profile=i+1)
        for i in range(len(filename_list))
    ]
    for ride, filename in zip(ride_pp_list, filename_list):
        ride.fit(filename)
    record_pp = RecordPowerProfile(max_duration_profile=1)
    assert_raises(ValueError, record_pp.fit, ride_pp_list)


def test_mmp_wrong_type():
    assert_raises(ValueError, maximal_mean_power, 1)
    filename_list = load_toy()
    ride_pp_list = [
        RidePowerProfile(max_duration_profile=i+1)
        for i in range(len(filename_list))
    ]
    for ride, filename in zip(ride_pp_list, filename_list):
        ride.fit(filename)
    ride_pp_list.append(1)
    assert_raises(ValueError, maximal_mean_power, ride_pp_list)
