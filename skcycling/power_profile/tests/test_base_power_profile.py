from numpy.testing import assert_allclose
from numpy.testing import assert_raises

from skcycling.datasets import load_toy
from skcycling.power_profile import RidePowerProfile


def test_save_load_ride_pp():
    filename = load_toy()[0]
    my_ride_rpp = RidePowerProfile(max_duration_profile=1)
    my_ride_rpp.fit(filename)
    val = my_ride_rpp.resampling_rpp(.5)
    assert_allclose(val, 376.9)


def test_save_load_ride_pp_weight():
    filename = load_toy()[0]
    my_ride_rpp = RidePowerProfile(max_duration_profile=1,
                                   cyclist_weight=60.)
    my_ride_rpp.fit(filename)
    val = my_ride_rpp.resampling_rpp(.5, normalized=True)
    assert_allclose(val, 6.281666666666666)


def test_save_load_ride_pp_wrong_norm():
    filename = load_toy()[0]
    my_ride_rpp = RidePowerProfile(max_duration_profile=1, cyclist_weight=None)
    my_ride_rpp.fit(filename)
    assert_raises(ValueError, my_ride_rpp.resampling_rpp, .5, normalized=True)
