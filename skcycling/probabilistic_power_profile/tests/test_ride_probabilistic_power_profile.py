"""Test the probabilistic power-profile for a ride"""

import unittest
from datetime import date

from numpy.testing import assert_equal

from skcycling.probabilistic_power_profile import RideProbabilisticPowerProfile
from skcycling.datasets import load_toy

_dummy = unittest.TestCase('__init__')
assert_true = _dummy.assertTrue
try:
    assert_raises_regex = _dummy.assertRaisesRegex
except AttributeError:
    # Python 2.7
    assert_raises_regex = _dummy.assertRaisesRegexp


def test_ride_rppp_fit():
    filename = load_toy()[0]
    ride_rppp = RideProbabilisticPowerProfile(max_duration_profile=1,
                                              cyclist_weight=None,
                                              n_jobs=-1)
    ride_rppp.fit(filename)
    filename = '2014-05-07-14-26-22.fit'
    assert_true(ride_rppp.filename_.endswith(filename))
    assert_equal(ride_rppp.date_profile_, date(2014, 5, 7))
    assert_equal(ride_rppp.data_.min(), 0.0)
    assert_equal(ride_rppp.data_.max(), 317)
    assert_equal(ride_rppp.data_.shape, (501, 60))
    assert_equal(ride_rppp.data_.mean(), 4.3435462408516283)
    assert_equal(len(ride_rppp.data_.data), 19383)


def test_error_plot():
    filename = load_toy()[0]
    ride_rppp = RideProbabilisticPowerProfile(max_duration_profile=1,
                                              cyclist_weight=None,
                                              n_jobs=-1)
    assert_raises_regex(ValueError, 'Fit the data before to plot them.',
                        ride_rppp.plot_heatmap)
    ride_rppp.fit(filename)
    assert_raises_regex(ValueError, 'You need to provide a weight to get a'
                        ' normalized plot.', ride_rppp.plot_heatmap,
                        normalized=True)


def test_plot_heatmap():
    filename = load_toy()[0]
    ride_rppp = RideProbabilisticPowerProfile(max_duration_profile=1,
                                              cyclist_weight=60,
                                              n_jobs=-1)
    ride_rppp.fit(filename)
    ride_rppp.plot_heatmap(False)
    ride_rppp.plot_heatmap(True)
