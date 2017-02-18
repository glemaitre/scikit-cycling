import unittest
import numpy as np

from numpy.testing import assert_allclose

from skcycling.restoration import outliers_rejection
from skcycling.restoration import moving_average

_dummy = _dummy = unittest.TestCase('__init__')
try:
    assert_raises_regex = _dummy.assertRaisesRegex
except AttributeError:
    # Python 2.7
    assert_raises_regex = _dummy.assertRaisesRegexp

pow_ride_1 = np.linspace(-10., 3000., 20)
pow_ride_2 = np.array([200.] * 4 + [300] * 4)

X_COMP = np.array([
    1495., 148.42105263, 306.84210526, 465.26315789, 623.68421053,
    782.10526316, 940.52631579, 1098.94736842, 1257.36842105, 1415.78947368,
    1574.21052632, 1732.63157895, 1891.05263158, 2049.47368421, 2207.89473684,
    2366.31578947, 1495., 1495., 1495., 1495.
])

X_GT = np.array([200., 200., 233.333333, 266.666667, 300., 300.])


def test_outliers_thres_rejection():
    x_free_outliers = outliers_rejection(pow_ride_1)
    assert_allclose(x_free_outliers, X_COMP)


def test_outliers_unknown_method():
    assert_raises_regex(ValueError, "The outliers detection method is"
                        " unknown.", outliers_rejection, pow_ride_1, '')


def test_moving_average():
    x_denoise = moving_average(pow_ride_2, win=3)
    assert_allclose(x_denoise, X_GT)
