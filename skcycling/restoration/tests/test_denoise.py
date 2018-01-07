import numpy as np
from numpy.testing import assert_allclose

from skcycling.restoration import moving_average


def test_moving_average():
    power = np.array([200.] * 4 + [300] * 4)
    expected_power = np.array([200., 200., 233.333333, 266.666667, 300., 300.])
    power_denoise = moving_average(power, win=3)
    assert_allclose(power_denoise, expected_power)
