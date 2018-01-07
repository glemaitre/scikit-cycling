# Authors: Guillaume Lemaitre <g.lemaitre58@gmail.com>
#          Cedric Lemaitre
# License: BSD 3 clause

from datetime import time

import pytest

from skcycling.io import bikeread
from skcycling.datasets import load_fit
from skcycling.extraction import activity_power_profile


@pytest.mark.parametrize(
    "max_duration, power_profile_shape, first_element",
    [(None, (2256,), 174.404255),
     (10, (9,), 450.5555),
     ('00:00:10', (9,), 450.5555),
     (time(0, 0, 10), (9,), 450.5555)]
)
def test_activity_power_profile(max_duration, power_profile_shape,
                                first_element):
    activity = bikeread(load_fit()[0])
    power_profile = activity_power_profile(activity, max_duration=max_duration)
    assert power_profile.shape == power_profile_shape
    assert power_profile.iloc[-1] == pytest.approx(first_element)
