"""Test the gradient module."""

# Authors: Guillaume Lemaitre <g.lemaitre58@gmail.com>
#          Cedric Lemaitre
# License: BSD 3 clause

import numpy as np
import pandas as pd

import pytest

from skcycling.extraction import gradient_elevation
from skcycling.exceptions import MissingDataError


def test_gradient_elevation_error():
    activity = pd.DataFrame({'A': np.random.random(1000)})
    msg = "elevation and distance data are required"
    with pytest.raises(MissingDataError, message=msg):
        gradient_elevation(activity)


@pytest.mark.parametrize(
    "activity, append, type_output, shape",
    [(pd.DataFrame({'elevation': np.random.random(100),
                    'distance': np.random.random(100)}),
      False, pd.Series, (100,)),
     (pd.DataFrame({'elevation': np.random.random(100),
                    'distance': np.random.random(100)}),
      True, pd.DataFrame, (100, 3))])
def test_gradient_elevation(activity, append, type_output, shape):
    output = gradient_elevation(activity, append=append)
    assert isinstance(output, type_output)
    assert output.shape == shape
