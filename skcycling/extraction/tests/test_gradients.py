"""Test the gradient module."""

# Authors: Guillaume Lemaitre <g.lemaitre58@gmail.com>
#          Cedric Lemaitre
# License: BSD 3 clause

import numpy as np
import pandas as pd

import pytest

from skcycling.extraction import acceleration
from skcycling.extraction import gradient_elevation
from skcycling.extraction import gradient_heart_rate
from skcycling.exceptions import MissingDataError


def test_acceleration_error():
    activity = pd.DataFrame({'A': np.random.random(1000)})
    msg = "speed data are required"
    with pytest.raises(MissingDataError, message=msg):
        acceleration(activity)


@pytest.mark.parametrize(
    "activity, append, type_output, shape",
    [(pd.DataFrame({'speed': np.random.random(100)}),
      False, pd.Series, (100,)),
     (pd.DataFrame({'speed': np.random.random(100)}),
      True, pd.DataFrame, (100, 2))])
def test_acceleration(activity, append, type_output, shape):
    output = acceleration(activity, append=append)
    assert isinstance(output, type_output)
    assert output.shape == shape


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


def test_gradient_heart_rate_error():
    activity = pd.DataFrame({'A': np.random.random(1000)})
    msg = "heart-rate data are required"
    with pytest.raises(MissingDataError, message=msg):
        gradient_heart_rate(activity)


@pytest.mark.parametrize(
    "activity, append, type_output, shape",
    [(pd.DataFrame({'heart-rate': np.random.random(100)}),
      False, pd.Series, (100,)),
     (pd.DataFrame({'heart-rate': np.random.random(100)}),
      True, pd.DataFrame, (100, 2))])
def test_gradient_heart_rate(activity, append, type_output, shape):
    output = gradient_heart_rate(activity, append=append)
    assert isinstance(output, type_output)
    assert output.shape == shape
