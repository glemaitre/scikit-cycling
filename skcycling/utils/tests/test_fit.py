""" Test the functions for mathematical fitting. """

import numpy as np

from numpy.testing import assert_equal
from numpy.testing import assert_almost_equal
from numpy.testing import assert_raises

from skcycling.utils import res_std_dev
from skcycling.utils import r_squared
from skcycling.utils import log_linear_fitting
from skcycling.utils import linear_model
from skcycling.utils import log_linear_model

PRECISION_TEST = 3


def test_res_std_dec_wrong_sz():
    """ Test either if an error is raised if the estimate and model sizes
    are not consistent. """
    model = np.arange(10)
    estimate = np.ones((5, )) * 5.

    assert_raises(ValueError, res_std_dev, model, estimate)


def test_res_std_dec():
    """ Test the function which compute the residual standard deviation. """
    model = np.arange(10)
    estimate = np.ones((10, )) * 5.

    assert_almost_equal(
        res_std_dev(model, estimate), 3.2596012026, decimal=PRECISION_TEST)


def test_r_squared_wrong_sz():
    """ Test either if an error is raised if the estimate and model sizes
    are not consistent. """
    model = np.arange(10)
    estimate = np.ones((5, )) * 5.

    assert_raises(ValueError, r_squared, model, estimate)


def test_r_squared():
    """ Test the function which compute the coefficient of determination. """
    model = np.arange(10)
    estimate = np.ones((10, )) * 5.

    assert_almost_equal(
        r_squared(model, estimate),
        -0.030303030303030276,
        decimal=PRECISION_TEST)


def test_linear_model():
    """ Test the linear model routine. """
    assert_equal(linear_model(3., 4., 2.), 14.)


def test_log_linear_model():
    """ Test the linear model routine. """
    assert_almost_equal(
        log_linear_model(3., 4., 2.),
        6.3944491546724391,
        decimal=PRECISION_TEST)


def test_log_linear_fitting_wrong_size():
    """ Test either if an error is raised when the size of x and y are
    not the same. """
    x = np.arange(2., 10.)
    y = np.arange(2., 9.)

    assert_raises(ValueError, log_linear_fitting, x, y)


def test_log_linear_fitting_lsq():
    """ Test either if an error is raised when the size of x and y are
    not the same. """
    x = np.arange(3, 10, dtype=float)
    y = linear_model(x, 2., 3.)

    method = 'lsq'
    slope, intercept, std_err, coeff_det = log_linear_fitting(x, y, method)

    assert_almost_equal(slope, 10.884469280692462, decimal=PRECISION_TEST)
    assert_almost_equal(intercept, -3.8280798214097231, decimal=PRECISION_TEST)
    assert_almost_equal(std_err, 0.74166428170326482, decimal=PRECISION_TEST)
    assert_almost_equal(coeff_det, 0.97544348630560629, decimal=PRECISION_TEST)


def test_log_linear_fitting_lm():
    """ Test either if an error is raised when the size of x and y are
    not the same. """
    x = np.arange(3, 10, dtype=float)
    y = linear_model(x, 2., 3.)

    method = 'lm'
    slope, intercept, std_err, coeff_det = log_linear_fitting(x, y, method)

    assert_almost_equal(slope, 10.884469280692462, decimal=PRECISION_TEST)
    assert_almost_equal(intercept, -3.8280798214097231, decimal=PRECISION_TEST)
    assert_almost_equal(std_err, 0.74166428170326482, decimal=PRECISION_TEST)
    assert_almost_equal(coeff_det, 0.97544348630560629, decimal=PRECISION_TEST)
