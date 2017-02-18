"""Methods to help for mathematical fitting."""

import numpy as np

from scipy.stats import linregress
from scipy.optimize import curve_fit


def res_std_dev(model, estimate):
    """Function to compute the residual standard deviation.

    Parameters
    ----------
    model : array-like, shape (n_sample, )
         Value used to made the fitting.

    estimate : array-like, shape (n_sample, )
         Value obtained by fitting.

    Returns
    -------
    residual : float
        Residual standard deviation.

    """

    if model.shape != estimate.shape:
        raise ValueError('The model and estimate arrays should have'
                         ' the same size.')

    return np.sqrt(np.sum((model - estimate)**2) / (float(model.size) - 2.))


def r_squared(model, estimate):
    """Function to compute the coefficient of determination.

    Parameters
    ----------
    model : array-like, shape (n_sample, )
         Value used to made the fitting.

    estimate : array-like, shape (n_sample, )
         Value obtained by fitting.

    Returns
    -------
    coeff_det : float
        Coefficient of determination.

    """

    if model.shape != estimate.shape:
        raise ValueError('The model and estimate arrays should have'
                         ' the same size.')

    # Compute of the observed data
    model_mean = np.mean(model)

    # Compute the total sum of squares
    ss_tot = np.sum((model - model_mean)**2)

    # Compute the sum of squares residual
    ss_res = np.sum((model - estimate)**2)

    return 1. - (ss_res / ss_tot)


def log_linear_fitting(x, y, method='lsq'):
    """Function to perform log linear regression.

    Parameters
    ----------
    x : ndarray, shape (n_samples)
        Corresponds to the x. In our case, it should be the time.

    y : ndarray, shape (n_samples)
        Corresponds to the y. In our case, it should be the rpp.

    method : string, optional (default='lsq')
        The method to use to perform the regression. The choices are:

        - If 'lsq', an ordinary least-square approach is used.
        - If 'lm', the Levenberg-Marquardt is used.

    Returns
    -------
    slope : float
        slope of the regression line.

    intercept : float
        intercept of the regression line.

    std_err : float
        Standard error of the estimate.

    coeff_det : float
        Coefficient of determination.

    """

    # Check that the array x and y have the same size
    if x.shape != y.shape:
        raise ValueError('The size of x and y should be the same.')

    if method == 'lsq':
        # Perform the fitting using least-square
        slope, intercept, _, _, _ = linregress(np.log(x), y)

        std_err = res_std_dev(y, log_linear_model(x, slope, intercept))

        coeff_det = r_squared(y, log_linear_model(x, slope, intercept))
    elif method == 'lm':
        # Perform the fitting using non-linear least-square
        # Levenberg-Marquardt
        popt, _ = curve_fit(linear_model, np.log(x), y)

        slope = popt[0]
        intercept = popt[1]

        std_err = res_std_dev(y, log_linear_model(x, slope, intercept))

        coeff_det = r_squared(y, log_linear_model(x, slope, intercept))
    else:
        raise NotImplementedError

    return slope, intercept, std_err, coeff_det


def linear_model(x, slope, intercept):
    """Function wich return value of a linear model.

    Parameters
    ----------
    x : ndarray, shape (n_samples, )
        Values for which to compute the output of the linear model.

    slope : float
        Slope of the linear equation.

    intercept : float
        Intercept of the linear equation.

    Returns
    -------
    y : ndarray, shape (n_samples)
        Output of the linear model.

    """

    return slope * x + intercept


def log_linear_model(x, slope, intercept):
    """Function wich return value of a log linear model.

    Parameters
    ----------
    x : ndarray, shape (n_samples, )
        Values for which to compute the output of the linear model.

    slope : float
        Slope of the linear equation.

    intercept : float
        Intercept of the linear equation.

    Returns
    -------
    y : ndarray, shape (n_samples)
        Output of the linear model.

    """

    return slope * np.log(x) + intercept
