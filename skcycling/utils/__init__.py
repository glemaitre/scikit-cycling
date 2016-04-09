"""
The :mod:`skcycling.utils` module include utility functions.
"""

from .io_fit import load_power_from_fit

from .fit import res_std_dev
from .fit import r_squared
from .fit import log_linear_fitting
from .fit import linear_model
from .fit import log_linear_model

from .checker import check_X
from .checker import check_float
from .checker import check_tuple_date
from .checker import check_filename_fit
from .checker import check_filename_pickle_load
from .checker import check_filename_pickle_save


__all__ = ['load_power_from_fit',
           'res_std_dev',
           'r_squared',
           'log_linear_fitting',
           'linear_model',
           'log_linear_model',
           'check_X',
           'check_float',
           'check_tuple_date',
           'check_filename_fit',
           'check_filename_pickle_load',
           'check_filename_pickle_save']