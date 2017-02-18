""" Testing the checker methods """
import unittest
from datetime import date

from numpy.testing import assert_equal
from numpy.testing import assert_raises

from skcycling.datasets import load_toy
from skcycling.utils import check_tuple_date
from skcycling.utils import check_filename_fit

_dummy = _dummy = unittest.TestCase('__init__')
try:
    assert_raises_regex = _dummy.assertRaisesRegex
except AttributeError:
    # Python 2.7
    assert_raises_regex = _dummy.assertRaisesRegexp


def test_check_check_filename_fit_wrong_type():
    assert_raises_regex(ValueError, 'filename needs to be a string',
                        check_filename_fit, 1)


def test_check_filename_fit_not_fit():
    assert_raises_regex(ValueError, 'The file is not a fit file.',
                        check_filename_fit, 'file.rnd')


def test_check_filename_fit_not_exist():
    assert_raises_regex(ValueError, 'The file does not exist.',
                        check_filename_fit, 'file.fit')


def test_check_filename_fit():
    filename = load_toy()[0]
    my_filename = check_filename_fit(filename)
    assert_equal(my_filename, filename)


def test_tuple_date_wrong_type_dim():
    assert_raises(ValueError, check_tuple_date, 1)
    assert_raises(ValueError, check_tuple_date, (1, 1, 1))


def test_tuple_date_wrong_type_tuple():
    assert_raises(ValueError, check_tuple_date, (1, 1))


def test_tuple_date_wrong_order():
    assert_raises(ValueError, check_tuple_date, (date(2014, 1, 1),
                                                 date(2013, 1, 1)))


def test_tuple_date():
    dt = (date(2014, 1, 1), date(2015, 1, 1))
    dt_out = check_tuple_date(dt)

    assert_equal(dt_out, dt)
