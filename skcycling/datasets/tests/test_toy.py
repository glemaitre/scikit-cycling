from skcycling.datasets import load_toy

import unittest
_dummy = unittest.TestCase('__init__')
assert_true = _dummy.assertTrue


def test_load_toy_list_file():
    filenames = load_toy()

    gt_filenames = [
        '2014-05-11-11-39-38.fit', '2014-05-07-14-26-22.fit',
        '2014-07-26-18-50-56.fit'
    ]

    for f, gt in zip(filenames, gt_filenames):
        assert_true(gt in f)


def test_load_toy_path():
    path = load_toy(returned_type='path')

    gt_path = 'data'

    assert_true(gt_path in path)
