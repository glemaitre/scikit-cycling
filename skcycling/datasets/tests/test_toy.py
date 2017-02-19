from skcycling.datasets import load_toy

import unittest
_dummy = unittest.TestCase('__init__')
assert_true = _dummy.assertTrue


def test_load_toy_list_file():
    filenames = load_toy()
    gt_filenames = sorted([
        '2014-05-11-11-39-38.fit', '2014-05-07-14-26-22.fit',
        '2014-07-26-18-50-56.fit'
    ])
    for f, gt in zip(filenames, gt_filenames):
        assert_true(gt in f)


def test_load_toy_path():
    path = load_toy(returned_type='path')
    gt_path = 'data'
    assert_true(gt_path in path)


def test_load_toy_list_file_corrupted():
    filenames = load_toy(set_data='corrupted')
    gt_filenames = sorted([
        '2013-04-24-22-22-25.fit', '2014-05-17-10-44-53.fit',
        '2015-11-27-18-54-57.fit'
    ])
    for f, gt in zip(filenames, gt_filenames):
        assert_true(gt in f)


def test_load_toy_path_corrupted():
    path = load_toy(returned_type='path', set_data='corrupted')
    print(path)
    gt_path = 'corrupted_data'
    assert_true(gt_path in path)
