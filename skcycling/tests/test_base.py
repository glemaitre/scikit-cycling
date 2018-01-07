# Authors: Guillaume Lemaitre <g.lemaitre58@gmail.com>
#          Cedric Lemaitre
# License: BSD 3 clause

import os
import shutil
from tempfile import mkdtemp

import pytest
from pandas.testing import assert_frame_equal

from skcycling.base import Rider
from skcycling.datasets import load_fit
from skcycling.datasets import load_rider


def test_rider_add_activities():
    rider = Rider.from_csv(load_rider(), n_jobs=-1)

    # update a previous DataFrame
    rider.add_activities(load_fit()[0])
    assert rider.power_profile_.shape == (6703, 4)

    # add a first power-profile
    rider = Rider(n_jobs=-1)
    rider.add_activities(load_fit()[0])
    assert rider.power_profile_.shape == (2256, 1)


@pytest.mark.parametrize(
    "dates, expected_shape",
    [('07 May 2014', (6703, 2)),
     (['07 May 2014'], (6703, 2)),
     (tuple(['07 May 2014', '11 May 2014']), (6703, 1))])
def test_rider_delete_activities(dates, expected_shape):
    rider = Rider.from_csv(load_rider(), n_jobs=-1)
    rider.delete_activities(dates)
    assert rider.power_profile_.shape == expected_shape


@pytest.mark.parametrize(
    "dates",
    [(tuple(['07 May 2014'])),
     (tuple(['07 May 2014', '10 May 2014', '11 May 2014']))])
def test_rider_delete_activities_error(dates):
    rider = Rider.from_csv(load_rider(), n_jobs=-1)

    msg = "Wrong tuple format"
    with pytest.raises(ValueError, message=msg):
        rider.delete_activities(dates)


def test_dump_load_rider():
    filenames = load_fit()[:1]
    rider = Rider(n_jobs=-1)
    rider.add_activities(filenames)

    tmpdir = mkdtemp()
    csv_filename = os.path.join(tmpdir, 'rider.csv')
    try:
        rider.to_csv(csv_filename)
        rider2 = Rider.from_csv(csv_filename)
        assert_frame_equal(rider.power_profile_, rider2.power_profile_)
    finally:
        shutil.rmtree(tmpdir)
