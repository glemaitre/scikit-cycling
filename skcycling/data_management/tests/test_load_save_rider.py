""" Test the loading and saving using pickles. """

import os

from tempfile import mkdtemp
import shutil

from numpy.testing import assert_array_equal
from numpy.testing import assert_equal

from skcycling.datasets import load_toy
from skcycling.data_management import Rider
from skcycling.power_profile import RidePowerProfile


def test_save_load_rider():
    filename_list = load_toy()
    ride_pp_list = [
        RidePowerProfile(max_duration_profile=1)
        for i in range(len(filename_list))
    ]
    for ride, filename in zip(ride_pp_list, filename_list):
        ride.fit(filename)

    rider = Rider(
        cyclist_weight=60., max_duration_profile=1, rides_pp=ride_pp_list)
    rider.compute_record_pp()

    # temporary directory
    tmp_dir = mkdtemp()
    try:
        filename = os.path.join(tmp_dir, 'rider.pkl')
        rider.save_to_pickles(filename)
        obj = rider.load_from_pickles(filename)

        assert_array_equal(rider.rides_pp_[0].data_, obj.rides_pp_[0].data_)
        assert_array_equal(rider.record_pp_.data_, obj.record_pp_.data_)
        assert_equal(rider.max_duration_profile, obj.max_duration_profile)
        assert_equal(rider.cyclist_weight, obj.cyclist_weight)
    finally:
        shutil.rmtree(tmp_dir)
