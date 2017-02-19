""" Test the loading and saving using pickles. """

import os
import shutil
from temp import mkdtemp

from numpy.testing import assert_array_equal
from numpy.testing import assert_equal

from skcycling.datasets import load_toy
from skcycling.power_profile import RidePowerProfile


def test_save_load_ride_pp():
    filename = load_toy()[0]
    my_ride_rpp = RidePowerProfile(max_duration_profile=1)
    my_ride_rpp.fit(filename)

    tmp_dir = mkdtemp()
    try:
        store_filename = os.path.join(tmp_dir, 'ride_rpp.pkl')
        my_ride_rpp.save_to_pickles()
        obj = RidePowerProfile.load_from_pickles(store_filename)

        assert_array_equal(my_ride_rpp.data_, obj.data_)
        assert_equal(my_ride_rpp.data_norm_, obj.data_norm_)
        assert_equal(my_ride_rpp.cyclist_weight, obj.cyclist_weight)
        assert_equal(my_ride_rpp.max_duration_profile,
                     obj.max_duration_profile)
        assert_equal(my_ride_rpp.date_profile_, obj.date_profile_)
        assert_equal(my_ride_rpp.filename_, obj.filename_)
    finally:
        shutil.rmtree(store_filename)
