""" Testing the metrics developed to asses performance of a ride """

import pytest

import pandas as pd
import numpy as np

from skcycling.metrics import normalized_power_score
from skcycling.metrics import intensity_factor_ftp_score
from skcycling.metrics import intensity_factor_mpa_score
from skcycling.metrics import training_stress_ftp_score
from skcycling.metrics import training_stress_mpa_score
from skcycling.metrics import mpa2ftp
from skcycling.metrics import ftp2mpa
from skcycling.metrics import training_stress_mpa_grappe_score
from skcycling.metrics import training_stress_ftp_grappe_score

mpa = 400.
ftp = 304.

ride = np.array([300.] * 200 + [0.] * 200 + [200.] * 200)
ride = pd.Series(ride,
                 index=pd.date_range('1/1/2011',
                                     periods=ride.size,
                                     freq='1S'),
                 name='power')

ride_2 = np.array([140.] * 20 + [220.] * 20 + [250.] * 20 + [310.] * 20 +
                  [350.] * 20 + [410.] * 20 + [800.] * 20)
ride_2 = pd.Series(ride_2,
                   index=pd.date_range('1/1/2011',
                                       periods=ride_2.size,
                                       freq='1S'),
                   name='power')


@pytest.mark.parametrize(
    "score_func, params, expected_score",
    [(normalized_power_score, (ride, mpa), 260.7611),
     (intensity_factor_ftp_score, (ride, ftp), 0.857766),
     (intensity_factor_mpa_score, (ride, mpa), 0.857766),
     (training_stress_ftp_score, (ride, ftp), 12.26273),
     (training_stress_mpa_score, (ride, mpa), 12.26273),
     (training_stress_mpa_grappe_score, (ride_2, mpa), 11.16666),
     (training_stress_ftp_grappe_score, (ride_2, ftp), 11.16666)])
def test_scores(score_func, params, expected_score):
    assert score_func(*params) == pytest.approx(expected_score)


def test_convert_mpa_ftp():
    assert mpa2ftp(ftp2mpa(ftp)) == pytest.approx(ftp)
