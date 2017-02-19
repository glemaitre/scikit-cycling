""" Testing the metrics developed to asses performance of a ride """

import numpy as np

from numpy.testing import assert_allclose
from numpy.testing import assert_equal

from skcycling.metrics import normalized_power_score
from skcycling.metrics import intensity_factor_ftp_score
from skcycling.metrics import intensity_factor_pma_score
from skcycling.metrics import training_stress_ftp_score
from skcycling.metrics import training_stress_pma_score
from skcycling.metrics import pma2ftp
from skcycling.metrics import ftp2pma
from skcycling.metrics import training_stress_pma_grappe_score
from skcycling.metrics import training_stress_ftp_grappe_score

pma = 400.
ftp = 304.
ride = np.array([300.]*200 + [0.]*200 + [200.]*200)
ride_2 = np.array([140.]*20 + [220.]*20 + [250.]*20 + [310.]*20 +
                  [350.]*20 + [410.]*20 + [800.]*20)


def test_normalized_power_score():
    np_score_gt = 260.42569651622745
    np_score = normalized_power_score(ride, pma)
    assert_allclose(np_score, np_score_gt)


def test_intensity_factor_ftp_score():
    if_score_ftp_gt = 0.85666347538232712
    if_score_ftp = intensity_factor_ftp_score(ride, ftp)
    assert_allclose(if_score_ftp, if_score_ftp_gt)


def test_intensity_factor_pma_score():
    if_score_pma_gt = 0.85666347538232712
    if_score_pma = intensity_factor_pma_score(ride, pma)
    assert_allclose(if_score_pma, if_score_pma_gt)


def test_training_stress_ftp_score():
    tss_score_ftp_gt = 12.231205167568783
    tss_score_ftp = training_stress_ftp_score(ride, ftp)
    assert_allclose(tss_score_ftp, tss_score_ftp_gt)


def test_training_stress_pma_score():
    tss_score_pma_gt = 12.231205167568783
    tss_score_pma = training_stress_pma_score(ride, pma)
    assert_allclose(tss_score_pma, tss_score_pma_gt)


def test_pma2ftp():
    ftp_score = pma2ftp(pma)
    assert_equal(ftp_score, ftp)


def test_ftp2pma():
    pma_score = ftp2pma(ftp)
    assert_equal(pma_score, pma)


def test_training_stress_pma_grappe_score():
    ts_score_gt = 11.166666666666664
    ts_score = training_stress_pma_grappe_score(ride_2, pma)
    assert_allclose(ts_score, ts_score_gt)


def test_training_stress_ftp_grappe_score():
    ts_score_gt = 11.166666666666664
    ts_score = training_stress_ftp_grappe_score(ride_2, ftp)
    assert_allclose(ts_score, ts_score_gt)
