from skcycling.datasets import load_toy
from skcycling.probabilistic_power_profile import (
    RideProbabilisticPowerProfile,
    RecordProbabilisticPowerProfile)


def test_record_rppp():
    filenames = load_toy()

    rides_rppp = [RideProbabilisticPowerProfile(max_duration_profile=20,
                                                cyclist_weight=60.,
                                                n_jobs=-1).fit(filename)
                  for filename in filenames]
    record_rppp = RecordProbabilisticPowerProfile(max_duration_profile=20,
                                                  cyclist_weight=60)
    record_rppp.fit(rides_rppp)
