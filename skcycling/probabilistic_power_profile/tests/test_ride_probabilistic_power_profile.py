"""Test the probabilistic power-profile for a ride"""

from skcycling.probabilistic_power_profile import RideProbabilisticPowerProfile

from skcycling.datasets import load_toy


def test_ride_rppp_fit():
    filename = load_toy()[0]
    ride_rppp = RideProbabilisticPowerProfile(max_duration_profile=20,
                                              cyclist_weight=60.,
                                              n_jobs=-1)
    ride_rppp.fit(filename)
    return ride_rppp


if __name__ == '__main__':
    a = test_ride_rppp_fit()
