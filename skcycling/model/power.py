"""Module to estimate power from data."""

# Authors: Guillaume Lemaitre <g.lemaitre58@gmail.com>
#          Cedric Lemaitre
# License: BSD 3 clause

from __future__ import division

import scipy as sp
import numpy as np


def strava_model(activity, cyclist_weight, bike_weight=6.8,
                 coef_roll_res=0.005, pressure=101325.0,
                 temperature=15.0, drag_coef=1, surface_rider=1):
    """Strava model used to estimate power.

    Parameters
    ----------
    activity : DataFrame
        The activity containing the ride information.

    cyclist_weight : float
        The cyclist weight in kg.

    bike_weight : float, default=6.8
        The bike weight in kg.

    coef_roll_res : float, default=0.005
        Rolling resistance coefficient.

    pressure : float, default=101325.0
        Pressure in Pascal.

    temperature : float, default=15.0
        Temperature in Celsius.

    drag_coeff : float, default=1
        The drag coefficient also known as Cx.

    surface_rider : float, default=1
        Surface area of the rider facing wind also known as S. The unit is m^2.

    Returns
    -------
    power : Series
        The power estimated.

    """
    speed = activity['speed']
    power_roll_res = (coef_roll_res *
                      sp.constants.g * (cyclist_weight + bike_weight) *
                      speed)

    temperature_kelvin = sp.constants.convert_temperature(
        temperature, 'Celsius', 'Kelvin')
    # air density at 0 degree Celsius and a standard atmosphere
    molar_mass_dry_air = 28.97 / 1000  # kg.mol^-1
    standard_atmosphere = sp.constants.physical_constants[
        'standard atmosphere'][0]  # Pa
    zero_celsius_kelvin = sp.constants.convert_temperature(
        0, 'Celsius', 'Kelvin')  # 273.15 K
    air_density_ref = ((standard_atmosphere * molar_mass_dry_air) /
                       (sp.constants.gas_constant * zero_celsius_kelvin))
    air_density = air_density_ref * (
        (pressure * zero_celsius_kelvin) /
        (standard_atmosphere * temperature_kelvin))
    power_wind = 0.5 * air_density * surface_rider * drag_coef * speed**2

    slope = activity['gradient-elevation']
    power_gravity = ((cyclist_weight + bike_weight) * sp.constants.g *
                     np.sin(np.arctan(slope)) * speed)

    acceleration = activity['acceleration']
    power_acceleration = (cyclist_weight + bike_weight) * acceleration * speed

    return power_roll_res + power_wind + power_gravity + power_acceleration
