"""
The :mod:`skcycling.model` module includes algorithms to model cycling data.
"""

# Authors: Guillaume Lemaitre <g.lemaitre58@gmail.com>
#          Cedric Lemaitre
# License: BSD 3 clause

from .power import strava_power_model

__all__ = ['strava_power_model']
