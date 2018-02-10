"""
The :mod:`skcycling.extraction` module includes algorithms to extract
information from cycling data.
"""

# Authors: Guillaume Lemaitre <g.lemaitre58@gmail.com>
#          Cedric Lemaitre
# License: BSD 3 clause

from .gradient import acceleration
from .gradient import gradient_elevation

from .power_profile import activity_power_profile


__all__ = ['acceleration',
           'gradient_elevation',
           'activity_power_profile']
