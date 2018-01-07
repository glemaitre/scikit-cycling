"""
The :mod:`skcycling.extraction` module includes algorithms to extract
information from cycling data.
"""

# Authors: Guillaume Lemaitre <g.lemaitre58@gmail.com>
#          Cedric Lemaitre
# License: BSD 3 clause

from .power_profile import activity_power_profile

__all__ = ['activity_power_profile']
