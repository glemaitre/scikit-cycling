"""Cycling Processing Toolbox (Toolbox for SciPy)

``scikit-cycling`` (a.k.a ``skcycling``) is a set of python methods to
analyse file extracted from powermeters.

Subpackages
-----------
datasets
    Modules with helper for datasets.
metrics
    Metrics to quantify cyclist ride.
power_profile
    Record power-profile of cyclist.
utils
    Utility to read and save cycling ride.
"""

from ._version import __version__

from . import __check_build
from .base import Rider
