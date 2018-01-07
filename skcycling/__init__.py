"""Cycling Processing Toolbox (Toolbox for SciPy)

``scikit-cycling`` (a.k.a ``skcycling``) is a set of python methods to
analyse file extracted from powermeters.

Subpackages
-----------
data_management
    Class to manage the data for a rider.
datasets
    Modules with helper for datasets.
metrics
    Metrics to quantify cyclist ride.
power_profile
    Record power-profile of cyclist.
restoration
    Utility for denoising cyclist ride.
utils
    Utility to read and save cycling ride.
"""

from ._version import __version__

from . import __check_build
from .base import Rider
