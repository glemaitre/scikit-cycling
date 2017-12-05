"""
The :mod:`skcycling.datasets` module includes utilities to load datasets.
Mainly this is used for performing the tests of this package.
"""
from .toy import load_toy
from .toy import load_toy_rider

__all__ = ['load_toy',
           'load_toy_rider']
