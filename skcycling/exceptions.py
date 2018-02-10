"""Module containing the exceptions used in scikit-cycling."""

# Authors: Guillaume Lemaitre <g.lemaitre58@gmail.com>
#          Cedric Lemaitre
# License: BSD 3 clause

__all__ = ['MissingDataError']


class MissingDataError(ValueError):
    """Error raised when there is not the required data to make some
    computation.

    For instance, :func:`skcycling.extraction.gradient_elevation` required
    elevation and distance data which might not be provided. In this case, this
    type of error is raised.

    """
