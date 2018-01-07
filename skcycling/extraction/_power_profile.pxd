# Authors: Guillaume Lemaitre <g.lemaitre58@gmail.com>
#          Cedric Lemaitre
# License: BSD 3 clause

from cython cimport floating


cpdef double max_mean_power_interval(floating[:] activity_power,
                                     Py_ssize_t time_interval)
