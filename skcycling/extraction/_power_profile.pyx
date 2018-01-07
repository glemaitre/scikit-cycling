#cython: cdivision=True
#cython: boundscheck=False
#cython: nonecheck=False
#cython: wraparound=False

# Authors: Guillaume Lemaitre <g.lemaitre58@gmail.com>
#          Cedric Lemaitre
# License: BSD 3 clause

cpdef double max_mean_power_interval(floating[:] activity_power,
                                     Py_ssize_t time_interval):
    """Compute the maximum power delivered for a specific amount of time.

    Parameters
    ----------
    activity_power : ndarray, shape (n_data_point,)
        The power data of the activity.

    time_interval : int
        The time interval for which we compute the mean power.

    Returns
    -------
    max_mean : double
        The maximum power delivered for a specific amount of time.

    """

    cdef:
        Py_ssize_t n_element = activity_power.shape[0]
        Py_ssize_t idx_element, idx_interval
        double acc = 0.0
        double max_mean = 0.0

    with nogil:
        for idx_element in range(n_element - time_interval):
            acc = 0.0
            for idx_interval in range(time_interval):
                acc += activity_power[idx_element + idx_interval]
            if acc > max_mean:
                max_mean = acc

        return max_mean / time_interval
