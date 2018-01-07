from cython cimport floating


cpdef double max_mean_power_interval(floating[:] activity_power,
                                     Py_ssize_t time_interval)
