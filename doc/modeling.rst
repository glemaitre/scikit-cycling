.. _modeling:

.. currentmodule:: skcycling

==============================
Analysis of Cyclist Power Data
==============================

`scikit-cycling` provides tools to analyze power data of a cyclist over time.

Cyclist record power-profile
----------------------------

The record power-profile is a data derivative computed from the power
information of several file. It is used to assess the potential and performance
of cyclist [P2011]_ and calibrate the training of cyclist.

Before to focus on the record power-profile and the facilities provided by
`scikit-cycling` to provide this analysis, we will first define how to compute
a power-profile.

Power-profile for a single activity
...................................

A power-profile is computed for an activity and it is computed by taking the
maximum power delivered for different amount of time (e.g. from 1 seconds to 3
hours). The function :func:`extraction.activity_power_profile` computes this
profile for a given max duration::

  >>> from skcycling.datasets import load_fit
  >>> from skcycling.io import bikeread
  >>> from skcycling.extraction import activity_power_profile
  >>> ride = bikeread(load_fit()[0], drop_nan='columns')
  >>> power_profile = activity_power_profile(ride, max_duration='00:08:00')
  
Record power-profile
....................

The record power-profile of a cyclist is computed by taking the maximum of the
all available power-profile of activities for the different duration. It would
be possible to concatenate all pandas Series returned by the function
:func:`extraction.activity_power_profile` and create a pandas
DataFrame. However, it is quite tedious and `scikit-learn` solve this issue
with the class :class:`Rider`.

:class:`Rider` makes it possible to add and remove power-profile of activities
as well as compute record power-profile for a specific period of time during the
year::

  >>> from skcycling import Rider
  >>> rider = Rider()
  >>> rider.add_activities(load_fit())

Once, the power-profile for each activity is added, they can be accessed via the attributes `rider.power_profile_` which is a pandas DataFrame::

  >>> print(rider.power_profile_.head())
                    2014-05-07 12:26:22  2014-05-11 09:39:38  \
  cadence 00:00:01            78.000000           100.000000   
          00:00:02            64.000000            89.000000   
          00:00:03            62.666667            68.333333   
          00:00:04            62.500000            59.500000   
          00:00:05            64.400000            63.200000   
  <BLANKLINE>
                    2014-07-26 16:50:56  
  cadence 00:00:01            60.000000  
          00:00:02            58.000000  
          00:00:03            56.333333  
          00:00:04            59.250000  
          00:00:05            61.000000

The record power-profile is computed such as::

  >>> record_power_profile = rider.record_power_profile()

Note that `record_power_profile` accepts two parameters `range_dates` and
`columns` which limit to some dates or type of data the computation of the
record.

Store and load power-profile for a rider
........................................

The methods `to_csv` and `from_csv` allows to store and load a cyclist
power-profile.
  
.. topic:: References

.. [P2011] Pinot, J., and F. Grappe. "The record power profile to assess
   performance in elite cyclists." International journal of sports medicine
   32.11 (2011): 839-844.


Determination of the Maximum Power Aerobic
------------------------------------------

Effort quantification based on power data
-----------------------------------------


