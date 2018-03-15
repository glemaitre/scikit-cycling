.. _modeling:

.. currentmodule:: skcycling

==============================
Analysis of Cyclist Power Data
==============================

``scikit-cycling`` provides tools to analyze power data of a cyclist over time.

Cyclist record power-profile
----------------------------

The record power-profile is a data derivative computed from the power
information of several file. It is used to assess the potential and performance
of cyclist [P2011]_ and calibrate the training of cyclist.

Before to focus on the record power-profile and the facilities provided by
``scikit-cycling`` to provide this analysis, we will first define how to compute
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
DataFrame. However, it is quite tedious and ``scikit-learn`` solve this issue
with the class :class:`Rider`.

:class:`Rider` makes it possible to add and remove power-profile of activities
as well as compute record power-profile for a specific period of time during the
year::

  >>> from skcycling import Rider
  >>> rider = Rider()
  >>> rider.add_activities(load_fit())

Once, the power-profile for each activity is added, they can be accessed via the attributes ``rider.power_profile_`` which is a pandas DataFrame::

  >>> print(rider.power_profile_.head()) # doctest: +ELLIPSIS
                    2014-05-07 12:26:22  2014-05-11 09:39:38  \
  cadence 00:00:01            78.000...           100.000...   
          00:00:02            64.000...            89.000...   
          00:00:03            62.666...            68.333...   
          00:00:04            62.500...            59.500...   
          00:00:05            64.400...            63.200...   
  <BLANKLINE>
                    2014-07-26 16:50:56  
  cadence 00:00:01            60.000...  
          00:00:02            58.000...  
          00:00:03            56.333...  
          00:00:04            59.250...  
          00:00:05            61.000...

The record power-profile is computed such as::

  >>> record_power_profile = rider.record_power_profile()

Note that ``record_power_profile`` accepts two parameters ``range_dates`` and
``columns`` which limit to some dates or type of data the computation of the
record.

Store and load power-profile for a rider
........................................

The methods ``to_csv`` and ``from_csv`` allows to store and load a cyclist
power-profile.

Determination of the Maximum Power Aerobic
------------------------------------------

Using the record power-profile, Pinot et al. proposes a method to estimate the
Maximum Power Aerobic [P2014]_. The function :func:`metrics.aerobic_meta_model`
implements the algorithm::

  >>> from skcycling.metrics import aerobic_meta_model
  >>> mpa, t_mpa, aei, _, _ = aerobic_meta_model(rider.record_power_profile()) # doctest: +SKIP

Effort quantification based on power data
-----------------------------------------

During a ride, different scores can be computed to quantify the intensity of an
activity using the power data.

Maximum power aerobic and functional threshold power
....................................................

All those methods need an estimation of the
maximum power aerobic or alternatively the equivalent functional threshold
power. Both metrics are related such that:

.. math::
   MPA = FTP \times \frac{1}{0.76}

.. math::
   FTP = MPA \times 0.76

The functions :func:`metrics.mpa2ftp` and :func:`metrics.ftp2mpa` converts one
metric to another.

Normalized power score
......................

During a ride, it is common to have low power intensity during the ride which
reduce the average power. The normalized power is a metric which does not
under-estimate the average power by rejecting low power intensity (i.e. < 30%
of the maximum power aerobic) and smoothing the power before to compute the
average such as

.. math::
   NPS = \left( \frac{1}{N} \sum_{n=1}^{N} p_{n}^{4} \right)^{\frac{1}{4}}

Intensity factor
................

The intensity factor is defined as the normalized power score normalized by the
functional threshold power such as

.. math::
   IF = \frac{NPS}{FP}

Training stress score
.....................

There is two definitions of the training stress score. The first one is based
on the intensity factor and it is defined as

.. math::
   TSS = \frac{100 \times N \times IF^{2}}{3600}

The second definition 

.. topic:: References

   .. [P2011] Pinot, J., and F. Grappe. "The record power profile to assess
      performance in elite cyclists." International journal of sports medicine
      32.11 (2011): 839-844.

   .. [P2014] Pinot, J., and F. Grappe. "Determination of Maximal Aerobic Power
      from the Record Power Profile to improve cycling training." Journal of
      Science and Cycling 3.1 (2014): 26.
