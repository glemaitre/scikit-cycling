####################
`scikit-cycling` API
####################

This is the full API documentation of `scikit-cycling`

.. _data_management_ref:

Data-management
===============

.. currentmodule:: skcycling

.. autosummary::
   :toctree: generated/

   Rider

.. _extraction_ref:

Extraction
==========

.. automodule:: skcycling.extraction
    :no-members:
    :no-inherited-members:

.. currentmodule:: skcycling

.. autosummary::
   :toctree: generated/
   :template: function.rst

   extraction.activity_power_profile
   extraction.acceleration
   extraction.gradient_activity
   extraction.gradient_elevation
   extraction.gradient_heart_rate

.. _metrics_ref:

Extraction
==========

.. automodule:: skcycling.metrics
    :no-members:
    :no-inherited-members:

.. currentmodule:: skcycling

Single cycling activity
-----------------------

.. autosummary::
   :toctree: generated/
   :template: function.rst

   metrics.normalized_power_score
   metrics.intensity_factor_ftp_score
   metrics.intensity_factor_mpa_score
   metrics.training_stress_ftp_score
   metrics.training_stress_mpa_score
   metrics.training_stress_mpa_grappe_score
   metrics.training_stress_ftp_grappe_score

Power-profile
-------------

.. autosummary::
   :toctree: generated/
   :template: function.rst

   metrics.aerobic_meta_model

.. _utils_ref:

Utilities
=========

.. automodule:: skcycling.utils
    :no-members:
    :no-inherited-members:

.. currentmodule:: skcycling

.. autosummary::
   :toctree: generated/
   :template: function.rst

   utils.validate_filenames

.. _io_ref:

IO interaction
==============

.. automodule:: skcycling.io
    :no-members:
    :no-inherited-members:

.. currentmodule:: skcycling

.. autosummary::
   :toctree: generated/
   :template: function.rst

   io.bikeread

.. _datasets_ref:

Datasets
========

.. automodule:: skcycling.datasets
    :no-members:
    :no-inherited-members:

.. currentmodule:: skcycling

.. autosummary::
   :toctree: generated/
   :template: function.rst

   datasets.load_fit
   datasets.load_rider
