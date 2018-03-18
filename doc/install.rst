Getting Started
===============

Install
-------

``scikit-cycling`` is currently available on the PyPi’s reporitories and you can
install it via pip::

  pip install -U scikit-cycling

The package is release also in conda-forge::

  conda install -c conda-forge scikit-cycling

If you prefer, you can clone it and run the ``setup.py`` file. Use the
following commands to get a copy from Github and install all dependencies::

  git clone https://github.com/scikit-cycling/scikit-cycling.git
  cd scikit-cycling
  pip install .

Or install using ``pip`` and GitHub::

  pip install -U git+https://github.com/scikit-cycling/scikit-cycling.git

Test and coverage
-----------------

You want to test the code before to install::

  $ make test

You wish to test the coverage of your version::

  $ make coverage

Contribute
----------

You can contribute to this code through Pull Request on GitHub_. Please, make
sure that your code is coming with unit tests to ensure full coverage and
continuous integration in the API.

.. _GitHub: https://github.com/glemaitre/scikit-cycling
