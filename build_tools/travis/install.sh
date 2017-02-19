#!/bin/bash
# This script is meant to be called by the "install" step defined in
# .travis.yml. See http://docs.travis-ci.com/ for more details.
# The behavior of the script is controlled by environment variabled defined
# in the .travis.yml in the top level folder of the project.

# License: 3-clause BSD

# Travis clone scikit-learn/scikit-learn repository in to a local repository.
# We use a cached directory with three scikit-learn repositories (one for each
# matrix entry) from which we pull from local Travis repository. This allows
# us to keep build artefact for gcc + cython, and gain time

set -e

# Fix the compilers to workaround avoid having the Python 3.4 build
# lookup for g++44 unexpectedly.
export CC=gcc
export CXX=g++

echo 'List files from cached directories'
echo 'pip:'
ls $HOME/.cache/pip


if [[ "$DISTRIB" == "conda" ]]; then
    # Deactivate the travis-provided virtual environment and setup a
    # conda-based environment instead
    deactivate

    # Use the miniconda installer for faster download / install of conda
    # itself
    pushd .
    cd
    mkdir -p download
    cd download
    echo "Cached in $HOME/download :"
    ls -l
    echo
    wget http://repo.continuum.io/miniconda/Miniconda-3.6.0-Linux-x86_64.sh \
         -O miniconda.sh
    chmod +x miniconda.sh && ./miniconda.sh -b
    cd ..
    export PATH=/home/travis/miniconda/bin:$PATH
    conda update --yes conda
    popd

    conda create -n testenv --yes python=$PYTHON_VERSION pip nose six \
          numpy=$NUMPY_VERSION scipy=$SCIPY_VERSION
    source activate testenv

    # Install nose-timer via pip
    pip install nose-timer
fi

if [[ "$COVERAGE" == "true" ]]; then
    pip install coverage coveralls
fi

# check the version that have been installed
python --version
python -c "import numpy; print('numpy %s' % numpy.__version__)"
python -c "import scipy; print('scipy %s' % scipy.__version__)"

cd $TRAVIS_BUILD_DIR/third-party/python-fitparse
python setup.py install
cd $TRAVIS_BUILD_DIR
python setup.py develop
