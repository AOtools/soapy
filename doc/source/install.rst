Installation
************

Firstly, you'll need Python. This comes with pretty much every linux distribution and is also installed by default on Mac OS X. For Windows, I'd recommend using a Python distribution, such as anaconda or enthought canopy. The code should be compatible with Windows as Python and all the libraries used are platform independent. Nonetheless, I only use it on Mac OSX and Linux so I can't guarantee that there won't be bugs not present on other OSs.

============
Installation
============
Once all the requirements outlined below are met, you are ready to install Soapy. Download the source from `github <https://github.com/andrewpaulreeves/soapy>`_, either as a zip file, or clone the git repository with::

    git clone https://github.com/soapy/soapy.git

If downloading the code as a zip, you can choose which version to use with the drop down box on the left of the page, entitled ``branch:master``. Whilst I try not to, the master branch will occasionally be broken so you might want to get the latest stable version by clicking "tags" in the dropdown list, and selecting the most recent version number.

Once the code is downloaded (and unzipped) or cloned, navigate to the resulting directory using the command line. You can import it into python straight away from this directory. To use the ``soapy`` script, run::

    python soapy <options> <configfile>


If you wish to have it available elsewhere on your system, either set the relavant ``PATH`` and ``PYTHONPATH`` variables to ``<soapy dir>/bin`` and ``<soapy dir>/`` respectively, or run the install script with::

    python setup.py install

This latter method may require superuser permissions for your system and should setup the paths for you. You should now be able to run ``soapy`` and import soapy into python from any directory on your system.

==================
Required Libraries
==================

Soapy doesn't have too many requirements in terms of external libraries, though it does rely on some. Performance of the simulation is made reasonable (for ELT scale operation) by using pyfftw and the numba library. Pyfftw simply wraps the FFTW library for fast fourier transforms. Numba, is a clever library that leverages the LLVM compiler infrastructure to compile python code directly to machine code. A library of functions has been written for the most computationally challenging algorithms, which are in pure python so can be easily read and improved, but operate quickly with the option of using multiple threads.  There are also some optional libraries which are recommended for plotting.

--------
Required
--------

::

    numpy
    scipy
    astropy
    pyfftw
    numba
    yaml

-------    
For GUI
-------
::

    PyQt5 (PyQt4 supported)
    matplotlib
    ipython
    

=====
Linux
=====
If your starting with python from scratch, there a couple of options. For Ubuntu (14.04+) linux users, all these packages can be installed via apt-get::

    sudo apt-get install python-numpy python-scipy python-fftw python-astropy python-qt4 python-matplotlib ipython ipython-qtconsole python-yaml python-numba


for Red-hat based systems these packages should also be available from repositories, though I'm not sure of they're names.


=======
Mac OSX
=======

for mac os, all of these packages can be install via macports, with::

    sudo port install python36 py36-numpy py36-scipy py36-astropy py36-pyqt5 py36-ipython py36-jupyter py36-numba py36-yaml py36-qtconsole

`pyfftw <https://github.com/pyFFTW/pyFFTW>`_ is not available for python3.6 on macports, so must be installed with another method, such as pip (see below)

If you're using Python 2.7::

    sudo port install python27 py27-numpy py27-scipy py27-astropy py27-pyfftw py27-pyqt5 py27-ipython py27-jupyter py27-numba py27-qtconsole py27-yaml


======
Any OS
======

---------------
Anaconda Python
---------------
For any OS, including Windows, python distributions exist which include lots of python packages useful for science.
A couple of good examples are Enthought Canopy (https://www.enthought.com), which is free for academics, and Anaconda (https://store.continuum.io/cshop/anaconda/) which is also free.
Anaconda includes most of the required libraries by default apart from pyfftw and pyyaml. These can be installed with::

    conda install pyyaml
    pip install pyfftw


---
pip
---

A lot of python packages are also listed on `pypi <https://pypi.python.org/pypi>`_. Usually when python is installed, a script called ``easy_install`` is installed also, which can be used to get any package on pypi with ``easy_install <package>``. Confusingly, ``pip`` is now the recommended Python package manager instead of ``easy_install``. If you've only got ``easy_install`` you can install ``pip`` using ``easy_install pip``, or it can be installed using the script linked `here <https://pip.readthedocs.org/en/latest/installing.html>`_.

Once you have ``pip``, the required libraries can be installed by using the ``requirements.txt`` file. From the soapy directory, just run (may need to be as ``sudo``)::

    pip install numpy scipy astropy pyfftw pyyaml numba
    
and all the requirements should be installed for the simulation, though not the GUI. For the GUI PyQt4 or PyQt5 is required, I dont think these are available from pip.

Sometimes pyfftw has a hard time finding your installation of fftw to link against. On a Mac, these lines usually help before running the pip command::

    export DYLIB_LIBRARY_PATH=$DYLIB_LIBRARY_PATH:<path/to/fftw>/lib
    export LDFLAGS=-L<path/to/fftw>/lib
    export CFLAGS=-I<path/to/fftw>/include/

=======
Testing
=======
Once you think everything is installed, tests can be run by navigating to the ``test`` directory and running::

    python testSimulation.py

Currently, this only runs system wide tests, but further, more atomic tests will be added in future. To run the tests, soapy must be either "installed", or manually put into the PYTHONPATH.
