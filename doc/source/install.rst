Installation
************

Firstly, you'll need Python. This comes with pretty much every linux distribution and is also installed by default on Mac OS X. For Windows, I'd recommend using a Python distribution, such as anaconda or enthought canopy. The code should be compatible with Windows as Python and all the libraries used are platform independent. Nonetheless, I only use it on Mac OSX and Linux so I can't guarantee that there won't be bugs not present on other OSs.


=====================
Required Libraries
=====================
pyAOS doesn't have too many requirements in terms of external libraries, though it does rely on some. There are also some optional libraries which are recommended for plotting or performance.

--------
Required
--------

::

    numpy
    scipy
    pyfits *or* astropy

-----------    
Recommended
-----------

^^^^^^^^^^^^^^^
for performance
^^^^^^^^^^^^^^^
::

    pyfftw (Highly Recommended!)

^^^^^^^    
for gui
^^^^^^^

::
    
    PyQt4
    pyqtgraph (http://www.pyqtgraph.org)
    matplotlib
    ipython
    

=====
Linux
=====
If your starting with python from scratch, there a couple of options. For Ubuntu (and probably debian) linux users, all these packages can be installed via apt-get::
    
    sudo apt-get install python-numpy python-scipy python-fftw python-pyfits python-qt4 python-matplotlib ipython
    
for pyqtgraph, go to http://www.pyqtgraph.org and download the .deb file
    
for Red-hat based systems these packages should also be available from repositories, though I'm not sure of they're names. Again, get pyqtgraph from http://www.pyqtgraph.org, but download the source. pyqtgraph, like most python packages is pretty easy to install from source, just download the package, unpack, navigate into the package and run ``sudo python setup.py install``
    
=======
Mac OSX
=======

for mac os, all of these packages can be install via macports, with::
    
    sudo port install python27 py27-numpy py27-scipy py27-pyfits py27-pyfftw py27-pyqt4 py27-ipython

again, pyqtgraph must be downloaded from http://www.pyqtgraph.org and installed with ``sudo python setup.py install``

=======
Windows
=======

For any OS, including Windows, python distributions exist which include lots of python packages useful for science. A couple of good examples are Enthought Canopy (https://www.enthought.com), which is free for academics, and Anaconda (https://store.continuum.io/cshop/anaconda/) which is also free.


======
Any OS
======
A lot of python packages are listed on `pypi <https://pypi.python.org/pypi>`_. Usually when python is installed, a script called ``easy_install`` is installed also, which can be used to get any package on pypi with ``easy_install <package>``. Confusingly, ``pip`` is now the recommended Python package manager instead of ``easy_install``. If you've only got ``easy_install`` you can install ``pip`` using ``easy_install pip``, or it can be installed using the script linked `here <https://pip.readthedocs.org/en/latest/installing.html>`_.

Once you have ``pip``, the required libraries can be installed by using the ``requirements.txt`` file. From the pyAOS directory, just run (may need to be as ``sudo``)::

    pip -r requirements.txt
    
and all the requirements should be installed for you!

=======
Testing
=======
Once you think everything is installed, tests can be run by navigating to the ``test`` directory and running ::
    
    python testSimulation.py

Currenlty, this only runs system wide tests, but further, more atomic tests will be added in future.
