#Simulation 'Optique Adaptative' with Python
(formerly PyAOS)

[![Build Status](https://travis-ci.org/AOtools/soapy.svg?branch=master)](https://travis-ci.org/AOtools/soapy))
[![Build status](https://ci.appveyor.com/api/projects/status/ea65yv0p7s32ejxx/branch/master?svg=true)](https://ci.appveyor.com/project/Soapy/soapy/branch/master)
[![codecov](https://codecov.io/gh/AOtools/soapy/branch/master/graph/badge.svg)](https://codecov.io/gh/AOtools/soapy)
[![Documentation Status](http://readthedocs.org/projects/soapy/badge/?version=latest)](http://soapy.readthedocs.io/en/latest/?badge=latest)


## Introduction


Soapy is a Monte-Carlo Adaptive Optics Simulation toolkit written in Python. soapy can be used as a conventional end-to-end simulation, where a large number of AO configurations can be created simply by editing a configuration file. Its real power lays in modular nature of objects, such as WFSs, DMs and reconstructors which can be taken and used as building blocks to construct new and complex AO ideas and configurations.

Please keep in mind that soapy is very much a work-in-progress and under heavy development. I've not yet settled on a completely stable API, but I will try and say when something big has changed. **For these reasons I would strongly reccomend against using soapy for critical work and would suggest contacting me to discuss its suitability for any work to be published.**

There is documentation at http://soapy.readthedocs.io/en/latest/, again this is also being developed at this time!

## Quick-Start


Try out some of the code examples in the ``conf`` directory, either run the ``soapy`` script in ``bin``, or load a python or IPython terminal: 

    import soapy
    sim = soapy.Sim("configFilename")
    sim.aoinit()
    sim.makeIMat()
    sim.aoloop()
    
All the data from the simulation exists in the ``sim`` object, the data available will depend upon parameters set in the configuration file. e.g. Slopes can be accessed by ``sim.allSlopes``.

## Required Libraries
Soapy doesn't have too many requirements in terms of external libraries, though it does rely on some. There are also some optional libraries which are recommended for plotting or performance.

### Required

    numpy => 1.7.0
    scipy => 0.10
    pyfits *or* astropy
    pyfftw
    
### For GUI

    PyQt5 (attempted to be compatibile with PyQt4 but not guaranteed)
    pyqtgraph (http://www.pyqtgraph.org)
    matplotlib
    ipython

If your starting with python from scratch, there a couple of options. For Ubuntu linux (14.04+) users, all these packages can be installed via apt-get:
    
    sudo apt-get install python-numpy python-scipy python-pyfftw python-astropy python-qt4 python-matplotlib ipython ipython-qtconsole python-pyqtgraph

For Red-hat based systems these packages should also be available from repositories, though I'm not sure of their names. 
    
for mac os, all of these packages can be install via macports, with 
    
    sudo port install python27 py27-numpy py27-scipy py27-astropy py27-pyfftw py27-pyqt4 py27-ipython py27-pyqtgraph py27-jupyter

    
For any OS (including Windows), python distributions exist which include lots of python packages useful for science. A couple of good examples are Enthought Canopy (https://www.enthought.com), which is free for academics, and Anaconda (https://store.continuum.io/cshop/anaconda/) which is also free.

A lot of python packages are listed on https://pypi.python.org/pypi. Usually when python is installed, a script called ``easy_install`` is installed also, which can be used to get any package on pypi with ``easy_install <package>``. Pip is a more recent python package manager which is currently reccommended for use, which can be found either through your system package manager or with ``easy_install pip``. Then to install packages use ``pip install <package>``.


