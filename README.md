# Introduction


pyAOS is a Monte-Carlo Adaptive Optics Simulation toolkit written in Python. pyAOS can be used as a conventional end-to-end simulation, where a large number of AO configurations can be created simply by editing a configuration file. Its real power lays in modular nature of objects, such as WFSs, DMs and reconstructors which can be taken and used as building blocks to construct new and complex AO ideas and configurations.

Please keep in mind that pyAOS is very much a work-in-progress and under heavy development. I've not yet settled on a completely stable API, but I will try and say when something big has changed. **For these reasons I would strongly reccomend against using pyAOS for critical work and would suggest contacting me to discuss its suitability for any work to be published.**

There is documentation at http://pyaos.readthedocs.org/en/latest/index.html, again this is also being developed at this time!

## Quick-Start


Try out some of the code examples in the ``conf`` directory, either run the ``pyAOS`` script in ``bin``, or load a python or IPython terminal: 

    import pyAOS
    sim = pyAOS.Sim("configFilename")
    sim.aoinit()
    sim.makeIMat()
    sim.aoloop()
    
All the data from the simulation exists in the ``sim`` object, the data available will depend upon parameters set in the configuration file. e.g. Slopes can be accessed by ``sim.allSlopes``.

## Required Libraries
pyAOS doesn't have too many requirements in terms of external libraries, though it does rely on some. There are also some optional libraries which are recommended for plotting or performance.

### Required

    numpy 
    scipy
    pyfits *or* astropy
    
### Recommended
    
#### for performance:
    pyfftw (Highly Recommended!)
    
#### for gui
    PyQt4
    pyqtgraph (http://www.pyqtgraph.org)
    matplotlib
    ipython
    

If your starting with python from scratch, there a couple of options. For Ubuntu (and probably debian) linux users, all these packages can be installed via apt-get:
    
    sudo apt-get install python-numpy python-scipy python-pyfftw python-pyfits python-qt4 python-matplotlib ipython ipython-qtconsole python-pyqtgraph
    
for pyqtgraph, go to http://www.pyqtgraph.org and download the .deb file
    
for Red-hat based systems these packages should also be available from repositories, though I'm not sure of they're names. Again, get pyqtgraph from http://www.pyqtgraph.org, but download the source. pyqtgraph, like most python packages is pretty easy to install from source, just download the package, unpack, navigate into the package and run ``sudo python setup.py install``
    
for mac os, all of these packages can be install via macports, with 
    
    sudo port install python27 py27-numpy py27-scipy py27-pyfits py27-pyfftw py27-pyqt4 py27-ipython py27-pyqtgraph

    
For any OS (including Windows), python distributions exist which include lots of python packages useful for science. A couple of good examples are Enthought Canopy (https://www.enthought.com), which is free for academics, and Anaconda (https://store.continuum.io/cshop/anaconda/) which is also free.

A lot of python packages are listed on https://pypi.python.org/pypi. Usually when python is installed, a script called ``easy_install`` is installed also, which can be used to get any package on pypi with ``easy_install <package>``.


