Simple Tutorial
===============

This tutorial will go through some example AO systems using Soapy. We'll see how to make configuration files to run the AO system that you'd like to, then extract data which can be subsequently analysed. CANARY is an AO system on the 4.2m William Herschel Telescope on La Palma. It is designed to be very flexible to run various "modes" of AO, so makes a nice test bed for us to simulate. We'll simulate it in SCAO mode, in GLAO with multiple guide-stars and in SCAO with a LGS.


Running an Example Configuration
--------------------------------

Before making new configuration files though, its a pretty good idea to make sure everything is working as expected by running one of the examples. First, lets create a directory where we do this tutorial, call it something like ``soapy_tutorial``, make a further directory called ``conf`` inside and copy the example configuration file ``sh_8x8.py`` form the downloaded or cloned Soapy directory into it.

To open the Graphical User Interface (GUI), type in the command line::
    
    soapy --gui conf/sh_8x8.py

This relies on ``soapy`` being in you're ``PATH``. If thats not the case, run::

    python <path/to/soapy>/bin/soapy --gui conf/sh8x8.py

You should see a window which looks a bit like this pop up:

.. image:: imgs/annotatedGUI.png
    :align: center

If you don't want to run the GUI, then open a python terminal and run::

    import soapy
    sim = soapy.Sim("conf/sh8x8.py")

Before the simulation can be started, some initialisation routines must be run. If running the GUI, then this will automatically when you start it up. In the command line, to initialise run::

    sim.aoinit()

Next, the interaction matrixes between the DMs and the WFSs. In the GUI this is achieved by clicking "makIMat", and in the command line with::

    sim.makeIMat()

This simulation will save command matrices, interaction matrices and DM influence functions for a simulation, so that it doesn't alway have to remake them. If you'd like to override the loading them from file and make them from scratch, tick the "force new" button in the GUI, or pass the argument ``forceNew=True`` to the ``makeIMat`` command.

To actually run the simulation, click "aoloop" in the GUI, or type::
    
    sim.aoloop()

at the command line. This will run the simulation for the configured number of iterations, and estimate the performance of the specified AO system.

Creating a new configuration file
---------------------------------

Now the simulation is working, lets start to simulate CANARY. We'll use the ``sh_8x8.py`` configuration file as a template, copy it to another file called ``CANARY_SCAO.py``,  and open this file in your favourite text editor. The configuration file contains all the parameters which determine the configuration of the simulated AO system. All the parameters are help in a Python dictionary, called ``simConfiguration``. Further to this, parameters are grouped into sub-dictionaries depending on which components they control. Descriptions of all possible parameters are given in the :ref:`configuration` section.


``Sim`` Parameters
^^^^^^^^^^^^^^^^^^

The first of these groups are parameters are those which have a system wide effect, so-called ``Sim`` parameters. The first parameter to change is the ``simName``, this is the directory where data will be saved during and after an AO run. Set it to ``CANARY_SCAO``. The ``logFile`` is the filename of a log which records all text output from the simulation, set it to ``CANARY_SCAO.log``. The value of ``loopTime`` specifies the frame rate of the simulation, which is usually, though not always, also the frame rate of the WFSs and DMs. More accurately though, it is the time between movements of the atmosphere. For CANARY, make the system run at 200Hz, so set this to ``0.005``, or ``1./200`` to be more explicit about the system frame rate. For the purposes of this tutorial, lets also set the number of iterations which will be run, ``nIters`` to around 500 so that it will run quickly. 

The ``Sim`` group also contains parameters which determine the data which will be stored and saved from the simulation. Set values to ``True`` if you'd like them to be continually saved in a memory buffer before being written to disk in a AO run specific, time-stamped directory within the ``simName`` directory.


``Atmosphere`` Parameters
^^^^^^^^^^^^^^^^^^^^^^^^^