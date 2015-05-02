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

Creating a new SCAO configuration file
--------------------------------------

Now the simulation is working, lets start to simulate CANARY. We'll use the ``sh_8x8.py`` configuration file as a template, copy it to another file called ``CANARY_SCAO.py``,  and open this file in your favourite text editor. The configuration file contains all the parameters which determine the configuration of the simulated AO system. All the parameters are help in a Python dictionary, called ``simConfiguration``. Further to this, parameters are grouped into sub-dictionaries depending on which components they control. Descriptions of all possible parameters are given in the :ref:`configuration` section.


``Sim`` Parameters
^^^^^^^^^^^^^^^^^^

The first of these groups are parameters are those which have a system wide effect, so-called ``Sim`` parameters. The first parameter to change is the ``simName``, this is the directory where data will be saved during and after an AO run. Set it to ``CANARY_SCAO``. The ``logFile`` is the filename of a log which records all text output from the simulation, set it to ``CANARY_SCAO.log``. The value of ``loopTime`` specifies the frame rate of the simulation, which is usually, though not always, also the frame rate of the WFSs and DMs. More accurately though, it is the time between movements of the atmosphere. For CANARY, make the system run at 200Hz, so set this to ``0.005``, or ``1./200`` to be more explicit about the system frame rate. For the purposes of this tutorial, lets also set the number of iterations which will be run, ``nIters`` to around 500 so that it will run quickly. 

The ``Sim`` group also contains parameters which determine the data which will be stored and saved from the simulation. Set values to ``True`` if you'd like them to be continually saved in a memory buffer before being written to disk in a AO run specific, time-stamped directory within the ``simName`` directory.


``Atmosphere`` Parameters
^^^^^^^^^^^^^^^^^^^^^^^^^

As would be expected, this group of parameters describe the nature of the atmospheric turbulence. Currently, this configuration file features an atmosphere with 4 discrete turbulence layers, increase that to 5 by setting ``scrnNo`` to ``5``.  The ``r0`` parameter is the Fried parameter in metres and controls the integrated seeing strength, set this to ``0.14``. ``screenHeights``, ``scrnStrengths``, ``windDirs`` and ``windSpeed`` control the layer heights, relative C\ :sub:`N`\ :sup:`2` strengths, wind directions and wind velocities. These must be formatted as a list or array at least as long as ``scrnNo``, so add another value to each. 

Phase screens can be either created on each simulation run, or can be loaded from file. To load screens from file a parameter, ``scrnNames``, must be set with the filename of each phase screen in a list.

``Telescope`` Parameters
^^^^^^^^^^^^^^^^^^^^^^^^
The diameter of the simulated telescope and its central obscuration are determined by the ``telDiam`` and ``obsDiam`` parameters in the ``Telescope`` parameters. The ``mask`` value determines the shape if the pupil mask. If set to ``circle``, this will simple be a circular telescope pupil, with a circular obscuration cut out the centre. If something more complex is desired, this value should be set to a 2-dimensional numpy array of size ``(sim.pupilSize, sim.pupilSize)``, set to ``0`` or ``False`` at opaque parts of the pupil and ``1`` or ``True`` at transparent parts.

CANARY is hosted by the WHT, which is a 4.2 metre diameter telescope with a central obscuration of approximately 1.2 metres. Set these values, and keep ``mask`` set to ``circle``.

``WFS`` Parameters
^^^^^^^^^^^^^^^^^^
Each value in the ``WFS`` group of parameters must be a list or numpy array which is at least as long as ``sim.nGS``, which configures each of the wavefront sensors. For this configuration file, each parameter will be lists with only a single entry. Set ``nxSubaps``, the number of Shack-Hartmann sub-apertures in a single dimension to ``7`` and  ``pxlsPerSubap`` to ``14``. The pixel scale is defined by the parameter ``subapFOV``, which is actually the FOV of the entire sub-aperture, set this to ``2.5``.

As we are not initially going to use a LGS, leave the ``LGS`` parameters empty for now.

``DM`` Parameters
^^^^^^^^^^^^^^^^^

As with ``WFS`` parameters, each value describing the DM is formatted as a list or numpy array at least as long as ``sim.nDM``. As this configuration contains 2 DMs, each value must have 2 elements. The first DM will be a Tip-tilt mirror, hence the ``type`` is set to ``TT``. The second is a higher spatial order stack array type denoted in the simulation as ``Piezo``. These names correspond to classes which are defined in the ``DM.py`` module. Set the number of actuators in one dimension to 8, by setting the second value in ``nxActuators`` to ``8``.

``Science`` Parameters
^^^^^^^^^^^^^^^^^^^^^^

The final group of parameters which define the simulation are the ``Science`` parameters which define the science targets and detectors to be used to measure AO performance. Again, these are formatted as list or array at least as long as ``sim.nSci``




