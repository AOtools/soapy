.. _dataSources:

Data Sources
============

In this section, the data sources which are stored in soapy are listed and a description of how they are obtained is given.


Simulation Run Data
-------------------
The following sources of data are recorded for each simulation run and are saved as a fits file in a time stamped run specific directory inside the ``simName`` directory. They can be accessed by ``sim.<data>``, where ``<data>`` is listed in the  "Internal data structure" column. As the storing of some of these data sources can increase  memory usage significantly, they are not all saved by default, and the flag must be set in the configuration file.

+-------------+-------------------+------------------+-------------------------+
|Data         | Saved filename    |Internal data     |Description              |
|             |                   |structure         |                         |
+=============+===================+==================+=========================+
|Instantaneous|``instStrehl.fits``|``instStrehl``    |The instantaneous        |
|Strehl ratio |                   |                  |strehl ratio for         |
|             |                   |                  |each science target      |
|             |                   |                  |frame                    |
+-------------+-------------------+------------------+-------------------------+
|Long exposure|``longStrehl.fits``|``longStrehl``    |The long exposure        |
|Strehl ratio |                   |                  |strehl ratio for         |
|             |                   |                  |each science target      |
|             |                   |                  |frame                    |
+-------------+-------------------+------------------+-------------------------+
|Wavefront    |``WFE.fits``       |``WFE``           |The corrected wave-      |
|Error        |                   |                  |front error for each     |
|             |                   |                  |science target in nm     |
+-------------+-------------------+------------------+-------------------------+
|Science PSF  |``sciPsf_n.fits``  |``sciImgs[n]``    |The science camera PSFs  |
|             |                   |                  |where ``n`` indicates the|
|             |                   |                  |camera number            |
+-------------+-------------------+------------------+-------------------------+
|Residual     |``sciResidual_n    |``sciPhase[n]``   |The residual uncorrected |
|Science phase|.fits``            |                  |phase across science     |
|             |                   |                  |target ``n``             |
+-------------+-------------------+------------------+-------------------------+
|WFS          |``slopes.fits``    | ``allSlopes``    |All WFS measurements     |
|measurements |                   |                  |stored in a numpy        |
|             |                   |                  |array of size            |
|             |                   |                  |(nIters, totalSlopes)    |
+-------------+-------------------+------------------+-------------------------+
|WFS Frames   |``wfsFPFrames/     |``sim.wfss[n].    |WFS detector image, only |
|             |wfs-n_frame-i      |wfsDetectorPlane``|last frame stored        |
|             |.fits``            |                  |in memory. Can save each |
|             |                   |                  |frame, ``i``, from wfs   |
|             |                   |                  |``n``                    |
+-------------+-------------------+------------------+-------------------------+
|DM Commands  |``dmCommands.fits``|``allDmCommands`` |DM commands for all      |
|             |                   |                  |DMs present in numpy     |
|             |                   |                  |of size                  |
|             |                   |                  |(nIters, totaldmCommands)|
+-------------+-------------------+------------------+-------------------------+
