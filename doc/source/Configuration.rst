.. _configuration:

Configuration
*************

Configuration of the system is handled by the ``confParse`` module, that reads the simulation parameters from a given configuration file. This file should be a YAML file, which contains groups for each simulation sub-module. Where a sub-module may consist of multiple components i.e. Wave-front sensors, each WFS must be specified seperately, with an integer index, for example::

		WFS:
			0:
				GSMag: 0
				GSPosition: (0, 0)
			1:
				GSMag: 1
				GSPosition: (1, 0)

Example configuration files can be found in the ``conf`` directory of the soapy package.
(Note: Previously, a Python file was used for configuration. This format is still supported but can lead to messy configuration files! There are still examples of these in the source repository if you prefer.)

Below is a list of all possible simulation parameters. Parameters which have a description ending in \** can be altered while the simulation is running. When others are changed and ``aoinit`` must be run before they will take effect and they may break a running simulation.

Simulation Parameters
---------------------
.. autoclass:: soapy.confParse.SimConfig
	:members:

Telescope Parameters
--------------------

.. autoclass:: soapy.confParse.TelConfig
	:members:

Atmosphere Parameters
---------------------

.. autoclass:: soapy.confParse.AtmosConfig
	:members:

Wave-front Sensor Parameters
----------------------------

.. autoclass:: soapy.confParse.WfsConfig
	:members:

Laser Guide Star Parameters
---------------------------

.. autoclass:: soapy.confParse.LgsConfig
	:members:

Deformable Mirror Parameters
----------------------------

.. autoclass:: soapy.confParse.DmConfig
	:members:

Science Camera Parameters
-------------------------

.. autoclass:: soapy.confParse.SciConfig
	:members:
