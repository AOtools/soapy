.. _configuration:

Configuration
*************

Configuration of the system is handled by the ``confParse`` module, that reads the simulation parameters from a given configuration file. This file must contain a ``simConfiguration`` dictionary, which contains sub-dictionaries for each simulation sub-module. Where a sub-module consists of multiple components i.e. Wave-front sensors, parameters must be given as lists at least as long as the number of components. Example configuration files can be found in the ``conf`` directory of the soapy package.

Below is a list of all possible simulation parameters.

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
