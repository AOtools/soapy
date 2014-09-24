Configuration
*************

Configuration of the system is handled by the ``confParse`` module. This module reads the simulation parameters from a given configuration file. This file must contain a ``simConfiguration`` dictionary, which contains sub-dictionaries for each simulation sub-module. Where a sub-module consists of multiple components i.e. Wave-front sensors, parameters must be given as lists at least as long as the number of components. Example configuration files can be found in the ``conf`` directory of the pyAOS package.

Below is a list of all possible simulation parameters.

Simulation Parameters
---------------------
.. autoclass:: pyAOS.confParse.SimConfig
	:members:

Telescope Parameters
--------------------

.. autoclass:: pyAOS.confParse.TelConfig

Atmosphere Parameters
---------------------
.. autoclass:: pyAOS.confParse.AtmosConfig

Wave-front Sensor Parameters
----------------------------
.. autoclass:: pyAOS.confParse.WfsConfig

Laser Guide Star Parameters
---------------------------
.. autoclass:: pyAOS.confParse.LgsConfig

Deformable Mirror Parameters
----------------------------
.. autoclass:: pyAOS.confParse.DmConfig

Science Camera Parameters
-------------------------
.. autoclass:: pyAOS.confParse.SciConfig

