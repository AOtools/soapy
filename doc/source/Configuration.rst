Configuration
*************

pyAOS can be configured from a configuration file. These files take the form of python scripts, which contains 1 dictionary named `simConfiguration`. This dictionary contains sub-dictionarys to configure: global simulation parameters, Atmosphere, Wave-front Sensors, the Telescope, Laser Guide Stars, Deformable Mirrors and Science cameras. The parameters for these sub-dictionaries are described below. For sub-modules where there may be more than one component, such as WFSs, parameters are given in python lists with as many entries as the number of components. If the list is longer than the number of components then the entries after that number are ignored.

This file is parsed by the :doc:`confParse.py` module, which 