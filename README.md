# Introduction


pyAOS is a Monte-Carlo Adaptive Optics Simulation toolkit written in Python. pyAOS can be used as a conventional end-to-end simulation, where a large number of AO configurations can be created simply by editing a configuration file. Its real power lays in modular nature of objects, such as WFSs, DMs and reconstructors which can be taken and used as building blocks to construct new and complex AO ideas and configurations.

## Quick-Start


Try out some of the code examples in the ``conf`` directory, either run the ``pyAOS`` script in ``bin``, or load a python or IPython terminal:

    import pyAOS
    sim = pyAOS.Sim("configFilename")
    sim.aoinit()
    sim.makeIMat()
    sim.aoloop()