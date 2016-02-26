*****************
Simulation Design
*****************

Soapy has been designed from the beginning to be extremely modular, where each AO component can be used individually. In fact, the file `simulation.py`, really only acts as a shepherd, moving data around between the components, with some fancy bits for saving data and printing nice outputs. A simple control loop for replace that file could be written from scratch in only 5-10 lines of Python!

This is well illustrated by a data flow diagram describing the simulations, show in the figure below.

.. image:: imgs/DataFlow.svg
        :align: center
