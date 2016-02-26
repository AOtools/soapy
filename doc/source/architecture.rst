*****************
Simulation Design
*****************

Data flow and modularity
------------------------
Soapy has been designed from the beginning to be extremely modular, where each AO component can be used individually. In fact, the file `simulation.py`, really only acts as a shepherd, moving data around between the components, with some fancy bits for saving data and printing nice outputs. A simple control loop to replace that file could be written from scratch in only 5-10 lines of Python!

This modularity is well illustrated by a data flow diagram describing the simulations, show in Figure 1, below.

.. image:: imgs/DataFlow.svg
        :align: center

        Figure 1. Soapy Data Flow

Class Hierarchy
---------------
Pythons Object Orientated nature has also been exploited. Categories of AO component have a `base class`, which deals with most of the interfaces to the main simulation module and other boiler-plate style code. The classes which represent actual AO modules inherit this base class, and hopefully need only add interesting functionality specific to that new component. This is illustrated in the class diagram in Figure 2, with some example methods and attributes of each class.

.. image:: imgs/FullClassDiagram.svg
        :align: center

        Figure 2. Class diagram with example attributes and methods
        
It is aimed that in future developments of Soapy, this philosophy will be extended. Currently the WFS, science camera and LGS modules all deal with optical propagation through turbulence separately, clearly this should be combined into one place to easy code readability and maintenance. This is currently under development.
