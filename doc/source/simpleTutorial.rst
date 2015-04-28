Simple Tutorial
===============

This tutorial will go through some example AO systems using Soapy. We'll see how to make configuration files to run the AO system that you'd like to then extract data which can be subsequently analysed. CANARY is an AO sysmem on the 4.2m William Herschel Telescope on La Palma. It is designed to be very flexible to run various "modes" of AO, so make a nice test bed for us to simulate. We'll simulate it in SCAO mode, in GLAO with multiple guide-stars, and in SCAO with a LGS.


Running an Example Configuration
--------------------------------

Before making new configuration files though, its a pretty good idea to make sure everything is working as expected by running one of the examples. First, lets create a directory where we do this tutorial, call it something like ``soapy_tutorial``, make a further directory called ``conf`` inside and copy the example configuration file ``sh_8x8.py`` form the downloaded or cloned Soapy directory into it.