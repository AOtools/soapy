#0.12.0
- Shift lots of light propagation code to a separate `LineOfSight` object, which is shared by WFSs, LGSs and Science targets
- Add optional YAML configuration style. This will now be the main supported configuration method
- Added ability to conjugate DMs at a finite altitude
- Re-worked API to objects so they only need the simulation config file 
- Fix "Gradient" WFS so slopes outputted in arcseconds
- Other fixes and small features

#0.11.0
-  Begun restructuring of components as files were getting very large. Now components will be kept in sub-modules. Done WFS module.
- Re-worked correlation WFSing - now there is an "ExtendedSH" WFS, which can take an extended object, that is convolved with the sub-aperture PSF to form a wide-field WFSer. A correlation centroiding technique is used to get centroids (thanks to @matthewtownson!)
- Added options of integer frame loop delay (thanks to @jwoillez!)
- Can set tau_0 parameters in atmosphere. Atmosphere also prints out atmosphere summary on init. (thanks to @jwoillez)

#0.10.0
- Phase is now internally passed around in nano-metres
- Added both WFS photon and read noise. Still needs comparisons to verify though
- Added a DM which is interpolated on each frame - better for larger systems
- Can now access and save the corrected EField (thanks @Robjharris37!)
- Added testing infrastructure and code coverage checking - also added more tests
- Data now saved with common FITS header


#0.9.1
- Tutorial added to docs

- Fix bug in off-axis science cameras, where Interp2d wasnt imported

- Can use a random phase screen for each loop iteration

#0.9
- Name change to Soapy!

- Some parameters changed name 
    ```python

    sim
        filePrefix  -->  simName

    tel
        obs      -->  obsDiam

    wfs
        subaps  --> nxSubaps
        subapOversamp -->  fftOversamp

    lgs
        lgsUplink -->  uplink
        lgsPupilDiam -->  pupilDiam
    
    dm
        dmType   -->  type
        dmActs    -->  nxActuators (Now only specify the 1-d size)
        dmCond  -->  svdConditioning

    sci
        oversamp   -->   fftOversamp

    ```
- Configuration parameters in lists are now accessed as plurals, e.g. `sim.config.wfs[0].nxSubaps` --> `sim.config.wfss[0].nxSubaps`.

#0.8.1
- Some bugs fixed with WFSs

- Made it easier to change shack-hartmann WFS centroider. Add new centroider to pyaos/tool/centroiders.py, then use the function name in the config file -- the simulation will now use the new centroider

- continued adding to docs and tests

#0.8.0
- `simSize` is now defined, which is the size of phase arrays to be passed around. This is currently sized at 1.2 x pupilSize, though this can be altered, and eliminates any edge effects which used to appear when interpolating near edges. This broke lots of things, which I think have all been fixed. If any exceptions which feature simSize or pupilSize occur, they could be caused by his change and if reported should be quickly fixed. Now, phase from DMs and the expected correction given to WFSs should be of shape `config.sim.simSize`, rather than `config.sim.pupilSize` as previously.

- A major bug in scaling phase for r0 has been fixed, which caused a big degradation in AO performance. AO Performance now matched YAO well for single NGS cases.

- A correlation centroiding method has been added by new contributor @mathewtownson

- Unit testing has begun to be integrated into the simulation, with the eventual aim of test driven development. Currently, the tests are system tests which run the entire simulation in a variety of configurations. More atomic testing will continue to be added.

- Documentation has been updated and is getting to the point where all existing code structures are explained. The target of the Docs will now turn to more explanatory stuff, such as a tutorial, and how to extract Data from a simulation run.

- Various arrays in the WFS module now use circular NumPy buffers to avoid excessive data array creation.

###Numba Branch
- Due to all the code using either pure python or NumPy or SciPy routines, the code is currently not competitive performance wise with other AO codes. To address this shortcoming, we've begun to develop a version of the code with performance critical sections accelerated using Numba, a free library which uses LLVM to compile pure python to machine code. This branch can be accessed through git and is kept up to date with the master branch, Numba is included with the Anaconda python distribution, or can be installed seperately, though it requires LLVM 3.5 to also be installed. Early results are very promising, showing over twice speed-ups for lots of applications.
