#Copyright Durham University and Andrew Reeves
#2014

# This file is part of soapy.

#     soapy is free software: you can redistribute it and/or modify
#     it under the terms of the GNU General Public License as published by
#     the Free Software Foundation, either version 3 of the License, or
#     (at your option) any later version.

#     soapy is distributed in the hope that it will be useful,
#     but WITHOUT ANY WARRANTY; without even the implied warranty of
#     MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
#     GNU General Public License for more details.

#     You should have received a copy of the GNU General Public License
#     along with soapy.  If not, see <http://www.gnu.org/licenses/>.
"""
A module to generate configuration objects for Soapy, given a parameter file.

This module defines a number of classes, which when instantiated, create objects used to configure the entire simulation, or just submodules. All configuration objects are stored in the ``Configurator`` object which deals with loading parameters from file, checking some potential conflicts and using parameters to calculate some other parameters used in parts of the simulation.

The ``ConfigObj`` provides a base class used by other module configuration objects, and provides methods to read the parameters from the dictionary read from file, and set defaults if appropriate. Each other module in the system has its own configuration object, and for components such as wave-front sensors (WFSs), Deformable Mirrors (DMs), Laser Guide Stars (LGSs) and Science Cameras,  lists of the config objects for each component are created.


"""

import numpy
import traceback
import copy

from . import logger

# Check if can use yaml configuration style
try:
    import yaml
    YAML = True
except ImportError:
    logger.info("Can't import pyyaml. Can only use old python config style")
    YAML = False

# Attributes that can be contained in all configs
CONFIG_ATTRIBUTES = [
        'N',
            ]

class ConfigurationError(Exception):
    pass


class PY_Configurator(object):
    """
    The configuration class holding all simulation configuration information

    This class is used to load the parameter dictionary from file, instantiate each configuration object and calculate some other parameters from the parameters given.

    The configuration file given to this class must contain a python dictionary, named ``simConfiguration``. This must contain other dictionaries for each sub-module of the system, ``Sim``, ``Atmosphere``, ``Telescope``, ``WFS``, ``LGS``, ``DM``, ``Science``. For the final 4 sub-dictionaries, each entry must be formatted as a list (or numpy array) where each value corresponds to that component.

    The number of components on the module will only depend on the number set in the ``Sim`` dict. For example, if ``nGS`` is set to 2 in ``Sim``, then in the ``WFS`` dict, each parameters must have at least 2 entries, e.g. ``subaps : [10,10]``. If the parameter has more than 2 entries, then only the first 2 will be noted and any others discarded.

    Descriptions of the available parameters for each sub-module are given in that that config classes documentation

    Args:
        filename (string): The name of the configuration file

    """

    def __init__(self, filename):
        self.filename = filename

        #placeholder for WFS param objs
        self.wfss = []
        self.lgss = []
        self.scis = []
        self.dms = []

        self.sim = SimConfig()
        self.atmos = AtmosConfig()
        self.tel = TelConfig()

    def readfile(self):

        #Exec the config file, which should contain a dict ``simConfiguration``
        try:
            with open(self.filename) as file_:
                exec(file_.read(), globals())
        except:
            traceback.print_exc()
            raise ConfigurationError(
                    "Error loading config file: {}".format(self.filename))

        self.configDict = simConfiguration

    def loadSimParams(self):

        self.readfile()

        logger.debug("\nLoad Sim Params...")
        self.sim.loadParams(self.configDict["Sim"])

        logger.debug("\nLoad Atmosphere Params...")
        self.atmos.loadParams(self.configDict["Atmosphere"])

        logger.debug("\nLoad Telescope Params...")
        self.tel.loadParams(self.configDict["Telescope"])

        for wfs in range(self.sim.nGS):
            logger.debug("Load WFS {} Params...".format(wfs))
            self.wfss.append(WfsConfig(wfs))
            self.wfss[wfs].loadParams(self.configDict["WFS"])

        for lgs in range(self.sim.nGS):
            logger.debug("Load LGS {} Params".format(lgs))
            self.lgss.append(LgsConfig(lgs))
            self.lgss[lgs].loadParams(self.configDict["LGS"])

        for dm in range(self.sim.nDM):
            logger.debug("Load DM {} Params".format(dm))
            self.dms.append(DmConfig(dm))
            self.dms[dm].loadParams(self.configDict["DM"])

        for sci in range(self.sim.nSci):
            logger.debug("Load Science {} Params".format(sci))
            self.scis.append(SciConfig(sci))
            self.scis[sci].loadParams(self.configDict["Science"])


        self.calcParams()

    def calcParams(self):
        """
        Calculates some parameters from the configuration parameters.
        """

        # Run calcparams on each config object
        self.sim.calcParams()
        self.atmos.calcParams()
        self.tel.calcParams()
        for w in self.wfss:
            if w is not None:
                w.calcParams()
        for l in self.lgss:
            if l is not None:
                l.calcParams()
        for d in self.dms:
            if d is not None:
                d.calcParams()
        for s in self.scis:
            if s is not None:
                s.calcParams()

        self.sim.pxlScale = (float(self.sim.pupilSize)/
                                    self.tel.telDiam)

        # We oversize the pupil to what we'll call the "simulation size"
        simPadRatio = (self.sim.simOversize-1)/2.
        self.sim.simPad = int(round(self.sim.pupilSize*simPadRatio))
        self.sim.simSize = self.sim.pupilSize + 2*self.sim.simPad


        # Furthest out GS or SCI target defines the sub-scrn size
        gsPos = []
        for gs in range(self.sim.nGS):
            pos = self.wfss[gs].GSPosition.astype('float')

            # Need to add bit if the GS is an elongated off-axis LGS
            if (hasattr(self.lgss[gs], 'elongationDepth')
                    and self.lgss[gs].elongationDepth is not 0):
                # This calculation is done more explicitely in the WFS module
                # in the ``calcElongPos`` method
                maxLaunch = abs(numpy.array(
                        self.lgss[gs].launchPosition)).max()*self.tel.telDiam/2.
                dh = numpy.array([  -1*self.lgss[gs].elongationDepth/2.,
                                    self.lgss[gs].elongationDepth/2.])
                H = self.wfss[gs].GSHeight
                theta_n = (max(pos) - (dh*maxLaunch)/(H*(H+dh))*
                        (3600.*180/numpy.pi)).max()
                pos += theta_n
            gsPos.append(abs(numpy.array(pos)))

        for sci in range(self.sim.nSci):
            gsPos.append(self.scis[sci].position)

        if len(gsPos)!=0:
            maxGSPos = numpy.array(gsPos).max()
        else:
            maxGSPos = 0

        self.sim.scrnSize = 2*numpy.ceil(
                self.sim.pxlScale*self.atmos.scrnHeights.max()
                *abs(maxGSPos)*numpy.pi/(3600.*180)
                )+self.sim.simSize

        # Make scrnSize even
        if self.sim.scrnSize % 2 != 0:
            self.sim.scrnSize += 1

        # Check if any WFS use physical propogation.
        # If so, make oversized phase scrns
        wfsPhys = False
        for wfs in range(self.sim.nGS):
            if self.wfss[wfs].propagationMode=="Physical":
                wfsPhys = True
                break
        if wfsPhys:
            self.sim.scrnSize *= 2

        # If any wfs exposure times set to None, set to the sim loopTime
        for wfs in self.wfss:
            if not wfs.exposureTime:
                wfs.exposureTime = self.sim.loopTime

        logger.info("Pixel Scale: {0:.2f} pxls/m".format(self.sim.pxlScale))
        logger.info("subScreenSize: {:d} simulation pixels".format(int(self.sim.scrnSize)))



        #If outer scale is None, set all to 100m
        if self.atmos.L0 is None:
            self.atmos.L0 = []
            for scrn in range(self.atmos.scrnNo):
                self.atmos.L0.append(100.)

        #Check if SH WFS with 1 subap. Feild stop must be FOV
        for wfs in self.wfss:
            if wfs.nxSubaps==1 and wfs.subapFieldStop==False:
                logger.warning("Setting WFS:{} to have field stop at sub-ap FOV as it only has 1 sub-aperture".format(wfs))
                wfs.subapFieldStop = True

        #Use the simulation ``procs`` value to determine how many threads in
        #multi-threaded/processed operations
        for wfs in self.wfss:
            if wfs is not None:
                wfs.fftwThreads = self.sim.procs
        for lgs in self.lgss:
            if lgs is not None:
                lgs.fftwThreads = self.sim.procs
        for sci in self.scis:
            if sci is not None:
                sci.fftwThreads = self.sim.procs




    def __iter__(self):
        objs = {'Sim': dict(self.sim),
                'Atmosphere': dict(self.atmos),
                'Telescope': dict(self.tel),
                'WFS': [],
                'LGS': [],
                'DM': [],
                'Science': []
                }

        for w in self.wfss:
            if w is not None:
                objs['WFS'].append(dict(w))
            else:
                objs['WFS'].append(None)

        for l in self.lgss:
            if l is not None:
                objs['LGS'].append(dict(l))
            else:
                objs['LGS'].append(None)

        for d in self.dms:
            if d is not None:
                objs['DM'].append(dict(d))
            else:
                objs['DM'].append(None)

        for s in self.scis:
            if s is not None:
                objs['Science'].append(dict(s))
            else:
                objs['Science'].append(None)

        for configName, configObj in objs.iteritems():
            yield configName, configObj

    def __len__(self):
        # Always have sim, atmos, tel, DMs, WFSs, LGSs, and Scis
        return 7

class YAML_Configurator(PY_Configurator):

    def readfile(self):

        # load config file from Yaml file
        with open(self.filename) as file_:
            self.configDict = yaml.load(file_)


    def loadSimParams(self):

        self.readfile()

        logger.debug("\nLoad Sim Params...")
        self.sim.loadParams(self.configDict)

        logger.debug("\nLoad Atmosphere Params...")
        self.atmos.loadParams(self.configDict["Atmosphere"])

        logger.debug("\nLoad Telescope Params...")
        self.tel.loadParams(self.configDict["Telescope"])

        for nWfs in range(self.sim.nGS):
            logger.debug("Load WFS {} Params...".format(nWfs))
            wfsType = self.configDict['WFS'][nWfs].keys()[0]
            wfsDict = self.configDict['WFS'][nWfs][wfsType]
            wfsDict['type'] = wfsType

            self.wfss.append(WfsConfig(None))
            self.wfss[nWfs].loadParams(wfsDict)

        for nLgs in range(self.sim.nGS):
            logger.debug("Load LGS {} Params".format(nLgs))
            try:
                lgsType = self.configDict['LGS'][nLgs].keys()[0]
                lgsDict = self.configDict['LGS'][nLgs][lgsType]
                lgsDict['type'] = lgsType

                self.lgss.append(LgsConfig(None))
                self.lgss[nLgs].loadParams(lgsDict)
            except:
                self.lgss.append(None)

        for nDm in range(self.sim.nDM):
            logger.debug("Load DM {} Params".format(nDm))
            dmType = self.configDict['DM'][nDm].keys()[0]
            dmDict = self.configDict['DM'][nDm][dmType]
            dmDict['type'] = dmType

            self.dms.append(DmConfig(None))
            self.dms[nDm].loadParams(dmDict)

        for nSci in range(self.sim.nSci):
            logger.debug("Load Science {} Params".format(nSci))
            sciType = self.configDict['Science'][nSci].keys()[0]
            sciDict = self.configDict['Science'][nSci][sciType]
            sciDict['type'] = sciType

            self.scis.append(SciConfig(None))
            self.scis[nSci].loadParams(sciDict)

        self.calcParams()

class ConfigObj(object):
    # Parameters that can be had by any configuration object

    def __init__(self, N=None):
        #This is the index of some config object, i.e. WFS 1, 2, 3..N
        self.N = N

    def warnAndExit(self, param):

        message = "{0} not set!".format(param)
        logger.warning(message)
        raise ConfigurationError(message)

    def warnAndDefault(self, param, newValue):
        message = "{0} not set, default to {1}".format(param, newValue)
        self.__setattr__(param, newValue)

        logger.debug(message)

    def initParams(self):
        for param in self.requiredParams:
            self.__setattr__(param, None)

    def loadParams(self, configDict):

        if self.N!=None:
            for param in self.requiredParams:
                try:
                    self.__setattr__(param, configDict[param][self.N])
                except KeyError:
                    self.warnAndExit(param)
                except IndexError:
                    raise ConfigurationError(
                                "Not enough values for {0}".format(param))
                except:
                    raise ConfigurationError(
                            "Failed while loading {0}. Check config file.".format(param))

            for param in self.optionalParams:
                try:
                    self.__setattr__(param[0], configDict[param[0]][self.N])
                except KeyError:
                    self.warnAndDefault(param[0], param[1])
                except IndexError:
                    raise ConfigurationError(
                                "Not enough values for {0}".format(param))
                except:
                    raise ConfigurationError(
                            "Failed while loading {0}. Check config file.".format(param))
        else:
            for param in self.requiredParams:
                try:
                    self.__setattr__(param, configDict[param])
                except KeyError:
                    self.warnAndExit(param)
                except:
                    raise ConfigurationError(
                            "Failed while loading {0}. Check config file.".format(param))

            for param in self.optionalParams:
                try:
                    self.__setattr__(param[0], configDict[param[0]])
                except KeyError:
                    self.warnAndDefault(param[0], param[1])
                except:
                    raise ConfigurationError(
                            "Failed while loading {0}. Check config file.".format(param))

        self.calcParams()

    def calcParams(self):
        """
        Dummy method to be overidden if required
        """
        pass

    def __iter__(self):
        for param in self.requiredParams:
            yield param, self.__dict__[param]
        for param in self.optionalParams:
            yield param[0], self.__dict__[param[0]]

    def __len__(self):
        return len(self.requiredParams)+len(self.optionalParams)

    def __setattr__(self, name, value):
        if name in self.allowedAttrs:
            self.__dict__[name] = value
        else:
            raise ConfigurationError("'{}' Attribute not a configuration parameter".format(name))

class SimConfig(ConfigObj):
    """
    Configuration parameters relavent for the entire simulation. These should be held in the ``Sim`` sub-dictionary of the ``simConfiguration`` dictionary in the parameter file.

    Required:
        =============   ===================
        **Parameter**   **Description**
        -------------   -------------------
        ``pupilSize``   int: Number of phase points across the simulation pupil
        ``nIters``      int: Number of iteration to run simulation
        ``loopTime``    float: Time between simulation frames (1/framerate)
        =============   ===================


    Optional:
        ==================  =================================   ===============
        **Parameter**       **Description**                         **Default**
        ------------------  ---------------------------------   ---------------
        ``nGS``             int: Number of Guide Stars and
                            WFS                                 ``0``
        ``nDM``             int: Number of deformable Mirrors   ``0``
        ``nSci``            int: Number of Science Cameras      ``0``
        ``reconstructor``   str: name of reconstructor
                            class to use. See
                            ``reconstructor`` module
                            for available reconstructors.       ``"MVM"``
        ``simName``         str: directory name to store
                            simulation data                     ``None``
        ``wfsMP``           bool: Each WFS uses its own
                            process                             ``False``
        ``verbosity``       int: debug output for the
                            simulation ranging from 0
                            (no-ouput) to 3 (all debug
                            output)                             ``2``
        ``logfile``         str: name of file to store
                            logging data,                       ``None``
        ``learnIters``      int: Number of `learn` iterations
                            for Learn & Apply reconstructor     ``0``
        ``learnAtmos``      str: if ``random``, then
                            random phase screens used for
                            `learn`                             ``random``
        ``procs``           int: number of processes to use
                            in multiprocessing operations       ``1``
        ``simOversize``     float: The fraction to pad the
                            pupil size with to reduce edge
                            effects                             ``1.2``
        ``loopDelay``       int: loop delay in integer count
                            of ``loopTime``                     ``0``


        ==================  =================================   ===============

    Data Saving (all default to False):
        ======================      ===================
        **Parameter**               **Description**
        ----------------------      -------------------
        ``saveSlopes``              Save all WFS slopes. Accessed from sim with
                                    ``sim.allSlopes``
        ``saveDmCommands``          Saves all DM Commands. Accessed from sim
                                    with ``sim.allDmCommands``
        ``saveWfsFrames``           Saves all WFS pixel data. Saves to disk a
                                    after every frame to avoid using too much
                                    memory
        ``saveStrehl``              Saves the science camera Strehl Ratio.
                                    Accessed from sim with ``sim.longStrehl``
                                    and ``sim.instStrehl``
        ``saveWfe``                 Saves the science camera wave front error.
                                    Accessed from sim with ``sim.WFE``.
        ``saveSciPsf``              Saves the science PSF.
        ``saveInstPsf``             Saves the instantenous science PSF.
        ``saveInstScieField``       Saves the instantaneous electric field at focal plane.
        ``saveSciRes``              Save Science residual phase
        ======================      ===================

    """
    requiredParams = [  "pupilSize",
                        "nIters",
                        "loopTime",
                        ]

    optionalParams = [ ("nGS", 0),
                            ("nDM", 0),
                            ("nSci", 0),
                            ("gain", 0.6),
                            ("reconstructor", "MVM"),
                            ("simName", None),
                            ("saveSlopes", False),
                            ("saveDmCommands", False),
                            ("saveLgsPsf", False),
                            ("saveLearn", False),
                            ("saveStrehl", False),
                            ("saveWfsFrames", False),
                            ("saveSciPsf", False),
                            ("saveInstPsf", False),
                            ("saveInstScieField", False),
                            ("saveWfe", False),
                            ("saveSciRes", False),
                            ("wfsMP", False),
                            ("verbosity", 2),
                            ("logfile", None),
                            ("learnIters", 0),
                            ("learnAtmos", "random"),
                            ("procs", 1),
                            ("simOversize", 1.2),
                            ("loopDelay", 0),
                        ]

    # Parameters which may be set at some point and are allowed
    calculatedParams = [    'pxlScale',
                            'simPad',
                            'simSize',
                            'scrnSize',
                            'totalWfsData',
                            'totalActs',
                            'saveHeader',
                    ]


    allowedAttrs = copy.copy(
            requiredParams + calculatedParams + CONFIG_ATTRIBUTES)
    for p in optionalParams:
        allowedAttrs.append(p[0])

class AtmosConfig(ConfigObj):
    """
    Configuration parameters characterising the atmosphere. These should be held in the ``Atmosphere`` sub-dictionary of the ``simConfiguration`` dictionary in the parameter file.

    Required:
        ==================      ===================
        **Parameter**           **Description**
        ------------------      -------------------
        ``scrnNo``              int: Number of turbulence layers
        ``scrnHeights``         list, int: Phase screen heights in metres
        ``scrnStrength``        list, float: Relative layer scrnStrength
        ``windDirs``            list, float: Wind directions in degrees.
        ``windSpeeds``          list, float: Wind velocities in m/s
        ``r0``                  float: integrated  seeing strength
                                (metres at 550nm)
        ``wholeScrnSize``       int: Size of the phase screens to store in the
                                ``atmosphere`` object
        ==================      ===================

    Optional:
        ==================  =================================   ===========
        **Parameter**       **Description**                     **Default**
        ------------------  ---------------------------------   -----------
        ``scrnNames``       list, string: filenames of phase
                            if loading from fits files. If
                            ``None`` will make new screens.     ``None``
        ``subHarmonics``    bool: Use sub-harmonic screen
                            generation algorithm for better
                            tip-tilt statistics - useful
                            for small phase screens.             ``False``
        ``L0``              list, float: Outer scale of each
                            layer. Kolmogorov turbulence if
                            ``None``.                           ``None``
        ``randomScrns``     bool: Use a random set of phase
                            phase screens for each loop
                            iteration?                          ``False``
        ``tau0``            float: Turbulence coherence time,
                            if set wind speeds are scaled.      ``None``
        ==================  =================================   ===========
    """

    requiredParams = [ "scrnNo",
                        "scrnHeights",
                        "scrnStrengths",
                        "r0",
                        "windDirs",
                        "windSpeeds",
                        "wholeScrnSize",
                        ]

    optionalParams = [ ("scrnNames",None),
                        ("subHarmonics",False),
                        ("L0", None),
                        ("randomScrns", False),
                        ("tau0", None),
                        ]

    # Parameters which may be set at some point and are allowed
    calculatedParams = [
                        'normScrnStrengths',
                        ]
    allowedAttrs = copy.copy(requiredParams + calculatedParams + CONFIG_ATTRIBUTES)
    for p in optionalParams:
        allowedAttrs.append(p[0])


    def calcParams(self):
        # Turn lists into numpy arrays
        self.scrnHeights = numpy.array(self.scrnHeights)
        self.scrnStrengths = numpy.array(self.scrnStrengths)
        self.windDirs = numpy.array(self.windDirs)
        self.windSpeeds = numpy.array(self.windSpeeds)
        if self.L0 is not None:
            self.L0 = numpy.array(self.L0)


class WfsConfig(ConfigObj):
    """
    Configuration parameters characterising Wave-front Sensors. These should be held in the ``WFS`` sub-dictionary of the ``simConfiguration`` dictionary in the parameter file. Each parameter must be in the form of a list, where each entry corresponds to a WFS. Any entries above ``sim.nGS`` will be ignored.

    Required:
        ==================      ===================
        **Parameter**           **Description**
        ------------------      -------------------
        ``GSPosition``          tuple: position of GS on-sky in arc-secs
        ``wavelength``          float: wavelength of GS light in metres
        ``nxSubaps``            int: number of SH sub-apertures
        ``pxlsPerSubap``        int: number of pixels per sub-apertures
        ``subapFOV``            float: Field of View of sub-aperture in
                                arc-secs
        ==================      ===================

    Optional:
        =================== ================================== ===========
        **Parameter**       **Description**                    **Default**
        ------------------- ---------------------------------- -----------
        ``type``            string: Which WFS object to load
                            from WFS.py?                        ``ShackHartmann``
        ``GSMag``           float: Apparent magnitude of the
                            guide star                         ``0``
        ``photonNoise``     bool: Include photon (shot) noise. ``False``
        ``eReadNoise``      float: Electrons of read noise     ``0``
        ``throughput``      float: Throughput of the entire
                            optical and electronic system
                            from guide star photons to
                            recorded WFS detector counts.
                            Includes atmospheric effects, the
                            optical train and detector gain.   ``1.``
        ``propagationMode`` string: Mode of light propogation
                            from GS. Can be "Physical" or
                            "Geometric"\**.                     ``"Geometric"``
        ``subapFieldStop``  bool: if True, add a field stop to
                            the wfs to prevent spots wandering
                            into adjacent sub-apertures. if
                            False, oversample subap FOV by a
                            factor of 2 to allow into adjacent
                            subaps.                             ``False``
        ``removeTT``        bool: if True, remove TT signal
                            from WFS slopes before
                            reconstruction.\**                  ``False``
        ``fftOversamp``     int: Multiplied by the number of
                            of phase points required for FOV
                            to increase fidelity from FFT.      ``3``
        ``GSHeight``        float: Height of GS beacon. ``0``
                            if at infinity.                     ``0``
        ``subapThreshold``  float: How full should subap be
                            to be used for wavefront sensing?   ``0.5``
        ``lgs``             bool: is WFS an LGS?                ``False``
        ``centMethod``      string: Method used for
                            Centroiding. Can be
                            ``centreOfGravity``,
                            ``brightestPxl``, or
                            ``correlation``.\**                 ``centreOfGravity``
        ``referenceImage``  array: Reference images used in
                            the correlation centroider. Full
                            image plane image, each subap has
                            a separate reference image          ``None``
        ``angleEquivNoise`` float: width of gaussian noise
                            added to slopes measurements
                            in arc-secs                         ``0``
        ``centThreshold``   float: Centroiding threshold as
                            a fraction of the max subap
                            value.\**                           ``0.1``
        ``exposureTime``    float: Exposure time of the WFS
                            camera - must be higher than
                            loopTime. If None, will be
                            set to loopTime.                    ``None``
        ``wvlBandWidth``    float: Width of wavelength
                            band sent to WFS in nm              ``100``
        ``extendedObject``  ndarray or str: The object used
                            as extended source for WFS, of
                            size 2*fftOversamp*pxlsPerSubap.
                            The FOV of the object should be
                            twice the FOV of the sub-aperture.  ``None``
        ``fftwThreads``     int: number of threads for fftw
                            to use. If ``0``, will use
                            system processor number.           ``1``
        ``fftwFlag``        str: Flag to pass to FFTW
                            when preparing plan.               ``FFTW_PATIENT``
        =================== ================================== ===========


        """

    requiredParams = [ "GSPosition",
                        "wavelength",
                        "nxSubaps",
                        "pxlsPerSubap",
                        "subapFOV",
                        ]
    optionalParams = [  ("propagationMode", "geometric"),
                        ("fftwThreads", 1),
                        ("fftwFlag", "FFTW_PATIENT"),
                        ("angleEquivNoise", 0),
                        ("subapFieldStop", False),
                        ("removeTT", "False"),
                        ("angleEquivNoise", 0),
                        ("fftOversamp", 3),
                        ("GSHeight", 0),
                        ("subapThreshold", 0.5),
                        ("lgs", False),
                        ("centThreshold", 0.3),
                        ("centMethod", "centreOfGravity"),
                        ("type", "ShackHartmann"),
                        ("exposureTime", None),
                        ("referenceImage", None),
                        ("throughput", 1.),
                        ("eReadNoise", 0),
                        ("photonNoise", False),
                        ("GSMag", 0.0),
                        ("wvlBandWidth", 100.),
                        ("extendedObject", None),
                        ]

        # Parameters which may be Set at some point and are allowed
    calculatedParams = [
                        'position',
                        'pxlsPerSubap2',
                        'dataStart',

                        ]

    allowedAttrs = copy.copy(
            requiredParams + calculatedParams + CONFIG_ATTRIBUTES)
    for p in optionalParams:
        allowedAttrs.append(p[0])

    def calcParams(self):
        # Set some parameters to correct type
        self.GSPosition = numpy.array(self.GSPosition)
        self.position = self.GSPosition # For compatability

        # Ensure wavelength is a float
        self.wavelength = float(self.wavelength)

class TelConfig(ConfigObj):
    """
        Configuration parameters characterising the Telescope. These should be held in the ``Tel`` sub-dictionary of the ``simConfiguration`` dictionary in the parameter file.

    Required:
        =============   ===================
        **Parameter**   **Description**
        -------------   -------------------
        ``telDiam``     float: Diameter of telescope pupil in metres
        =============   ===================

    Optional:
        ==================  =================================   ===========
        **Parameter**       **Description**                     **Default**
        ------------------  ---------------------------------   -----------
        ``obsDiam``         float: Diameter of central
                            obscuration                         ``0``
        ``mask``            str: Shape of pupil (only
                            accepts ``circle`` currently)       ``circle``
        ==================  =================================   ===========

    """


    requiredParams = [ "telDiam",
                            ]

    optionalParams = [ ("obsDiam", 0),
                        ("mask", "circle")
                        ]
    calculatedParams = [  ]

    allowedAttrs = copy.copy(requiredParams + calculatedParams + CONFIG_ATTRIBUTES)
    for p in optionalParams:
        allowedAttrs.append(p[0])


class LgsConfig(ConfigObj):
    """
        Configuration parameters characterising the Laser Guide Stars. These should be held in the ``LGS`` sub-dictionary of the ``simConfiguration`` dictionary in the parameter file. Each parameter must be in the form of a list, where each entry corresponds to a WFS. Any entries above ``sim.nGS`` will be ignored.


    Optional:
        ==================== =================================   ===========
        **Parameter**        **Description**                     **Default**
        -------------------- ---------------------------------   -----------
        ``uplink``           bool: Include LGS uplink effects    ``False``
        ``pupilDiam``        float: Diameter of LGS launch
                             aperture in metres.                 ``0.3``
        ``wavelength``       float: Wavelength of laser beam
                             in metres                           ``600e-9``
        ``propagationMode``  str: Mode of light propogation
                             from GS. Can be "Physical" or
                             "Geometric".                        ``"Phsyical"``
        ``height``           float: Height to use physical
                             propogation of LGS (does not
                             effect cone-effect) in metres       ``90000``
        ``elongationDepth``  float:
                             Depth of LGS elongation in metres   ``0``
        ``elongationLayers`` int:
                             Number of layers to simulate for
                             elongation.                         ``10``
        ``launchPosition``   tuple: The launch position of
                             the LGS in units of the pupil
                             radii, where ``(0,0)`` is the
                             centre launched case, and
                             ``(1,0)`` is side-launched.          ``(0,0)``
        ``fftwThreads``      int: number of threads for fftw
                             to use. If ``0``, will use
                             system processor number.             ``1``
        ``fftwFlag``         str: Flag to pass to FFTW
                             when preparing plan.                 ``FFTW_PATIENT``
        ``naProfile``        list: The relative sodium layer
                             strength for each elongation
                             layer. If None, all equal.          ``None``
        ==================== =================================   ===========

    """

    requiredParams = [ ]

    optionalParams = [  ("uplink", False),
                        ("pupilDiam", 0.3),
                        ("wavelength", 600e-9),
                        ("propagationMode", "physical"),
                        ("height", 90000),
                        ("fftwFlag", "FFTW_PATIENT"),
                        ("fftwThreads", 0),
                        ("elongationDepth", 0),
                        ("elongationLayers", 10),
                        ("launchPosition",  numpy.array([0,0])),
                        ("naProfile", None),
                        ]
    calculatedParams = []

    allowedAttrs = copy.copy(
            requiredParams + calculatedParams + CONFIG_ATTRIBUTES)
    for p in optionalParams:
        allowedAttrs.append(p[0])

    def calcParams(self):
        # If lgs sodium layer profile is none, set it to 1s for each layer
        if not hasattr(self, "naProfile") or self.naProfile is None:
            self.naProfile = numpy.ones(self.elongationLayers)
        if len(self.naProfile)<self.elongationLayers:
            raise ConfigurationError("Not enough values for naProfile")

class DmConfig(ConfigObj):
    """
    Configuration parameters characterising Deformable Mirrors. These should be held in the ``DM`` sub-dictionary of the ``simConfiguration`` dictionary in the parameter file. Each parameter must be in the form of a list, where each entry corresponds to a DM. Any entries above ``sim.nDM`` will be ignored.

    Required:
        ===================     ===============================================
        **Parameter**           **Description**
        -------------------     -----------------------------------------------
        ``type``                string: Type of DM. This must the name of a
                                class in the ``DM`` module.
        ``nxActuators``         int: Number independent DM shapes. e.g., for
                                stack-array DMs this is number of actuators in
                                one dimension,
                                for Zernike DMs this is number of Zernike
                                modes.
        ``gain``                float: The loop gain for the DM.\**
        ``svdConditioning``     float: The conditioning parameter used in the
                                pseudo inverse of the interaction matrix. This
                                is performed by `numpy.linalg.pinv <http://docs.scipy.org/doc/numpy/reference/generated/numpy.linalg.pinv.html>`_.
        ===================     ===============================================

    Optional:
        ==================== =================================   ===========
        **Parameter**        **Description**                     **Default**
        -------------------- ---------------------------------   -----------
        ``closed``           bool:Is DM closed loop of WFS?\**    ``True``
        ``iMatValue``        float: Value to push actuators
                             when making iMat                    ``10``
        ``wfs``              int: which Wfs to take iMat and
                             use to correct for.                 ``0``
        ``rotation``         float: A DM rotation with respect
                             to the pupil in degrees             ``0``
        ``interpOrder``      Order of interpolation for dm,
                             including piezo actuators and
                             rotation.                           ``1``
        ``gaussWidth``       float: Width of Guass DM actuator
                             as a fraction of the
                             inter-actuator spacing.             ``0.5``
        ==================== =================================   ===========
    """


    requiredParams = [ "type",
                        ]


    optionalParams = [
                    ("nxActuators", None),
                    ("svdConditioning", 0),
                    ("gain", 0.6),
                    ("closed", True),
                    ("iMatValue", 10),
                    ("wfs", None),
                    ("rotation", 0),
                    ("interpOrder", 2),
                    ("gaussWidth", 0.5),
                    ]

    calculatedParams = [
                        ]

    allowedAttrs = copy.copy(requiredParams + calculatedParams + CONFIG_ATTRIBUTES)
    for p in optionalParams:
        allowedAttrs.append(p[0])

    def calcParams(self):
        # Some params commonly written in Scientific notation
        self.iMatValue  = float(self.iMatValue)
        self.svdConditioning = float(self.svdConditioning)

class SciConfig(ConfigObj):
    """
    Configuration parameters characterising Science Cameras. These should be held in the ``Science`` sub-dictionary of the ``simConfiguration`` dictionary in the parameter file. Each parameter must be in the form of a list, where each entry corresponds to a science camera. Any entries above ``sim.nSci`` will be ignored.

    Required:
        ==================      ============================================
        **Parameter**           **Description**
        ------------------      --------------------------------------------
        ``position``            tuple: The position of the science camera
                                in the field in arc-seconds
        ``FOV``                 float: The field of fiew of the science
                                detector in arc-seconds
        ``wavelength``          float: The wavelength of the science
                                detector light
        ``pxls``                int: Number of pixels in the science detector
        ==================      ============================================

    Optional:
        ==================== =================================   ===========
        **Parameter**        **Description**                     **Default**
        -------------------- ---------------------------------   -----------
        ``type``             string: Type of science camera
                             This must the name of a class
                             in the ``SCI`` module.              ``PSF``
        ``fftOversamp``      int: Multiplied by the number of
                             of phase points required for FOV
                             to increase fidelity from FFT.      ``2``
        ``fftwThreads``      int: number of threads for fftw
                             to use. If ``0``, will use
                             system processor number.             ``1``
        ``fftwFlag``         str: Flag to pass to FFTW
                             when preparing plan.                 ``FFTW_MEASURE``
         ``height``          float: Altitude of the object.
                             0 denotes infinity.                  ``0``
        ``propagationMode``  str: Mode of light propogation
                             from object. Can be "Physical" or
                             "Geometric".                       ``"Geometric"``
        ``instStrehlWithTT`` bool: Whether or not to include
                             tip/tilt in instantaneous Strehl
                             calculations.                       ``False``

        ==================== =================================   ===========

    """


    requiredParams = [  "position",
                        "FOV",
                        "wavelength",
                        "pxls",
                        ]
    optionalParams = [  ("type", "PSF"),
                        ("fftOversamp", 2),
                        ("fftwFlag", "FFTW_MEASURE"),
                        ("fftwThreads", 1),
                        ("instStrehlWithTT", False),
                        ("height", 0),
                        ("propagationMode", "Geometric")
                        ]

    calculatedParams = [
                            ]

    allowedAttrs = copy.copy(requiredParams + calculatedParams + CONFIG_ATTRIBUTES)
    for p in optionalParams:
        allowedAttrs.append(p[0])

    def calcParams(self):
        # Set some parameters to correct type
        self.position = numpy.array(self.position)


def loadSoapyConfig(configfile):

    # Find configfile extension
    file_ext = configfile.split('.')[-1]

    # If YAML use yaml configurator
    if file_ext=='yml' or file_ext=='yaml':
        config = YAML_Configurator(configfile)

    # Otherwise, try and execute as python
    else:
        config = PY_Configurator(configfile)

    config.loadSimParams()

    return config

# compatability
Configurator = PY_Configurator

def test():
    C = Configurator("conf/testConfNew.py")
    C.readfile()
    C.loadSimParams()

    print("Test Passesd!")
    return 0


if __name__ == "__main__":
    test()
