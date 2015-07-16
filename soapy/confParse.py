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
from . import logger

#How much bigger to make the simulation than asked for. The value is a ratio
#of how much to pad on each side
SIM_OVERSIZE = 0.1


class ConfigurationError(Exception):
    pass


class Configurator(object):
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
        self.sim.pxlScale = (float(self.sim.pupilSize)/
                                    self.tel.telDiam)

        #We oversize the pupil to what we'll call the "simulation size"
        self.sim.simSize = int(self.sim.pupilSize 
                + 2*numpy.round(SIM_OVERSIZE*self.sim.pupilSize))
        self.sim.simPad = int(numpy.round(SIM_OVERSIZE*self.sim.pupilSize))


        #furthest out GS or SCI target defines the sub-scrn size
        gsPos = []
        for gs in range(self.sim.nGS):
            pos = self.wfss[gs].GSPosition
            #Need to add bit if the GS is an elongation off-axis LGS
            if self.lgss[gs].elongationDepth:
                #This calculation is done more explicitely in teh WFS module
                #in the ``calcElongPos`` method
                maxLaunch = abs(numpy.array(
                        self.lgss[gs].launchPosition)).max()*self.tel.telDiam/2.
                dh = numpy.array([  -1*self.lgss[gs].elongationDepth/2.,
                                    self.lgss[gs].elongationDepth/2.])
                H = self.wfss[gs].GSHeight
                theta_n = abs(max(pos) - (dh*maxLaunch)/(H*(H+dh))*
                        (3600*180/numpy.pi)).max()
                pos+=theta_n         
            gsPos.append(pos)
               
        for sci in range(self.sim.nSci):
            gsPos.append(self.scis[sci].position)

        if len(gsPos)!=0:
            maxGSPos = numpy.array(gsPos).max()
        else:
            maxGSPos = 0

        self.sim.scrnSize = numpy.ceil(
                2*self.sim.pxlScale*self.atmos.scrnHeights.max()
                *maxGSPos*numpy.pi/(3600.*180) 
                )+self.sim.simSize
        
        #Make scrnSize even
        if self.sim.scrnSize%2!=0:
            self.sim.scrnSize+=1

        #Check if any WFS use physical propogation.
        #If so, make oversize phase scrns
        wfsPhys = False
        for wfs in range(self.sim.nGS):
            if self.wfss[wfs].propagationMode=="physical":
                wfsPhys = True
                break
        if wfsPhys:
            self.sim.scrnSize*=2
            
        #If any wfs exposure times set to None, set to the sim loopTime
        for wfs in self.wfss:
            if not wfs.exposureTime:
                wfs.exposureTime = self.sim.loopTime

        logger.info("Pixel Scale: {0:.2f} pxls/m".format(self.sim.pxlScale))
        logger.info("subScreenSize: {:d} simulation pixels".format(int(self.sim.scrnSize)))

        #If lgs sodium layer profile is none, set it to 1s for each layer
        for lgs in self.lgss:
            if not numpy.any(lgs.naProfile):
                lgs.naProfile = numpy.ones(lgs.elongationLayers)
            if len(lgs.naProfile)<lgs.elongationLayers:
                raise ConfigurationError("Not enough values for naProfile")

        #If outer scale is None, set all to 100m
        if self.atmos.L0==None:
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
            wfs.fftwThreads = self.sim.procs
        for lgs in self.lgss:
            lgs.fftwThreads = self.sim.procs
        for sci in self.scis:
            sci.fftwThreads = self.sim.procs

        


class ConfigObj(object):
    def __init__(self):

        #This is the index of the config object, i.e. WFS 1, 2, 3..N
        self.N = None

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

    def calcParams(self):
        """
        Dummy method to be overidden if required
        """
        pass


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
        ``saveSciPsf``              Saves the science PSF.
        ``saveSciRes``              Save Science residual phase
        ======================      ===================

    """

    def __init__(self):
        """
        Set some initial parameters which will be used by default
        """

        super(SimConfig, self).__init__()

        self.requiredParams = [  "pupilSize",
                            "nIters",
                            "loopTime",
                            ]

        self.optionalParams = [ ("nGS", 0),
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
                                ("saveWFE", False),
                                ("saveSciRes", False),
                                ("wfsMP", False),
                                ("verbosity", 2),
                                ("logfile", None),
                                ("learnIters", 0),
                                ("learnAtmos", "random"), 
                                ("procs", 1),
                        ]

        self.initParams()


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
        ==================  =================================   ===========    
    """

    def __init__(self):
        super(AtmosConfig, self).__init__()

        self.requiredParams = [ "scrnNo",
                                "scrnHeights",
                                "scrnStrengths",
                                "r0",
                                "windDirs",
                                "windSpeeds",
                                "wholeScrnSize",
                                ]

        self.optionalParams = [ ("scrnNames",None),
                                ("subHarmonics",False),
                                ("L0", None),
                                ("randomScrns", False)
                                ]

        self.initParams()


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
        ``propagationMode`` string: Mode of light propogation 
                            from GS. Can be "physical" or 
                            "geometric"\**.                     ``"geometric"``
        ``subapFieldStop``  bool: if True, add a field stop to
                            the wfs to prevent spots wandering
                            into adjacent sub-apertures. if
                            False, oversample subap FOV by a
                            factor of 2 to allow into adjacent
                            subaps.                             ``False``
        ``bitDepth``        int: bitdepth of WFS detector       ``32``
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
        ``fftwThreads``     int: number of threads for fftw 
                            to use. If ``0``, will use 
                            system processor number.           ``1``
        ``fftwFlag``        str: Flag to pass to FFTW 
                            when preparing plan.               ``FFTW_PATIENT``
        =================== ================================== =========== 


        """
    def __init__(self, N):

        super(WfsConfig, self).__init__()

        self.N = N

        self.requiredParams = [ "GSPosition",
                                "wavelength",
                                "nxSubaps",
                                "pxlsPerSubap",
                                "subapFOV",
                            ]

        self.optionalParams = [ ("propagationMode", "geometric"),
                                ("fftwThreads", 1),
                                ("fftwFlag", "FFTW_PATIENT"),
                                ("SNR", 0),
                                ("angleEquivNoise", 0),
                                ("subapFieldStop", False),
                                ("bitDepth", 32),
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
                            ]
        self.initParams()


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
    def __init__(self):

        super(TelConfig, self).__init__()

        self.requiredParams = [ "telDiam",
                                ]

        self.optionalParams = [ ("obsDiam", 0),
                                ("mask", "circle")
                                ]

        self.initParams()
    

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
                             from GS. Can be "physical" or 
                             "geometric".                        ``"phsyical"``
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
    def __init__(self, N):
        super(LgsConfig, self).__init__()

        self.N = N

        self.requiredParams = [ ]

        self.optionalParams = [ ("uplink", False),
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


        self.initParams()


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


    def __init__(self, N):
        super(DmConfig, self).__init__()

        self.N = N

        self.requiredParams = [ "type",
                                "nxActuators",
                                "svdConditioning",
                                "gain",
                                ]


        self.optionalParams = [ 
                                ("closed", True),
                                ("iMatValue", 10),
                                ("wfs", None),
                                ("rotation", 0),
                                ("interpOrder", 2),
                                ("gaussWidth", 0.5),
                                ]
        self.initParams()


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
        ``fftOversamp``      int: Multiplied by the number of
                             of phase points required for FOV 
                             to increase fidelity from FFT.      ``2``
        ``fftwThreads``      int: number of threads for fftw 
                             to use. If ``0``, will use 
                             system processor number.             ``1``
        ``fftwFlag``         str: Flag to pass to FFTW 
                             when preparing plan.                 ``FFTW_MEASURE``            
        ==================== =================================   ===========   

    """
    def __init__(self, N):

        super(SciConfig, self).__init__()

        self.N = N

        self.requiredParams = [ "position",
                                "FOV",
                                "wavelength",
                                "pxls",
                                ]
        self.optionalParams = [ ("fftOversamp", 2),
                                ("fftwFlag", "FFTW_MEASURE"),
                                ("fftwThreads", 1)
                                ]

        self.initParams()


def test():
    C = Configurator("conf/testConfNew.py")
    C.readfile()
    C.loadSimParams()

    print("Test Passesd!")
    return 0


if __name__ == "__main__":
    test()







