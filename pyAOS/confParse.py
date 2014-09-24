#Copyright Durham University and Andrew Reeves
#2014

# This file is part of pyAOS.

#     pyAOS is free software: you can redistribute it and/or modify
#     it under the terms of the GNU General Public License as published by
#     the Free Software Foundation, either version 3 of the License, or
#     (at your option) any later version.

#     pyAOS is distributed in the hope that it will be useful,
#     but WITHOUT ANY WARRANTY; without even the implied warranty of
#     MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
#     GNU General Public License for more details.

#     You should have received a copy of the GNU General Public License
#     along with pyAOS.  If not, see <http://www.gnu.org/licenses/>.
"""
A module to generate configuration objects for the pyAOS, given a parameter file.

This module defines a number of classes, which when instantiated, create objects used to configure the entire simulation, or just submodules. All configuration objects are stored in the ``Configurator`` object which deals with loading parameters from file, checking some potential conflicts and using parameters to calculate some other parameters used in parts of the simulation.

The ``ConfigObj`` provides a base class used by other module configuration objects, and provides methods to read the parameters from the dictionary read from file, and set defaults if appropriate. Each other module in the system has its own configuration object, and for components such as wave-front sensors (WFSs), Deformable Mirrors (DMs), Laser Guide Stars (LGSs) and Science Cameras,  lists of the config objects for each component are created.


"""

import numpy
from . import logger

log = logger.Logger()


#######################
#New style configuration stuff
###

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
        self.wfs = []
        self.lgs = []
        self.sci = []
        self.dm = []

        self.sim = SimConfig()
        self.atmos = AtmosConfig()
        self.tel = TelConfig()

    def readfile(self):

        #Exec the config file, which should contain a dict ``simConfiguration``
        with open(self.filename) as file_:
            exec(file_.read(), globals())

        self.configDict = simConfiguration

    def loadSimParams(self):

        log.debug("Load Sim Params...")
        self.sim.loadParams(self.configDict["Sim"])

        log.debug("Load Atmosphere Params...")
        self.atmos.loadParams(self.configDict["Atmosphere"])

        log.debug("Load Telescope Params...")
        self.tel.loadParams(self.configDict["Telescope"])

        log.debug("Load WFS Params...")
        for wfs in range(self.sim.nGS):
            self.wfs.append(WfsConfig(wfs))
            self.wfs[wfs].loadParams(self.configDict["WFS"])

        log.debug("Load LGS Params")
        for lgs in range(self.sim.nGS):
            self.lgs.append(LgsConfig(lgs))
            self.lgs[lgs].loadParams(self.configDict["LGS"])

        log.debug("Load DM Params")
        for dm in range(self.sim.nDM):
            self.dm.append(DmConfig(dm))
            self.dm[dm].loadParams(self.configDict["DM"])

        log.debug("Load Science Params")
        for sci in range(self.sim.nSci):
            self.sci.append(SciConfig(sci))
            self.sci[sci].loadParams(self.configDict["Science"])


        self.sim.pxlScale = float(self.sim.pupilSize)/self.tel.telDiam

        #furthest out GS defines the sub-scrn size
        gsPos = []
        for gs in range(self.sim.nGS):
            gsPos.append(self.wfs[gs].GSPosition)
        maxGSPos = numpy.array(gsPos).max()
        self.sim.scrnSize = numpy.ceil(
                    2*self.sim.pxlScale*self.atmos.scrnHeights.max()
                    *maxGSPos*numpy.pi/(3600.*180) 
                    )+self.sim.pupilSize
        log.print_("ScreenSize: {}".format(self.sim.scrnSize))


class ConfigObj(object):
    def __init__(self):

        self.N = None

    def warnAndExit(self, param):

        message = "{0} not set!".format(param)
        log.warning(message)
        raise ConfigurationError(message)

    def warnAndDefault(self, param, newValue):
        message = "{0} not set, default to {1}".format(param, newValue)
        self.__setattr__(param, newValue)

        log.info(message)

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

            for param in self.optionalParams:
                try:
                    self.__setattr__(param[0], configDict[param[0]][self.N])
                except KeyError:
                    self.warnAndDefault(param[0], param[1])
                except IndexError:
                    raise ConfigurationError(
                                "Not enough values for {0}".format(param))
        else:
            for param in self.requiredParams:
                try:
                    self.__setattr__(param, configDict[param])
                except KeyError:
                    self.warnAndExit(param)

            for param in self.optionalParams:
                try:
                    self.__setattr__(param[0], configDict[param[0]])
                except KeyError:
                    self.warnAndDefault(param[0], param[1])

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
        pupilSize       int: Number of phase points across the simulation pupil
        nIters          int: Number of iteration to run simulation
        loopTime        float: Time between simulation frames (1/framerate)
        =============   ===================


    Optional:
        ==================  =================================   ===============
        **Parameter**       **Description**                         **Default**
        ------------------  ---------------------------------   ---------------
        ``nGS``             int: Number of Guide Stars and 
                            WFS                                 ``0``
        ``nDM``             int: Number of deformable Mirrors   ``0``
        ``nSci``            int: Number of Science Cameras      ``0``
        ``gain``            float: loop gain of system          ``0.6``
        ``aoloopMode``      string: loop "open" or "closed"     ``"closed"``
        ``reconstructor``   string: name of reconstructor 
                            class to use. See 
                            ``reconstructor`` module
                            for available reconstructors.       ``"MVM"``
        ``filePrefix``      string: directory name to store 
                            simulation data                     ``None``
        ``tipTilt``         bool: Does system use tip-tilt 
                            Mirror                              ``False``
        ``ttGain``          float: loop gain of tip-tilt 
                            Mirror                              ``0.6``
        ``wfsMP``           bool: Each WFS uses its own 
                            process                             ``False``
        ``verbosity``       int: debug output for the 
                            simulation ranging from 0 
                            (no-ouput) to 3 (all debug 
                            output)                             ``2``
        ``logfile``         string: name of file to store 
                            logging data,                       ``None``
        ``learnIters``      int: Number of `learn` iterations
                            for Learn & Apply reconstructor     ``0``
        ``learnAtmos``      string: if ``random``, then 
                            random phase screens used for 
                            `learn`                             ``random``
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
                                ("aoloopMode", "closed"),
                                ("reconstructor", "MVM"),
                                ("filePrefix", None),
                                ("saveSlopes", False),
                                ("saveDmCommands", False),
                                ("saveLgsPsf", False),
                                ("saveLearn", False),
                                ("saveStrehl", False),
                                ("saveWfsFrames", False),
                                ("saveSciPsf", False),
                                ("saveSciRes", False),
                                ("tipTilt", False),
                                ("ttGain", 0.6),
                                ("wfsMP", False),
                                ("verbosity", 2),
                                ("logfile", None),
                                ("learnIters", 0),
                                ("learnAtmos", "random"),
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
        ``r0``                  float: integrated seeing strength 
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

        self.optionalParams = [("scrnNames",None)]

        self.initParams()


class WfsConfig(ConfigObj):

    def __init__(self, N):
        """
        Configuration parameters characterising Wave-front Sensors. These should be held in the ``WFS`` sub-dictionary of the ``simConfiguration`` dictionary in the parameter file. Each parameter must be in the form of a list, where each entry corresponds to a WFS. Any entries above ``sim.nGS`` will be ignored.

        Required:
            ==================      ===================
            **Parameter**           **Description** 
            ------------------      -------------------

            ==================      ===================

        Optional:
            ==================  =================================   ===========
            **Parameter**       **Description**                     **Default**
            ------------------  ---------------------------------   -----------
            ``scrnNames``       list, string: filenames of phase
                                if loading from fits files. If 
                                ``None`` will make new screens.     ``None``
            ==================  =================================   =========== 


        """
        super(WfsConfig, self).__init__()

        self.N = N

        self.requiredParams = [ "GSPosition",
                                "wavelength",
                                "subaps",
                                "pxlsPerSubap",
                                "subapFOV",
                            ]

        self.optionalParams = [ ("propagationMode", "geometric"),
                                ("fftwThreads", 1),
                                ("fftwFlag", "FFTW_PATIENT"),
                                ("SNR", 0),
                                ("angleEquivNoise", 0),
                                ("bitDepth", 32),
                                ("removeTT", "False"),
                                ("angleEquivNoise", 0),
                                ("subapOversamp", 2),
                                ("GSHeight", 0),
                                ("subapThreshold", 0.5),
                                ("lgs",False),
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
        ``obs``             float: Diameter of central
                            obscuration                         ``0``
        ``mask``            string: Shape of pupil (only 
                            accepts ``circle`` currently)       ``circle``
        ==================  =================================   ===========  

    """
    def __init__(self):

        super(TelConfig, self).__init__()

        self.requiredParams = [ "telDiam",
                                ]

        self.optionalParams = [ ("obs", 0),
                                ("mask", "circle")
                                ]

        self.initParams()
    

class LgsConfig(ConfigObj):

    def __init__(self, N):
        super(LgsConfig, self).__init__()

        self.N = N

        self.requiredParams = [ ]

        self.optionalParams = [ ("lgsUplink", False),
                                ("lgsPupilDiam", 0.3),
                                ("wavelength", 600e-9),
                                ("propagationMode", "physical"),
                                ("height", 90000),
                                ("fftwFlag", "FFTW_PATIENT"),
                                ("fftwThreads", 0),
                                ("elongationDepth", 0),
                                ("elongationLayers", 10),
                                ("launchPosition",  numpy.array([0,0]))
                                ]


        self.initParams()


class DmConfig(ConfigObj):

    def __init__(self, N):
        super(DmConfig, self).__init__()

        self.N = N

        self.requiredParams = [ "dmType",
                                "dmActs",
                                "dmCond",
                                ]


        self.optionalParams = [ 
                                ]
        self.initParams()


class SciConfig(ConfigObj):

    def __init__(self, N):

        super(SciConfig, self).__init__()

        self.N = N

        self.requiredParams = [ "position",
                                "FOV",
                                "wavelength",
                                "pxls",
                                ]
        self.optionalParams = [ ("oversamp", 2),
                                ("pyfftw_FLAG", "FFTW_MEASURE"),
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







