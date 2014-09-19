import numpy
from . import logger

log = logger.Logger()


#######################
#New style configuration stuff
###

class ConfigurationError(Exception):
    pass


class Configurator(object):

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
        Dummy method to be overidden if requiredParams
        """
        pass

class SimConfig(ConfigObj):

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
                            ]
        self.initParams()


class TelConfig(ConfigObj):

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







