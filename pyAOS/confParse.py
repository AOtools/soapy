import numpy
import logging
import sys
import traceback
from . import logger

log = logger.Logger()

def readParams(sim, configFile):

    '''
    Loads all the parameter"/"+s from the config file
    and stores then as data in the Sim class

    Args:
        sim (simObj): The simulation object, all paramaters will be given as attributes to this object.
        configFile (string): The name of the simulation config file
    '''


    confFile = open(configFile)
    exec(confFile.read(),globals())
    confDict = parameters

    #simulation params
    sim.pupilSize = confDict["pupilSize"] #pupil size in pxls
    try:
        sim.filePrefix = confDict["filePrefix"]
    except KeyError:
        sim.filePrefix = None
        
    if sim.filePrefix!=None:
        sim.filePrefix="data/"+sim.filePrefix

    #Atmosphere Params
    sim.scrnNo = confDict["scrnNo"]
    sim.newScreens = confDict["newScreens"]
    try:
        sim.scrnNames = confDict["scrnNames"]
    except KeyError:
        sim.newScreens = True
        sim.scrnNames= [None]*sim.scrnNo
    sim.scrnHeights = confDict["scrnHeights"]
    sim.scrnStrengths = confDict["scrnStrengths"]
    sim.windSpeeds = confDict["windSpeeds"]
    sim.windDirs = confDict["windDirs"]
    
    sim.wholeScrnSize = confDict["scrnSize"]
    sim.r0 = confDict["r0"]
    
    try:
        sim.learnAtmos = confDict["learnAtmos"]
    except KeyError:
        sim.learnAtmos = "scrns"
        

    #Telescope Params
    sim.telDiam = confDict["telDiam"]
    try:
        sim.obs = confDict["obs"]
    except KeyError:
        sim.obs = 0
    try:
        sim.mask = confDict["mask"] #pupilmask
    except KeyError:
        sim.mask = "circle"
        
    #WFSs
        
    sim.GSNo = confDict["GSNo"]
    sim.GSPositions = confDict["GSPositions"]
    sim.GSHeight = confDict["GSHeight"]
    sim.subaps = confDict["subaps"]
    sim.pxlsPerSubap = confDict["pxlsPerSubap"]
    sim.subapFOV = confDict["subapFOV"]
    
    try:
        sim.wfsPropMode = confDict["wfsPropMode"]
    except KeyError:
        sim.wfsPropMode = ["geometric"]*sim.GSNo
    
    try:
        sim.subapOversamp = confDict["subapOversamp"]
    except KeyError:
        sim.subapOversamp = 4    
    
    try:
        sim.subapThreshold = confDict["subapThreshold"]
    except KeyError:
        sim.subapThreshold = 0.5
    
    try:
        sim.waveLengths = confDict["waveLengths"]
    except KeyError:
        sim.waveLengths = [550e-9]*im.GSNo
        
    try:
        sim.wfsFftProcs = confDict["wfsFftProcs"]
    except KeyError:
        sim.wfsFftProcs = 1

    try:
        sim.wfsPyfftw_FLAG = confDict["wfsPyfftw_FLAG"]
    except KeyError:
        sim.wfsPyfftw_FLAG = "FFTW_PATIENT"

    try:
        sim.mp_wfs = confDict["mp_wfs"]
    except KeyError:
        sim.mp_wfs = False

    try:
        sim.removeTT = confDict["removeTT"]
    except KeyError:
        sim.removeTT = numpy.array([False]*sim.GSNo)

    try:
        sim.wfsSNR = numpy.array(confDict["wfsSNR"])
    except KeyError:
        sim.wfsSNR = numpy.array( [0]*sim.GSNo )
    try:
        sim.angleEquivNoise = numpy.array(confDict["angleEquivNoise"])
    except KeyError:
        sim.angleEquivNoise = numpy.array([0]*sim.GSNo)

    try:
        sim.wfsBitDepth = numpy.array(confDict["wfsBitDepth"])
    except KeyError:
        sim.wfsBitDepth = numpy.array( [8]*sim.GSNo )


    #LGS
    try:
        sim.LGSUplink = confDict["LGSUplink"]
    except KeyError:
        sim.LGSUplink = [0]*sim.GSNo
        
    sim.LGSPupilSize = confDict["LGSPupilSize"]
    
    sim.LGSFFTPadding = confDict["LGSFFTPadding"]
    sim.LGSWL = confDict["LGSWL"]
    try:
        sim.lgsPropMode = confDict["lgsPropMode"]
    except KeyError:
        sim.lgsPropMode = ["geometric"]*sim.GSNo
        
    sim.lgsHeight = confDict["lgsHeight"]
    try:
        sim.lgsPyfftw_FLAG = confDict["lgsPyfftw_FLAG"]
    except KeyError:
        sim.lgsPyfftw_FLAG = "FFTW_MEASURE"

    try:
        sim.elongation = confDict["lgsElongation"]
    except KeyError:
        sim.elongation = 0

    try:
        sim.elongLayers = confDict["elongLayers"]
    except KeyError:
        sim.elongLayers = 10

    try:
        sim.lgsLaunchPos = confDict["lgsLaunchPos"]
    except KeyError:
        sim.lgsLaunchPos = numpy.array( [[0,0]]*sim.GSNo)

    #DM Params
    try:
        sim.tipTilt = confDict["tipTilt"]
    except KeyError:
        sim.tipTilt = False

    sim.ttGain = confDict["ttGain"]
    sim.dmNo = confDict["dmNo"]
    sim.dmTypes = confDict["dmTypes"]
    sim.dmActs = confDict["dmActs"]
    sim.dmCond = confDict["dmCond"]

    #Science Params
    sim.sciNo = confDict["sciNo"]
    sim.sciPos = confDict["sciPos"]
    sim.sciFOV = confDict["sciFOV"]
    sim.sciWvls = confDict["sciWvls"]
    sim.sciPxls = confDict["sciPxls"]
    sim.sciOversamp = confDict["sciOversamp"]
    try:
        sim.sciPyfftw_FLAG = confDict["lgsPyfftw_FLAG"]
    except KeyError:
        sim.sciPyfftw_FLAG = "FFTW_MEASURE"


    #Loop
    sim.nIters = confDict["nIters"]
    sim.loopTime = confDict["loopTime"]
    sim.gain = confDict["gain"]
    sim.aoloopMode = confDict["loopMode"]
    sim.reconstructor = confDict["reconstructor"]
    sim.learnIters = confDict["learnIters"]


    #Set logging
    sim.loggingMode = confDict["logging"]


    #dataSaving
    sim.saveSlopes = confDict["saveSlopes"]
    sim.saveDmCommands = confDict["saveDmCommands"]
    sim.saveLgsPsf = confDict["saveLgsPsf"]
    try:
        sim.saveLearn = confDict["saveLearn"]
    except KeyError:
        sim.saveLearn = False

    try:
        sim.saveStrehl = confDict["saveStrehl"]
    except KeyError:
        sim.saveStrehl = False

    try:
        sim.saveWfsFrames = confDict["saveWfsFrames"]
    except KeyError:
        sim.saveWfsFrames = False

    try:
        sim.saveSciPsf = confDict["saveSciPsf"]
    except KeyError:
        sim.saveSciPsf = True

    try:
        sim.saveSciRes = confDict["saveSciRes"]
    except KeyError:
        sim.saveSciRes = False

    sim.go=False

    sim.guiQueue = None
    sim.guiLock = None
    sim.waitingPlot = False




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

        self.optionalParams = [  ("nGS", 0),
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
                                "scrnSize",
                                ]

        self.optionalParams = [("scrnNames",None)]

        self.initParams()


class WfsConfig(ConfigObj):

    def __init__(self, N):

        super(WfsConfig, self).__init__()

        self.N = N

        self.requiredParams = [ "GSPosition",
                                "wavelength"
                            ]

        self.optionalParams = [ ("propogagationMode", "geometric"),
                                ("fftProcs", "1"),
                                ("pyfftw_FLAG", "FFTW_PATIENT"),
                                ("SNR", "0"),
                                ("angleEquivNoise", "0"),
                                ("bitDepth", "32"),
                                ("removeTT", "False")
                            ]
        self.initParams()
        print("N:{}".format(N))

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

        self.optionalParams = [ ("LgsUplink", False),
                                ("LgsPupilSize", 0.3),
                                ("wavelength", 600e-9),
                                ("propogagationMode", "physical"),
                                ("height", 90000),
                                ("pyfftw_FLAG", "FFTW_PATIENT"),
                                ("elongationDepth", 0),
                                ("elongationLayers", 10),
                                ("launchPosition",  [0,0])
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







