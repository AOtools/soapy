import numpy
import circle
import zerns
import sys
import pylab
import make_gammas
import progressbar
import aoToolsLib

class AO:
    '''
    Class which will do lots of things to analyse an AO system.
    Ideally made for DRAGON and Andrews AO Sim
    '''

    def loadSimDir(self,dir):
        '''
        Loads slopes, dmActs and configuration params
        from and AO Sim directory
        '''
        sys.path.append( dir )
        configObject = __import__("conf.py")
        confDict = configObject.parameters

        self.GSNo = confDict["GSNo"]
        self.subaps = confDict["subaps"]
        self.subapFOV = confDict["subapFOV"]
        self.wvls   = confDict["waveLengths"]
        self.pxlsPerSubap = confDict["pxlsPerSubap"]
        self.subapThreshold = confDict["threshold"]

        self.FR = 1./confDict["loopTime"]

        self.dmNo = confDict["dmNo"]

        self.rawSlopes = FITS.Read(dir+"/slopes.fits")
        self.slopes = self.rawSlopes.copy()

        self.dmActs = {}
        for i in range(self.dmNo):
            self.dmActs[i] = FITS.Read(dir+"/dm_%i.fits"%i)

    def loadSimObj(self,sim):

        self.GSNo = sim.GSNo
        self.subaps = sim.subaps
        self.subapFOV = sim.subaoFIV
        self.wvls = sim.waveLengths
        self.pxlsPerSubap = sim.pxlsPerSubap
        self.subapThreshold = sim.subapThreshold

        self.FR = 1./sim.loopTime

        self.dmNo = sim.dmNo

        self.rawSlopes = []
        slopes = sim.slopes.copy()
        slopeShape = slopes.shape
        slopes.shape = slopeShape[0],self.GSNo, slopeShape/self.GSNo

        self.maks = sim.mask

    def temporalPowerSpectrum(self):
        tPS = aoToolsLib.temporalPowerSpectrum(self.slopes)[0]
        return tPS

    def plotTPS = (self):
        tPS = temporalPowerSpectrum()
        pylab.figure()
        f = numpy.arange( tPS.shape[0] )

        pylab.loglog( f, tPS )
        pylab.title("Temporal Power Spectrum")
        pylab.xlabel("Frequency (Hz)")
        pylab.ylabel("Power")
        pylab.show()

    def spatialPowerSpectrum(self):
        slopes2d = self.remapSubaps()
        sPS = aoToolsLib.spatialPowerSpectrum(slopes2d)

        return sPS

    def plotSPS(self):
        sPS = self.spatialPowerSpectrum()

        pylab.figure()
        f = numpy.arange(sPS.shape)

    def remapSubaps(self):
        try:
            self.subapPositions
        else:
            self.subapPositions = aoToolsLib.getActiveSubaps(self.mask,
                                                             self.subaps,
                                                             self.threshold)
        slopes2d = aoToolsLib.remapSubaps(self.slopes, self.subaps,
                                          self.subapPositions)
        return slopes2d








