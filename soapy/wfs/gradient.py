import numpy
import numpy.random
from scipy.interpolate import interp2d
try:
    from astropy.io import fits
except ImportError:
    try:
        import pyfits as fits
    except ImportError:
        raise ImportError("PyAOS requires either pyfits or astropy")

from .. import AOFFT, aoSimLib, LGS, logger
from . import base
from ..tools import centroiders
from ..opticalPropagationLib import angularSpectrum

# xrange now just "range" in python3.
# Following code means fastest implementation used in 2 and 3
try:
    xrange
except NameError:
    xrange = range

# The data type of data arrays (complex and real respectively)
CDTYPE = numpy.complex64
DTYPE = numpy.float32

RAD2ASEC = 206264.849159
ASEC2RAD = 1./RAD2ASEC

class Gradient(base.WFS):

    def calcInitParams(self):
        super(Gradient, self).calcInitParams()
        self.subapSpacing = self.simConfig.pupilSize/self.wfsConfig.nxSubaps
        self.findActiveSubaps()

        # Normalise gradient measurement to 1 radian
        self.subapDiam = float(self.telConfig.telDiam) / self.wfsConfig.nxSubaps

        # Amp in m of 1 arcsecond tilt for single sub-aperture
        amp = self.telConfig.telDiam * 1. * ASEC2RAD
        
        # amp of 1" tilt in rads of the light
        amp *= ((2 * numpy.pi) / self.config.wavelength)

        # Arrays to be used for gradient calculation
        telCoord = numpy.linspace(0, amp, self.soapyConfig.sim.pupilSize)
        subapCoord = telCoord[:self.subapSpacing]
        
        # Remove piston
        subapCoord -= subapCoord.mean()
        subapCoord *= -1
        
        self.xGrad_1, self.yGrad_1 = numpy.meshgrid(subapCoord, subapCoord)
        
        self.xGrad = self.xGrad_1/((self.xGrad_1**2).sum())
        self.yGrad = self.yGrad_1/((self.yGrad_1**2).sum())

    def findActiveSubaps(self):
        '''
        Finds the subapertures which are not empty space
        determined if mean of subap coords of the mask is above threshold.
        '''
        pupilMask = self.mask[
                self.simConfig.simPad : -self.simConfig.simPad,
                self.simConfig.simPad : -self.simConfig.simPad
                ]
        self.subapCoords, self.subapFillFactor = aoSimLib.findActiveSubaps(
                self.wfsConfig.nxSubaps, pupilMask,
                self.wfsConfig.subapThreshold, returnFill=True)

        self.activeSubaps = self.subapCoords.shape[0]

    def allocDataArrays(self):
        """
        Allocate the data arrays the WFS will require

        Determines and allocates the various arrays the WFS will require to
        avoid having to re-alloc memory during the running of the WFS and
        keep it fast.
        """

        super(Gradient, self).allocDataArrays()

        self.subapArrays = numpy.zeros(
                (self.activeSubaps, self.subapSpacing, self.subapSpacing),
                dtype=DTYPE)

        self.slopes = numpy.zeros(2 * self.activeSubaps)


    def calcFocalPlane(self, intensity=1):
        '''
        Calculates the wfs focal plane, given the phase across the WFS. For this WFS, chops the pupil phase up into sub-apertures.

        Parameters:
            intensity (float): The relative intensity of this frame, is used when multiple WFS frames taken for extended sources.
        '''

        # Apply the scaled pupil mask
        self.los.phase *= self.mask

        # Now cut out only the phase across the pupilSize
        coord = self.simConfig.simPad
        self.pupilPhase = self.los.phase[coord:-coord, coord:-coord]

        # Create an array of individual subap phase
        for i, (x, y) in enumerate(self.subapCoords):
            x1 = int(round(x))
            x2 = int(round(x + self.subapSpacing))
            y1 = int(round(y))
            y2 = int(round(y + self.subapSpacing))
            self.subapArrays[i] = self.pupilPhase[x1: x2, y1: y2]


    def makeDetectorPlane(self):
        '''
        Creates a 'detector' image suitable for plotting
        '''
        self.wfsDetectorPlane = numpy.zeros((self.wfsConfig.nxSubaps,)*2)

        coords = (self.subapCoords/self.subapSpacing).astype('int')
        self.wfsDetectorPlane[coords[:, 0], coords[:, 1]] = self.subapArrays.mean((1, 2))

    def calculateSlopes(self):
        '''
        Calculates WFS slopes from wfsFocalPlane

        Returns:
            ndarray: array of all WFS measurements
        '''
        # Remove all piston from the sub-apertures
        # self.subapArrays = (self.subapArrays.T-self.subapArrays.mean((1,2))).T

        # Integrate with tilt/tip to get slope measurements
        for i, subap in enumerate(self.subapArrays):
            subap -= subap.mean()
            self.slopes[i] = (subap * self.xGrad).sum()
            self.slopes[i+self.activeSubaps] = (subap * self.yGrad).sum()

        # self.slopes[:self.activeSubaps] = self.xSlopes
        # self.slopes[self.activeSubaps:] = self.ySlopes

        # Remove tip-tilt if required
        if self.wfsConfig.removeTT == True:
            self.slopes[:self.activeSubaps] -= self.slopes[:self.activeSubaps].mean()
            self.slopes[self.activeSubaps:] -= self.slopes[self.activeSubaps:].mean()

        # Add 'angle equivalent noise' if asked for
        if self.wfsConfig.angleEquivNoise and not self.iMat:
            pxlEquivNoise = (
                    self.wfsConfig.angleEquivNoise *
                    float(self.wfsConfig.pxlsPerSubap)
                    /self.wfsConfig.subapFOV )
            self.slopes += numpy.random.normal( 0, pxlEquivNoise,
                                                2*self.activeSubaps)

        return self.slopes
