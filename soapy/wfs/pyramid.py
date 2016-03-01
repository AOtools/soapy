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


class Pyramid(base.WFS):
    """
    *Experimental* Pyramid WFS.

    This is an early prototype for a Pyramid WFS. Currently, its at a very early stage. It doesn't oscillate, so performance aint too good at the minute.

    To use, set the wfs parameter ``type'' to ``Pyramid'' type is a list of length number of wfs.
    """
    # oversampling for the first FFT from EField to focus (4 seems ok...)
    FOV_OVERSAMP = 4

    def calcInitParams(self):
        super(Pyramid, self).calcInitParams()
        self.FOVrad = self.wfsConfig.subapFOV * numpy.pi / (180. * 3600)

        self.FOVPxlNo = numpy.round(self.telDiam *
                                    self.FOVrad/self.wfsConfig.wavelength)

        self.detectorPxls = 2*self.wfsConfig.pxlsPerSubap
        self.scaledMask = aoSimLib.zoom(self.mask, self.FOVPxlNo)

        self.activeSubaps = self.wfsConfig.pxlsPerSubap**2

        while (self.wfsConfig.pxlsPerSubap*self.wfsConfig.fftOversamp
                    < self.FOVPxlNo):
            self.wfsConfig.fftOversamp += 1

    def initFFTs(self):

        self.FFT = AOFFT.FFT(   [self.FOV_OVERSAMP*self.FOVPxlNo,]*2,
                                axes=(0,1), mode="pyfftw",
                                fftw_FLAGS=("FFTW_DESTROY_INPUT",
                                            self.wfsConfig.fftwFlag),
                                THREADS=self.wfsConfig.fftwThreads
                                )

        self.iFFTPadding = self.FOV_OVERSAMP*(self.wfsConfig.fftOversamp*
                                            self.wfsConfig.pxlsPerSubap)
        self.iFFT = AOFFT.FFT(
                    [4, self.iFFTPadding, self.iFFTPadding],
                    axes=(1,2), mode="pyfftw",
                    THREADS = self.wfsConfig.fftwThreads,
                    fftw_FLAGS=("FFTW_DESTROY_INPUT", self.wfsConfig.fftwFlag),
                    direction="BACKWARD"
                    )

    def allocDataArrays(self):

        super(Pyramid, self).allocDataArrays()
        # Allocate arrays
        # Find sizes of detector planes

        self.paddedDetectorPxls = (2*self.wfsConfig.pxlsPerSubap
                                    *self.wfsConfig.fftOversamp)
        self.paddedDetectorPlane = numpy.zeros([self.paddedDetectorPxls]*2,
                                                dtype=DTYPE)

        self.focalPlane = numpy.zeros( [self.FOV_OVERSAMP*self.FOVPxlNo,]*2,
                                        dtype=CDTYPE)

        self.quads = numpy.zeros(
                    (4,self.focalPlane.shape[0]/2.,self.focalPlane.shape[1]/2.),
                    dtype=CDTYPE)

        self.wfsDetectorPlane = numpy.zeros([self.detectorPxls]*2,
                                            dtype=DTYPE)

        self.slopes = numpy.zeros(2*self.activeSubaps)

    def zeroData(self, detector=True, inter=True):
        """
        Sets data structures in WFS to zero.

        Parameters:
            detector (bool, optional): Zero the detector? default:True
            inter (bool, optional): Zero intermediate arrays? default:True
        """

        self.zeroPhaseData()

        if inter:
            self.paddedDetectorPlane[:] = 0

        if detector:
            self.wfsDetectorPlane[:] = 0

    def calcFocalPlane(self):
        '''
        takes the calculated pupil phase, and uses FFT
        to transform to the focal plane, and scales for correct FOV.
        '''
        # Apply tilt fix and scale EField for correct FOV
        self.pupilEField = self.EField[
                self.simConfig.simPad:-self.simConfig.simPad,
                self.simConfig.simPad:-self.simConfig.simPad
                ]
        self.pupilEField *= numpy.exp(1j*self.tiltFix)
        self.scaledEField = aoSimLib.zoom(
                self.pupilEField, self.FOVPxlNo)*self.scaledMask

        # Go to the focus
        self.FFT.inputData[:] = 0
        self.FFT.inputData[ :self.FOVPxlNo,
                            :self.FOVPxlNo ] = self.scaledEField
        self.focalPlane[:] = AOFFT.ftShift2d( self.FFT() )

        #Cut focus into 4
        shapeX, shapeY = self.focalPlane.shape
        n=0
        for x in xrange(2):
            for y in xrange(2):
                self.quads[n] = self.focalPlane[x*shapeX/2 : (x+1)*shapeX/2,
                                                y*shapeX/2 : (y+1)*shapeX/2]
                n+=1

        #Propogate each quadrant back to the pupil plane
        self.iFFT.inputData[:] = 0
        self.iFFT.inputData[:,
                            :0.5*self.FOV_OVERSAMP*self.FOVPxlNo,
                            :0.5*self.FOV_OVERSAMP*self.FOVPxlNo] = self.quads
        self.pupilImages = abs(AOFFT.ftShift2d(self.iFFT()))**2

        size = self.paddedDetectorPxls/2
        pSize = self.iFFTPadding/2.


        #add this onto the padded detector array
        for x in range(2):
            for y in range(2):
                self.paddedDetectorPlane[
                        x*size:(x+1)*size,
                        y*size:(y+1)*size] += self.pupilImages[
                                                2*x+y,
                                                pSize:pSize+size,
                                                pSize:pSize+size]

    def makeDetectorPlane(self):

        #Bin down to requried pixels
        self.wfsDetectorPlane[:] += aoSimLib.binImgs(
                        self.paddedDetectorPlane,
                        self.wfsConfig.fftOversamp
                        )

    def calculateSlopes(self):

        xDiff = (self.wfsDetectorPlane[ :self.wfsConfig.pxlsPerSubap,:]-
                    self.wfsDetectorPlane[  self.wfsConfig.pxlsPerSubap:,:])
        xSlopes = (xDiff[:,:self.wfsConfig.pxlsPerSubap]
                    +xDiff[:,self.wfsConfig.pxlsPerSubap:])

        yDiff = (self.wfsDetectorPlane[:, :self.wfsConfig.pxlsPerSubap]-
                    self.wfsDetectorPlane[:, self.wfsConfig.pxlsPerSubap:])
        ySlopes = (yDiff[:self.wfsConfig.pxlsPerSubap, :]
                    +yDiff[self.wfsConfig.pxlsPerSubap:, :])


        self.slopes[:] = numpy.append(xSlopes.flatten(), ySlopes.flatten())

    #Tilt optimisation
    ################################
    def calcTiltCorrect(self):
        """
        Calculates the required tilt to add to avoid the PSF being centred on
        only 1 pixel
        """
        if not self.wfsConfig.pxlsPerSubap%2:
            #Angle we need to correct
            theta = self.FOVrad/ (2*self.FOV_OVERSAMP*self.FOVPxlNo)

            A = theta*self.telDiam/(2*self.wfsConfig.wavelength)*2*numpy.pi

            coords = numpy.linspace(-1,1,self.simConfig.pupilSize)
            X,Y = numpy.meshgrid(coords,coords)

            self.tiltFix = -1*A*(X+Y)

        else:
            self.tiltFix = numpy.zeros((self.simConfig.pupilSize,)*2)
