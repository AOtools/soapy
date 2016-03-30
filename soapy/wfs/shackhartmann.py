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


class ShackHartmann(base.WFS):
    """Class to simulate a Shack-Hartmann WFS"""

    def calcInitParams(self):
        """
        Calculate some parameters to be used during initialisation
        """
        super(ShackHartmann, self).calcInitParams()

        self.subapFOVrad = self.config.subapFOV * numpy.pi / (180. * 3600)
        self.subapDiam = self.los.telDiam/self.config.nxSubaps

        # spacing between subaps in pupil Plane (size "pupilSize")
        self.PPSpacing = float(self.simConfig.pupilSize)/self.config.nxSubaps

        # Spacing on the "FOV Plane" - the number of elements required
        # for the correct subap FOV (from way FFT "phase" to "image" works)
        self.subapFOVSpacing = numpy.round(
                self.subapDiam * self.subapFOVrad/ self.config.wavelength)

        # make twice as big to double subap FOV
        if self.config.subapFieldStop==True:
            self.SUBAP_OVERSIZE = 1
        else:
            self.SUBAP_OVERSIZE = 2

        self.detectorPxls = self.config.pxlsPerSubap*self.config.nxSubaps
        self.subapFOVSpacing *= self.SUBAP_OVERSIZE
        self.config.pxlsPerSubap2 = (self.SUBAP_OVERSIZE
                                            *self.config.pxlsPerSubap)

        # The total size of the required EField for all subaps.
        # Extra scaling to account for simSize padding
        self.scaledEFieldSize = int(round(
                self.config.nxSubaps*self.subapFOVSpacing*
                (float(self.simConfig.simSize)/self.simConfig.pupilSize)
                ))

        # If physical prop, must always be at same pixel scale
        # If not, can use less phase points for speed
        if self.config.propagationMode=="Physical":
            # This the pixel scale required for the correct FOV
            outPxlScale = (float(self.simConfig.simSize)/float(self.scaledEFieldSize)) * (self.simConfig.pxlScale**-1)
            self.los.calcInitParams(
                    outPxlScale=outPxlScale, nOutPxls=self.scaledEFieldSize)


        # Calculate the subaps that are actually seen behind the pupil mask
        self.findActiveSubaps()

        self.referenceImage = self.wfsConfig.referenceImage

    def findActiveSubaps(self):
        '''
        Finds the subapertures which are not empty space
        determined if mean of subap coords of the mask is above threshold.
        '''

        mask = self.mask[
                self.simConfig.simPad : -self.simConfig.simPad,
                self.simConfig.simPad : -self.simConfig.simPad
                ]
        self.subapCoords, self.subapFillFactor = aoSimLib.findActiveSubaps(
                self.wfsConfig.nxSubaps, mask,
                self.wfsConfig.subapThreshold, returnFill=True)

        self.activeSubaps = self.subapCoords.shape[0]
        self.detectorSubapCoords = numpy.round(
                self.subapCoords*(
                        self.detectorPxls/float(self.simConfig.pupilSize) ) )

        self.setMask(self.mask)

    def setMask(self, mask):
        super(ShackHartmann, self).setMask(mask)

        # Find the mask to apply to the scaled EField
        self.scaledMask = numpy.round(aoSimLib.zoom(
                    self.mask, self.scaledEFieldSize))

        p = self.simConfig.simPad
        self.subapFillFactor = aoSimLib.computeFillFactor(
                self.mask[p:-p, p:-p],
                self.subapCoords,
                round(float(self.simConfig.pupilSize)/self.wfsConfig.nxSubaps)
                )


    def initFFTs(self):
        """
        Initialise the FFT Objects required for running the WFS

        Initialised various FFT objects which are used through the WFS,
        these include FFTs to calculate focal planes, and to convolve LGS
        PSFs with the focal planes
        """

        #Calculate the FFT padding to use
        self.subapFFTPadding = self.wfsConfig.pxlsPerSubap2 * self.wfsConfig.fftOversamp
        if self.subapFFTPadding < self.subapFOVSpacing:
            while self.subapFFTPadding<self.subapFOVSpacing:
                self.wfsConfig.fftOversamp+=1
                self.subapFFTPadding\
                        =self.wfsConfig.pxlsPerSubap2*self.wfsConfig.fftOversamp

            logger.warning("requested WFS FFT Padding less than FOV size... Setting oversampling to: %d"%self.wfsConfig.fftOversamp)

        #Init the FFT to the focal plane
        self.FFT = AOFFT.FFT(
                inputSize=(
                self.activeSubaps, self.subapFFTPadding, self.subapFFTPadding),
                axes=(-2,-1), mode="pyfftw",dtype=CDTYPE,
                THREADS=self.wfsConfig.fftwThreads,
                fftw_FLAGS=(self.wfsConfig.fftwFlag,"FFTW_DESTROY_INPUT"))

        # If LGS uplink, init FFTs to conovolve LGS PSF and WFS PSF(s)
        # This works even if no lgsConfig.uplink as ``and`` short circuits
        if self.lgsConfig and self.lgsConfig.uplink:
            self.iFFT = AOFFT.FFT(
                    inputSize = (self.activeSubaps,
                                        self.subapFFTPadding,
                                        self.subapFFTPadding),
                    axes=(-2,-1), mode="pyfftw",dtype=CDTYPE,
                    THREADS=self.wfsConfig.fftwThreads,
                    fftw_FLAGS=(self.wfsConfig.fftwFlag,"FFTW_DESTROY_INPUT")
                    )

            self.lgs_iFFT = AOFFT.FFT(
                    inputSize = (self.subapFFTPadding,
                                self.subapFFTPadding),
                    axes=(0,1), mode="pyfftw",dtype=CDTYPE,
                    THREADS=self.wfsConfig.fftwThreads,
                    fftw_FLAGS=(self.wfsConfig.fftwFlag,"FFTW_DESTROY_INPUT")
                    )

    def initLGS(self):
        super(ShackHartmann, self).initLGS()
        if self.lgsConfig.uplink:
            lgsObj = getattr(
                    LGS, "LGS_{}".format(self.lgsConfig.propagationMode))
            self.lgs = lgsObj(
                    self.config, self.soapyConfig,
                    nOutPxls=self.subapFFTPadding,
                    outPxlScale=float(self.config.subapFOV)/self.subapFFTPadding
                    )

    def allocDataArrays(self):
        """
        Allocate the data arrays the WFS will require

        Determines and allocates the various arrays the WFS will require to
        avoid having to re-alloc memory during the running of the WFS and
        keep it fast.
        """
        self.los.allocDataArrays()

        self.subapArrays=numpy.zeros((self.activeSubaps,
                                      self.subapFOVSpacing,
                                      self.subapFOVSpacing),
                                     dtype=CDTYPE)
        self.binnedFPSubapArrays = numpy.zeros( (self.activeSubaps,
                                                self.wfsConfig.pxlsPerSubap2,
                                                self.wfsConfig.pxlsPerSubap2),
                                                dtype=DTYPE)
        self.FPSubapArrays = numpy.zeros((self.activeSubaps,
                                          self.subapFFTPadding,
                                          self.subapFFTPadding),dtype=DTYPE)

        self.wfsDetectorPlane = numpy.zeros( (  self.detectorPxls,
                                                self.detectorPxls   ),
                                                dtype = DTYPE )
        #Array used when centroiding subaps
        self.centSubapArrays = numpy.zeros( (self.activeSubaps,
              self.config.pxlsPerSubap, self.wfsConfig.pxlsPerSubap) )

        self.slopes = numpy.zeros( 2*self.activeSubaps )


    def calcTiltCorrect(self):
        """
        Calculates the required tilt to add to avoid the PSF being centred on
        only 1 pixel
        """
        if not self.wfsConfig.pxlsPerSubap%2:
            # If pxlsPerSubap is even
            # Angle we need to correct for half a pixel
            theta = self.SUBAP_OVERSIZE*self.subapFOVrad/ (
                    2*self.subapFFTPadding)

            # Magnitude of tilt required to get that angle
            A = theta * self.subapDiam/(2*self.wfsConfig.wavelength)*2*numpy.pi

            # Create tilt arrays and apply magnitude
            coords = numpy.linspace(-1, 1, self.subapFOVSpacing)
            X,Y = numpy.meshgrid(coords,coords)

            self.tiltFix = -1 * A * (X+Y)

        else:
            self.tiltFix = numpy.zeros((self.subapFOVSpacing,)*2)

    def getStatic(self):
        """
        Computes the static measurements, i.e., slopes with flat wavefront
        """

        self.staticData = None

        # Make flat wavefront, and run through WFS in iMat mode to turn off features
        phs = numpy.zeros([self.simConfig.scrnSize]*2).astype(DTYPE)
        self.staticData = self.frame(
                phs, iMatFrame=True).copy().reshape(2,self.activeSubaps)
#######################################################################


    def zeroData(self, detector=True, FP=True):
        """
        Sets data structures in WFS to zero.

        Parameters:
            detector (bool, optional): Zero the detector? default:True
            FP (bool, optional): Zero intermediate focal plane arrays? default: True
        """

        self.zeroPhaseData()

        if FP:
            self.FPSubapArrays[:] = 0

        if detector:
            self.wfsDetectorPlane[:] = 0


    def calcFocalPlane(self, intensity=1):
        '''
        Calculates the wfs focal plane, given the phase across the WFS

        Parameters:
            intensity (float): The relative intensity of this frame, is used when multiple WFS frames taken for extended sources.
        '''

        if self.config.propagationMode=="Geometric":
            # Have to make phase the correct size if geometric prop
            scaledEField = aoSimLib.zoom(self.los.phase, self.scaledEFieldSize)
            scaledEField = numpy.exp(1j*scaledEField)
        else:
            scaledEField = self.EField

        # Apply the scaled pupil mask
        scaledEField *= self.scaledMask

        # Now cut out only the eField across the pupilSize
        coord = round(int(((self.scaledEFieldSize/2.)
                - (self.wfsConfig.nxSubaps*self.subapFOVSpacing)/2.)))
        self.cropEField = scaledEField[coord:-coord, coord:-coord]

        #create an array of individual subap EFields
        for i in xrange(self.activeSubaps):
            x,y = numpy.round(self.subapCoords[i] *
                                     self.subapFOVSpacing/self.PPSpacing)
            self.subapArrays[i] = self.cropEField[
                                    int(x):
                                    int(x+self.subapFOVSpacing) ,
                                    int(y):
                                    int(y+self.subapFOVSpacing)]

        #do the fft to all subaps at the same time
        # and convert into intensity
        self.FFT.inputData[:] = 0
        self.FFT.inputData[:,:int(round(self.subapFOVSpacing))
                        ,:int(round(self.subapFOVSpacing))] \
                = self.subapArrays*numpy.exp(1j*(self.tiltFix))

        if intensity==1:
            self.FPSubapArrays += numpy.abs(AOFFT.ftShift2d(self.FFT()))**2
        else:
            self.FPSubapArrays += intensity*numpy.abs(
                    AOFFT.ftShift2d(self.FFT()))**2

    def makeDetectorPlane(self):
        '''
        Scales and bins intensity data onto the detector with a given number of
        pixels.

        If required, will first convolve final PSF with LGS PSF, then bin
        PSF down to detector size. Finally puts back into ``wfsFocalPlane``
        array in correct order.
        '''

        # If required, convolve with LGS PSF
        if self.wfsConfig.lgs and self.lgs and self.lgsConfig.uplink and self.iMat!=True:
            self.applyLgsUplink()


        # bins back down to correct size and then
        # fits them back in to a focal plane array
        self.binnedFPSubapArrays[:] = aoSimLib.binImgs(self.FPSubapArrays,
                                            self.wfsConfig.fftOversamp)

        # In case of empty sub-aps, will get NaNs
        self.binnedFPSubapArrays[numpy.isnan(self.binnedFPSubapArrays)] = 0

        # Scale each sub-ap flux by sub-aperture fill-factor
        self.binnedFPSubapArrays\
                = (self.binnedFPSubapArrays.T * self.subapFillFactor).T

        for i in xrange(self.activeSubaps):
            x,y=self.detectorSubapCoords[i]

            #Set default position to put arrays into (SUBAP_OVERSIZE FOV)
            x1 = int(round(
                    x+self.wfsConfig.pxlsPerSubap/2.
                    -self.wfsConfig.pxlsPerSubap2/2.))
            x2 = int(round(
                    x+self.wfsConfig.pxlsPerSubap/2.
                    +self.wfsConfig.pxlsPerSubap2/2.))
            y1 = int(round(
                    y+self.wfsConfig.pxlsPerSubap/2.
                    -self.wfsConfig.pxlsPerSubap2/2.))
            y2 = int(round(
                    y+self.wfsConfig.pxlsPerSubap/2.
                    +self.wfsConfig.pxlsPerSubap2/2.))

            #Set defualt size of input array (i.e. all of it)
            x1_fp = int(0)
            x2_fp = int(round(self.wfsConfig.pxlsPerSubap2))
            y1_fp = int(0)
            y2_fp = int(round(self.wfsConfig.pxlsPerSubap2))

            # If at the edge of the field, may only fit a fraction in
            if x == 0:
                x1 = 0
                x1_fp = int(round(
                        self.wfsConfig.pxlsPerSubap2/2.
                        -self.wfsConfig.pxlsPerSubap/2.))

            elif x == (self.detectorPxls-self.wfsConfig.pxlsPerSubap):
                x2 = int(round(self.detectorPxls))
                x2_fp = int(round(
                        self.wfsConfig.pxlsPerSubap2/2.
                        +self.wfsConfig.pxlsPerSubap/2.))

            if y == 0:
                y1 = 0
                y1_fp = int(round(
                        self.wfsConfig.pxlsPerSubap2/2.
                        -self.wfsConfig.pxlsPerSubap/2.))

            elif y == (self.detectorPxls-self.wfsConfig.pxlsPerSubap):
                y2 = int(self.detectorPxls)
                y2_fp = int(round(
                        self.wfsConfig.pxlsPerSubap2/2.
                        +self.wfsConfig.pxlsPerSubap/2.))

            self.wfsDetectorPlane[x1:x2, y1:y2] += (
                    self.binnedFPSubapArrays[i, x1_fp:x2_fp, y1_fp:y2_fp])

        # Scale data for correct number of photons
        self.wfsDetectorPlane /= self.wfsDetectorPlane.sum()
        self.wfsDetectorPlane *= aoSimLib.photonsPerMag(
                self.wfsConfig.GSMag, self.mask, self.simConfig.pxlScale**(-1),
                self.wfsConfig.wvlBandWidth, self.wfsConfig.exposureTime
                ) * self.wfsConfig.throughput

        if self.wfsConfig.photonNoise:
            self.addPhotonNoise()

        if self.wfsConfig.eReadNoise!=0:
            self.addReadNoise()

    def applyLgsUplink(self):
        '''
        A method to deal with convolving the LGS PSF
        with the subap focal plane.
        '''

        self.lgs.getLgsPsf(self.los.scrns)

        self.lgs_iFFT.inputData[:] = self.lgs.psf
        self.iFFTLGSPSF = self.lgs_iFFT()

        self.iFFT.inputData[:] = self.FPSubapArrays
        self.iFFTFPSubapsArray = self.iFFT()

        # Do convolution
        self.iFFTFPSubapsArray *= self.iFFTLGSPSF

        # back to Focal Plane.
        self.FFT.inputData[:] = self.iFFTFPSubapsArray
        self.FPSubapArrays[:] = AOFFT.ftShift2d(self.FFT()).real

    def calculateSlopes(self):
        '''
        Calculates WFS slopes from wfsFocalPlane

        Returns:
            ndarray: array of all WFS measurements
        '''

        # Sort out FP into subaps
        for i in xrange(self.activeSubaps):
            x, y = self.detectorSubapCoords[i]
            x = int(x)
            y = int(y)
            self.centSubapArrays[i] = self.wfsDetectorPlane[
                    x:x+self.wfsConfig.pxlsPerSubap,
                    y:y+self.wfsConfig.pxlsPerSubap ].astype(DTYPE)

        slopes = getattr(centroiders, self.wfsConfig.centMethod)(
                self.centSubapArrays,
                threshold=self.wfsConfig.centThreshold,
                ref=self.referenceImage
                )


        # shift slopes relative to subap centre and remove static offsets
        slopes -= self.wfsConfig.pxlsPerSubap/2.0

        if numpy.any(self.staticData):
            slopes -= self.staticData

        self.slopes[:] = slopes.reshape(self.activeSubaps*2)

        if self.wfsConfig.removeTT == True:
            self.slopes[:self.activeSubaps] -= self.slopes[:self.activeSubaps].mean()
            self.slopes[self.activeSubaps:] -= self.slopes[self.activeSubaps:].mean()

        if self.wfsConfig.angleEquivNoise and not self.iMat:
            pxlEquivNoise = (
                    self.wfsConfig.angleEquivNoise *
                    float(self.wfsConfig.pxlsPerSubap)
                    /self.wfsConfig.subapFOV )
            self.slopes += numpy.random.normal(
                    0, pxlEquivNoise, 2*self.activeSubaps)

        return self.slopes
