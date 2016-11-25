"""
A Shack Hartmann WFS for use with extended reference sources, such as solar AO, where correlation centroiding techniques are required.

"""

import numpy
import scipy.signal

try:
    from astropy.io import fits
except ImportError:
    try:
        import pyfits as fits
    except ImportError:
        raise ImportError("PyAOS requires either pyfits or astropy")

from .. import AOFFT, logger
from . import shackhartmann
from ..aotools import centroiders

# xrange now just "range" in python3.
# Following code means fastest implementation used in 2 and 3
try:
    xrange
except NameError:
    xrange = range

# The data type of data arrays (complex and real respectively)
CDTYPE = numpy.complex64
DTYPE = numpy.float32

class ExtendedSH(shackhartmann.ShackHartmann):

    def calcInitParams(self):
        super(ExtendedSH, self).calcInitParams()

        # For correlation centroider, open reference image.
        self.referenceImage = self.wfsConfig.referenceImage

    def initFFTs(self):
        """
        Initialise an extra FFT for the convolution in the correlation
        """
        super(ExtendedSH, self).initFFTs()

        self.corrFFT = AOFFT.FFT(
                inputSize=(
                    self.activeSubaps, self.wfsConfig.pxlsPerSubap,
                    self.wfsConfig.pxlsPerSubap),
                axes=(-2,-1), mode="pyfftw",dtype=CDTYPE,
                THREADS=self.wfsConfig.fftwThreads,
                fftw_FLAGS=(self.wfsConfig.fftwFlag,"FFTW_DESTROY_INPUT"),
                )

        self.corrIFFT = AOFFT.FFT(
                inputSize=(
                    self.activeSubaps, self.wfsConfig.pxlsPerSubap,
                    self.wfsConfig.pxlsPerSubap),
                axes=(-2,-1), mode="pyfftw",dtype=CDTYPE,
                THREADS=self.wfsConfig.fftwThreads,
                fftw_FLAGS=(self.wfsConfig.fftwFlag,"FFTW_DESTROY_INPUT"),
                direction="BACKWARD")

        # Also open object if its given
        self.extendedObject = self.wfsConfig.extendedObject

    def allocDataArrays(self):
        super(ExtendedSH, self).allocDataArrays()

        # Make a convolution object to apply the object
        if self.extendedObject is None:
            self.objectConv = None
        else:
            self.objectConv = AOFFT.Convolve(
                    self.FPSubapArrays.shape, self.extendedObject.shape,
                    threads=self.wfsConfig.fftwThreads, axes=(-2, -1)
                    )

        self.corrSubapArrays = numpy.zeros(self.centSubapArrays.shape, dtype=DTYPE)

    def makeDetectorPlane(self):
        """
        If an extended object is supplied, convolve with spots to make
        the detector images
        """
        if self.extendedObject is not None:
            # Perform correlation to get subap images
            self.FPSubapArrays[:] = self.objectConv(
                    self.FPSubapArrays, self.extendedObject).real

        # If sub-ap is oversized, apply field mask (TODO:make more general)
        if self.SUBAP_OVERSIZE!=1:
            coord = int(self.subapFFTPadding/(2*self.SUBAP_OVERSIZE))
            fieldMask = numpy.zeros((self.subapFFTPadding,)*2)
            fieldMask[coord:-coord, coord:-coord] = 1

            self.FPSubapArrays *= fieldMask

        # Finally, run put these arrays onto the simulated detector
        super(ExtendedSH, self).makeDetectorPlane()

    def makeCorrelationImgs(self):
        """
        Use a convolution method to retrieve the 2d correlation peak between the subaperture and reference images.
        """

        # Remove the min from each sub-ap to increase contrast
        self.centSubapArrays[:] = (
                self.centSubapArrays.T-self.centSubapArrays.min((1,2))).T

        # Now do convolution
        # Get inverse FT of subaps
        # iCentSubapArrays = self.corrFFT(self.centSubapArrays)
        #
        # # Multiply by inverse of reference image FFT (made when set in property)
        # # Do FFT to get correlation
        # self.corrSubapArrays = self.corrIFFT(
        #         iCentSubapArrays*self.iReferenceImage).real

        # for i, subap in enumerate(self.centSubapArrays):
        #     self.corrSubapArrays[i] = scipy.signal.fftconvolve(subap, self.referenceImage[i], mode='same')

        if self.config.correlationFFTPad is None:
            subap_pad = self.centSubapArrays
            ref_pad = self.referenceImage
        else:
            PAD = round(0.5*(self.config.correlationFFTPad - self.config.pxlsPerSubap))
            subap_pad = numpy.pad(
                    self.centSubapArrays, mode='constant',
                    pad_width=((0,0), (PAD, PAD), (PAD, PAD)))
            ref_pad = numpy.pad(
                    self.referenceImage, mode='constant',
                    pad_width=((0,0), (PAD, PAD), (PAD, PAD)))

        self.corrSubapArrays = numpy.fft.fftshift(numpy.fft.ifft2(
                numpy.fft.fft2(subap_pad, axes=(1,2)) * numpy.fft.fft2(ref_pad, axes=(1,2)))).real


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
            self.centSubapArrays[i] = self.wfsDetectorPlane[x:x+self.wfsConfig.pxlsPerSubap,
                                                    y:y+self.wfsConfig.pxlsPerSubap ].astype(DTYPE)
        # If a reference image is supplied, use it for correlation centroiding
        if self.referenceImage is not None:
            self.makeCorrelationImgs()
        # Else: Just centroid on the extended image
        else:
            self.corrSubapArrays = self.FPSubapArrays

        slopes = eval("centroiders."+self.wfsConfig.centMethod)(
                self.corrSubapArrays,
                threshold=self.wfsConfig.centThreshold,
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

    @property
    def referenceImage(self):
        """
        A reference image to be used by a correlation centroider.
        """
        return self._referenceImage

    @referenceImage.setter
    def referenceImage(self, referenceImage):
        if referenceImage is not None:
            # If given value is a string, assume a filename of fits file
            if isinstance(referenceImage, str):
                referenceImage = fits.getdata(referenceImage)

            # Shape of expected ref values
            refShape = (
                    self.activeSubaps, self.wfsConfig.pxlsPerSubap,
                    self.wfsConfig.pxlsPerSubap)
            self._referenceImage = numpy.zeros(refShape)

            # if its an array of sub-aps, no work needed
            if referenceImage.shape == refShape:
                self._referenceImage = referenceImage

            # If its the size of a sub-ap, set all subaps to that value
            elif referenceImage.shape == (self.wfsConfig.pxlsPerSubap,)*2:
                # Make a placeholder for the reference image
                self._referenceImage = numpy.zeros(
                        (self.activeSubaps, self.wfsConfig.pxlsPerSubap,
                        self.wfsConfig.pxlsPerSubap))
                self._referenceImage[:] = referenceImage

            # If its the size of the detector, assume its a tiled array of sub-aps
            elif referenceImage.shape == (self.detectorPxls,)*2:

                for i, (x, y) in enumerate(self.detectorSubapCoords):
                    self._referenceImage[i] = referenceImage[
                            x:x+self.wfsConfig.pxlsPerSubap,
                            y:y+self.wfsConfig.pxlsPerSubap]

            # Do the FFT of the reference image for the correlation
            self.iReferenceImage = numpy.fft.ifft2(
                    self._referenceImage, axes=(1,2))
        else:
            self._referenceImage = None

    @property
    def extendedObject(self):
        return self._extendedObject

    @extendedObject.setter
    def extendedObject(self, extendedObject):
        if extendedObject is not None:
            # If a string, assume a fits file
            if isinstance(extendedObject, str):
                extendedObject = fits.getdata(extendedObject)

            if extendedObject.shape!=(self.subapFFTPadding,)*2:
                raise ValueError("Shape of extended object must be ({}, {}). This is `pxlsPersubap * fftOversamp`".format(self.subapFFTPadding, self.subapFFTPadding))

            self._extendedObject = extendedObject
        else:
            self._extendedObject = None
