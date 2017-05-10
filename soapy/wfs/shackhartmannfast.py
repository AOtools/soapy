import numpy
import numpy.random

import pyfftw

import aotools
from aotools.image_processing import centroiders
from aotools import wfs

from .. import LGS, logger, lineofsight_fast, AOFFT, interp
from . import base
from .. import numbalib

# xrange now just "range" in python3.
# Following code means fastest implementation used in 2 and 3
try:
    xrange
except NameError:
    xrange = range

# The data type of data arrays (complex and real respectively)
CDTYPE = numpy.complex64
DTYPE = numpy.float32


class ShackHartmannFast(base.WFS):
    """Class to simulate a Shack-Hartmann WFS"""

    def calcInitParams(self):
        """
        Calculate some parameters to be used during initialisation
        """
        super(ShackHartmannFast, self).calcInitParams()

        # Sort out some required parameters
        self.nx_subaps = self.config.nxSubaps
        self.subap_fov = self.config.subapFOV
        self.nx_subap_pixels = self.config.pxlsPerSubap
        self.fft_oversamp = self.config.fftOversamp
        self.nx_guard_pixels = self.config.nx_guard_pixels
        self.subap_threshold = self.config.subapThreshold


        # Calculate some others
        self.pixel_scale =self.subap_fov / self.nx_subap_pixels
        self.subap_fov_rad = self.subap_fov * numpy.pi / (180. * 3600)
        self.subap_diam = self.telescope_diameter/self.nx_subaps
        self.nm_to_rad = 1e-9 * (2 * numpy.pi) / self.wavelength

        # spacing between subaps in pupil Plane (size "pupilSize")
        self.nx_subap_pupil = float(self.pupil_size)/self.nx_subaps

        # Spacing on the "FOV Plane" - the number of elements required
        # for the correct subap FOV (from way FFT "phase" to "image" works)
        self.nx_subap_interp = int(round(
                self.subap_diam * self.subap_fov_rad/ self.config.wavelength))

        # make twice as big to double subap FOV (unless told not to!)
        if self.config.subapFieldStop==True:
            self.SUBAP_OVERSIZE = 1
        else:
            self.SUBAP_OVERSIZE = 2
        
        self.nx_detector_pixels = self.nx_subaps * (self.nx_subap_pixels + self.nx_guard_pixels) + self.nx_guard_pixels

        self.nx_subap_interp *= self.SUBAP_OVERSIZE
        self.nx_subap_pixels_oversize = self.SUBAP_OVERSIZE * self.nx_subap_pixels

        # The total size of the required EField for all subaps.
        # Extra scaling to account for simSize padding
        self.nx_interp_efield = int(round(
                self.nx_subaps*self.nx_subap_interp*
                (float(self.sim_size)/self.pupil_size)
                ))

        # If physical prop, must always be at same pixel scale
        # If not, can use less phase points for speed
        if self.config.propagationMode=="Physical":
            # This the pixel scale required for the correct FOV
            out_pixel_scale = (float(self.sim_size) / float(self.nx_interp_efield)) * self.phase_scale
            self.los.calcInitParams(
                    out_pixel_scale=out_pixel_scale, nx_out_pixels=self.nx_interp_efield)

        # Calculate the subaps that are actually seen behind the pupil mask
        self.findActiveSubaps()

        self.referenceImage = self.config.referenceImage
        self.n_measurements = 2 * self.n_subaps

        self.thread_pool = numbalib.ThreadPool(self.threads)

    def initLos(self):
        """
        Initialises the ``LineOfSight`` object, which gets the phase or EField in a given direction through turbulence.
        """
        self.los = lineofsight_fast.LineOfSight(
                self.config, self.soapy_config,
                propagation_direction="down")


    def findActiveSubaps(self):
        '''
        Finds the subapertures which are not empty space
        determined if mean of subap coords of the mask is above threshold.
        '''

        mask = self.mask[
                self.sim_pad : -self.sim_pad,
                self.sim_pad : -self.sim_pad
                ]

        (self.pupil_subap_coords, self.detector_subap_coords,
         self.valid_subap_coords, self.detector_cent_coords,
         self.subap_fill_factor) = findActiveSubaps(
            self.nx_subaps, mask, self.subap_threshold, self.nx_subap_pixels,
            self.SUBAP_OVERSIZE, self.nx_guard_pixels)

        self.interp_subap_coords = numpy.round(
            (self.pupil_subap_coords + self.sim_pad) * self.nx_subap_interp / self.nx_subap_pupil)


        self.n_subaps = int(self.pupil_subap_coords.shape[0])
        self.n_measurements = 2 * self.n_subaps

        self.setMask(self.mask)

    def setMask(self, mask):
        super(ShackHartmannFast, self).setMask(mask)

        # Find the mask to apply to the scaled EField
        self.scaledMask = numpy.round(interp.zoom(
                    self.mask, self.nx_interp_efield))

        p = self.sim_pad
        self.subapFillFactor = wfs.computeFillFactor(
                self.mask[p:-p, p:-p],
                self.pupil_subap_coords,
                round(float(self.pupil_size)/self.nx_subaps)
                )


    def initFFTs(self):
        """
        Initialise the FFT Objects required for running the WFS

        Initialised various FFT objects which are used through the WFS,
        these include FFTs to calculate focal planes, and to convolve LGS
        PSFs with the focal planes
        """

        #Calculate the FFT padding to use
        self.subapFFTPadding = self.nx_subap_pixels_oversize * self.config.fftOversamp
        if self.subapFFTPadding < self.nx_subap_interp:
            while self.subapFFTPadding<self.nx_subap_interp:
                self.config.fftOversamp+=1
                self.subapFFTPadding\
                        =self.nx_subap_pixels_oversize*self.config.fftOversamp

            logger.warning("requested WFS FFT Padding less than FOV size... Setting oversampling to: %d"%self.config.fftOversamp)

        #Init the FFT to the focal plane
        # self.FFT = AOFFT.FFT(
        #         inputSize=(
        #             self.n_subaps, self.subapFFTPadding, self.subapFFTPadding),
        #         axes=(-2,-1), mode="pyfftw",dtype=CDTYPE,
        #         THREADS=self.threads,
        #         fftw_FLAGS=(self.config.fftwFlag,"FFTW_DESTROY_INPUT"))

        self.fft_input_data = pyfftw.empty_aligned(
                (self.n_subaps, self.subapFFTPadding, self.subapFFTPadding), dtype=CDTYPE)
        self.fft_output_data = pyfftw.empty_aligned(
                (self.n_subaps, self.subapFFTPadding, self.subapFFTPadding), dtype=CDTYPE)
        self.FFT = pyfftw.FFTW(
                self.fft_input_data, self.fft_output_data, axes=(-2, -1),
                threads=self.threads, flags=(self.config.fftwFlag, "FFTW_DESTROY_INPUT")
                )

        # If LGS uplink, init FFTs to conovolve LGS PSF and WFS PSF(s)
        # This works even if no lgsConfig.uplink as ``and`` short circuits
        if self.lgsConfig and self.lgsConfig.uplink:

            self.ifft_input_data = pyfftw.empty_aligned(
                    (self.n_subaps, self.subapFFTPadding, self.subapFFTPadding), dtype=CDTYPE)
            self.ifft_output_data = pyfftw.empty_aligned(
                    (self.n_subaps, self.subapFFTPadding, self.subapFFTPadding), dtype=CDTYPE)
            self.FFT = pyfftw.FFTW(
                self.ifft_input_data, self.ifft_output_data, axes=(-2, -1),
                threads=self.threads, flags=(self.config.fftwFlag, "FFTW_DESTROY_INPUT"),
                direction="FFTW_BACKWARD"
                )

            self.lgss_ifft_input_data = pyfftw.empty_aligned(
                (self.subapFFTPadding, self.subapFFTPadding), dtype=CDTYPE)
            self.lgss_ifft_output_data = pyfftw.empty_aligned(
                (self.subapFFTPadding, self.subapFFTPadding), dtype=CDTYPE)
            self.FFT = pyfftw.FFTW(
                self.lgs_ifft_input_data, self.lgs_ifft_output_data, axes=(0, 1),
                threads=self.threads, flags=(self.config.fftwFlag, "FFTW_DESTROY_INPUT"),
                direction="FFTW_BACKWARD"
                )

    def initLGS(self):
        super(ShackHartmannFast, self).initLGS()
        if self.lgsConfig.uplink:
            lgsObj = getattr(
                    LGS, "LGS_{}".format(self.lgsConfig.propagationMode))
            self.lgs = lgsObj(
                    self.config, self.soapy_config,
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

        self.interp_phase = numpy.zeros(
                (self.nx_interp_efield, self.nx_interp_efield), dtype=DTYPE)
        self.interp_efield = numpy.zeros(
                (self.nx_interp_efield, self.nx_interp_efield), dtype=CDTYPE)
        self.subap_interp_efield=numpy.zeros(
                (self.n_subaps, self.nx_subap_interp, self.nx_subap_interp),
                dtype=CDTYPE)
        self.binnedFPSubapArrays = numpy.zeros(
                (self.n_subaps, self.nx_subap_pixels_oversize, self.nx_subap_pixels_oversize),
                dtype=DTYPE)
        self.subap_focus_intensity = numpy.zeros(
                (self.n_subaps, self.subapFFTPadding, self.subapFFTPadding),
                dtype=DTYPE)
        self.temp_subap_intensity = self.subap_focus_intensity.copy()

        self.detector = numpy.zeros(
                (self.nx_detector_pixels, self.nx_detector_pixels),
                dtype=DTYPE)

        #Array used when centroiding subaps
        self.centSubapArrays = numpy.zeros(
                (self.n_subaps, self.config.pxlsPerSubap, self.config.pxlsPerSubap))

        self.slopes = numpy.zeros(2 * self.n_subaps)

        # compatablity...
        self.wfsDetectorPlane = self.detector


    def calcTiltCorrect(self):
        """
        Calculates the required tilt to add to avoid the PSF being centred on
        only 1 pixel
        """
        if not self.config.pxlsPerSubap%2:
            # If pxlsPerSubap is even
            # Angle we need to correct for half a pixel
            theta = self.SUBAP_OVERSIZE*self.subap_fov_rad/ (
                    2*self.subapFFTPadding)

            # Magnitude of tilt required to get that angle
            A = theta * self.subap_diam/(2*self.config.wavelength)*2*numpy.pi

            # Create tilt arrays and apply magnitude
            coords = numpy.linspace(-1, 1, self.nx_subap_interp)
            X,Y = numpy.meshgrid(coords,coords)

            self.tilt_fix = -1 * A * (X + Y)

        else:
            self.tilt_fix = numpy.zeros((self.nx_subap_interp,) * 2)

        self.tilt_fix_efield = numpy.exp(1j * (self.tilt_fix))

    def getStatic(self):
        """
        Computes the static measurements, i.e., slopes with flat wavefront
        """

        self.staticData = None

        # Make flat wavefront, and run through WFS in iMat mode to turn off features
        phs = numpy.zeros([self.los.n_layers, self.screen_size, self.screen_size]).astype(DTYPE)
        self.staticData = self.frame(
                phs, iMatFrame=True).copy().reshape(2, self.n_subaps)
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
            self.subap_focus_intensity[:] = 0

        if detector:
            self.detector[:] = 0


    def calcFocalPlane(self, intensity=1):
        '''
        Calculates the wfs focal plane, given the phase across the WFS

        Parameters:
            intensity (float): The relative intensity of this frame, is used when multiple WFS frames taken for extended sources.
        '''

        if self.config.propagationMode=="Geometric":
            # Have to make phase the correct size if geometric prop
            numbalib.wfs.zoomtoefield(self.los.phase, self.interp_efield, thread_pool=self.thread_pool)

        else:
            self.interp_efield = self.EField

        # Create an array of individual subap EFields
        self.fft_input_data[:] = 0
        numbalib.wfs.chop_subaps_mask_pool(
                self.interp_efield, self.interp_subap_coords, self.nx_subap_interp,
                self.fft_input_data, self.scaledMask, thread_pool=self.thread_pool)
        self.fft_input_data[:, :self.nx_subap_interp, :self.nx_subap_interp] *= self.tilt_fix_efield
        self.FFT()

        self.temp_subap_focus = AOFFT.ftShift2d(self.fft_output_data)

        numbalib.abs_squared(self.temp_subap_focus, out=self.subap_focus_intensity)

        if intensity != 1:
            self.subap_focus_intensity *= intensity

    def makeDetectorPlane(self):
        '''
        Scales and bins intensity data onto the detector with a given number of
        pixels.

        If required, will first convolve final PSF with LGS PSF, then bin
        PSF down to detector size. Finally puts back into ``wfsFocalPlane``
        array in correct order.
        '''

        # If required, convolve with LGS PSF
        if self.config.lgs and self.lgs and self.lgsConfig.uplink and self.iMat!=True:
            self.applyLgsUplink()


        # bins back down to correct size and then
        # fits them back in to a focal plane array
        self.binnedFPSubapArrays[:] = 0
        numbalib.wfs.bin_imgs_pool(
                self.subap_focus_intensity, self.config.fftOversamp, self.binnedFPSubapArrays,
                thread_pool=self.thread_pool)

        # Scale each sub-ap flux by sub-aperture fill-factor
        self.binnedFPSubapArrays\
                = (self.binnedFPSubapArrays.T * self.subapFillFactor).T

        numbalib.wfs.place_subaps_on_detector(
                self.binnedFPSubapArrays, self.detector, self.detector_subap_coords, self.valid_subap_coords,
                threads=self.threads
        )
        # numbalib.wfs.place_subaps_on_detector_pool(
        #         self.binnedFPSubapArrays, self.detector, self.detector_subap_coords, self.valid_subap_coords,
        #         thread_pool=self.thread_pool
        # )

        # Scale data for correct number of photons
        self.detector /= self.detector.sum()
        self.detector *= aotools.photonsPerMag(
                self.config.GSMag, self.mask, self.phase_scale**(-1),
                self.config.wvlBandWidth, self.config.exposureTime
                ) * self.config.throughput

        if self.config.photonNoise:
            self.addPhotonNoise()

        if self.config.eReadNoise!=0:
            self.addReadNoise()

    def applyLgsUplink(self):
        """
        A method to deal with convolving the LGS PSF
        with the subap focal plane.
        """

        self.lgs.getLgsPsf(self.los.scrns)

        self.lgs_ifft_input_data[:] = self.lgs.psf
        self.lgs_iFFT()

        self.ifft_input_data[:] = self.subap_focus_intensity
        self.iFFT()

        # Do convolution
        self.ifft_output_data *= self.lgss_ifft_output_data

        # back to Focal Plane.
        self.fft_input_data[:] = self.iFFTFPSubapsArray
        self.FFT()
        self.subap_focus_intensity[:] = AOFFT.ftShift2d(self.fft_output_data).real

    def calculateSlopes(self):
        '''
        Calculates WFS slopes from wfsFocalPlane

        Returns:
            ndarray: array of all WFS measurements
        '''
        numbalib.wfs.chop_subaps(
                self.detector, self.detector_cent_coords, self.nx_subap_pixels,
                self.centSubapArrays, threads=self.threads)

        slopes = getattr(centroiders, self.config.centMethod)(
                self.centSubapArrays,
                threshold=self.config.centThreshold,
                ref=self.referenceImage
                )


        # shift slopes relative to subap centre and remove static offsets
        slopes -= self.config.pxlsPerSubap/2.0

        if numpy.any(self.staticData):
            slopes -= self.staticData

        self.slopes[:] = slopes.reshape(self.n_subaps * 2)

        if self.config.removeTT == True:
            self.slopes[:self.n_subaps] -= self.slopes[:self.n_subaps].mean()
            self.slopes[self.n_subaps:] -= self.slopes[self.n_subaps:].mean()

        if self.config.angleEquivNoise and not self.iMat:
            pxlEquivNoise = (
                    self.config.angleEquivNoise * float(self.config.pxlsPerSubap)
                    /self.config.subapFOV )
            self.slopes += numpy.random.normal(
                    0, pxlEquivNoise, 2*self.n_subaps)

        return self.slopes


def findActiveSubaps(
            nx_subaps, mask, threshold, nx_subap_pixels, subap_oversize, guard=0):
    '''
    Finds the subapertures which are "seen" be through the
    pupil function. Returns the coords of those subaps

    Parameters:
        nx_subaps (int): The number of subaps in x (assumes square)
        mask (ndarray): A pupil mask, where is transparent when 1, and opaque when 0
        threshold (float): The mean value across a subap to make it "active"
        nx_subap_pixels (int): Pixels per subaperture on detector
        subap_oversize (int): Factor that subap is "oversized" on detector
        guard (int, optional): Guard pixels between sub-apertures

    Returns:
        ndarray: An array of active subap coords
    '''

    pupil_coords = []
    x_spacing = mask.shape[0]/float(nx_subaps)
    y_spacing = mask.shape[1]/float(nx_subaps)

    detector_coords = []
    subap_coords = []
    detector_cent_coords = []

    fills = []

    nx_oversize_subap = subap_oversize * nx_subap_pixels # number of pixels on oversized subap
    detector_pad = (nx_oversize_subap - nx_subap_pixels) / 2. # Pad on each side of subap
    nx_detector_pixels = nx_subaps * nx_subap_pixels + (nx_subaps + 1) * guard # Total number of detector pixels


    for x in range(nx_subaps):
        for y in range(nx_subaps):
            subap = mask[
                    int(numpy.round(x*x_spacing)): int(numpy.round((x+1)*x_spacing)),
                    int(numpy.round(y*y_spacing)): int(numpy.round((y+1)*y_spacing))
                    ]

            if subap.mean() >= threshold:
                pupil_coords.append( [x * x_spacing, y * y_spacing])
                fills.append(subap.mean())

                detector_cent_coords.append([
                        x * nx_subap_pixels + (x+1) * guard, y * nx_subap_pixels + (y+1) * guard])

                dx1 = x * nx_subap_pixels - detector_pad + (x+1) * guard
                dx2 = (x + 1) * nx_subap_pixels + detector_pad + (x+1) * guard
                dy1 = y * nx_subap_pixels - detector_pad + (y+1) * guard
                dy2 = (y + 1) * nx_subap_pixels + detector_pad + (y+1) * guard
                detector_coords.append([dx1, dx2, dy1, dy2])

                if dx1 < 0:
                    sx1 = -dx1
                else:
                    sx1 = 0

                if dx2 > nx_detector_pixels:
                    sx2 = -(dx2 - nx_detector_pixels)
                else:
                    sx2 = nx_oversize_subap

                if dy1 < 0:
                    sy1 = -dy1
                else:
                    sy1 = 0

                if dy2 > nx_detector_pixels:
                    sy2 = -(dy2 - nx_detector_pixels)
                else:
                    sy2 = nx_oversize_subap

                subap_coords.append([sx1, sx2, sy1, sy2])

    pupil_coords = numpy.array(pupil_coords).astype('int32')
    detector_coords = numpy.array(detector_coords)
    detector_coords = detector_coords.clip(0, nx_detector_pixels).astype('int32')
    subap_coords = numpy.array(subap_coords).astype('int32')
    detector_cent_coords = numpy.array(detector_cent_coords)

    return pupil_coords, detector_coords, subap_coords, detector_cent_coords, numpy.array(fills)
