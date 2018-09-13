# Copyright Durham University and Andrew Reeves
# 2014

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

import numpy
import scipy.optimize as opt

import aotools

from . import AOFFT, logger, lineofsight, numbalib, interp
DTYPE = numpy.float32
CDTYPE = numpy.complex64


class PSF(object):

    def __init__(self, soapyConfig, nSci=0, mask=None):

        self.soapy_config = soapyConfig
        self.config = self.sciConfig = self.soapy_config.scis[nSci]

        self.simConfig = soapyConfig.sim


        # Get some vital params from config
        self.sim_size = self.soapy_config.sim.simSize
        self.pupil_size = self.soapy_config.sim.pupilSize
        self.sim_pad = self.soapy_config.sim.simPad
        self.fov = self.config.FOV
        self.threads = self.soapy_config.sim.threads
        self.telescope_diameter = self.soapy_config.tel.telDiam
        self.nx_pixels = self.config.pxls

        self.fov_rad = self.config.FOV * numpy.pi / (180. * 3600)

        self.setMask(mask)

        self.thread_pool = numbalib.ThreadPool(self.threads)

        self.FOVPxlNo = int(numpy.round(
            self.telescope_diameter * self.fov_rad
            / self.config.wavelength))

        self.padFOVPxlNo = int(round(
            self.FOVPxlNo * float(self.sim_size)
            / self.pupil_size)
        )
        if self.padFOVPxlNo % 2 != self.FOVPxlNo % 2:
            self.padFOVPxlNo += 1

        # Init line of sight - Get the phase at the right size for the FOV
        self.los = lineofsight.LineOfSight(
                self.config, self.soapy_config,
                propagation_direction="down")

        self.scaledMask = numpy.round(interp.zoom(self.mask, self.FOVPxlNo)
                                      ).astype("int32")

        # Init FFT object
        self.FFTPadding = self.nx_pixels * self.config.fftOversamp
        if self.FFTPadding < self.FOVPxlNo:
            while self.FFTPadding < self.FOVPxlNo:
                self.config.fftOversamp += 1
                self.FFTPadding\
                    = self.nx_pixels * self.config.fftOversamp
            logger.info(
                "SCI FFT Padding less than FOV size... Setting oversampling to %d" % self.config.fftOversamp)

        self.FFT = AOFFT.FFT(
                inputSize=(self.FFTPadding, self.FFTPadding), axes=(0, 1),
                mode="pyfftw", dtype="complex64",
                fftw_FLAGS=(self.config.fftwFlag, "FFTW_DESTROY_INPUT"),
                THREADS=self.config.fftwThreads)

        # Convert phase in nm to radians at science wavelength
        self.phsNm2Rad = 2*numpy.pi/(self.sciConfig.wavelength*10**9)

        # Allocate some useful arrays
        self.interp_coords = numpy.linspace(self.sim_pad, self.pupil_size + self.sim_pad, self.FOVPxlNo).astype(DTYPE)
        self.interp_coords = self.interp_coords.clip(0, self.los.nx_out_pixels-1.00001)

        self.interp_phase = numpy.zeros((self.FOVPxlNo, self.FOVPxlNo), DTYPE)
        self.focus_efield = numpy.zeros((self.FFTPadding, self.FFTPadding), dtype=CDTYPE)
        self.focus_intensity = numpy.zeros((self.FFTPadding, self.FFTPadding), dtype=DTYPE)
        self.detector = numpy.zeros((self.nx_pixels, self.nx_pixels), dtype=DTYPE)

        # Calculate ideal PSF for purposes of strehl calculation
        self.los.phase[:] = 1
        self.calcFocalPlane()
        self.bestPSF = self.detector.copy()
        self.psfMax = self.bestPSF.max()
        self.longExpStrehl = 0
        self.instStrehl = 0

    def setMask(self, mask):
        """
        Sets the pupil mask as seen by the WFS.

        This method can be called during a simulation
        """

        # If supplied use the mask
        if numpy.any(mask):
            self.mask = mask
        else:
            self.mask = aotools.circle(
                    self.pupil_size/2., self.sim_size,
                    )

    def calcFocalPlane(self):
        '''
        Takes the calculated pupil phase, scales for the correct FOV,
        and uses an FFT to transform to the focal plane.
        '''

        # Store the residual phase for later analysis and plotting
        self.residual = self.los.residual.copy() * self.mask

        numbalib.bilinear_interp(
                self.los.phase, self.interp_coords, self.interp_coords, self.interp_phase,
                self.thread_pool, bounds_check=False)

        self.EField_fov = numpy.exp(1j * self.interp_phase) * self.scaledMask

        # Get the focal plan using an FFT
        self.FFT.inputData[:self.FOVPxlNo, :self.FOVPxlNo] = self.EField_fov
        self.focus_efield = AOFFT.ftShift2d(self.FFT())

        # Turn complex efield into intensity
        numbalib.abs_squared(self.focus_efield, out=self.focus_intensity)

        # Bin down to detector number of pixels
        numbalib.bin_img(self.focus_intensity, self.config.fftOversamp, self.detector)

        # Normalise the psf
        self.detector /= self.detector.sum()

    def calcInstStrehl(self):
        """
        Calculates the instantaneous Strehl, including TT if configured.
        """
        if self.sciConfig.instStrehlWithTT:
            self.instStrehl = self.detector[self.sciConfig.pxls // 2, self.sciConfig.pxls // 2] / self.detector.sum() / self.psfMax
        else:
            self.instStrehl = self.detector.max() / self.detector.sum() / self.psfMax


    def calc_wavefronterror(self):
        """
        Calculates the wavefront error across the telescope pupil 
        
        Returns:
             float: RMS WFE across pupil in nm
        """

        res = (self.los.phase.copy() * self.mask) / self.los.phs2Rad

        # Piston is mean across aperture
        piston = res.sum() / self.mask.sum()

        # remove from WFE measurements as its not a problem
        res -= (piston*self.mask)

        ms_wfe = numpy.square(res).sum() / self.mask.sum()
        rms_wfe = numpy.sqrt(ms_wfe)

        return rms_wfe


    def frame(self, scrns, correction=None):
        """
        Runs a single science camera frame with one or more phase screens

        Parameters:
            scrns (ndarray, list, dict): One or more 2-d phase screens. Phase in units of nm.
            phaseCorrection (ndarray): Correction phase in nm

        Returns:
            ndarray: Resulting science PSF
        """
        self.los.frame(scrns, correction=correction)
        self.calcFocalPlane()

        self.calcInstStrehl()

        # Here so when viewing data, that outside of the pupil isn't visible.
        # self.residual*=self.mask

        return self.detector


class singleModeFibre(PSF):

    def __init__(self, soapyConfig, nSci=0, mask=None):
        scienceCam.__init__(self, soapyConfig, nSci, mask)

        self.normMask = self.mask / numpy.sqrt(numpy.sum(numpy.abs(self.mask)**2))
        self.fibreSize = opt.minimize_scalar(self.refCouplingLoss, bracket=[1.0, self.sim_size]).x
        self.refStrehl = 1.0 - self.refCouplingLoss(self.fibreSize)
        self.fibre_efield = self.fibreEfield(self.fibreSize)
        print("Coupling efficiency: {0:.3f}".format(self.refStrehl))

    def fibreEfield(self, size):
        fibre_efield = aotools.gaussian2d((self.sim_size, self.sim_size), (size, size))
        fibre_efield /= numpy.sqrt(numpy.sum(numpy.abs(aotools.gaussian2d((self.sim_size*3, self.sim_size*3), (size, size)))**2))
        return fibre_efield

    def refCouplingLoss(self, size):
        return 1.0 - numpy.abs(numpy.sum(self.fibreEfield(size) * self.normMask))**2

    def calcInstStrehl(self):
        self.instStrehl = numpy.abs(numpy.sum(self.fibre_efield * self.los.EField * self.normMask))**2


# Compatability with older versions
scienceCam = ScienceCam = PSF
