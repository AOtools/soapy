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
"""
Science Instruments
===================

In this module, several 'science' instruments are defined. These are devices that 
observe a target with the purpose of assessing AO performance
"""

import numpy
import scipy.optimize as opt

import aotools

from . import AOFFT, logger, lineofsight, numbalib, interp
DTYPE = numpy.float32
CDTYPE = numpy.complex64


class PSFCamera(object):
    """
    A detector observing the Point Spread Function of the telescope

    Parameters:
        soapyConfig (soapy.confParse.Config): Simulation configuration object
        nSci (int, optional): index of this science instrument. default is ``0``
        mask (ndarray, optional): Mask, 1 where telescope is transparent, 0 where opaque. 
    """
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

        # Calculate the number of pixels required in the aperture plane
        # To generate the correct FOV in the focal plane
        self.FOVPxlNo = int(numpy.round(
            self.telescope_diameter * self.fov_rad
            / self.config.wavelength))

        # WIP: We dont' want to down sample too much from the users
        # specifed pupil_size, so find an integer factor of elements greater than required
        # This gives a focus FOV n times larger than specified,
        # but later we will crop the focal plane back down to size
        self.crop_fov_factor = 1 + self.pupil_size // self.FOVPxlNo
        self.fov_crop_elements = (self.FOVPxlNo * self.crop_fov_factor - self.FOVPxlNo) // 2
        self.FOVPxlNo *= self.crop_fov_factor

        # And then pad it by the required telescope padding.
        # Will later cut out the FOVPxlNo of pixels for focussing
        self.padFOVPxlNo = int(round(
            self.FOVPxlNo * float(self.sim_size)
            / self.pupil_size)
        )

        # If odd, keep odd, if even, keep even - keeps padding an integer on each side
        if self.padFOVPxlNo % 2 != self.FOVPxlNo % 2:
            self.padFOVPxlNo += 1
        self.fov_sim_pad = int((self.padFOVPxlNo - self.FOVPxlNo) // 2.)

        # If propagation direction is up, need to consider a mask in the optical propagation
        # Otherwise, we'll apply it later
        if self.config.propagationDir == "up":
            los_mask = self.mask
        else:
            los_mask = None

        self.los = lineofsight.LineOfSight(
                self.config, self.soapy_config,
                propagation_direction=self.config.propagationDir, mask=los_mask)

        # Init line of sight - Get the phase at the right size for the FOV
        if self.config.propagationMode == "Physical":
            # If physical prop, must do propagation at 
            # at FOV size to avoid interpolation of EField
            out_pixel_scale = float(self.telescope_diameter) / float(self.FOVPxlNo)
            self.los.calcInitParams(
                    out_pixel_scale=out_pixel_scale,
                    nx_out_pixels=self.padFOVPxlNo
            )
        
        # Cut out the mask just around the telescope aperture
        simpad = self.simConfig.simPad
        mask_pupil = self.mask[simpad: -simpad, simpad: -simpad]
        self.scaledMask = numpy.round(interp.zoom(mask_pupil, self.FOVPxlNo)
                                      ).astype("int32")

        # Init FFT object
        # fft padding must be oversampled from nx_pixels, and an integer number of FOVPxlNo
        self.FFTPadding = self.nx_pixels * self.config.fftOversamp
        if self.FFTPadding < self.FOVPxlNo:
            while self.FFTPadding < self.FOVPxlNo:
                self.config.fftOversamp += 1
                self.FFTPadding\
                    = self.nx_pixels * self.config.fftOversamp
            logger.info(
                "SCI FFT Padding less than FOV size... Setting oversampling to %d" % self.config.fftOversamp)

        self.fft_crop_elements = (self.FFTPadding * self.crop_fov_factor - self.FFTPadding)//2
        self.FFTPadding *= self.crop_fov_factor

        self.FFT = AOFFT.FFT(
                inputSize=(self.FFTPadding, self.FFTPadding), axes=(0, 1),
                mode="pyfftw", dtype="complex64",
                fftw_FLAGS=(self.config.fftwFlag, "FFTW_DESTROY_INPUT"),
                THREADS=self.config.fftwThreads)

        # Convert phase in nm to radians at science wavelength
        self.phsNm2Rad = 2*numpy.pi/(self.sciConfig.wavelength*10**9)

        # Allocate some useful arrays
        self.interp_coords = numpy.linspace(
                self.sim_pad, self.pupil_size + self.sim_pad, self.FOVPxlNo).astype(DTYPE)
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
        if self.config.propagationMode == "Physical":
            # If physical propagation, efield should already be the correct
            # size for the Field of View
            crop_efield = self.los.EField[
                    self.fov_sim_pad: -self.fov_sim_pad,
                    self.fov_sim_pad: -self.fov_sim_pad] # crop 
                    
            self.EField_fov = crop_efield * self.scaledMask
        else:
            # If geo prop...

            # Store the residual phase for later analysis and plotting
            self.residual = self.los.residual.copy() * self.mask

            # Interpolate to the correct number of pixels for the specced
            # Field of View on the detector
            numbalib.bilinear_interp(
                    self.los.phase, self.interp_coords, self.interp_coords, self.interp_phase,
                    bounds_check=False)

            self.EField_fov = numpy.exp(1j * self.interp_phase) * self.scaledMask

        # Get the focal plane using an FFT
        # Reset the FFT from the previous iteration
        self.FFT.inputData[:] = 0
        # place the array in the centre of the padding
        self.FFT.inputData[
                (self.FFTPadding - self.FOVPxlNo)//2:
                (self.FFTPadding + self.FOVPxlNo)//2, 
                (self.FFTPadding - self.FOVPxlNo)//2:
                (self.FFTPadding + self.FOVPxlNo)//2
                ] = self.EField_fov
        # This means we can do a pre-fft shift properly. This is neccessary for anythign that 
        # cares about the EField of the focal plane, not just the intensity pattern
        self.FFT.inputData[:] = numpy.fft.fftshift(self.FFT.inputData)
        self.focus_efield = AOFFT.ftShift2d(self.FFT())

        # Turn complex efield into intensity
        numbalib.abs_squared(self.focus_efield, out=self.focus_intensity)

        if self.fft_crop_elements != 0:
        # Bin down to detector number of pixels
            fov_focus_intensity = self.focus_intensity[
                    self.fft_crop_elements: -self.fft_crop_elements,
                    self.fft_crop_elements: -self.fft_crop_elements
            ]
        else:
            fov_focus_intensity = self.focus_intensity
        # numbalib.bin_img(self.focus_intensity, self.config.fftOversamp, self.detector)
        numbalib.bin_img(fov_focus_intensity, self.config.fftOversamp, self.detector)
        # Normalise the psf
        self.detector /= self.detector.sum()


    def calcInstStrehl(self):
        """
        Calculates the instantaneous Strehl, including TT if configured.
        """
        if self.sciConfig.instStrehlWithTT:
            self.instStrehl = self.detector[self.sciConfig.pxls // 2 - 1, self.sciConfig.pxls // 2 - 1] / self.detector.sum() / self.psfMax
        else:
            self.instStrehl = self.detector.max() / self.detector.sum() / self.psfMax


    def calc_wavefronterror(self):
        """
        Calculates the wavefront error across the telescope pupil 
        
        Returns:
             float: RMS WFE across pupil in nm
        """
        if self.config.propagationMode == "Physical":
            return 0
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

        return self.detector


class singleModeFibre(PSFCamera):

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
scienceCam = ScienceCam = PSF = PSFCamera
