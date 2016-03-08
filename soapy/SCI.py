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
from . import aoSimLib, AOFFT, logger, lineofsight
from scipy.interpolate import interp2d

DTYPE = numpy.float32
CDTYPE = numpy.complex64

class ScienceCam(object):

    def __init__(self, simConfig, telConfig, atmosConfig, sciConfig, mask):

        self.simConfig = simConfig
        self.telConfig = telConfig
        self.config = self.sciConfig = sciConfig # For compatability
        self.atmosConfig = atmosConfig
        self.mask = mask
        self.FOVrad = self.config.FOV * numpy.pi / (180. * 3600)

        self.FOVPxlNo = numpy.round(
            self.telConfig.telDiam * self.FOVrad
            / self.config.wavelength)

        self.padFOVPxlNo = int(round(
            self.FOVPxlNo * float(self.simConfig.simSize)
            / self.simConfig.pupilSize)
        )
        if self.padFOVPxlNo % 2 != self.FOVPxlNo % 2:
            self.padFOVPxlNo += 1

        # Init line of sight - Get the phase at the right size for the FOV
        self.los = lineofsight.LineOfSight(
                self.config, self.simConfig, self.atmosConfig,
                propagationDirection="down")

        mask = self.mask[
                self.simConfig.simPad:-self.simConfig.simPad,
                self.simConfig.simPad:-self.simConfig.simPad
                ]

        self.scaledMask = numpy.round(aoSimLib.zoom(mask, self.FOVPxlNo)
                                      ).astype("int32")

        # Init FFT object
        self.FFTPadding = self.config.pxls * self.config.fftOversamp
        if self.FFTPadding < self.FOVPxlNo:
            while self.FFTPadding < self.FOVPxlNo:
                self.config.fftOversamp += 1
                self.FFTPadding\
                    = self.config.pxls * self.config.fftOversamp
            logger.info(
                "SCI FFT Padding less than FOV size... Setting oversampling to %d" % self.config.fftOversamp)

        self.FFT = AOFFT.FFT(
                inputSize=(self.FFTPadding, self.FFTPadding), axes=(0, 1),
                mode="pyfftw", dtype="complex64",
                fftw_FLAGS=(sciConfig.fftwFlag, "FFTW_DESTROY_INPUT"),
                THREADS=sciConfig.fftwThreads)

        # Convert phase to radians at science wavelength
        self.phs2Rad = 2*numpy.pi/(self.config.wavelength*10**9)

        # Calculate ideal PSF for purposes of strehl calculation
        self.los.EField[:] = numpy.ones(
                (self.los.nOutPxls,) * 2, dtype=CDTYPE)
        self.calcFocalPlane()
        self.bestPSF = self.focalPlane.copy()
        self.psfMax = self.bestPSF.max()
        self.longExpStrehl = 0
        self.instStrehl = 0

    def calcTiltCorrect(self):
        """
        Calculates the required tilt to add to avoid the PSF being centred
        on one pixel only
        """
        # Only required if pxl number is even
        if not self.config.pxls % 2:
            # Need to correct for half a pixel angle
            theta = float(self.FOVrad) / (2 * self.FFTPadding)

            # Find magnitude of tilt from this angle
            A = theta * self.telConfig.telDiam / \
                (2 * self.config.wavelength) * 2 * numpy.pi

            coords = numpy.linspace(-1, 1, self.FOVPxlNo)
            X, Y = numpy.meshgrid(coords, coords)
            self.tiltFix = -1 * A * (X + Y)
        else:
            self.tiltFix = numpy.zeros((self.FOVPxlNo,) * 2)

    def calcFocalPlane(self):
        '''
        Takes the calculated pupil phase, scales for the correct FOV,
        and uses an FFT to transform to the focal plane.
        '''
        self.EField = self.los.EField

        # Apply the system mask
        self.EField *= self.mask

        # Scaled the padded phase to the right size for the requried FOV
        self.EField_fov = aoSimLib.zoom(self.EField, self.padFOVPxlNo)

        # Chop out the phase across the pupil before the fft
        coord = int(round((self.padFOVPxlNo - self.FOVPxlNo) / 2.))
        self.EField_fov = self.EField_fov[coord:-coord, coord:-coord]

        # Get the focal plan using an FFT
        self.FFT.inputData[:self.FOVPxlNo, :self.FOVPxlNo] = self.EField_fov
        focalPlane_efield = AOFFT.ftShift2d(self.FFT())

        # Bin down to the required number of pixels
        self.focalPlane_efield = aoSimLib.binImgs(
            focalPlane_efield, self.config.fftOversamp)

        self.focalPlane = numpy.abs(self.focalPlane_efield.copy())**2

        # Normalise the psf
        self.focalPlane /= self.focalPlane.sum()

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

        # Here so when viewing data, that outside of the pupil isn't visible.
        # self.residual*=self.mask

        if self.sciConfig.instStrehlWithTT:
            self.instStrehl = self.focalPlane[self.sciConfig.pxls//2,self.sciConfig.pxls//2]/self.focalPlane.sum()/self.psfMax
        else:
            self.instStrehl = self.focalPlane.max()/self.focalPlane.sum()/ self.psfMax

        return self.focalPlane


# For compatability
scienceCam = ScienceCam
