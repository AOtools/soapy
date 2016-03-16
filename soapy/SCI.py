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
from . import aoSimLib, AOFFT, logger
from scipy.interpolate import interp2d
import scipy.optimize as opt


class PSF(object):

    def __init__(self, simConfig, telConfig, atmosConfig, sciConfig, mask):

        self.simConfig = simConfig
        self.telConfig = telConfig
        self.sciConfig = sciConfig
        self.atmosConfig = atmosConfig
        self.mask = mask

        self.FOVrad = self.sciConfig.FOV * numpy.pi / (180. * 3600)

        self.FOVPxlNo = numpy.round(
            self.telConfig.telDiam * self.FOVrad
            / self.sciConfig.wavelength)

        self.padFOVPxlNo = int(round(
            self.FOVPxlNo * float(self.simConfig.simSize)
            / self.simConfig.pupilSize)
        )
        if self.padFOVPxlNo % 2 != self.FOVPxlNo % 2:
            self.padFOVPxlNo += 1

        mask = self.mask[
            self.simConfig.simPad:-self.simConfig.simPad,
            self.simConfig.simPad:-self.simConfig.simPad
        ]
        self.scaledMask = numpy.round(aoSimLib.zoom(mask, self.FOVPxlNo)
                                      ).astype("int32")

        # Init FFT object
        self.FFTPadding = self.sciConfig.pxls * self.sciConfig.fftOversamp
        if self.FFTPadding < self.FOVPxlNo:
            while self.FFTPadding < self.FOVPxlNo:
                self.sciConfig.fftOversamp += 1
                self.FFTPadding\
                    = self.sciConfig.pxls * self.sciConfig.fftOversamp
            logger.info(
                "SCI FFT Padding less than FOV size... Setting oversampling to %d" % self.sciConfig.fftOversamp)

        self.FFT = AOFFT.FFT(inputSize=(self.FFTPadding, self.FFTPadding),
                             axes=(0, 1), mode="pyfftw", dtype="complex64",
                             fftw_FLAGS=(
                                 sciConfig.fftwFlag, "FFTW_DESTROY_INPUT"),
                             THREADS=sciConfig.fftwThreads)

        # Get phase scaling factor to get r0 in other wavelength
        # phsWvl = 500e-9
        # self.r0Scale = phsWvl / self.sciConfig.wavelength
        # Convert phase in nm to radians at science wavelength
        self.phsNm2Rad = 2*numpy.pi/(self.sciConfig.wavelength*10**9)

        # Calculate ideal PSF for purposes of strehl calculation
        self.residual = numpy.zeros((self.simConfig.simSize,) * 2)
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
        if not self.sciConfig.pxls % 2:
            # Need to correct for half a pixel angle
            theta = float(self.FOVrad) / (2 * self.FFTPadding)

            # Find magnitude of tilt from this angle
            A = theta * self.telConfig.telDiam / \
                (2 * self.sciConfig.wavelength) * 2 * numpy.pi

            coords = numpy.linspace(-1, 1, self.FOVPxlNo)
            X, Y = numpy.meshgrid(coords, coords)
            self.tiltFix = -1 * A * (X + Y)
        else:
            self.tiltFix = numpy.zeros((self.FOVPxlNo,) * 2)

    def getMetaPupilPos(self, height):
        '''
        Finds the centre of a metapupil at a given height,
        when offset by a given angle in arcsecs
        Arguments:
            height (float): Height of the layer in metres

        Returns:
            ndarray: The position of the centre of the metapupil in metres
        '''

        # Convert positions into radians
        sciPos = numpy.array(
            self.sciConfig.position) * numpy.pi / (3600.0 * 180.0)

        # Position of centre of GS metapupil off axis at required height
        sciCent = numpy.tan(sciPos) * height

        return sciCent

    def getMetaPupilPhase(self, scrn, height):
        '''
        Returns the phase across a metaPupil at some height
        and angular offset in arcsec

        Parameters:
            scrn (ndarray): An array representing the phase screen
            height (float): Height of the phase screen

        Return:
            ndarray: The meta pupil at the specified height
        '''

        sciCent = self.getMetaPupilPos(height) * self.simConfig.pxlScale
        logger.debug("SciCent:({0},{1})".format(sciCent[0], sciCent[1]))
        scrnX, scrnY = scrn.shape

        x1 = scrnX / 2. + sciCent[0] - self.simConfig.simSize / 2.0
        x2 = scrnX / 2. + sciCent[0] + self.simConfig.simSize / 2.0
        y1 = scrnY / 2. + sciCent[1] - self.simConfig.simSize / 2.0
        y2 = scrnY / 2. + sciCent[1] + self.simConfig.simSize / 2.0

        logger.debug("Sci scrn Coords: ({0}:{1}, {2}:{3})".format(
            x1, x2, y1, y2))

        if x1 < 0 or x2 > scrnX or y1 < 0 or y2 > scrnY:
            raise ValueError(  "Sci Position seperation requires larger scrn size")

        if (x1.is_integer() and x2.is_integer()
                and y1.is_integer() and y2.is_integer()):
            # Old, simple integer based solution
            metaPupil = scrn[x1:x2, y1:y2]
        else:
            # Dirty, temporary fix to interpolate between phase points
            xCoords = numpy.linspace(x1, x2 - 1, self.simConfig.simSize)
            yCoords = numpy.linspace(y1, y2 - 1, self.simConfig.simSize)
            scrnCoords = numpy.arange(scrnX)
            interpObj = interp2d(scrnCoords, scrnCoords, scrn, copy=False)
            metaPupil = interpObj(yCoords, xCoords)

        return metaPupil

    def calcPupilPhase(self):
        '''
        Returns the total phase on a science Camera which is offset
        by a given angle
        '''
        totalPhase = numpy.zeros([self.simConfig.simSize] * 2)
        for i in range(len(self.scrns)):
            phase = self.getMetaPupilPhase(
                self.scrns[i], self.atmosConfig.scrnHeights[i])

            totalPhase += phase

        self.phase = totalPhase

    def calcFocalPlane(self):
        '''
        Takes the calculated pupil phase, scales for the correct FOV,
        and uses an FFT to transform to the focal plane.
        '''

        # Scaled the padded phase to the right size for the requried FOV
        phs = aoSimLib.zoom(self.residual, self.padFOVPxlNo)

        # Convert phase deviation in nm to radians
        phs *= self.phsNm2Rad

        # Chop out the phase across the pupil before the fft
        coord = int(round((self.padFOVPxlNo - self.FOVPxlNo) / 2.))
        phs = phs[coord:-coord, coord:-coord]

        eField = numpy.exp(1j * (phs)) * self.scaledMask

        self.FFT.inputData[:self.FOVPxlNo, :self.FOVPxlNo] = eField
        focalPlane_efield = AOFFT.ftShift2d(self.FFT())

        self.focalPlane_efield = aoSimLib.binImgs(
            focalPlane_efield, self.sciConfig.fftOversamp)

        self.focalPlane = numpy.abs(self.focalPlane_efield.copy())**2

        # Normalise the psf
        self.focalPlane /= self.focalPlane.sum()

    def calcInstStrehl(self):
        """
        Calculates the instantaneous Strehl, including TT if configured.
        """
        if self.sciConfig.instStrehlWithTT:
            self.instStrehl = self.focalPlane[self.sciConfig.pxls//2,self.sciConfig.pxls//2]/self.focalPlane.sum()/self.psfMax
        else:
            self.instStrehl = self.focalPlane.max()/self.focalPlane.sum()/ self.psfMax

    def frame(self, scrns, phaseCorrection=None):
        """
        Runs a single science camera frame with one or more phase screens

        Parameters:
            scrns (ndarray, list, dict): One or more 2-d phase screens. Phase in units of nm.
            phaseCorrection (ndarray): Correction phase in nm

        Returns:
            ndarray: Resulting science PSF
        """
        # If scrns is not dict or list, assume array and put in list
        t = type(scrns)
        if t!=dict and t!=list:
            scrns = [scrns]

        self.scrns = scrns
        self.calcPupilPhase()

        if numpy.any(phaseCorrection):
            self.residual = self.phase - (phaseCorrection)
        else:
            self.residual = self.phase

        self.calcFocalPlane()

        self.calcInstStrehl()

        # Here so when viewing data, that outside of the pupil isn't visible.
        # self.residual*=self.mask

        return self.focalPlane


class singleModeFibre(PSF):

    def __init__(self, simConfig, telConfig, atmosConfig, sciConfig, mask):
        scienceCam.__init__(self, simConfig, telConfig, atmosConfig, sciConfig, mask)
        self.normMask = self.mask / numpy.sqrt(numpy.sum(numpy.abs(self.mask)**2))
        self.fibreSize = opt.minimize_scalar(self.refCouplingLoss, bracket=[1.0, self.simConfig.simSize]).x
        self.refStrehl = 1.0 - self.refCouplingLoss(self.fibreSize)
        self.fibre_efield = self.fibreEfield(self.fibreSize)
        print("Coupling efficiency: {0:.3f}".format(self.refStrehl))

    def fibreEfield(self, size):
        fibre_efield = aoSimLib.gaussian2d((self.simConfig.simSize, self.simConfig.simSize), (size, size))
        fibre_efield /= numpy.sqrt(numpy.sum(numpy.abs(aoSimLib.gaussian2d((self.simConfig.simSize*3, self.simConfig.simSize*3), (size, size)))**2))
        return fibre_efield

    def refCouplingLoss(self, size):
        return 1.0 - numpy.abs(numpy.sum(self.fibreEfield(size) * self.normMask))**2

    def calcInstStrehl(self):
        self.instStrehl = numpy.abs(numpy.sum(self.fibre_efield * numpy.exp(1j*self.residual*self.phsNm2Rad) * self.normMask))**2

# Compatability with older versions
scienceCam = PSF
