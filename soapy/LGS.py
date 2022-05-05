#Copyright Durham University and Andrew Reeves
#2014

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

import aotools

from . import AOFFT, logger, lineofsight, interp
# from .aotools import circle, interp

#xrange now just "range" in python3.
#Following code means fastest implementation used in 2 and 3
try:
    xrange
except NameError:
    xrange = range

RAD2ASEC = 206265.
ASEC2RAD = 1./RAD2ASEC

class LGS(object):
    '''
    A class to simulate the propogation of a laser up through turbulence.
    Given a set of phase screens, this will return the PSF which would be present on-sky.

    Parameters:
        simConfig: The Soapy simulation config
        wfsConfig: The relavent Soapy WFS configuration
        atmosConfig: The relavent Soapy atmosphere configuration
        nOutPxls (int): Number of pixels required in output LGS
        outPxlScale (float): The pixel scale of the output LGS PSF in arcsecs per pixel
    '''

    def __init__(
            self, wfsConfig, soapyConfig, nOutPxls=None, outPxlScale=None):

        # The LGS WFS config
        self.wfsConfig = wfsConfig
        self.config = wfsConfig.lgs

        self.soapyConfig = soapyConfig
        self.simConfig = soapyConfig.sim
        self.atmosConfig = soapyConfig.atmos

        self.outPxlScale = outPxlScale
        self.nOutPxls = nOutPxls

        # get this no of pixels from the LOS if not other wise told
        # Change in ``calcInitParams`` if you want more or less
        self.losNOutPxls = self.nOutPxls
        self.losOutPxlScale = self.outPxlScale

        self.config.position = self.wfsConfig.GSPosition
        self.losMask = None

        # correction phase for LGS precompensation
        self.precorrection = None            

        self.calcInitParams()

        self.initLos()

        self.initFFTs()

    def calcInitParams(self):
        pass

    def initFFTs(self):
        """
        Virtual Method as many LGS implentations will require extra FFTs
        """
        pass

    def initLos(self):
        """
        Initialises the ``LineOfSight`` object, which gets the phase or EField in a given direction through turbulence.
        """
        # Init the line of sight object for light propation through turbulence
        self.los = lineofsight.LineOfSight(
                    self.config, self.soapyConfig,
                    propagation_direction="up", nx_out_pixels=self.losNOutPxls,
                    mask=self.losMask, out_pixel_scale=self.losOutPxlScale,
                    )

        # Find central position of the LGS pupil at each altitude.
        self.los.metaPupilPos = {}
        for i in xrange(self.atmosConfig.scrnNo):
            self.los.metaPupilPos[i] = lgsOALaunchMetaPupilPos(
                    self.config.position,
                    numpy.array(self.config.launchPosition)*self.los.telescope_diameter/2.,
                    self.config.height, self.atmosConfig.scrnHeights[i]
                    )
        # Check position not too far from centre. May need more phase!
        maxPos = numpy.array(
                list(self.los.metaPupilPos.values())).max()
        maxPos*=self.simConfig.pxlScale
        if 2*(maxPos+self.simConfig.pupilSize/2.) > self.simConfig.simSize:
            logger.warning(
                    "LGS far off-axis - likely need to make simOversize bigger")

    def getLgsPsf(self, scrns):
        logger.debug("Get LGS PSF")
        self.los.frame(scrns, self.precorrection)
        self.EField = self.los.EField
        self.phase = self.los.phase


class LGS_Geometric(LGS):
    '''
    A class to simulate the propogation of a laser up through turbulence using a geometric algorithm.
    Given a set of phase screens, this will return the PSF which would be present on-sky.

    Parameters:
        simConfig: The Soapy simulation config
        wfsConfig: The relavent Soapy WFS configuration
        atmosConfig: The relavent Soapy atmosphere configuration
        nOutPxls (int): Number of pixels required in output LGS
        outPxlScale (float): The pixel scale of the output LGS PSF in arcsecs per pixel
    '''

    def calcInitParams(self):
        """
        Calculate some useful paramters to be used later
        """
        self.lgsPupilPxls = int(
                round(self.config.pupilDiam * self.simConfig.pxlScale))

        if self.outPxlScale is None:
            self.outPxlScale_m = 1./self.simConfig.pxlScale
        else:
            # The pixel scale in metres per pixel at the LGS altitude
            self.outPxlScale_m = (self.outPxlScale/3600.)*(numpy.pi/180.) * self.config.height

        # Get the angular scale in radians of the output array
        self.outPxlScale_rad = self.outPxlScale_m/self.config.height

        # The number of pixels required across the LGS image
        if self.nOutPxls is None:
            self.nOutPxls = self.simConfig.simSize

        # Field of fov of the requested LGS PSF image
        self.fov = (self.nOutPxls * self.outPxlScale_rad) * RAD2ASEC

        # The number of points required to get the correct FOV after the FFT
        fov_rad = self.fov / RAD2ASEC
        self.nFovPxls = int(round(fov_rad * self.config.pupilDiam
                / self.config.wavelength))

        # The mask to apply before geometric FFTing
        self.mask = aotools.circle(self.nFovPxls/2., self.nFovPxls)
        if self.config.obsDiam > 0:
            obspxls = self.nFovPxls * (self.config.obsDiam/self.config.pupilDiam)
            self.mask -= aotools.circle(obspxls/2, self.nFovPxls)

        self.losNOutPxls = self.lgsPupilPxls
        self.losOutPxlScale = self.config.pupilDiam/self.lgsPupilPxls


    def initFFTs(self):
        # FFT for geometric propagation
        self.FFT = AOFFT.FFT(
                (self.nOutPxls, self.nOutPxls),
                axes=(0,1), mode="pyfftw",
                dtype = "complex64",direction="FORWARD",
                THREADS=self.config.fftwThreads,
                fftw_FLAGS=(self.config.fftwFlag,"FFTW_DESTROY_INPUT")
                )

    def getLgsPsf(self, scrns):
        super(LGS_Geometric, self).getLgsPsf(scrns)

        # Pick out lgs Pupil sized chunk of field in middle
        # coord = (self.los.EField.shape[0]-self.lgsPupilPxls)/2.
        # lgsEField = self.los.EField[coord: -coord, coord: -coord]

        # Scale to the desired size for LGS FOV
        lgsEField = interp.zoom(self.EField, self.nFovPxls)*self.mask

        self.FFT.inputData[:self.nFovPxls, :self.nFovPxls] = lgsEField
        self.psf = abs(AOFFT.ftShift2d(self.FFT())**2)

        return self.psf


class LGS_Physical(LGS):
    '''
    A class to simulate the propogation of a laser up through turbulence using a geometric algorithm.
    Given a set of phase screens, this will return the PSF which would be present on-sky.

    Parameters:
        simConfig: The Soapy simulation config
        wfsConfig: The relavent Soapy WFS configuration
        atmosConfig: The relavent Soapy atmosphere configuration
        nOutPxls (int): Number of pixels required in output LGS
        outPxlScale (float): The pixel scale of the output LGS PSF in arcsecs per pixel
    '''

    def calcInitParams(self):
        """
        Calculate some useful paramters to be used later
        """

        self.mask = aotools.circle(
                0.5*self.config.pupilDiam*self.simConfig.pxlScale,
                self.simConfig.simSize)

        if self.outPxlScale is None:
            self.outPxlScale_m = 1./self.simConfig.pxlScale
        else:
            # The pixel scale in metres per pixel at the LGS altitude
            self.outPxlScale_m = (self.outPxlScale / RAD2ASEC) * self.config.height

        # Get the angular scale in radians of the output array
        self.outPxlScale_rad = self.outPxlScale_m/self.config.height


        # The number of pixels required across the LGS image
        if self.nOutPxls is None:
            self.nOutPxls = self.simConfig.simSize

        # Field of fov of the requested LGS PSF image
        self.fov = (self.nOutPxls * self.outPxlScale_rad) * RAD2ASEC

        # The number of points required to get the correct FOV after the FFT
        fov_rad = self.fov / RAD2ASEC
        self.nFovPxls = (fov_rad * self.config.pupilDiam
                / self.config.wavelength)

        # The mask to apply before physical propagation
        self.lgsPupilPxls = int(round(self.config.pupilDiam/self.outPxlScale_m))
        self.mask = aotools.circle(self.lgsPupilPxls/2., 3*self.nOutPxls)
        if self.config.obsDiam > 0:
            obspxls = self.config.obsDiam/self.outPxlScale_m
            self.mask -= aotools.circle(obspxls/2., self.mask.shape[-1])

        # this is the geometric focus assuming you want to focus at the LGS height
        focus_rms = self.config.height / ( 2.*numpy.sqrt(3.) * 8. * (self.config.height/self.config.pupilDiam)**2 ) * (2*numpy.pi)/self.config.wavelength

        focus = -focus_rms * aotools.zernike_noll(4, self.lgsPupilPxls)
        focus = numpy.pad(focus, (self.mask.shape[-1]-self.lgsPupilPxls)//2)

        # NOTE applying focus here makes mask complex, doesn't seem to break anything
        self.mask = self.mask.astype(complex) * numpy.exp(1j * focus)

        self.losMask = self.mask

        self.losOutPxlScale = self.outPxlScale_m
        self.losNOutPxls = 3*self.nOutPxls

    def getLgsPsf(self, scrns=None):
        """
        Return the LGS PSF to be used in WFS calculation
        """
        super(LGS_Physical, self).getLgsPsf(scrns)

        # Pick out middle of oversized fov

        self.psf = abs(self.EField[
                self.nOutPxls: -self.nOutPxls,
                self.nOutPxls: -self.nOutPxls])**2
        return self.psf


def lgsOALaunchMetaPupilPos(gsPos, launchPos, lgsHt, layerHt):
    """
    Finds the centre of a meta-pupil in the atmosphere sampled by an LGS launched from a position off-axis from the centre of the telescope.

    Parameters:
        gsPos (ndarray): The X,Y position of the guide star in arcsecs
        launchPos (ndarray): The X, Y launch position of the telescope in metres from the telescope centre
        lgsHt (float): The altitude of the LGS beacon
        layerHt (float): The height of the meta-pupil of interest

    Returns:
        ndarray: Position in X,Y from the on-axis line-of-sight of the meta-pupil centre.
    """

    gsPos_rad = numpy.array(gsPos)/RAD2ASEC
    launchPos = numpy.array(launchPos)
    # Equation worked out painstakingly with vast number of triangles...
    # (please try out and verify!)
    pos = launchPos + layerHt * gsPos_rad - (layerHt * launchPos / lgsHt)

    return pos
