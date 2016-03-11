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
from .import aoSimLib, AOFFT, logger, lineofsight

import scipy.optimize
from scipy.interpolate import interp2d

#xrange now just "range" in python3.
#Following code means fastest implementation used in 2 and 3
try:
    xrange
except NameError:
    xrange = range


class LGS(object):
    '''
    A class to simulate the propogation of a laser up through turbulence.
    given a set of phase screens, this will return the PSF which would be present on-sky.

    Parameters:
        simConfig: The Soapy simulation config
        wfsConfig: The relavent Soapy WFS configuration
        atmosConfig: The relavent Soapy atmosphere configuration
        nOutPxls (int): Number of pixels required in output LGS
        outPxlScale (float): The pixel scale of the output LGS PSF in arcsecs per pixel
    '''
    def __init__(
            self, simConfig, wfsConfig, lgsConfig, atmosConfig,
            nOutPxls=None, outPxlScale=None):

        self.simConfig = simConfig
        self.wfsConfig = wfsConfig
        self.config = lgsConfig
        self.atmosConfig = atmosConfig

        self.outPxlScale = outPxlScale
        if outPxlScale is None:
            self.outPxlScale_m = 1./self.simConfig.pxlScale
        else:
            # The pixel scale in metres per pixel at the LGS altitude
            self.outPxlScale_m = (outPxlScale/3600.)*(numpy.pi/180.) * self.config.height
        # Get the angular scale in radians of the output array
        self.outPxlScale_rad = self.outPxlScale_m/self.config.height

        # The number of pixels required across the LGS image
        if nOutPxls is None:
            self.nOutPxls = self.simConfig.simSize
        else:
            self.nOutPxls = nOutPxls

        self.FOV = self.nOutPxls * self.outPxlScale

        self.config.position = self.wfsConfig.GSPosition

        self.LGSPupilSize = int(numpy.round(self.config.pupilDiam
                                            * self.simConfig.pxlScale))

        self.mask = aoSimLib.circle(
                0.5*self.config.pupilDiam*self.simConfig.pxlScale,
                self.simConfig.simSize)
        self.geoMask = aoSimLib.circle(self.LGSPupilSize/2., self.LGSPupilSize)

        self.initLos()

        # Find central position of the LGS pupil at each altitude.
        self.pupilPos = {}
        for i in xrange(self.atmosConfig.scrnNo):
            self.pupilPos[i] = self.los.getMetaPupilPos(
                self.atmosConfig.scrnHeights[i]
                )

        self.initFFTs()

    def initLos(self):
        """
        Initialises the ``LineOfSight`` object, which gets the phase or EField in a given direction through turbulence.
        """
        # print("LGS: self.outPxlScale: {}".format(self.outPxlScale))
        self.los = lineofsight.LineOfSight(
                    self.config, self.simConfig, self.atmosConfig,
                    propagationDirection="up",
                    outPxlScale=self.outPxlScale, nOutPxls=self.nOutPxls,
                    mask=self.mask
                    )

    def initFFTs(self):
        # FFT for geometric propagation
        self.geoFPSize = self.nOutPxls*self.los.telDiam

        self.FFT = AOFFT.FFT(
                (self.nOutPxls, self.nOutPxls),
                axes=(0,1),mode="pyfftw",
                dtype = "complex64",direction="FORWARD",
                THREADS=self.config.fftwThreads,
                fftw_FLAGS=(self.config.fftwFlag,"FFTW_DESTROY_INPUT")
                )

    def calcInitParams(self):
        pass


    def getLgsPsf(self, scrns=None):

        self.los.frame(scrns)
        if self.config.propagationMode=="physical":
            return self.getLgsPsf_physical(scrns)

        elif self.config.propagationMode=="geometric":
            return self.getLgsPsf_geometric(scrns)

        else:
            raise ValueError("Don't know that LGS propagation mode")

    def getLgsPsf_geometric(self, scrns=None):
        self.geoFFT.inputData[:] = self.los.EField
        fPlane = abs(AOFFT.ftShift2d(self.geoFFT())**2)

        # Crop to required FOV
        crop = self.subapFFTPadding*0.5/ self.fovOversize
        self.psf1 = fPlane[self.LGSFFTPadding*0.5 - crop:
                        self.LGSFFTPadding*0.5 + crop,
                        self.LGSFFTPadding*0.5 - crop:
                        self.LGSFFTPadding*0.5 + crop    ]

        return self.psf

    def getLgsPsf_physical(self, scrns=None):

        self.psf = abs(self.los.EField)**2
        return self.psf
