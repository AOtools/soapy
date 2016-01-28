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
        nLgsPxls (float): The number of pixels required across the LGS image
        outPxlScale (float): The total angular field required by the WFS
    '''

    def __init__(
            self, simConfig, wfsConfig, lgsConfig, atmosConfig, lgsOutDiam=None,
            outPxlScale=None):

        self.simConfig = simConfig
        self.wfsConfig = wfsConfig
        self.config = lgsConfig
        self.atmosConfig = atmosConfig
 
        if outPxlScale is None:
            self.outPxlScale = 1./self.simConfig.pxlScale
        else:
            # The pixel scale in metres per pixel at the LGS altitude
            self.outPxlScale = (outPxlScale/3600.)*(180/numpy.pi) * self.lgsConfig.height
        
        # The number of pixels required across the LGS image
        if lgsOutDiam is None:
            self.nOutPxls = self.simConfig.simSize/self.simConfig.pxlScale/self.outPxlScale
        else:
            self.nOutPxls = lgsOutDiam/self.outPxlScale
        
        self.initLos()

        self.LGSPupilSize = int(numpy.round(self.config.pupilDiam
                                            * self.simConfig.pxlScale))
        self.mask = aoSimLib.circle(
                0.5*self.config.pupilDiam*self.simConfig.pxlScale,
                self.simConfig.simSize)
        self.geoMask = aoSimLib.circle(self.LGSPupilSize/2., self.LGSPupilSize)

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
        self.los = lineofsight.LineOfSight(
                    self.config, self.simConfig, self.atmosConfig,
                    propagationDirection="up",
                    outPxlScale=self.outPxlScale
                    )


    def initFFTs(self):
        self.FFT = AOFFT.FFT(
                (simConfig.simSize, simConfig.simSize),
                axes=(0,1),mode="pyfftw",
                dtype = "complex64",direction="FORWARD",
                THREADS=lgsConfig.fftwThreads,
                fftw_FLAGS=(lgsConfig.fftwFlag,"FFTW_DESTROY_INPUT")
                )
        self.iFFT = AOFFT.FFT((simConfig.simSize,simConfig.simSize),
                axes=(0,1),mode="pyfftw",
                dtype="complex64",direction="BACKWARD",
                THREADS=lgsConfig.fftwThreads,
                fftw_FLAGS=(lgsConfig.fftwFlag,"FFTW_DESTROY_INPUT")
                )


    def getLgsPsf(self, scrns=None, phs=None):

        self.los.makePhase()
    
        self.lgsEField = aoSimLib.padCropImg(self.los.EField, self.nOutPxls)

        if self.config.propagationMode=="physical":
            self.psf1 = abs(self.los.EField)**2

        else:
            self.geoFFT.inputData[  :self.LGSFOVOversize,
                                    :self.LGSFOVOversize] = self.los.EField
            fPlane = abs(AOFFT.ftShift2d(self.geoFFT())**2)

            #Crop to required FOV
            crop = self.subapFFTPadding*0.5/ self.fovOversize
            fPlane = fPlane[self.LGSFFTPadding*0.5 - crop:
                            self.LGSFFTPadding*0.5 + crop,
                            self.LGSFFTPadding*0.5 - crop:
                            self.LGSFFTPadding*0.5 + crop    ]

 
