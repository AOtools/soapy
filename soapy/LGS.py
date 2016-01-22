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
from .import aoSimLib, AOFFT
from . import logger

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
    This class gives a simple, stacked wavefront approach, with no deviation between each layer.
    it assumes that the laser is focussed at the height it is propogated to.
    '''

    def __init__(
            self, simConfig, wfsConfig, lgsConfig, atmosConfig, fieldDiam=None, 
            outPxlScale=None):

        self.simConfig = simConfig
        self.wfsConfig = wfsConfig
        self.config = lgsConfig
        self.atmosConfig = atmosConfig

        self.fieldDiam = fieldDiam
        self.outPxlScale = outPxlScale
        self.phaseSize = round(int(self.fieldDiam/self.outPxlScale))

        self.LGSPupilSize = int(numpy.round(self.config.pupilDiam
                                            * self.simConfig.pxlScale))
        self.mask = aoSimLib.circle(
                0.5*self.config.pupilDiam*self.simConfig.pxlScale,
                self.simConfig.simSize)
        self.geoMask = aoSimLib.circle(self.LGSPupilSize/2., self.LGSPupilSize)

        self.pupilPos = {}
        for i in xrange(self.atmosConfig.scrnNo):
            self.pupilPos[i] = self.metaPupilPos(
                self.atmosConfig.scrnHeights[i]
                )

        self.initFFTs()

    def initLos(self):
        """
        Initialises the ``LineOfSight`` object, which gets the phase or EField in a given direction through turbulence.
        """
        self.los = lineofsight.LineOfSight(
                    self.config, self.simConfig, self.atmosConfig,
                    outDiam=self.fieldDiam, propagationDirection="up",
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

    def setWFSParams(self, subapFOVRad, subapOversamp, subapFFTPadding):

        self.subapFOVRad = subapFOVRad
        self.subapOversamp = subapOversamp
        self.subapFFTPadding = subapFFTPadding

        # This is the size of the patch of sky the subap sees at the LGS height
        self.FOVPatch = self.subapFOVRad*self.config.height

    def getLgsPsf(self, scrns=None, phs=None):
        # if self.config.propagationMode=="physical":
        #     return self.getLgsPsfPhys(scrns, phs)
        # else:
        #     return self.getLgsPsfGeo(scrns, phs)

        self.los.makePhase()

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




    def getLgsPsfGeo(self, scrns=None, phs=None):

        if scrns!=None:
            self.pupilWavefront = self.getPupilPhase(scrns)
        else:
            self.pupilWavefront = phs

        self.scaledWavefront = aoSimLib.zoom(self.pupilWavefront,
                                                         self.LGSFOVOversize)
        #Convert phase to complex amplitude (-1 as opposite direction)
        self.EField = numpy.exp(-1j*self.scaledWavefront)*self.geoMask

        self.geoFFT.inputData[  :self.LGSFOVOversize,
                                :self.LGSFOVOversize] = self.EField
        fPlane = abs(AOFFT.ftShift2d(self.geoFFT())**2)

        #Crop to required FOV
        crop = self.subapFFTPadding*0.5/ self.fovOversize
        fPlane = fPlane[self.LGSFFTPadding*0.5 - crop:
                        self.LGSFFTPadding*0.5 + crop,
                        self.LGSFFTPadding*0.5 - crop:
                        self.LGSFFTPadding*0.5 + crop    ]

        #now bin to size of wfs
        self.PSF = aoSimLib.binImgs(fPlane, self.padFactor)


    def getLgsPsfPhys(self, scrns=None, phs=None):
        '''
        Computes the LGS PSF for each frame by propagating
        light to each turbulent layer with a fresnel algorithm
        '''

        self.U = self.mask.astype("complex64")

        #Keep same scale through intermediate calculations,
        #This keeps the array at the telescope diameter size.
        self.d1 = self.d2 = (self.simConfig.pxlScale)**(-1.)

        #if the 1st screen is at the ground, no propagation neccessary
        #if not, then propagate to that height.
        self.phs = self.metaPupilPhase(scrns[0], self.atmosConfig.scrnHeights[0],self.pupilPos[0])
        if self.atmosConfig.scrnHeights[0] == 0:
            self.z=0
            self.U *= numpy.exp(-1j*self.phs)

        elif self.atmosConfig.scrnHeights[0] > self.config.height:
            self.z = 0

        else:
            self.z = self.atmosConfig.scrnHeights[0]
            self.U = self.angularSpectrum( self.U, self.config.wavelength, self.d1, self.d2, self.z )
            self.U *= numpy.exp(-1j*self.phs)

        #Loop over remaining screens and propagate to the screen heights
        #Keep track of total height for use later.
        self.ht = self.z
        for i in xrange(1,len(scrns)):
            logger.debug("Propagate layer: {}".format(i))
            self.phs = self.metaPupilPhase(scrns[i],
                                      self.atmosConfig.scrnHeights[i],
                                      self.pupilPos[i] )

            self.z = self.atmosConfig.scrnHeights[i]-self.atmosConfig.scrnHeights[i-1]

            if self.z != 0:
                self.U = self.angularSpectrum(self.U,
                                           self.config.wavelength, self.d1, self.d2, self.z)

            self.U*=self.phs
            self.ht += self.z

        #Finally, propagate to the last layer.
        #Here we scale the array differently to get the correct
        #output scaling for the oversampled WFS
        self.z = self.config.height - self.ht

        #Perform calculation onto grid with same pixel scale as WFS subap
        self.gridScale = float(self.FOVPatch)/self.subapFFTPadding
        self.Ufinal = self.angularSpectrum(
                self.U, self.config.wavelength, self.d1, self.gridScale, self.z)
        self.psf1 = abs(self.Ufinal)**2

        #Now fit this grid onto size WFS is expecting
        crop = self.subapFFTPadding/2.
        if self.simConfig.simSize>self.subapFFTPadding:

            self.PSF = self.psf1[
                    int(numpy.round(0.5*self.simConfig.simSize-crop)):
                    int(numpy.round(0.5*self.simConfig.simSize+crop)),
                    int(numpy.round(0.5*self.simConfig.simSize-crop)):
                    int(numpy.round(0.5*self.simConfig.simSize+crop))
                    ]

        else:
            self.PSF = numpy.zeros((self.subapFFTPadding, self.subapFFTPadding))
            self.PSF[   int(numpy.round(crop-self.simConfig.simSize*0.5)):
                        int(numpy.round(crop+self.simConfig.simSize*0.5)),
                        int(numpy.round(crop-self.simConfig.simSize*0.5)):
                        int(numpy.round(crop+self.simConfig.simSize*0.5))] = self.psf1
