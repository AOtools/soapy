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
from .import aoSimLib, AOFFT, lineofsight, opticalPropagationLib, logger

import numpy
import scipy.optimize
from scipy.interpolate import interp2d

#xrange now just "range" in python3.
#Following code means fastest implementation used in 2 and 3
try:
    xrange
except NameError:
    xrange = range


class LGSObj(object):
    '''
    A class to simulate the propogation of a laser up through turbulence.
    given a set of phase screens, this will return the PSF which would be present on-sky.
    This class gives a simple, stacked wavefront approach, with no deviation between each layer.
    it assumes that the laser is focussed at the height it is propogated to.
    '''

    def __init__(self, simConfig, wfsConfig, lgsConfig, atmosConfig):

        self.simConfig = simConfig
        self.wfsConfig = wfsConfig
        self.lgsConfig = lgsConfig
        self.atmosConfig = atmosConfig

        self.LGSPupilSize = int(numpy.round(lgsConfig.pupilDiam
                                            * self.simConfig.pxlScale))

        #Phase power scaling factor for lgs wavelength
        phsWvl = 550e-9
        self.r0scale = phsWvl / self.lgsConfig.wavelength

        self.mask = aoSimLib.circle(
                0.5*lgsConfig.pupilDiam*self.simConfig.pxlScale,
                self.simConfig.simSize)
        self.geoMask = aoSimLib.circle(self.LGSPupilSize/2., self.LGSPupilSize)
        self.pupilPos = {}
        for i in xrange(self.atmosConfig.scrnNo):
            self.pupilPos[i] = self.getMetaPupilPos(
                self.atmosConfig.scrnHeights[i], 
                pos=self.wfsConfig.GSPosition
                )

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

        #This is the size of the patch of sky the subap sees at the LGS height
        self.FOVPatch = self.subapFOVRad*self.lgsConfig.height

        #This is the number of pxls used for correct FOV (same as a WFS subap)
        self.LGSFOVPxls = numpy.round(
                                self.lgsConfig.pupilDiam * 
                                self.subapFOVRad/self.lgsConfig.wavelength 
                                )
        #Dont want to interpolate down as throwing away info - make sure we
        #interpolate up - but will have to crop down again
        self.fovOversize = 1
        self.LGSFOVOversize = self.LGSFOVPxls
        while self.LGSFOVPxls<self.LGSPupilSize:
            self.fovOversize+=1
            self.LGSFOVOversize = self.LGSFOVPxls * self.fovOversize
        
            
        #This is the size padding size applied to the LGSFFT
        #It is deliberatiely an integer number of wfs subap padding sizes
        #so we can easily bin down to size again
        self.padFactor = 1
        self.LGSFFTPadding = self.subapFFTPadding
        while self.LGSFFTPadding < self.LGSFOVOversize:
            self.padFactor+=1
            self.LGSFFTPadding = self.padFactor * self.subapFFTPadding
        self.LGSFFTPadding*=self.fovOversize
        
        #Set up requried FFTs
        self.geoFFT = AOFFT.FFT(
            inputSize=(self.LGSFFTPadding, self.LGSFFTPadding), axes=(0,1),
            mode="pyfftw", dtype="complex64", 
            THREADS=self.lgsConfig.fftwThreads,
            fftw_FLAGS=(self.lgsConfig.fftwFlag,"FFTW_DESTROY_INPUT"))
            
        #Make new mask
        self.geoMask = aoSimLib.circle(self.LGSFOVOversize/2.,
             self.LGSFOVOversize)

        #Get Tilt Correction
        self.optimiseTilt()
   

 
class GeometricLGS(LGSObj, lineofsight.LineOfSight_Geometric):
     
    def setWFSParams(self, subapFOVRad, subapOversamp, subapFFTPadding):
        
        self.subapFOVRad = subapFOVRad
        self.subapOversamp = subapOversamp
        self.subapFFTPadding = subapFFTPadding
        
        #This is the size of the patch of sky the subap sees at the LGS height
        self.FOVPatch = self.subapFOVRad*self.lgsConfig.height
        
        #This is the number of pxls used for correct FOV (same as a WFS subap)
        self.LGSFOVPxls = numpy.round(
                    self.lgsConfig.pupilDiam * 
                    self.subapFOVRad/self.lgsConfig.wavelength
                    )
        #Dont want to interpolate down as throwing away info - make sure we
        #interpolate up - but will have to crop down again
        self.fovOversize = 1
        self.LGSFOVOversize = self.LGSFOVPxls
        while self.LGSFOVPxls<self.LGSPupilSize:
            self.fovOversize+=1
            self.LGSFOVOversize = self.LGSFOVPxls * self.fovOversize
        
        
        logger.info("fovOversamp: {}".format(self.fovOversize))
            
        #This is the size padding size applied to the LGSFFT
        #It is deliberatiely an integer number of wfs subap padding sizes
        #so we can easily bin down to size again
        self.padFactor = 1
        self.LGSFFTPadding = self.subapFFTPadding
        while self.LGSFFTPadding < self.LGSFOVOversize:
            self.padFactor+=1
            self.LGSFFTPadding = self.padFactor * self.subapFFTPadding
        self.LGSFFTPadding*=self.fovOversize
        
        #Set up requried FFTs
        self.geoFFT = AOFFT.FFT(
            inputSize=(self.LGSFFTPadding, self.LGSFFTPadding), axes=(0,1),
            mode="pyfftw", dtype="complex64", 
            THREADS=self.lgsConfig.fftwThreads,
            fftw_FLAGS=(self.lgsConfig.fftwFlag,"FFTW_DESTROY_INPUT"))
            
        #Make new mask
        self.geoMask = aoSimLib.circle(self.LGSFOVOversize/2.,
             self.LGSFOVOversize)

        #Get Tilt Correction
        self.optimiseTilt()
    
#    def optimiseTilt(self):
#        '''
#        Finds the optimimum correction tilt to apply to each
#        subap to account for an odd even number of points in the subap
#        FFT.
#        to be executed during WFS initialisation.
#        '''
#        X,Y = numpy.meshgrid(numpy.linspace(-1,1,self.LGSFOVOversize),
#                             numpy.linspace(-1,1,self.LGSFOVOversize) )
# 
#        O = numpy.ones((self.LGSFOVOversize, self.LGSFOVOversize))
#        res = scipy.optimize.minimize(self.optFunc,0,args=(X,Y,O),tol=0.0001,
#                                options={"maxiter":100})
#        A = res["x"]
#        print(res)
#        print("Found A!:%f"%A)
#        self.cTilt = A * (X+Y)
#
#        
#    def optFunc(self,A,X,Y,O):
#        '''
#        Function the <optimiseTilt> function uses to optimise
#        corrective even pixel number tilt
#        '''
#
#        self.cTilt = A*(X+Y)
#        self.LGSPSF(phs=O)
#        cent = aoSimLib.simpleCentroid(self.PSF)
#        
#        cent -= self.PSF.shape[0]/2.
        
#        return abs(cent[1]+cent[0])
    
#    def getPupilPhase(self, scrns):
#        self.pupilWavefront = numpy.zeros( 
#                (self.simConfig.simSize,self.simConfig.simSize), dtype="complex64")
#
#        #Stack and sum up the wavefront front the 
#        #phase screens in the LGS meta pupils
#        for layer in scrns:
#            logger.debug("layer: {}".format(layer))
#            self.pupilWavefront += self.getMetaPupilPhase(
#                            scrns[layer],
#                            self.atmosConfig.scrnHeights[layer], 
#                            pos=self.pupilPos[layer]
#                            )
#        
#        #reduce to size of LGS pupil
#        self.pupilWavefront = self.pupilWavefront[
#                        0.5*(self.simConfig.simSize-self.LGSPupilSize):
#                        0.5*(self.simConfig.simSize+self.LGSPupilSize),
#                        0.5*(self.simConfig.simSize-self.LGSPupilSize):
#                        0.5*(self.simConfig.simSize+self.LGSPupilSize)  ]
#                        
#        return self.pupilWavefront

    def LGSPSF(self,scrns=None, phs=None):

        self.EField =  numpy.zeros( 
               (self.simConfig.simSize, self.simConfig.simSize),
               dtype="complex64")
        if scrns!=None:
            self.scrns = scrns
            self.pupilWavefront = self.makePhase(pos=self.pupilPos)
        elif phs!=None:
            self.pupilWavefront = phs
        else:
            raise ValueError("Must providei either scrns or phs")
        
        self.scaledWavefront = aoSimLib.zoom( self.pupilWavefront,
                                                         self.LGSFOVOversize)
        
        #Add corrective Tilts to phase
        self.scaledWavefront += self.cTilt
        
        #Convert phase to complex amplitude (-1 as opposite direction)
        self.EField =  numpy.exp(-1j*self.scaledWavefront)*self.geoMask
        
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


class PhysicalLGS(LGSObj, lineofsight.LineOfSight_Physical ):
    
    def setWFSParams(self, subapFOVRad, subapOversamp, subapFFTPadding):
        
        self.subapFOVRad = subapFOVRad
        self.subapOversamp = subapOversamp
        self.subapFFTPadding = subapFFTPadding
        
        #This is the size of the patch of sky the subap sees at the LGS height
        self.FOVPatch = self.subapFOVRad*self.lgsConfig.height
        
    

    def LGSPSF(self, scrns):
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
        self.phs = self.getMetaPupilPhase(
                    scrns[0], self.atmosConfig.scrnHeights[0],
                    pos=self.pupilPos[0])
        if self.atmosConfig.scrnHeights[0] == 0:
            self.z=0
            self.U *= numpy.exp(-1j*self.phs)
            
        elif self.atmosConfig.scrnHeights[0] > self.lgsConfig.height:
            self.z = 0
            
        else:
            self.z = self.atmosConfig.scrnHeights[0]
            self.U = self.angularSpectrum( self.U, self.lgsConfig.wavelength, self.d1, self.d2, self.z )
            self.U *= numpy.exp(-1j*self.phs)
            
        #Loop over remaining screens and propagate to the screen heights
        #Keep track of total height for use later.
        self.ht = self.z
        for i in xrange(1,len(scrns)):
            logger.debug("Propagate layer: {}".format(i))
            self.phs = self.getMetaPupilPhase(scrns[i],
                                      self.atmosConfig.scrnHeights[i],
                                      pos=self.pupilPos[i] )

            self.z = self.atmosConfig.scrnHeights[i]-self.atmosConfig.scrnHeights[i-1]

            if self.z != 0:
                self.U = opticalPropagationLib.angularSpectrum(
                        self.U, self.lgsConfig.wavelength, self.d1, self.d2, 
                        self.z)
            
            self.U*=self.phs
            self.ht += self.z

        #Finally, propagate to the last layer.
        #Here we scale the array differently to get the correct
        #output scaling for the oversampled WFS
        self.z = self.lgsConfig.height - self.ht

        #Perform calculation onto grid with same pixel scale as WFS subap
        self.gridScale = float(self.FOVPatch)/self.subapFFTPadding
        self.Ufinal = opticalPropagationLib.angularSpectrum(
                self.U, self.lgsConfig.wavelength,
                self.d1, self.gridScale, self.z)
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
                                                                     
                                                                    
