import numpy
from .import aoSimLib, AOFFT
import logging
from . import opticalPropagationLib as O

import scipy.optimize


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


    def __init__(self, pupilSize, GSPos, pxlScale, scrnHeights, LGSHeight,
                 LGSPupilSize, scrns, LGSFFTPadding, LGSWL,
                 subapFOV, pxlsPerSubap, subapOversamp, phsWvl, 
                 pyfftw_THREADS, pyfftw_FLAG="FFTW_MEASURE"):

        self.pupilSize = pupilSize
        self.GSPos = GSPos
        self.pxlScale = pxlScale
        self.scrnHeights = scrnHeights
        self.LGSHeight = LGSHeight
        self.LGSPupilDiam = LGSPupilSize
        self.LGSPupilSize = int(numpy.round(LGSPupilSize * self.pxlScale))
        self.LGSWL = LGSWL
        self.LGSFFTPadding = LGSFFTPadding
        self.subapFOV = subapFOV
        self.subapOversamp = subapOversamp
        self.pxlsPerSubap = pxlsPerSubap
        self.fftw_FLAG = pyfftw_FLAG
        self.fftw_THREADS = pyfftw_THREADS
        
        #Phase power scaling factor for lgs wavelength
        self.r0scale = phsWvl/ self.LGSWL

        self.mask = aoSimLib.circle(self.LGSPupilSize/2,self.pupilSize)
        self.geoMask = aoSimLib.circle(self.LGSPupilSize/2., self.LGSPupilSize)
        self.pupilPos = {}
        for i in scrns:
            self.pupilPos[i] = self.metaPupilPos(scrnHeights[i])

        self.FPPxlScale = self.LGSWL * self.pxlScale / self.LGSFFTPadding
        self.FPPxlScale *= 3600. * 180./numpy.pi
        logging.debug("LGS FPPxlsScale: %f"%self.FPPxlScale)

        self.FFT = AOFFT.FFT((pupilSize,pupilSize), axes=(0,1),mode="pyfftw",
            dtype = "complex64",direction="FORWARD", THREADS=pyfftw_THREADS,
            fftw_FLAGS=(pyfftw_FLAG,))
        self.iFFT = AOFFT.FFT((pupilSize,pupilSize),axes=(0,1),mode="pyfftw",
            dtype="complex64",direction="BACKWARD", 
            THREADS=pyfftw_THREADS, fftw_FLAGS=(pyfftw_FLAG,))
        
 

    def setWFSParams(self, subapFOVRad, subapOversamp, subapFFTPadding):
        
        self.subapFOVRad = subapFOVRad
        self.subapOversamp = subapOversamp
        self.subapFFTPadding = subapFFTPadding
        
        #This is the size of the patch of sky the subap sees at the LGS height
        self.FOVPatch = self.subapFOVRad*self.LGSHeight
        
        #This is the number of pxls used for correct FOV (same as a WFS subap)
        self.LGSFOVPxls = numpy.round(self.LGSPupilDiam * 
                                                self.subapFOVRad/self.LGSWL )
        #Dont want to interpolate down as throwing away info - make sure we
        #interpolate up - but will have to crop down again
        self.fovOversize = 1
        self.LGSFOVOversize = self.LGSFOVPxls
        while self.LGSFOVPxls<self.LGSPupilSize:
            self.fovOversize+=1
            self.LGSFOVOversize = self.LGSFOVPxls * self.fovOversize
        
        
        print("fovOversamp: %d"%self.fovOversize)
            
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
            mode="pyfftw", dtype="complex64", THREADS=self.fftw_THREADS,
            fftw_FLAGS=(self.fftw_FLAG,))
            
        #Make new mask
        self.geoMask = aoSimLib.circle(self.LGSFOVOversize/2.,
             self.LGSFOVOversize)

        #Get Tilt Correction
        self.optimiseTilt()



    def metaPupilPos(self,height):

        #Convert positions into radians

        GSPos = numpy.array(self.GSPos)*numpy.pi/(3600*180.0)

        #Position of centre of GS metapupil off axis at required height
        GSCent = numpy.tan(GSPos)*height


        return GSCent


    
    def metaPupilPhase(self,scrn,height,pos):

        GSCent = pos*self.pxlScale
        scrnX,scrnY = scrn.shape

        x1 = int(round(scrnX/2. + GSCent[0] - self.pupilSize/2.))
        x2 = int(round(scrnX/2. + GSCent[0] + self.pupilSize/2.))
        y1 = int(round(scrnY/2. + GSCent[1] - self.pupilSize/2.))
        y2 = int(round(scrnY/2. + GSCent[1] + self.pupilSize/2.))

        logging.debug("LGS MetaPupil Coords: %i:%i,%i:%i"%(x1,x2,y1,y2))

        metaPupil = scrn[ x1:x2, y1:y2 ].copy()

        return metaPupil

    
    def getPupilPhase(self,scrns):
        self.pupilWavefront = numpy.zeros( 
                (self.pupilSize,self.pupilSize), dtype="complex64")

        #Stack and sum up the wavefront front the 
        #phase screens in the LGS meta pupils
        for layer in scrns:
            self.pupilWavefront += self.metaPupilPhase(scrns[layer],
                            self.scrnHeights[layer], self.pupilPos[layer])
        
        #reduce to size of LGS pupil
        self.pupilWavefront = self.pupilWavefront[
                        0.5*(self.pupilSize-self.LGSPupilSize):
                        0.5*(self.pupilSize+self.LGSPupilSize),
                        0.5*(self.pupilSize-self.LGSPupilSize):
                        0.5*(self.pupilSize+self.LGSPupilSize)  ]
                        
        return self.pupilWavefront
        
 
class GeometricLGS(LGSObj):
    
    def setWFSParams(self, subapFOVRad, subapOversamp, subapFFTPadding):
        
        self.subapFOVRad = subapFOVRad
        self.subapOversamp = subapOversamp
        self.subapFFTPadding = subapFFTPadding
        
        #This is the size of the patch of sky the subap sees at the LGS height
        self.FOVPatch = self.subapFOVRad*self.LGSHeight
        
        #This is the number of pxls used for correct FOV (same as a WFS subap)
        self.LGSFOVPxls = numpy.round(self.LGSPupilDiam * 
                                                self.subapFOVRad/self.LGSWL )
        #Dont want to interpolate down as throwing away info - make sure we
        #interpolate up - but will have to crop down again
        self.fovOversize = 1
        self.LGSFOVOversize = self.LGSFOVPxls
        while self.LGSFOVPxls<self.LGSPupilSize:
            self.fovOversize+=1
            self.LGSFOVOversize = self.LGSFOVPxls * self.fovOversize
        
        
        print("fovOversamp: %d"%self.fovOversize)
            
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
            mode="pyfftw", dtype="complex64", THREADS=self.fftw_THREADS,
            fftw_FLAGS=(self.fftw_FLAG,))
            
        #Make new mask
        self.geoMask = aoSimLib.circle(self.LGSFOVOversize/2.,
             self.LGSFOVOversize)

        #Get Tilt Correction
        self.optimiseTilt()
    
    def optimiseTilt(self):
        '''
        Finds the optimimum correction tilt to apply to each
        subap to account for an odd even number of points in the subap
        FFT.
        to be executed during WFS initialisation.
        '''
        X,Y = numpy.meshgrid(numpy.linspace(-1,1,self.LGSFOVOversize),
                             numpy.linspace(-1,1,self.LGSFOVOversize) )

        
        O = numpy.ones((self.LGSFOVOversize, self.LGSFOVOversize))
        res = scipy.optimize.minimize(self.optFunc,0,args=(X,Y,O),tol=0.0001,
                                options={"maxiter":100})
        A = res["x"]
        print(res)
        print("Found A!:%f"%A)
        self.cTilt = A * (X+Y)

        
    def optFunc(self,A,X,Y,O):
        '''
        Function the <optimiseTilt> function uses to optimise
        corrective even pixel number tilt
        '''

        self.cTilt = A*(X+Y)
        self.LGSPSF(phs=O)
        cent = aoSimLib.simpleCentroid(self.PSF)
        
        cent -= self.PSF.shape[0]/2.
        
        return abs(cent[1]+cent[0])
    
    def LGSPSF(self,scrns=None, phs=None):

        if scrns!=None:
            self.pupilWavefront = self.getPupilPhase(scrns)
        else:
            self.pupilWavefront = phs
        
        self.scaledWavefront = aoSimLib.zoom( self.pupilWavefront,
                                                         self.LGSFOVOversize)
        
        #Add corrective Tilts to phase
        self.scaledWavefront += self.cTilt
        
        #Convert phase to complex amplitude (-1 as opposite direction)
        self.EField =  numpy.exp(-1j*self.scaledWavefront)*self.geoMask
        
        self.geoFFT.inputData[  :self.LGSFOVOversize,
                                :self.LGSFOVOversize] = self.EField
        fPlane = abs(numpy.fft.fftshift(self.geoFFT())**2)
        
        #Crop to required FOV
        crop = self.subapFFTPadding*0.5/ self.fovOversize
        fPlane = fPlane[self.LGSFFTPadding*0.5 - crop:
                        self.LGSFFTPadding*0.5 + crop,
                        self.LGSFFTPadding*0.5 - crop:
                        self.LGSFFTPadding*0.5 + crop    ]
                        
        #now bin to size of wfs
        LGS.PSF = aoSimLib.binImgs(fPlane, self.padFactor)

class PhysicalLGS(LGSObj):
    
    def setWFSParams(self, subapFOVRad, subapOversamp, subapFFTPadding):
        
        self.subapFOVRad = subapFOVRad
        self.subapOversamp = subapOversamp
        self.subapFFTPadding = subapFFTPadding
        
        #This is the size of the patch of sky the subap sees at the LGS height
        self.FOVPatch = self.subapFOVRad*self.LGSHeight
        
    
    def angularSpectrum(self,Uin,wvl,d1,d2,z):
        N = Uin.shape[0] #Assumes Uin is square.
        k = 2*numpy.pi/wvl     #optical wavevector

        (x1,y1) = numpy.meshgrid(d1*numpy.arange(-N/2,N/2),
                                 d1*numpy.arange(-N/2,N/2),)
        r1sq = (x1**2 + y1**2) + 1e-10

        #Spatial Frequencies (of source plane)
        df1 = 1. / (N*d1)
        fX,fY = numpy.meshgrid(df1*numpy.arange(-N/2,N/2),
                               df1*numpy.arange(-N/2,N/2))
        fsq = fX**2 + fY**2

        #Scaling Param
        m = float(d2)/d1

        #Observation Plane Co-ords
        x2,y2 = numpy.meshgrid( d2*numpy.arange(-N/2,N/2),
                                d2*numpy.arange(-N/2,N/2) )
        r2sq = x2**2 + y2**2

        #Quadratic phase factors
        Q1 = numpy.exp( 1j * k/2. * (1-m)/z * r1sq)

        Q2 = numpy.exp(-1j * numpy.pi**2 * 2 * z/m/k*fsq)

        Q3 = numpy.exp(1j * k/2. * (m-1)/(m*z) * r2sq)

        #Compute propagated field
        Uout = Q3 * self.ift2( Q2 * self.ft2(Q1 * Uin/m,d1),df1 )
        
        return Uout
        
    def ft2(self,g,delta):
        self.FFT.inputData[:] = numpy.fft.fftshift(g)
        G = numpy.fft.fftshift( self.FFT() * delta**2 )

        return G

    def ift2(self,G,delta_f):
        N = G.shape[0]
        self.iFFT.inputData[:] = numpy.fft.ifftshift(G)
        g = numpy.fft.ifftshift(self.iFFT()) * (N * delta_f)**2
        return g
        
    def LGSPSF(self,scrns):
        '''
        Computes the LGS PSF for each frame by propagating
        light to each turbulent layer with a fresnel algorithm
        '''

        self.U = self.mask.astype("complex64")

        #Keep same scale through intermediate calculations,
        #This keeps the array at the telescope diameter size.
        self.d1 = self.d2 = (self.pxlScale)**(-1.)

        #if the 1st screen is at the ground, no propagation neccessary
        #if not, then propagate to that height.
        self.phs = self.metaPupilPhase(scrns[0], self.scrnHeights[0],self.pupilPos[0])
        if self.scrnHeights[0] == 0:
            self.z=0
            self.U *= numpy.exp(-1j*self.phs)
            
        elif self.scrnHeights[0] > self.LGSHeight:
            self.z = 0
            
        else:
            self.z = self.scrnHeights[0]
            self.U = self.angularSpectrum( self.U, self.LGSWL, self.d1, self.d2, self.z )
            self.U *= numpy.exp(-1j*self.phs)
            
        #Loop over remaining screens and propagate to the screen heights
        #Keep track of total height for use later.
        self.ht = self.z
        for i in xrange(1,len(scrns)):
            self.phs = self.metaPupilPhase(scrns[i],
                                      self.scrnHeights[i],
                                      self.pupilPos[i] )

            self.z = self.scrnHeights[i]-self.scrnHeights[i-1]

            if self.z == 0:
                self.U *= numpy.exp(-1j * self.phs)
            else:
                self.U = self.angularSpectrum(self.U*numpy.exp(-1j*self.phs),
                                           self.LGSWL, self.d1, self.d2, self.z)
            self.ht += self.z

        #Finally, propagate to the last layer.
        #Here we scale the array differently to get the correct
        #output scaling for the oversampled WFS
        self.z = self.LGSHeight - self.ht

        #Perform calculation onto grid with same pixel scale as WFS subap
        self.gridScale = float(self.FOVPatch)/self.subapFFTPadding
        self.Ufinal = self.angularSpectrum( self.U, self.LGSWL,
                                        self.d1, self.gridScale, self.z)
        self.psf1 = abs(self.Ufinal)**2
        
        #Now fit this grid onto size WFS is expecting
        crop = self.subapFFTPadding/2.
        if self.pupilSize>self.subapFFTPadding:
            
            self.PSF = self.psf1[   int(numpy.round(0.5*self.pupilSize-crop)):
                                    int(numpy.round(0.5*self.pupilSize+crop)),
                                    int(numpy.round(0.5*self.pupilSize-crop)):
                                    int(numpy.round(0.5*self.pupilSize+crop))  ]
                                    
        else:
            self.PSF = numpy.zeros((self.subapFFTPadding, self.subapFFTPadding))
            self.PSF[   int(numpy.round(crop-self.pupilSize*0.5)):
                        int(numpy.round(crop+self.pupilSize*0.5)),
                        int(numpy.round(crop-self.pupilSize*0.5)):
                        int(numpy.round(crop+self.pupilSize*0.5))] = self.psf1
                                                                     
                                                                    