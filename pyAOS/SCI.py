import numpy
import logging
import scipy.ndimage

from . import aoSimLib, AOFFT



class scienceCam:

    def __init__(self, pupilSize,pxlScale, sciPos,
                 scrnHeights, mask,
                 sciFOV, waveLength, telDiam,
                 sciPxls, oversamp, phsWvl):

        self.pupilSize = pupilSize
        self.sciPos = sciPos
        self.scrnHeights = scrnHeights
        self.FOV = sciFOV
        self.wvl = waveLength
        self.telDiam = telDiam
        self.sciPxls = sciPxls
        self.oversamp = oversamp
        self.pxlScale = pxlScale
        self.mask = mask
        self.FOVrad = self.FOV * numpy.pi / (180.*3600)
        self.FOVPxlNo = numpy.round( self.telDiam * self.FOVrad/self.wvl)

        self.scaleFactor = float(self.FOVPxlNo)/self.pupilSize
        self.scaledMask = numpy.round(aoSimLib.zoom(self.mask,self.FOVPxlNo))
        

        #Init FFT object
        self.FFTPadding = self.sciPxls * self.oversamp
        if self.FFTPadding < self.FOVPxlNo:
            while self.FFTPadding<self.FOVPxlNo:
                self.oversamp+=1
                self.FFTPadding\
                        =self.sciPxls*self.oversamp
            logging.info("SCI FFT Padding less than FOV size... Setting oversampling to %d"%self.oversamp)

        self.FFT = AOFFT.FFT(inputSize=(self.FFTPadding,self.FFTPadding),
                            axes=(0,1),
                            mode="pyfftw",
                            dtype="complex64",
                            fftw_FLAGS=("FFTW_MEASURE",))

                            
        #Calculate ideal PSF for purposes of strehl calculation
        self.FFT.inputData[:self.FOVPxlNo,:self.FOVPxlNo] \
                                            =(numpy.exp(1j*self.scaledMask)
                                                    *self.scaledMask)
        fp = abs(numpy.fft.fftshift(self.FFT()))**2
        binFp = aoSimLib.binImgs(fp,self.oversamp)
        print("max Strehl Point: {0}".format(binFp.max()))
        self.psfMax = binFp.max()        
        self.longExpStrehl = 0
        self.instStrehl = 0 
        #Get phase scaling factor to get r0 in other wavelength     
        self.r0Scale = phsWvl/self.wvl

    def metaPupilPos(self,height):
        '''
        Finds the centre of a metapupil at a given height,
        when offset by a given angle in arcsecs
        '''

        #Convert positions into radians
        sciPos = numpy.array(self.sciPos)*numpy.pi/(3600.0*180.0)

        #Position of centre of GS metapupil off axis at required height
        sciCent=numpy.tan(sciPos)*height

        return sciCent



    def metaPupilPhase(self,scrn,height):
        '''
        Returns the phase across a metaPupil at some height
        and angular offset in arcsec
        '''

        sciCent = self.metaPupilPos(height) * self.pxlScale
        logging.debug("SciCents:(%i,%i)"%(sciCent[0],sciCent[1]))

        scrnX,scrnY=scrn.shape


        if      (scrnX/2+sciCent[0]-self.pupilSize/2.0) < 0 \
           or   (scrnX/2+sciCent[0]-self.pupilSize/2.0) > scrnX \
           or   (scrnX/2+sciCent[0]-self.pupilSize/2.0) < 0 \
           or   (scrnY/2+sciCent[1]-self.pupilSize/2.0) > scrnY :

            raise ValueError(  "Sci Position seperation\
                                requires larger scrn size" )

        X1 = scrnX/2+sciCent[0]-self.pupilSize/2.0
        X2 = scrnX/2+sciCent[0]+self.pupilSize/2.0
        Y1 = scrnY/2+sciCent[1]-self.pupilSize/2.0
        Y2 = scrnY/2+sciCent[1]+self.pupilSize/2.0

        metaPupil=scrn[ X1:X2, Y1:Y2 ]

        return metaPupil


    def calcPupilPhase(self):

        '''
        Returns the total phase on a science Camera which is offset
        by a given angle
        '''
        totalPhase=numpy.zeros([self.pupilSize]*2)
        for i in self.scrns:

            phase=self.metaPupilPhase(self.scrns[i],self.scrnHeights[i])
            #print("phase Shape:",phase.shape)

            totalPhase+=phase
        totalPhase *= self.mask

        self.phase=totalPhase

    def calcFocalPlane(self):
        '''
        takes the calculated pupil phase, and uses FFT
        to transform to the focal plane, and scales for correct FOV.
        '''

        phs = aoSimLib.zoom(self.residual, self.FOVPxlNo) *self.r0Scale

        eField = numpy.exp(1j*phs)*self.scaledMask

        self.FFT.inputData[:self.FOVPxlNo,:self.FOVPxlNo] = eField
        focalPlane = numpy.fft.fftshift( self.FFT() )

        focalPlane = numpy.abs(focalPlane)**2

        self.focalPlane = aoSimLib.binImgs( focalPlane , self.oversamp )



    def frame(self,scrns,phaseCorrection=None):

        self.scrns = scrns
        self.calcPupilPhase()

        if phaseCorrection!=None:
            self.residual = self.phase - (phaseCorrection)#/self.wvl)

        else:
            self.residual = self.phase
        self.calcFocalPlane()

        self.instStrehl =  self.focalPlane.max()/self.psfMax


        return self.focalPlane

