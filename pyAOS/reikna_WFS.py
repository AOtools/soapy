from pyAOS import WFS

from reikna import cluda
from reikna.core import Type

from . import reiknalib, AOFFT, aoSimLib

import numpy

CDTYPE = WFS.CDTYPE
DTYPE = WFS.DTYPE

class ShackHartmann(WFS.ShackHartmann):
    """
    Class to provide GPU acceleration for the pyAOS Shack Hartmann WFS.
    """
    def __init__(self, simConfig, wfsConfig, atmosConfig, lgsConfig, mask, 
        thr=None, api="ocl"):
        
        self.thr = thr
        self.api = api
        
        super(ShackHartmann, self).__init__(simConfig, wfsConfig, atmosConfig, lgsConfig, mask)
        
        self.initCalculations()
        
    def allocDataArrays(self):
        """
        Allocates data arrays on the GPU corresponding with those allocated on
        the host.
        """
        
        super(ShackHartmann, self).allocDataArrays()
        
        if self.api is not "cuda":
            self.gpuApi = cluda.ocl_api()
        else:
            self.gpuApi = cluda.cuda_api()
            
        if not self.thr:
            try:
                self.thr = self.gpuApi.Thread.create()
            except IndexError:
                self.thr  = self.gpuApi.Thread(
                        self.gpuApi.get_platforms()[0].get_devices()[0])
                
        self.subapArraysGPU = self.thr.to_device(self.subapArrays)
        self.binnedFPSubapArraysGPU = self.thr.to_device(
                                            self.binnedFPSubapArrays)
        self.FPSubapArraysGPU = self.thr.to_device(self.FPSubapArrays)
        self.wfsDetectorPlaneGPU = self.thr.to_device(self.wfsDetectorPlane)
        
        #Other things not usually thought of as "allocated data" in the cpu version
        self.scaledMaskGPU = self.thr.to_device(self.scaledMask.astype(CDTYPE))
        
        #The coords to scale the EField
        self.scaleEFieldXCoords = numpy.linspace(0, self.simConfig.pupilSize-1,
                               self.wfsConfig.subaps*self.subapFOVSpacing
                                                           ).astype("float32")
        self.scaleEFieldYCoords = numpy.linspace(0, self.simConfig.pupilSize-1,
                               self.wfsConfig.subaps*self.subapFOVSpacing
                                                           ).astype("float32")
        self.scaleEFieldXCoords[-1] -= 1e-6
        self.scaleEFieldYCoords[-1] -= 1e-6
        self.scaleEFieldXCoordsGPU = self.thr.to_device(self.scaleEFieldXCoords)
        self.scaleEFieldYCoordsGPU = self.thr.to_device(self.scaleEFieldYCoords)
        
        #And the scaled EField itself

        
        #subapCoords for sorting subaps
        #scaled subap coords to scaled EField size
        self.subapCoordsGPU = self.thr.to_device(
                numpy.round(self.subapCoords*(self.subapFOVSpacing/self.PPSpacing)
                            ))

        

    def initCalculations(self):
        
        self.scaledEFieldGPU = self.thr.to_device(
                numpy.zeros(
                        (self.wfsConfig.subaps*self.subapFOVSpacing,)*2
                        ).astype(CDTYPE)
                )
        

        scaleEFieldGPU = reiknalib.Interp2dGPU(
                self.scaledEFieldGPU, self.EField, self.scaleEFieldXCoordsGPU
                )
        self.scaleEFieldGPU = scaleEFieldGPU.compile(self.thr)
        
        mulScaledEField = reiknalib.MultiplyArrays2d(self.scaledEFieldGPU)     
        self.mulScaledEField = mulScaledEField.compile(self.thr)
    
        #used to turn EField into subaps
        self.makeSubaps = reiknalib.MakeSubaps( 
                self.subapArrays, self.scaledEFieldGPU,
                self.subapCoords
                ).compile(self.thr)
        

        #calc to add the tilt fix
        self.ETiltFixGPU = self.thr.to_device(self.ETiltFix)
        self.addTiltFix = reiknalib.Mul2dto3d(
                self.subapArraysGPU, self.ETiltFix).compile(self.thr)

        #calc to pad arrays before they get FFTed
        self.paddedSubapArraysGPU = self.thr.to_device(
                numpy.zeros(self.FPSubapArrays.shape).astype(CDTYPE))
        self.padFFT = reiknalib.PadArrays(
                self.paddedSubapArraysGPU, self.subapArraysGPU).compile(self.thr)

        #The forward FFTs for the subaps
        self.gpuFFT = reiknalib.ftAbs(
                self.FPSubapArraysGPU, self.paddedSubapArraysGPU,
                axes=(1,2)).compile(self.thr)
                

        self.binSubaps = reiknalib.BinImgs(
                self.binnedFPSubapArraysGPU, self.FPSubapArraysGPU,
                self.wfsConfig.subapOversamp).compile(self.thr)
        
        
        
    def calcFocalPlane(self):
        """
        Calculates the focal plane of the SH WFS, given the EField across
        the WFS.
        
        Also splits focal plane up into subap peices.
        
        This is the start of current work of the GPU, so must load the stacked 
        EField on the device first.
        """

        self.EFieldGPU = self.thr.to_device(self.EField)
        
        self.scaleEFieldGPU(
                self.scaledEFieldGPU, self.EFieldGPU, 
                self.scaleEFieldXCoordsGPU, self.scaleEFieldYCoordsGPU)

        self.mulScaledEField(
                self.scaledEFieldGPU, self.scaledEFieldGPU, self.scaledMaskGPU)

        self.makeSubaps(
                self.subapArraysGPU, self.scaledEFieldGPU, self.subapCoordsGPU
                )

        self.addTiltFix(self.subapArraysGPU, self.ETiltFixGPU)
        self.padFFT(self.paddedSubapArraysGPU, self.subapArraysGPU)
        self.gpuFFT(self.FPSubapArraysGPU, self.paddedSubapArraysGPU)

        #self.FPSubapArrays[:] += AOFFT.ftShift2d(self.FPSubapArraysGPU.get())

    def makeDetectorPlane(self):
        '''
        if required, will convolve final psf with LGS psf, then bins
        psf down to detector size. Finally puts back into wfsFocalPlane Array
        in correct order.
        '''

        #If required, convolve with LGS PSF
        # if self.LGS and self.lgsConfig.lgsUplink and self.iMat!=True:
#    self.LGSUplink()

        #bins back down to correct size and then
        #fits them back in to a focal plane array
        # self.binnedFPSubapArrays[:] = aoSimLib.binImgs(self.FPSubapArrays,
                                            # self.wfsConfig.subapOversamp)

        self.binSubaps(
                self.binnedFPSubapArraysGPU, self.FPSubapArraysGPU, 
                self.wfsConfig.subapOversamp
                )
        self.binnedFPSubapArrays[:] = AOFFT.ftShift2d(self.binnedFPSubapArraysGPU.get())

        self.binnedFPSubapArrays[:] = self.maxFlux* (self.binnedFPSubapArrays.T/
                                            self.binnedFPSubapArrays.max((1,2))
                                                                         ).T

        for i in xrange(self.activeSubaps):

            x,y=self.detectorSubapCoords[i]

            #Set default position to put arrays into (2 subap FOV)
            x1 = int(round(x-self.wfsConfig.pxlsPerSubap/2.))
            x2 = int(round(x+self.wfsConfig.pxlsPerSubap*3./2))
            y1 = int(round(y-self.wfsConfig.pxlsPerSubap/2.))
            y2 = int(round(y+self.wfsConfig.pxlsPerSubap*3./2.))

            #Set defualt size of input array (i.e. all of it)
            x1_fp = int(0)
            x2_fp = int(round(self.wfsConfig.pxlsPerSubap2))
            y1_fp = int(round(0))
            y2_fp = int(round(self.wfsConfig.pxlsPerSubap2))

            #If at the edge of the field, may only fit a fraction in 
            if x==0:
                x1 = int(0)
                x1_fp = int(round(self.wfsConfig.pxlsPerSubap/2.))

            elif x==(self.detectorPxls-self.wfsConfig.pxlsPerSubap):
                x2 = int(round(self.detectorPxls))
                x2_fp = int(round(-1*self.wfsConfig.pxlsPerSubap/2.))

            if y==0:
                y1 = int(0)
                y1_fp = int(round(self.wfsConfig.pxlsPerSubap/2.))

            elif y==(self.detectorPxls-self.wfsConfig.pxlsPerSubap):
                y2 = int(self.detectorPxls)
                y2_fp = int(round(-1*self.wfsConfig.pxlsPerSubap/2.))

            self.wfsDetectorPlane[x1:x2, y1:y2] += (
                    self.binnedFPSubapArrays[i, x1_fp:x2_fp, y1_fp:y2_fp] )

        if self.wfsConfig.SNR and self.iMat!=True:

            self.photonNoise()
            self.readNoise(self.wfsDetectorPlane)
    

