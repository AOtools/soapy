from pyAOS import WFS

from reikna import cluda
from reikna.core import Type

from . import reiknalib, AOFFT

import numpy

CDTYPE = numpy.complex64
DTYPE = numpy.float32

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
        self.scaledEFieldGPU = self.thr.to_device(
                numpy.zeros(
                        (self.wfsConfig.subaps*self.subapFOVSpacing,)*2
                        ).astype(CDTYPE)
                )
        
        #subapCoords for sorting subaps
        self.subapCoordsGPU = self.thr.to_device(self.subapCoords)
        
    def initCalculations(self):
        
        self.scaledEField = numpy.empty(
                (self.wfsConfig.subaps*self.subapFOVSpacing,)*2, dtype=CDTYPE)
        
        print(self.scaledEField.shape)
        print(self.EField.shape)
        print(self.scaleEFieldXCoords.shape)
                
        scaleEFieldGPU = reiknalib.Interp2dGPU(
                self.scaledEField, self.EField, self.scaleEFieldXCoords
                )
        self.scaleEFieldGPU = scaleEFieldGPU.compile(self.thr)
        
        mulScaledEField = reiknalib.MultiplyArrays2d(self.scaledEField)     
        self.mulScaledEField = mulScaledEField.compile(self.thr)
    
        #used to turn EField into subaps
        self.makeSubaps = reiknalib.MakeSubaps( 
                self.subapArrays, self.scaledEField,
                self.subapCoords
                ).compile(self.thr)
        
        #The forward FFTs for the subaps
        self.gpuFFT = reiknalib.ftAbs(
                self.FPSubapArrays, self.subapArraysGPU, axes=(1,2)).compile(self.thr)
        
    def calcFocalPlane(self):
        """
        Calculates the focal plane of the SH WFS, given the EField across
        the WFS.
        
        Also splits focal plane up into subap peices.
        
        This is the start of current work of the GPU, so must load the stacked 
        EField on the device first.
        """
        print('send efield...')
        self.EFieldGPU = self.thr.to_device(self.EField)
        
        print('scale efield...')
        self.scaleEFieldGPU(
                self.scaledEFieldGPU, self.EFieldGPU, 
                self.scaleEFieldXCoordsGPU, self.scaleEFieldYCoordsGPU)
        
        print('mul efield...')
        self.mulScaledEField(
                self.scaledEFieldGPU, self.scaledEFieldGPU, self.scaledMaskGPU)
        
        print('make subaps...')
        self.makeSubaps(
                self.subapArraysGPU, self.scaledEFieldGPU, self.subapCoordsGPU
                )
    
        print('do fft...')
        print(self.FPSubapArraysGPU.shape, self.FPSubapArraysGPU.dtype)
        print(self.subapArraysGPU.shape, self.subapArraysGPU.dtype)
        self.gpuFFT(self.FPSubapArraysGPU, self.subapArraysGPU)
        
        print('get data...')

        self.FPSubapArrays[:] += AOFFT.ftShift2d(self.FPSubapArraysGPU.get())
        