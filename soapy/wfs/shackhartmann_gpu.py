"""
The Shack Hartmann WFS accelerated using numba cuda
"""

import numpy
from numba import cuda
from accelerate.cuda.fft.binding import Plan, CUFFT_C2C
from .. import AOFFT, aoSimLib, LGS, logger, lineofsight
from . import shackhartmann
from .. import gpulib

# The data type of data arrays (complex and real respectively)
CDTYPE = numpy.complex64
DTYPE = numpy.float32

class ShackHartmannGPU(shackhartmann.ShackHartmann):
    def initLos(self):
        """
        Initialises the ``LineOfSight`` object, which gets the phase or EField in a given direction through turbulence.
        """
        self.los = lineofsight.LineOfSightGPU(
                self.config, self.soapyConfig,
                propagationDirection="down")


    def allocDataArrays(self):
        super(ShackHartmannGPU, self).allocDataArrays()

        self.subapArrays_gpu = cuda.to_device(self.FPSubapArrays.copy().astype(CDTYPE))
        self.ftSubapArrays_gpu = cuda.device_array_like(self.subapArrays_gpu)
        self.binnedFPSubapArrays_gpu = cuda.to_device(self.binnedFPSubapArrays)
        self.FPSubapArrays_gpu = cuda.to_device(self.FPSubapArrays)
        self.wfsDetectorPlane_gpu = cuda.to_device(self.wfsDetectorPlane)

        self.losPhase_gpu = cuda.device_array(self.los.phase.shape, dtype=DTYPE)
        self.scaledPhase_gpu = cuda.device_array((self.scaledEFieldSize,)*2, dtype=DTYPE)
        self.scaledEField_gpu = cuda.device_array((self.scaledEFieldSize,)*2, dtype=CDTYPE)

        # Data that must be transferred to the GPU
        self.scaledMask_gpu = cuda.to_device(self.scaledMask)
        self.scaledSubapCoords = numpy.round(
                self.subapCoords * self.subapFOVSpacing/self.PPSpacing
                ).astype('int32')
        scaledSimOffset = int(round(self.simConfig.simPad * self.subapFOVSpacing/self.PPSpacing))
        self.scaledSubapCoords += scaledSimOffset
        self.scaledSubapCoords_gpu = cuda.to_device(self.scaledSubapCoords)

        self.tiltfixEField = numpy.exp(1j*self.tiltFix)
        self.tiltfixEField_gpu = cuda.to_device(self.tiltfixEField)

        self.fftShape = ((self.subapFFTPadding,)*2)
        self.ftplan_gpu = Plan.many(
                self.fftShape, CUFFT_C2C,
                batch=self.activeSubaps)



    def calcFocalPlane(self, intensity=1):
        '''
        Calculates the wfs focal plane, given the phase across the WFS

        Parameters:
            intensity (float): The relative intensity of this frame, is used when multiple WFS frames taken for extended sources.
        '''

        if self.config.propagationMode=="Geometric":
            # Have to make phase the correct size if geometric prop
            # Put phase on GPU
            self.losPhase_gpu = cuda.to_device(self.los.phase.astype(DTYPE))
            # Scale to required size and make complex amp
            self.scaledEField_gpu = gpulib.wfs.zoomToEField(
                    self.losPhase_gpu, self.scaledEField_gpu)

        else:
            # If physical prop, correct size already. Put on GPU
            self.scaledEField_gpu = cuda.to_device(self.los.EField)

        self.subapArrays_gpu = gpulib.wfs.maskCrop2Subaps(
                 self.subapArrays_gpu, self.scaledEField_gpu,
                 self.scaledMask_gpu, self.subapFOVSpacing,
                 self.scaledSubapCoords_gpu, self.tiltfixEField_gpu)

        # Do the FFT with numba accelerate
        self.ftplan_gpu.forward(self.subapArrays_gpu, self.ftSubapArrays_gpu)
        gpulib.absSquared3d(
                self.ftSubapArrays_gpu, outputData=self.FPSubapArrays_gpu)
        self.FPSubapArrays[:] = self.FPSubapArrays_gpu.copy_to_host()
        self.FPSubapArrays = AOFFT.ftShift2d(self.FPSubapArrays)

        # if intensity==1:
        #     self.FPSubapArrays += numpy.abs(AOFFT.ftShift2d(self.FFT()))**2
        # else:
        #     self.FPSubapArrays += intensity*numpy.abs(
        #             AOFFT.ftShift2d(self.FFT()))**2
        #######################################
