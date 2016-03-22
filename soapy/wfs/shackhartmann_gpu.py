"""
The Shack Hartmann WFS accelerated using numba cuda
"""

import numpy
from numba import cuda

from . import shackhartmann
from .. import gpulib

# The data type of data arrays (complex and real respectively)
CDTYPE = numpy.complex64
DTYPE = numpy.float32



class ShackHartmannGPU(shackhartmann.ShackHartmann):
    def allocDataArrays(self):
        super(ShackHartmannGPU, self).allocDataArrays()

        self.subapArrays_gpu = cuda.to_device(self.subapArrays)
        self.binnedFPSubapArrays_gpu = cuda.to_device(self.binnedFPSubapArrays)
        self.FPSubapArrays_gpu = cuda.to_device(self.FPSubapArrays)
        self.wfsDetectorPlane_gpu = cuda.to_device(self.wfsDetectorPlane)

        self.losPhase_gpu = cuda.array(self.los.phase.shape, dtype=DTYPE)
        self.scaledPhase_gpu = cuda.array((scaledEFieldSize,)*2, dtype=DTYPE)
        self.scaledEField_gpu = cuda.array((scaledEFieldSize,)*2, dtype=CDTYPE)

    def calcFocalPlane(self, intensity=1):
        '''
        Calculates the wfs focal plane, given the phase across the WFS

        Parameters:
            intensity (float): The relative intensity of this frame, is used when multiple WFS frames taken for extended sources.
        '''

        if self.config.propagationMode=="Geometric":
            # Have to make phase the correct size if geometric prop
            # Put phase on GPU
            self.losPhase_gpu = cuda.to_device(self.los.phase)
            # Scale to required size
            scaledPhase = gpulib.zoom(
                    self.los.phase_gpu, self.scaledPhase_gpu)

            # Turn in to complex Amp
            gpulib.phs2EField(self.scaledPhase_gpu, self.scaledEField_gpu)

        else:
            # If physical prop, correct size already. Put on GPU
            self.scaledEField_gpu = cuda.to_device(self.los.EField)
