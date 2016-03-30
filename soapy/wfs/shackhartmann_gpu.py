"""
The Shack Hartmann WFS accelerated using numba cuda
"""

import numpy
from numba import cuda

from .. import AOFFT, aoSimLib, LGS, logger
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

        self.losPhase_gpu = cuda.device_array(self.los.phase.shape, dtype=DTYPE)
        self.scaledPhase_gpu = cuda.device_array((self.scaledEFieldSize,)*2, dtype=DTYPE)
        self.scaledEField_gpu = cuda.device_array((self.scaledEFieldSize,)*2, dtype=CDTYPE)

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
            scaledPhase = gpulib.wfs.zoomToEField(
                    self.losPhase_gpu, self.scaledEField_gpu)

        else:
            # If physical prop, correct size already. Put on GPU
            self.scaledEField_gpu = cuda.to_device(self.los.EField)

        scaledEField = self.scaledEField_gpu.copy_to_host()

        # Copied from shackhartmann CPU verision
        ########################################
        # Apply the scaled pupil mask
        scaledEField *= self.scaledMask

        # Now cut out only the eField across the pupilSize
        coord = round(int(((self.scaledEFieldSize/2.)
                - (self.wfsConfig.nxSubaps*self.subapFOVSpacing)/2.)))
        self.cropEField = scaledEField[coord:-coord, coord:-coord]

        #create an array of individual subap EFields
        for i in xrange(self.activeSubaps):
            x,y = numpy.round(self.subapCoords[i] *
                                     self.subapFOVSpacing/self.PPSpacing)
            self.subapArrays[i] = self.cropEField[
                                    int(x):
                                    int(x+self.subapFOVSpacing) ,
                                    int(y):
                                    int(y+self.subapFOVSpacing)]

        # do the fft to all subaps at the same time
        # and convert into intensity
        self.FFT.inputData[:] = 0
        self.FFT.inputData[:,:int(round(self.subapFOVSpacing))
                        ,:int(round(self.subapFOVSpacing))] \
                = self.subapArrays*numpy.exp(1j*(self.tiltFix))

        if intensity==1:
            self.FPSubapArrays += numpy.abs(AOFFT.ftShift2d(self.FFT()))**2
        else:
            self.FPSubapArrays += intensity*numpy.abs(
                    AOFFT.ftShift2d(self.FFT()))**2
        #######################################
