from soapy import confParse, aoSimLib, DM, WFS, gpulib
import unittest
import numpy
from numba import cuda
import os
CONFIG_PATH = os.path.join(os.path.dirname(os.path.abspath(__file__)), "../conf/")

class testGpuWfs(unittest.TestCase):

    def testMaskCrop2Subaps(self):
        PAD = 10
        # Make simple pupil
        mask = aoSimLib.circle(64, 128)
        # pad like in sim
        mask = numpy.pad(mask, ((PAD, PAD),(PAD, PAD)), mode='constant')
        EField = mask.astype('complex128')

        # find coords for 4x4 subaps
        subapCoords = numpy.zeros((16, 2))
        for x in range(4):
            for y in range(4):
                subapCoords[x*4 + y] = x*32, y*32

        # Populate the arrays formed by this procedure on the CPU
        subapArrays_cpu = numpy.zeros((16, 32, 32)).astype('complex128')
        EField_cpu = EField.copy()
        EField_cpu*=mask
        EField_crop = EField_cpu[PAD:-PAD, PAD:-PAD]

        for i, coords in enumerate(subapCoords):
            x, y = coords
            subapArrays_cpu[i] = EField_crop[x: x+32, y: y+32]

        # Now do the same for the GPU...
        subapArrays_gpu = cuda.device_array_like(subapArrays_cpu)
        coords_gpu = cuda.to_device(subapCoords)
        EField_gpu = cuda.to_device(EField)
        mask_gpu = cuda.to_device(mask)

        gpulib.wfs.maskCrop2Subaps(subapArrays_gpu, EField_gpu, mask_gpu, 10, coords_gpu)

        print(subapArrays_cpu, subapArrays_gpu)

        assert numpy.allclose(subapArrays_cpu, subapArrays_gpu.copy_to_host())
