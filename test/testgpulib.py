from soapy import confParse, aoSimLib, DM, WFS, gpulib
import unittest
import numpy
from numba import cuda
from scipy.interpolate import interp2d
import os
CONFIG_PATH = os.path.join(os.path.dirname(os.path.abspath(__file__)), "../conf/")
import pylab


def testRegularBilinearInterp():
    inSize = 3
    outSize = 16

    inputData = numpy.arange(inSize**2).reshape(inSize, inSize).astype('float32')
    outputData =  numpy.zeros((outSize, outSize)).astype('float32')
    print(inputData.shape)
    xMin = numpy.float32(1)
    xMax = numpy.float32(2)
    xSize = float(outputData.shape[0])
    yMin = numpy.float32(1)
    yMax = numpy.float32(2.)
    ySize = float(outputData.shape[1])

    # Do CPU version#############
    coords = numpy.arange(inputData.shape[0])
    interpObj = interp2d(
            coords, coords, inputData, copy=False)

    xCoords = numpy.linspace(xMin, xMax-1, xSize)
    yCoords = numpy.linspace(yMin, yMax-1, ySize)
    outputData_cpu = interpObj(xCoords, yCoords).astype('float32')
    ################
    
    # Do GPU Version
    outputData_gpu = cuda.to_device(outputData)
    gpulib.bilinterp2d_regular(
            inputData, xMin, xMax, xSize, yMin, yMax, ySize, outputData_gpu)
    outputData_gpu_host = outputData_gpu.copy_to_host()
    ########################

    pylab.figure()
    pylab.title("CPU")
    pylab.subplot(1,2,1)
    pylab.imshow(outputData_cpu)
    pylab.colorbar()
    pylab.draw()

    pylab.subplot(1,2,2)
    pylab.title("GPU")
    pylab.imshow(outputData_gpu_host)
    pylab.colorbar()
    pylab.show()
    #assert(numpy.allclose(outputData_cpu, outputData_gpu.copy_to_host()))
    return outputData_cpu, outputData_gpu_host


if __name__=="__main__":
    cpu, gpu = testRegularBilinearInterp()
