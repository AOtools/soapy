from accelerate.cuda.fft.binding import Plan, CUFFT_C2C
from numba import cuda
import numpy

def testBatch2dFFT():

    fftShape = 64
    phsSize = 32
    BATCH = 16

    ftplan = Plan.many((fftShape,)*2, CUFFT_C2C, batch=BATCH)

    data = numpy.ones((BATCH*phsSize*phsSize)).reshape(
            BATCH, phsSize, phsSize
            ).astype('complex64')
    # Pad data
    p = fftShape - phsSize
    data = numpy.pad(data, ((0,0), (0, p), (0, p)), mode='constant')

    print(data.shape)
    data_gpu = cuda.to_device(data)
    assert(numpy.allclose(data_gpu.copy_to_host(), data))
    outputData = numpy.zeros_like(data)
    outputData_gpu = cuda.to_device(outputData)

    # Do CPU FFT
    outputData_cpu = numpy.fft.fft2(data, axes=(1,2))

    # Do GPU FFT
    ftplan.forward(data_gpu, outputData_gpu)

    assert(numpy.allclose(outputData_gpu.copy_to_host(), outputData_cpu))
