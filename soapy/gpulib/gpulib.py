from numba import cuda
import numba
import numpy
import math
# Cuda threads per block
CUDA_TPB = 16

def linterp2d(data, xCoords, yCoords, interpArray, threadsPerBlock=None):
    """
    2-D interpolation using purely python - fast if compiled with numba
    Parameters:
        array (ndarray): The 2-D array to interpolate
        xCoords (ndarray): A 1-D array of x-coordinates
        yCoords (ndarray): A 2-D array of y-coordinates
        interpArray (ndarray): The array to place the calculation
    Returns:
        interpArray (ndarray): A pointer to the calculated ``interpArray''
    """
    if threadsPerBlock is None:
        threadsPerBlock = CUDA_TPB

    tpb = (threadsPerBlock, )*2
    # blocks per grid
    bpg = (
            numpy.ceil(interpArray.shape[0]/tpb),
            numpy.ceil(interpArray.shape[1]/tpb)
            )

    linterp2d_kernel[tpb, bpg](data, xCoords, yCoords, interpArray)

    return interpArray

@cuda.jit
def linterp2d_kernel(data, xCoords, yCoords, interpArray):
    """
    2-D interpolation using purely python - fast if compiled with numba
    Parameters:
        array (ndarray): The 2-D array to interpolate
        xCoords (ndarray): A 1-D array of x-coordinates
        yCoords (ndarray): A 2-D array of y-coordinates
        interpArray (ndarray): The array to place the calculation
    Returns:
        interpArray (ndarray): A pointer to the calculated ``interpArray''
    """
    # Thread id in a 1D block
    i, j = cuda.grid(2)

    x = xCoords[i]
    x1 = numba.int32(x)

    y = yCoords[j]
    y1 = numba.int32(y)

    xGrad1 = data[x1+1, y1] - data[x1, y1]
    a1 = data[x1, y1] + xGrad1*(x-x1)

    xGrad2 = data[x1+1, y1+1] - data[x1, y1+1]
    a2 = data[x1, y1+1] + xGrad2*(x-x1)

    yGrad = a2 - a1
    interpArray[i,j] = a1 + yGrad*(y-y1)

def zoom(data, zoomArray, threadsPerBlock=None):
    """
    2-D zoom interpolation using purely python - fast if compiled with numba.
    Both the array to zoom and the output array are required as arguments, the
    zoom level is calculated from the size of the new array.
    Parameters:
        array (ndarray): The 2-D array to zoom
        zoomArray (ndarray): The array to place the calculation
    Returns:
        ndarray: A pointer to the zoomArray
    """
    if threadsPerBlock is None:
        threadsPerBlock = CUDA_TPB

    tpb = (threadsPerBlock, )*2
    # blocks per grid
    bpg = (
            numpy.ceil(float(zoomArray.shape[0])/tpb),
            numpy.ceil(float(zoomArray.shape[1])/tpb)
            )

    zoom_kernel[tpb, bpg](data, zoomArray)

    return zoomArray

@cuda.jit
def zoom_kernel(data, zoomArray):
    """
    2-D zoom interpolation using purely python - fast if compiled with numba.
    Both the array to zoom and the output array are required as arguments, the
    zoom level is calculated from the size of the new array.
    Parameters:
        array (ndarray): The 2-D array to zoom
        zoomArray (ndarray): The array to place the calculation
    """
    i, j = cuda.grid(2)

    x = i*numba.float32(data.shape[0]-1)/(zoomArray.shape[0]-0.99999999)
    x1 = numba.int32(x)

    y = j*numba.float32(data.shape[1]-1)/(zoomArray.shape[1]-0.99999999)
    y1 = numba.int32(y)

    xGrad1 = data[x1+1, y1] - data[x1, y1]
    a1 = data[x1, y1] + xGrad1*(x-x1)

    xGrad2 = data[x1+1, y1+1] - data[x1, y1+1]
    a2 = data[x1, y1+1] + xGrad2*(x-x1)

    yGrad = a2 - a1
    zoomArray[i,j] = a1 + yGrad*(y-y1)


def phs2EField(phase, EField):
    """
    Converts phase to an efield on the GPU
    """
    if threadsPerBlock is None:
        threadsPerBlock = CUDA_TPB

    tpb = (threadsPerBlock, )*2
    # blocks per grid
    bpg = (
            numpy.ceil(float(phase.shape[0])/tpb),
            numpy.ceil(float(phase.shape[1])/tpb)
            )

    phs2EField_kernel[tpb, bpg](phase, EField)

    return EField

@cuda.jit
def phs2EField_kernel(phase, EField):
    i, j = cuda.grid(2)

    EField[i, j] = math.exp(phs[i, j])

def absSquared3d(inputData, outputData=None, threadsPerBlock=None):

    if threadsPerBlock is None:
        threadsPerBlock = CUDA_TPB

    tpb = (threadsPerBlock,)*3
    # blocks per grid
    bpg = (
            int(numpy.ceil(float(inputData.shape[0])/threadsPerBlock)),
            int(numpy.ceil(float(inputData.shape[1])/threadsPerBlock)),
            int(numpy.ceil(float(inputData.shape[2])/threadsPerBlock))
            )

    if outputData == None:
        outputData = inputData

    absSquared3d_kernel[tpb, bpg](inputData, outputData)

    return outputData

@cuda.jit
def absSquared3d_kernel(inputData, outputData):
    i, j, k = cuda.grid(3)
    if i<inputData.shape[0] and j<inputData.shape[1] and k<inputdata.shape[2]:     
        outputData[i, j, k] = inputData[i, j, k].real**2 + inputData[i, j, k].imag**2


