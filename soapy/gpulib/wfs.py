from numba import cuda
import numba
import math
import numpy

# Cuda threads per block
CUDA_TPB = 16

def zoomToEField(data, zoomArray, threadsPerBlock=None):
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
            numpy.ceil(float(zoomArray.shape[0])/threadsPerBlock),
            numpy.ceil(float(zoomArray.shape[1])/threadsPerBlock)
            )

    zoomToEField_kernel[tpb, bpg](data, zoomArray)

    return zoomArray

@cuda.jit
def zoomToEField_kernel(data, zoomArray):
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
    phsVal = a1 + yGrad*(y-y1)

    zoomArray[i,j] = math.cos(phsVal) + 1j * math.sin(phsVal)
