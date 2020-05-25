import multiprocessing
N_CPU = multiprocessing.cpu_count()
from threading import Thread

# python3 has queue, python2 has Queue
try:
    import queue
except ImportError:
    import Queue as queue

import numpy
import numba

def bilinear_interp(data, xCoords, yCoords, interpArray, bounds_check=True):
    """
    A function which deals with numba interpolation.

    Parameters:
        array (ndarray): The 2-D array to interpolate
        xCoords (ndarray): A 1-D array of x-coordinates
        yCoords (ndarray): A 2-D array of y-coordinates
        interpArray (ndarray): The array to place the calculation
        bounds_check (bool, optional): Do bounds checkign in algorithm? Faster if False, but dangerous! Default is True
    Returns:
        interpArray (ndarray): A pointer to the calculated ``interpArray''
    """
    if bounds_check:
        bilinear_interp_numba(data, xCoords, yCoords, interpArray)
    else:
        bilinear_interp_numba_inbounds(data, xCoords, yCoords, interpArray)

    return interpArray




@numba.jit(nopython=True, parallel=True)
def bilinear_interp_numba(data, xCoords, yCoords, interpArray):
    """
    2-D interpolation using purely python - fast if compiled with numba
    This version also accepts a parameter specifying how much of the array
    to operate on. This is useful for multi-threading applications.

    Bounds are checks to ensure no out of bounds access is attempted to avoid
    mysterious seg-faults

    Parameters:
        array (ndarray): The 2-D array to interpolate
        xCoords (ndarray): A 1-D array of x-coordinates
        yCoords (ndarray): A 2-D array of y-coordinates
        interpArray (ndarray): The array to place the calculation

    Returns:
        interpArray (ndarray): A pointer to the calculated ``interpArray''
    """
    jRange = range(yCoords.shape[0])
    for i in numba.prange(xCoords.shape[0]):
        x = xCoords[i]
        if x >= data.shape[0] - 1:
            x = data.shape[0] - 1 - 1e-9
        x1 = numba.int32(x)
        for j in jRange:
            y = yCoords[j]
            if y >= data.shape[1] - 1:
                y = data.shape[1] - 1 - 1e-9
            y1 = numba.int32(y)

            xGrad1 = data[x1 + 1, y1] - data[x1, y1]
            a1 = data[x1, y1] + xGrad1 * (x - x1)

            xGrad2 = data[x1 + 1, y1 + 1] - data[x1, y1 + 1]
            a2 = data[x1, y1 + 1] + xGrad2 * (x - x1)

            yGrad = a2 - a1
            interpArray[i, j] = a1 + yGrad * (y - y1)
    return interpArray


@numba.jit(nopython=True, nogil=True)
def bilinear_interp_numba_inbounds(data, xCoords, yCoords, interpArray):
    """
    2-D interpolation using purely python - fast if compiled with numba
    This version also accepts a parameter specifying how much of the array
    to operate on. This is useful for multi-threading applications.

    **NO BOUNDS CHECKS ARE PERFORMED - IF COORDS REFERENCE OUT-OF-BOUNDS
    ELEMENTS THEN MYSTERIOUS SEGFAULTS WILL OCCURR!!!**

    Parameters:
        array (ndarray): The 2-D array to interpolate
        xCoords (ndarray): A 1-D array of x-coordinates
        yCoords (ndarray): A 2-D array of y-coordinates
        interpArray (ndarray): The array to place the calculation

    Returns:
        interpArray (ndarray): A pointer to the calculated ``interpArray''
    """
    jRange = range(yCoords.shape[0])
    for i in range(xCoords.shape[0]):
        x = xCoords[i]
        x1 = numba.int32(x)
        for j in jRange:
            y = yCoords[j]
            y1 = numba.int32(y)

            xGrad1 = data[x1 + 1, y1] - data[x1, y1]
            a1 = data[x1, y1] + xGrad1 * (x - x1)

            xGrad2 = data[x1 + 1, y1 + 1] - data[x1, y1 + 1]
            a2 = data[x1, y1 + 1] + xGrad2 * (x - x1)

            yGrad = a2 - a1
            interpArray[i, j] = a1 + yGrad * (y - y1)
    return interpArray


@numba.jit(nopython=True, nogil=True)
def rotate(data, interpArray, rotation_angle):
    for i in range(interpArray.shape[0]):
        for j in range(interpArray.shape[1]):

            i1 = i - (interpArray.shape[0] / 2. - 0.5)
            j1 = j - (interpArray.shape[1] / 2. - 0.5)
            x = i1 * numpy.cos(rotation_angle) - j1 * numpy.sin(rotation_angle)
            y = i1 * numpy.sin(rotation_angle) + j1 * numpy.cos(rotation_angle)

            x += data.shape[0] / 2. - 0.5
            y += data.shape[1] / 2. - 0.5

            if x >= data.shape[0] - 1:
                x = data.shape[0] - 1.1
            x1 = numpy.int32(x)

            if y >= data.shape[1] - 1:
                y = data.shape[1] - 1.1
            y1 = numpy.int32(y)

            xGrad1 = data[x1 + 1, y1] - data[x1, y1]
            a1 = data[x1, y1] + xGrad1 * (x - x1)

            xGrad2 = data[x1 + 1, y1 + 1] - data[x1, y1 + 1]
            a2 = data[x1, y1 + 1] + xGrad2 * (x - x1)

            yGrad = a2 - a1
            interpArray[i, j] = a1 + yGrad * (y - y1)
    return interpArray


@numba.vectorize(["float32(complex64)"], nopython=True, target="parallel")
def abs_squared(data):
    return abs(data)**2


@numba.jit(nopython=True, parallel=True)
def bin_img(imgs, bin_size, new_img):
    # loop over each element in new array
    for i in numba.prange(new_img.shape[0]):
        x1 = i * bin_size

        for j in range(new_img.shape[1]):
            y1 = j * bin_size
            new_img[i, j] = 0

            # loop over the values to sum
            for x in range(bin_size):
                for y in range(bin_size):
                    new_img[i, j] += imgs[x1 + x, y1 + y]


def bin_img_slow(img, bin_size, new_img):

    # loop over each element in new array
    for i in range(new_img.shape[0]):
        x1 = i * bin_size
        for j in range(new_img.shape[1]):
            y1 = j * bin_size
            new_img[i, j] = 0
            # loop over the values to sum
            for x in range(bin_size):
                for y in range(bin_size):
                    new_img[i, j] += img[x + x1, y + y1]


@numba.jit(nopython=True, parallel=True)
def zoom(data, zoomArray):
    """
    2-D zoom interpolation using purely python - fast if compiled with numba.
    Both the array to zoom and the output array are required as arguments, the
    zoom level is calculated from the size of the new array.

    Parameters:
        array (ndarray): The 2-D array to zoom
        zoomArray (ndarray): The array to place the calculation

    Returns:
        interpArray (ndarray): A pointer to the calculated ``zoomArray''
    """
    for i in numba.prange(zoomArray.shape[0]):
        x = i*numba.float32(data.shape[0]-1)/(zoomArray.shape[0]-0.99999999)
        x1 = numba.int32(x)
        for j in range(zoomArray.shape[1]):
            y = j*numba.float32(data.shape[1]-1)/(zoomArray.shape[1]-0.99999999)
            y1 = numba.int32(y)

            xGrad1 = data[x1+1, y1] - data[x1, y1]
            a1 = data[x1, y1] + xGrad1*(x-x1)

            xGrad2 = data[x1+1, y1+1] - data[x1, y1+1]
            a2 = data[x1, y1+1] + xGrad2*(x-x1)

            yGrad = a2 - a1
            zoomArray[i,j] = a1 + yGrad*(y-y1)

    return zoomArray


@numba.jit(nopython=True, parallel=True)
def fftshift_2d_inplace(data):

    for x in numba.prange(data.shape[0]//2):
        for y in range(data.shape[1]//2):
            
            tmp = data[x, y]
            data[x, y] = data[x + data.shape[0]//2, y + data.shape[1]//2]
            data[x + data.shape[0]//2, y + data.shape[1]//2] = tmp

            tmp = data[x + data.shape[0]//2, y] 
            data[x + data.shape[0]//2, y] = data[x, y + data.shape[1]//2]
            data[x, y + data.shape[1]//2] = tmp

    return data