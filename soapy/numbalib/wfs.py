import multiprocessing
N_CPU = multiprocessing.cpu_count()
from threading import Thread
import queue

import numpy
import numba

def zoom(data, zoomArray, threads=None):
    """
    A function which deals with threaded numba interpolation.

    Parameters:
        array (ndarray): The 2-D array to interpolate
        zoomArray (ndarray, tuple): The array to place the calculation, or the shape to return
        threads (int): Number of threads to use for calculation

    Returns:
        interpArray (ndarray): A pointer to the calculated ``interpArray''
    """

    if threads is None:
        threads = N_CPU

    nx = zoomArray.shape[0]

    Ts = []
    for t in range(threads):
        Ts.append(Thread(target=zoom_numbaThread,
                         args=(
                             data,
                             numpy.array([int(t * nx / threads), int((t + 1) * nx / threads)]),
                             zoomArray)
                         ))
        Ts[t].start()

    for T in Ts:
        T.join()

    return zoomArray


@numba.jit(nopython=True, nogil=True)
def zoom_numbaThread(data, chunkIndices, zoomArray):
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

    for i in range(chunkIndices[0], chunkIndices[1]):
        x = i * numba.float32(data.shape[0] - 1) / (zoomArray.shape[0] - 0.99999999)
        x1 = numba.int32(x)
        for j in range(zoomArray.shape[1]):
            y = j * numba.float32(data.shape[1] - 1) / (zoomArray.shape[1] - 0.99999999)
            y1 = numba.int32(y)

            xGrad1 = data[x1 + 1, y1] - data[x1, y1]
            a1 = data[x1, y1] + xGrad1 * (x - x1)

            xGrad2 = data[x1 + 1, y1 + 1] - data[x1, y1 + 1]
            a2 = data[x1, y1 + 1] + xGrad2 * (x - x1)

            yGrad = a2 - a1
            zoomArray[i, j] = a1 + yGrad * (y - y1)

    return zoomArray


def chop_subaps(phase, subap_coords, nx_subap_size, subap_array, threads=None):
    if threads is None:
        threads = N_CPU

    n_subaps = subap_coords.shape[0]
    Ts = []
    for t in range(threads):
        Ts.append(Thread(target=chop_subaps_numba,
                         args=(
                             phase, subap_coords, nx_subap_size, subap_array,
                             numpy.array([int(t * n_subaps / threads), int((t + 1) * n_subaps / threads)]),
                         )
                         ))
        Ts[t].start()

    for T in Ts:
        T.join()

    return subap_array


@numba.jit(nopython=True, nogil=True)
def chop_subaps_numba(phase, subap_coords, nx_subap_size, subap_array, subap_indices):
    for i in range(subap_indices[0], subap_indices[1]):
        x = subap_coords[i, 0]
        y = subap_coords[i, 1]

        subap_array[i, :nx_subap_size, :nx_subap_size] = phase[x:x + nx_subap_size, y:y + nx_subap_size]

    return subap_array


def chop_subaps_slow(phase, subap_coords, nx_subap_size, subap_array, threads=None):
    for i in range(len(subap_coords)):
        x = subap_coords[i, 0]
        y = subap_coords[i, 1]

        subap_array[i, :nx_subap_size, :nx_subap_size] = phase[x: x + nx_subap_size, y: y + nx_subap_size]

    return subap_array


def chop_subaps_efield(phase, subap_coords, nx_subap_size, subap_array, threads=None):
    if threads is None:
        threads = N_CPU

    n_subaps = subap_coords.shape[0]
    Ts = []
    for t in range(threads):
        Ts.append(Thread(target=chop_subaps_efield_numba,
                         args=(
                             phase, subap_coords, nx_subap_size, subap_array,
                             numpy.array([int(t * n_subaps / threads), int((t + 1) * n_subaps / threads)]),
                         )
                         ))
        Ts[t].start()

    for T in Ts:
        T.join()

    return subap_array


@numba.jit(nopython=True, nogil=True)
def chop_subaps_efield_numba(phase, subap_coords, nx_subap_size, subap_array, subap_indices):
    for i in range(subap_indices[0], subap_indices[1]):
        x = subap_coords[i, 0]
        y = subap_coords[i, 1]

        subap_array[i, :nx_subap_size, :nx_subap_size] = numpy.exp(1j * phase[x:x + nx_subap_size, y:y + nx_subap_size])

    return subap_array


def chop_subaps_efield_slow(phase, subap_coords, nx_subap_size, subap_array, threads=None):
    for i in range(len(subap_coords)):
        x = subap_coords[i, 0]
        y = subap_coords[i, 1]

        subap_array[i, :nx_subap_size, :nx_subap_size] = numpy.exp(
            1j * phase[x: x + nx_subap_size, y: y + nx_subap_size])

    return subap_array


def place_subaps_on_detector(subap_imgs, detector_img, detector_positions, subap_coords, threads=None):
    if threads is None:
        threads = N_CPU

    n_subaps = detector_positions.shape[0]

    Ts = []
    for t in range(threads):
        Ts.append(Thread(target=place_subaps_on_detector_numba,
                         args=(
                             subap_imgs, detector_img, detector_positions, subap_coords,
                             numpy.array([int(t * n_subaps / threads), int((t + 1) * n_subaps / threads)]),
                         )
                         ))
        Ts[t].start()

    for T in Ts:
        T.join()

    return detector_img


@numba.jit(nopython=True, nogil=True)
def place_subaps_on_detector_numba(subap_imgs, detector_img, detector_positions, subap_coords, subap_indices):
    """
    Puts a set of sub-apertures onto a detector image
    """

    for i in range(subap_indices[0], subap_indices[1]):
        x1, x2, y1, y2 = detector_positions[i]
        sx1 ,sx2, sy1, sy2 = subap_coords[i]
        detector_img[x1: x2, y1: y2] += subap_imgs[i, sx1: sx2, sy1: sy2]

    return detector_img


def place_subaps_on_detector_slow(subap_imgs, detector_img, subap_positions, threads=None):
    """
    Puts a set of sub-apertures onto a detector image
    """

    for i in range(subap_positions.shape[0]):
        x1, x2, y1, y2 = subap_positions[i]

        detector_img[x1: x2, y1: y2] = subap_imgs[i]

    return detector_img


def bin_imgs(subap_imgs, bin_size, binned_imgs, threads=None):
    if threads is None:
        threads = N_CPU

    n_subaps = subap_imgs.shape[0]

    Ts = []
    for t in range(threads):
        Ts.append(Thread(target=bin_imgs_numba,
                         args=(
                             subap_imgs, bin_size, binned_imgs,
                             numpy.array([int(t * n_subaps / threads), int((t + 1) * n_subaps / threads)]),
                         )
                         ))
        Ts[t].start()

    for T in Ts:
        T.join()

    return binned_imgs


@numba.jit(nopython=True, nogil=True)
def bin_imgs_numba(imgs, bin_size, new_img, subap_range):
    # loop over subaps
    for n in range(subap_range[0], subap_range[1]):
        # loop over each element in new array
        for i in range(new_img.shape[1]):
            x1 = i * bin_size
            for j in range(new_img.shape[2]):
                y1 = j * bin_size
                new_img[n, i, j] = 0
                # loop over the values to sum
                for x in range(bin_size):
                    for y in range(bin_size):
                        new_img[n, i, j] += imgs[n, x + x1, y + y1]


def bin_imgs_slow(imgs, bin_size, new_img):
    # loop over subaps
    for n in range(imgs.shape[0]):
        # loop over each element in new array
        for i in range(new_img.shape[1]):
            x1 = i * bin_size
            for j in range(new_img.shape[2]):
                y1 = j * bin_size
                new_img[n, i, j] = 0
                # loop over the values to sum
                for x in range(bin_size):
                    for y in range(bin_size):
                        new_img[n, i, j] += imgs[n, x + x1, y + y1]


class Centroider(object):
    def __init__(self, n_subaps, nx_subap_pxls, threads=None):

        if threads is None:
            self.threads = 1
        else:
            self.threads = threads

        self.n_subaps = n_subaps
        self.nx_subap_pxls = nx_subap_pxls

        self.indices = numpy.indices((self.nx_subap_pxls, self.nx_subap_pxls))

        self.centroids = numpy.zeros((n_subaps, 2))

    def __call__(self, subaps):
        self.centre_of_gravity_numpy(subaps)
        return self.centroids

    def centre_of_gravity_numpy(self, subaps):
        self.centroids[:, 1] = ((self.indices[0] * subaps).sum((1, 2)) / subaps.sum((1, 2))) + 0.5 - subaps.shape[
                                                                                                         1] * 0.5
        self.centroids[:, 0] = ((self.indices[1] * subaps).sum((1, 2)) / subaps.sum((1, 2))) + 0.5 - subaps.shape[
                                                                                                         2] * 0.5
        return self.centroids

    def centre_of_gravity_numba(self, subaps):

        centre_of_gravity(subaps, self.indices, self.centroids, self.threads)
        return self.centroids


def centre_of_gravity(subaps, indices, centroids, threads=None):
    if threads is None:
        threads = N_CPU

    n_subaps = subaps.shape[0]

    Ts = []
    for t in range(threads):
        Ts.append(Thread(target=centre_of_gravity_numba,
                         args=(
                             subaps, indices, centroids,
                             numpy.array([int(t * n_subaps / threads), int((t + 1) * n_subaps / threads)]),
                         )))
        Ts[t].start()

    for T in Ts:
        T.join()

    return centroids


@numba.jit(nopython=True, nogil=True)
def centre_of_gravity_numba(subaps, indices, centroids, thread_indices):
    s1, s2 = thread_indices
    nx_subap_size = subaps.shape[1]
    subaps = subaps[s1:s2]

    centroids[s1:s2, 0] = (
                              indices[0] * subaps).sum((1, 2)) / subaps.sum((1, 2)) + 0.5 - nx_subap_size * 0.5
    centroids[s1:s2, 1] = (
                              indices[1] * subaps).sum((1, 2)) / subaps.sum((1, 2)) + 0.5 - nx_subap_size * 0.5


def abs_squared(subap_data, subap_output, threads=None):
    if threads is None:
        threads = N_CPU

    n_subaps = subap_data.shape[0]

    Ts = []
    for t in range(threads):
        x1 = int(t * n_subaps / threads)
        x2 = int((t + 1) * n_subaps / threads)
        Ts.append(Thread(target=abs_squared_numba,
                         args=(
                             subap_data, subap_output,
                             numpy.array([x1, x2]),
                         )))

        Ts[t].start()

    for T in Ts:
        T.join()

    return subap_output


@numba.jit(nopython=True)
def abs_squared_numba(data, output_data, indices):
    for n in range(indices[0], indices[1]):
        for x in range(data.shape[1]):
            for y in range(data.shape[2]):
                output_data[n, x, y] = data[n, x, y].real ** 2 + data[n, x, y].imag ** 2


def abs_squared_slow(data, output_data, threads=None):
    for n in range(data.shape[0]):
        for x in range(data.shape[1]):
            for y in range(data.shape[2]):
                output_data[n, x, y] = data[n, x, y].real ** 2 + data[n, x, y].imag ** 2

