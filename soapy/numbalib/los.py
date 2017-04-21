import numba
import numpy

from . import numbalib

def get_phase_slices(raw_phase_screens, layer_metapupil_coords, phase_screens, thread_pool):

    for i in range(raw_phase_screens.shape[0]):
        # print(raw_phase_screens[i])
        numbalib.bilinear_interp(
                raw_phase_screens[i], layer_metapupil_coords[i, 0], layer_metapupil_coords[i, 1],
                phase_screens[i], thread_pool)




def geometric_propagation(phase_screens, metapupil_coords, output_phase, thread_pool):
    """
    A function which deals with threaded numba interpolation.

    Parameters:
        array (ndarray): The 2-D array to interpolate
        xCoords (ndarray): A 1-D array of x-coordinates
        yCoords (ndarray): A 2-D array of y-coordinates
        interpArray (ndarray): The array to place the calculation
        threads (int): Number of threads to use for calculation

    Returns:
        interpArray (ndarray): A pointer to the calculated ``interpArray''
    """
    nx = output_phase.shape[0]

    n_threads = thread_pool.n_threads

    args = []
    for nt in range(n_threads):
        args.append(
                (phase_screens, metapupil_coords, output_phase,
                numpy.array([int(nt * nx / n_threads), int((nt + 1) * nx / n_threads)])))

    thread_pool.run(geometric_propagation_numba, args)

    return output_phase

# @numba.jit(nopython=True, nogil=True)
def geometric_propagation_numba(phase_screens, metapupil_coords, output_phase, thread_indices):
    """
    2-D interpolation using purely python - fast if compiled with numba
    This version also accepts a parameter specifying how much of the array
    to operate on. This is useful for multi-threading applications.

    Parameters:
        array (ndarray): The 2-D array to interpolate
        xCoords (ndarray): A 1-D array of x-coordinates
        yCoords (ndarray): A 2-D array of y-coordinates
        chunkIndices (ndarray): A 2 element array, with (start Index, stop Index) to work on for the x-dimension.
        interpArray (ndarray): The array to place the calculation

    Returns:
        interpArray (ndarray): A pointer to the calculated ``interpArray''
    """

    jRange = range(metapupil_coords.shape[0])
    for layer in range(phase_screens.shape[0]):
        if metapupil_coords[layer, 0, -1] == phase_screens.shape[1] - 1:
            metapupil_coords[layer, 0, -1] -= 1e-6
        if metapupil_coords[layer, 1, -1] == phase_screens.shape[2] - 1:
            metapupil_coords[layer, 1, -1] -= 1e-6
        for i in range(thread_indices[0], thread_indices[1]):
            print(i)
            x = metapupil_coords[layer, 0, i]
            x1 = numba.int32(x)
            for j in jRange:
                y = metapupil_coords[layer, 1, j]
                y1 = numba.int32(y)

                print(layer, x, y)

                xGrad1 = phase_screens[layer, x1 + 1, y1] - phase_screens[layer, x1, y1]
                a1 = phase_screens[layer, x1, y1] + xGrad1 * (x - x1)

                xGrad2 = phase_screens[layer, x1 + 1, y1 + 1] - phase_screens[layer, x1, y1 + 1]
                a2 = phase_screens[layer, x1, y1 + 1] + xGrad2 * (x - x1)

                yGrad = a2 - a1

                output_phase[i, j] += a1 + yGrad * (y - y1)

    return output_phase
