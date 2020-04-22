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

def bilinear_interp_new(data, xCoords, yCoords, interpArray, bounds_check=True):
    """
    A function which deals with threaded numba interpolation.

    Parameters:
        array (ndarray): The 2-D array to interpolate
        xCoords (ndarray): A 1-D array of x-coordinates
        yCoords (ndarray): A 2-D array of y-coordinates
        interpArray (ndarray): The array to place the calculation
        threads (int): Number of threads to use for calculation
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
        chunkIndices (ndarray): A 2 element array, with (start Index, stop Index) to work on for the x-dimension.
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


def bilinear_interp(data, xCoords, yCoords, interpArray, thread_pool=None, bounds_check=True,):
    """
    A function which deals with threaded numba interpolation.

    Parameters:
        array (ndarray): The 2-D array to interpolate
        xCoords (ndarray): A 1-D array of x-coordinates
        yCoords (ndarray): A 2-D array of y-coordinates
        interpArray (ndarray): The array to place the calculation
        threads (int): Number of threads to use for calculation
        bounds_check (bool, optional): Do bounds checkign in algorithm? Faster if False, but dangerous! Default is True
    Returns:
        interpArray (ndarray): A pointer to the calculated ``interpArray''
    """
    if thread_pool is None:
        thread_pool = ThreadPool(N_CPU)

    elif isinstance(thread_pool, int):
        thread_pool = ThreadPool(thread_pool)

    nx = interpArray.shape[0]

    n_threads = thread_pool.n_threads

    args = []
    for nt in range(n_threads):
        args.append(
            (data, xCoords, yCoords,
             numpy.array([int(nt * nx / n_threads), int((nt + 1) * nx / n_threads)]),
             interpArray))

    if bounds_check:
        thread_pool.run(bilinear_interp_numba_old, args)
    else:
        thread_pool.run(bilinear_interp_numba_inbounds_old, args)

    return interpArray


@numba.jit(nopython=True, nogil=True)
def bilinear_interp_numba_old(data, xCoords, yCoords, chunkIndices, interpArray):
    """
    2-D interpolation using purely python - fast if compiled with numba
    This version also accepts a parameter specifying how much of the array
    to operate on. This is useful for multi-threading applications.

    Parameters:
        array (ndarray): The 2-D array to interpolate
        xCoords (ndarray): A 1-D array of x-coordinates
        yCoords (ndarray): A 2-D array of y-coordinates
        interpArray (ndarray): The array to place the calculation

    Returns:
        interpArray (ndarray): A pointer to the calculated ``interpArray''
    """
    jRange = range(yCoords.shape[0])
    for i in range(chunkIndices[0], chunkIndices[1]):
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
def bilinear_interp_numba_inbounds_old(data, xCoords, yCoords, chunkIndices, interpArray):
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
    jRange = range(yCoords.shape[0])
    for i in range(chunkIndices[0], chunkIndices[1]):
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
def bin_img(input_img, bin_size, binned_img):

    # loop over each element in new array
    for i in numba.prange(binned_img.shape[0]):
        x1 = i * bin_size

        for j in range(binned_img.shape[1]):
            y1 = j * bin_size
            binned_img[i, j] = 0

            # loop over the values to sum
            for x in range(bin_size):
                for y in range(bin_size):
                    binned_img[i, j] += input_img[x1 + x, y1 + y]


def bin_img_old(input_img, bin_size, binned_img, threads=None):
    if threads is None:
        threads = N_CPU

    n_rows = binned_img.shape[0]

    Ts = []
    for t in range(threads):
        Ts.append(Thread(target=bin_img_numba_old,
                         args=(
                             input_img, bin_size, binned_img,
                             numpy.array([int(t * n_rows / threads), int((t + 1) * n_rows / threads)]),
                         )
                         ))
        Ts[t].start()

    for T in Ts:
        T.join()

    return binned_img


@numba.jit(nopython=True, nogil=True, parallel=True)
def bin_img_numba_old(imgs, bin_size, new_img, row_indices):
    # loop over each element in new array
    for i in numba.prange(row_indices[0], row_indices[1]):
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
def fft_shift_2d(data):

    output_data = numpy.zeros_like(data)

    for i in numba.prange(data.shape[0]//2):
        for j in numba.prange(data.shape[1]//2):
            tmp = data[i, j] 
            output_data[i, j] = data[i + data.shape[0]//2, j + data.shape[1]//2]
            output_data[i + data.shape[0]//2, j + data.shape[1]//2] = tmp

            tmp = data[i + data.shape[0]//2, j]
            output_data[i + data.shape[0]//2, j] = data[i, j + data.shape[1]//2]
            output_data[i, j + data.shape[1]//2] = tmp

    return output_data


@numba.jit(nopython=True, parallel=True)
def fft_shift_2d_inplace(data):
    # if output_data is None:
    #     output_data = data

    for i in numba.prange(data.shape[0]//2):
        for j in numba.prange(data.shape[1]//2):
            tmp = data[i, j] 
            data[i, j] = data[i + data.shape[0]//2, j + data.shape[1]//2]
            data[i + data.shape[0]//2, j + data.shape[1]//2] = tmp

            tmp = data[i + data.shape[0]//2, j]
            data[i + data.shape[0]//2, j] = data[i, j + data.shape[1]//2]
            data[i, j + data.shape[1]//2] = tmp

    return data


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
            args = (
                data,
                numpy.array([int(t*nx/threads), int((t+1)*nx/threads)]),
                zoomArray)
            ))
        Ts[t].start()

    for T in Ts:
        T.join()


    return zoomArray

@numba.jit(nopython=True, nogil=True)
def zoom_numbaThread(data,  chunkIndices, zoomArray):
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

def zoom_pool(data, zoomArray, thread_pool):
    """
    A function which deals with threaded numba interpolation.

    Parameters:
        array (ndarray): The 2-D array to interpolate
        zoomArray (ndarray, tuple): The array to place the calculation, or the shape to return
        threads (int): Number of threads to use for calculation

    Returns:
        interpArray (ndarray): A pointer to the calculated ``interpArray''
    """
    nx = zoomArray.shape[0]

    n_threads = thread_pool.n_threads

    args = []
    for nt in range(n_threads):
        args.append((data,
                numpy.array([int(nt*nx/n_threads), int((nt+1)*nx/n_threads)]),
                zoomArray))

    thread_pool.run(zoom_numbaThread, args)

    return zoomArray

class ThreadPool(object):
    """
    A 'pool' of threads that can be used by GIL released Numba functions

    A number of threads are initialised and set to watch a queue for input data
    A function can then be specified and some parameters given, and the arguments are send to the queue,
    where they are seen by each thread as a signal to start work. Once done, the threads will return the
    result of their computation down a different queue. The main thread gathers these results and returns
    them in a list. In practice its probably more useful to get the threads to operate on a static NumPy
    array to avoid sorting out these results.

    Paramters
        n_threads (int): Number of threads in the pool
    """
    def __init__(self, n_threads):

        self.n_threads = n_threads

        self._running = True

        self.input_queues = []
        self.output_queues = []
        self.exception_queues = []
        self.threads = []
        for i in range(n_threads):
            input_queue = queue.Queue()
            output_queue = queue.Queue()
            exception_queue = queue.Queue()
            thread = Thread(
                    target=self._thread_func,
                    args=(input_queue, output_queue, exception_queue))
            thread.daemon = True

            self.input_queues.append(input_queue)
            self.output_queues.append(output_queue)
            self.exception_queues.append(exception_queue)
            self.threads.append(thread)

            thread.start()

        # def cleanup():
        #     self.stop()
        # atexit.register(cleanup)


    def _thread_func(self, input_queue, output_queue, exception_queue):

        while self._running:
            input_args = input_queue.get()
            try:
                result = self._func(*input_args)
                output_queue.put(result)
                exception_queue.put(None)

            except Exception as exc:
                output_queue.put(None)
                exception_queue.put(exc)

    def run(self, func, args):

        results = []
        self._func = func
        for i, q in enumerate(self.input_queues):
            q.put(args[i])

        for i, q in enumerate(self.output_queues):
            results.append(q.get())
            exc = self.exception_queues[i].get()
            if exc is not None:
                raise exc

        return results

    def _stop_func(self, arg):
        pass

    def stop(self):
        print("Stopping threads...")
        self._running = False
        self.run(self._stop_func, [(None, )]*self.n_threads)

        for t in self.threads:
            t.join()
        print("Stopped!")


    def __del__(self):
        if self._running:
            self.stop()








