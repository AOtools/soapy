import multiprocessing
N_CPU = multiprocessing.cpu_count()
from threading import Thread
import queue

import numpy
import numba

def abs_squared(input_data, output_data, threads=None):
    if threads is None:
        threads = N_CPU

    n_rows = input_data.shape[0]

    Ts = []
    for t in range(threads):
        x1 = int(t * n_rows / threads)
        x2 = int((t+1) * n_rows / threads)
        Ts.append(Thread(target=abs_squared_numba,
                         args=(
                             input_data, output_data,
                             numpy.array([x1, x2]),
                         )))
        Ts[t].start()

    for T in Ts:
        T.join()

    return output_data

@numba.jit(nopython=True)
def abs_squared_numba(data, output_data, indices):

    for x in range(indices[0], indices[1]):
        for y in range(data.shape[1]):
            output_data[x, y] = data[x, y].real**2 + data[x, y].imag**2


def abs_squared_slow(data, output_data, threads=None):

    for x in range(data.shape[0]):
        for y in range(data.shape[1]):
             output_data[x, y] = data[x, y].real**2 + data[x, y].imag**2



def bin_img(input_img, bin_size, binned_img, threads=None):
    if threads is None:
        threads = N_CPU

    n_rows = binned_img.shape[0]

    Ts = []
    for t in range(threads):
        Ts.append(Thread(target=bin_img_numba,
                         args=(
                             input_img, bin_size, binned_img,
                             numpy.array([int(t * n_rows / threads), int((t + 1) * n_rows / threads)]),
                         )
                         ))
        Ts[t].start()

    for T in Ts:
        T.join()

    return binned_img


@numba.jit(nopython=True, nogil=True)
def bin_img_numba(imgs, bin_size, new_img, row_indices):
    # loop over each element in new array
    for i in range(row_indices[0], row_indices[1]):
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
        self.threads = []
        for i in range(n_threads):
            input_queue = queue.Queue()
            output_queue = queue.Queue()
            thread = Thread(target=self._thread_func, args=(input_queue, output_queue))

            self.input_queues.append(input_queue)
            self.output_queues.append(output_queue)
            self.threads.append(thread)

            thread.start()



    def _thread_func(self, input_queue, output_queue):

        while self._running:
            input_args = input_queue.get()
            result = self._func(*input_args)

            output_queue.put(result)


    def run(self, func, args):

        results = []
        self._func = func
        for i, q in enumerate(self.input_queues):
            q.put(args[i])

        for q in self.output_queues:
            results.append(q.get())

        return results

    def _stop_func(self, arg):
        pass

    def stop(self):
        print("Stopping threads!")
        self._running = False
        self.run(self._stop_func, [(None, )]*self.n_threads)

        for t in self.threads:
            t.join()

    def __del__(self):
        self.stop()








