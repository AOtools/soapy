from threading import Thread
from queue import Queue
import multiprocessing
N_CPU = multiprocessing.cpu_count()

import numpy
import numba

ASEC2RAD = (numpy.pi/(180 * 3600))

class LineOfSight(object):

    def __init__(self, obj_config, soapy_config, mask=None):


        self.direction = obj_config.position
        self.src_altitude = obj_config.source_altitude

        self.n_layers = soapy_config.atmos.scrnNo
        self.layer_altitudes = soapy_config.atmos.scrnHeights

        self.n_dm = soapy_config.sim.nDM
        self.dm_altitudes = [soapy_config.dms[i].altitude for i in range(soapy_config.sim.nDM)]

        self.phase_pxl_scale = soapy_config.sim.pxlScale**(-1)
        self.pupil_size = soapy_config.sim.pupilSize
        self.nx_scrn_size = soapy_config.sim.scrnSize

        self.threads = soapy_config.sim.threads

        # Calculate coords of phase at each altitude
        self.layer_metapupil_coords = numpy.zeros((self.n_layers, 2, self.pupil_size))
        for i in range(self.n_layers):
            x1, x2, y1, y2 = self.calculate_altitude_coords(self.layer_altitudes[i])
            self.layer_metapupil_coords[i, 0] = numpy.linspace(x1, x2, self.pupil_size) + self.nx_scrn_size/2.
            self.layer_metapupil_coords[i, 1] = numpy.linspace(y1, y2, self.pupil_size) + self.nx_scrn_size/2.

        # Calculate coords of phase at each DM altitude
        self.dm_metapupil_coords = numpy.zeros((self.n_dm, 2, self.pupil_size))
        for i in range(self.n_dm):
            x1, x2, y1, y2 = self.calculate_altitude_coords(self.dm_altitudes[i])
            self.dm_metapupil_coords[i, 0] = numpy.linspace(x1, x2, self.pupil_size) + self.nx_scrn_size/2.
            self.dm_metapupil_coords[i, 1] = numpy.linspace(y1, y2, self.pupil_size) + self.nx_scrn_size/2.

        self.raw_phase_screens = numpy.zeros((self.n_layers, self.nx_scrn_size, self.nx_scrn_size))
        self.raw_phase_correction = numpy.zeros((self.n_dm, self.nx_scrn_size, self.nx_scrn_size))

        # The phase chopped out of each layer at correction position and scaling
        self.phase_screens = numpy.zeros((self.n_layers, self.pupil_size, self.pupil_size))
        self.phase_correction = numpy.zeros((self.n_dm, self.pupil_size, self.pupil_size))

        # set mask
        if mask.shape == (soapy_config.sim.simSize, soapy_config.sim.simSize):
            p = soapy_config.sim.simPad
            self.mask = mask[p:-p, p:-p]
        elif mask.shape == (soapy_config.sim.pupilSize, soapy_config.sim.pupilSize):
            self.mask = mask
        else:
            raise ValueError("LineOfSight Mask shape not compatible")

        self.output_phase = numpy.zeros((self.pupil_size, self.pupil_size))

    def calculate_altitude_coords(self, layer_altitude):
        """

        :param layer_altitude:
        :return:
        """
        direction_radians = ASEC2RAD * numpy.array(self.direction)

        centre = (direction_radians * layer_altitude) / self.phase_pxl_scale

        if self.src_altitude != 0:
            meta_pupil_size = self.pupil_size * (1 - layer_altitude/self.src_altitude)
        else:
            meta_pupil_size = self.pupil_size

        x1 = centre[0] - meta_pupil_size/2.
        x2 = centre[0] + meta_pupil_size/2.
        y1 = centre[1] - meta_pupil_size/2.
        y2 = centre[1] + meta_pupil_size/2.

        return x1, x2, y1, y2

    def get_phase_slices(self):
        """
        Calculates the phase seen by the los at each altitude. Compiles a list of each phase.
        """
        get_phase_slices(
                self.raw_phase_screens, self.layer_metapupil_coords, self.phase_screens, self.threads)

    def propagate_light(self):
        """
        Propagates light through each layer
        """
        self.output_phase = self.phase_screens.sum(0)

    def get_phase_correction_slices(self):
        get_phase_slices(
                self.raw_phase_correction, self.dm_metapupil_coords, self.phase_correction, self.threads)

    def perform_correction(self):
        self.correction = self.phase_correction.sum(0)

        self.output_phase += self.correction

    def frame(self, phase_screens=None, phase_correction=None):
        """
        Calculates the phase through a line of sight above a telescope for a single frame

        Parameters:
            phase_screens (ndarray): A 3-d array of phase screen large enough to fit the line of sight
            phase_correction (ndarray): A 3-d array of phase correction

        Returns:
            ndarray: Output phase
        """
        if phase_screens is not None:
            if phase_screens.ndim == 3:
                self.raw_phase_screens = phase_screens
                # print("Get Phase Slices")
                self.get_phase_slices()
                # print("Propagate Light")
                self.propagate_light()

            elif phase_screens.ndim == 2:
                self.output_phase[:] = phase_screens


        else:
            self.output_phase[:] = 0
            self.raw_phase_screens[:] = 0



        if phase_correction is not None:
            self.raw_phase_correction = phase_correction
            # Now at telescope, so apply mask
            # self.output_phase *= self.mask
            self.get_phase_correction_slices()

            self.perform_correction()
        else:
            self.raw_phase_correction[:] = 0


        # Now at telescope, so apply mask
        # self.output_phase *= self.mask


        # # apply mask
        # if self.mask is not None:
        #     self.output_phase *= self.mask

        return self.output_phase



# LOS Functions
# -------------


def get_phase_slices(raw_phase_screens, layer_metapupil_coords, phase_screens, threads=None):

    if threads is None:
        threads = N_CPU

    for i in range(raw_phase_screens.shape[0]):
        bilinear_interp(raw_phase_screens[i], layer_metapupil_coords[i, 0], layer_metapupil_coords[i, 1], phase_screens[i], threads)


def bilinear_interp_notb(data, xCoords, yCoords, interpArray, threads=None):
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
    nx = xCoords.shape[0]

    Ts = []
    for t in range(threads):

        Ts.append(Thread(target=bilinear_interp_numba,
                         args=(
                             data, xCoords, yCoords,
                             numpy.array([int(t * nx / threads), int((t + 1) * nx / threads)]),
                             interpArray)
                         ))
        Ts[t].start()

    for T in Ts:
        T.join()

    return interpArray

def bilinear_interp(data, xCoords, yCoords, interpArray, threads=None):
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
    nx = xCoords.shape[0]

    Ts = []
    tc_Qs = []
    for t in range(threads):
        tc_Q = Queue()
        tc_Qs.append(tc_Q)
        Ts.append(Thread(target=run_threaded_func,
                         args=(bilinear_interp_numba,
                               (data, xCoords, yCoords,
                             numpy.array([int(t * nx / threads), int((t + 1) * nx / threads)]),
                             interpArray), tc_Q)
                         ))
        Ts[t].start()

    for T in Ts:
        T.join()

    # Check for errors:
    for n_t, tc_Q in enumerate(tc_Qs):
        potential_exception = tc_Q.get()
        if potential_exception!=False:
            raise potential_exception
    return interpArray

@numba.jit(nopython=True, nogil=True)
def bilinear_interp_numba(data, xCoords, yCoords, chunkIndices, interpArray):
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

    if xCoords[-1] == data.shape[0] - 1:
        xCoords[-1] -= 1e-6
    if yCoords[-1] == data.shape[1] - 1:
        yCoords[-1] -= 1e-6

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

def run_threaded_func(func, args, traceback_queue):
    """
    Runs a function and sends any potential errors down a queue to be checked later.
    If no tracebacks caused, sends `False` down Queue

    Parameters:
            func (function): Function to run
            args (tuple): Arguments to give to function
            traceback_queue (Queue): python queue.Queue object to send traceback.
    """
    try:
        func(*args)
        traceback_queue.put(False)
    except Exception as exc:
        traceback_queue.put(exc)





