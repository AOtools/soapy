from soapy import numbalib
import numpy

import aotools

def test_zoomtoefield():
    """
    Checks that when zooming to efield, the same result is found as when zooming
    then using numpy.exp to get efield.
    """
    input_data = numpy.arange(100).reshape(10,10).astype("float32")

    output_data = numpy.zeros((100, 100), dtype="float32")
    output_efield2 = numpy.zeros((100, 100), dtype="complex64")
    
    numbalib.zoom(input_data, output_data)

    output_efield1 = numpy.exp(1j * output_data)

    numbalib.wfslib.zoomtoefield(input_data, output_efield2)

    assert numpy.allclose(output_efield1, output_efield2)


def test_chop_subaps_mask():
    """
    Tests that the numba routing chops phase into sub-apertures in the same way
    as using numpy indices
    """
    nx_phase = 12
    nx_subap_size = 3
    nx_subaps = nx_phase // nx_subap_size

    phase = (numpy.random.random((nx_phase, nx_phase)) 
                + 1j * numpy.random.random((nx_phase, nx_phase))
                ).astype("complex64")
    subap_array = numpy.zeros((nx_subaps * nx_subaps, nx_subap_size, nx_subap_size)).astype("complex64")
    numpy_subap_array = subap_array.copy()

    mask = aotools.circle(nx_phase/2., nx_phase)

    x_coords, y_coords = numpy.meshgrid(
            numpy.arange(0, nx_phase, nx_subap_size),
            numpy.arange(0, nx_phase, nx_subap_size))
    subap_coords = numpy.array([x_coords.flatten(), y_coords.flatten()]).T

    numpy_chop(phase, subap_coords, nx_subap_size, numpy_subap_array, mask)
    numbalib.wfslib.chop_subaps_mask(
            phase, subap_coords, nx_subap_size, subap_array, mask)
    assert numpy.array_equal(numpy_subap_array, subap_array)


# def test_chop_subaps_mask_threads():
#     """
#     Tests that the numba routing chops phase into sub-apertures in the same way
#     as using numpy indices
#     Runs with multiple threads many times to detectect potential intermittant errors
#     """
#     nx_phase = 12
#     nx_subap_size = 3
#     nx_subaps = nx_phase // nx_subap_size

#     subap_array = numpy.zeros((nx_subaps * nx_subaps, nx_subap_size, nx_subap_size)).astype("complex64")
#     numpy_subap_array = subap_array.copy()

#     mask = aotools.circle(nx_phase/2., nx_phase)

#     x_coords, y_coords = numpy.meshgrid(
#             numpy.arange(0, nx_phase, nx_subap_size),
#             numpy.arange(0, nx_phase, nx_subap_size))
#     subap_coords = numpy.array([y_coords.flatten(),x_coords.flatten()]).T


#     for i in range(50):
#         phase = (numpy.random.random((nx_phase, nx_phase)) 
#                 + 1j * numpy.random.random((nx_phase, nx_phase))
#                 ).astype("complex64")

#         numpy_chop(phase, subap_coords, nx_subap_size, numpy_subap_array, mask)
#         numbalib.wfslib.chop_subaps_mask(
#                 phase, subap_coords, nx_subap_size, subap_array, mask)

#         assert numpy.array_equal(numpy_subap_array, subap_array)

def numpy_chop(phase, subap_coords, nx_subap_size, subap_array, mask):
    """
    Numpy vesion of chop subaps tests
    """
    mask_phase = mask * phase
    for n, (x, y) in enumerate(subap_coords):
        subap_array[n] = mask_phase[
                x: x + nx_subap_size,
                y: y + nx_subap_size
        ]
    return subap_array
    

def test_abs_squared():
    """
    Tests that the numba vectorised and parallelised abs squared gives the same result as numpy
    """
    data = (numpy.random.random((100, 20, 20))
            + 1j * numpy.random.random((100, 20, 20))).astype("complex64")

    output_data = numpy.zeros((100, 20, 20), dtype="float32")

    numbalib.abs_squared(data, out=output_data)

    assert numpy.array_equal(output_data, numpy.abs(data)**2)

def test_place_subaps_detector():

    nx_subaps = 4
    pxls_per_subap = 4
    tot_pxls_per_subap = 2 * pxls_per_subap # More for total FOV
    tot_subaps = nx_subaps * nx_subaps
    nx_pxls = nx_subaps * pxls_per_subap

    detector = numpy.zeros((nx_pxls, nx_pxls))
    detector_numpy = detector.copy()

    subaps = numpy.random.random((tot_subaps, tot_pxls_per_subap, tot_pxls_per_subap))
    # Find the coordinates of the vertices of the subap on teh detector
    detector_coords = []
    subap_coords = []
    for ix in range(nx_subaps):
        x1 = ix * pxls_per_subap - pxls_per_subap/2
        x2 = (ix + 1) * pxls_per_subap + pxls_per_subap/2
        sx1 = 0
        sx2 = tot_pxls_per_subap
        for iy in range(nx_subaps):
            y1 = iy * pxls_per_subap - pxls_per_subap/2
            y2 = (iy + 1) * pxls_per_subap + pxls_per_subap/2
            
            sy1 = 0
            sy2 = tot_pxls_per_subap
            # Check for edge subaps that would be out of bounds
            if x1 < 0:
                x1 = 0
                sx1 = tot_pxls_per_subap / 4
            if x2 > nx_pxls:
                x2 = nx_pxls
                sx2 = 3 * tot_pxls_per_subap / 4
            
            if y1 < 0:
                y1 = 0
                sy1 = tot_pxls_per_subap / 4
            if y2 > nx_pxls:
                y2 = nx_pxls
                sy2 = 3 * tot_pxls_per_subap / 4
            
            detector_coords.append(numpy.array([x1, x2, y1, y2]))
            subap_coords.append(numpy.array([sx1, sx2, sy1, sy2]))

    detector_coords = numpy.array(detector_coords).astype("int")
    subap_coords = numpy.array(subap_coords).astype("int")

    numbalib.wfslib.place_subaps_on_detector(
        subaps, detector, detector_coords, subap_coords)

    numpy_place_subaps(subaps, detector_numpy, detector_coords, subap_coords)

    assert numpy.array_equal(detector, detector_numpy)

def numpy_place_subaps( subap_arrays, detector, detector_subap_coords, valid_subap_coords):
    
    for n, (x1, x2, y1, y2) in enumerate(detector_subap_coords):
        sx1, sx2, sy1, sy2 = valid_subap_coords[n]
        subap = subap_arrays[n]    
        detector[x1: x2, y1: y2] += subap[sx1: sx2, sy1: sy2]
    
    return detector

if __name__ == "__main__":
    test_zoomtoefield()