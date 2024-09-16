import numpy
from scipy.interpolate import RectBivariateSpline, griddata

def zoom(array, newSize, order=3):
    """
    A Class to zoom 2-dimensional arrays using interpolation

    Uses the scipy `RectBivariateSpline` interpolation routine to zoom into an array. Can cope with real of complex data.

    Parameters:
        array (ndarray): 2-dimensional array to zoom
        newSize (tuple): the new size of the required array
        order (int, optional): Order of interpolation to use. default is 3

    Returns:
        ndarray : zoom array of new size.
    """
    try:
        xSize = newSize[0]
        ySize = newSize[1]

    except (IndexError, TypeError):
        xSize = ySize = newSize

    coordsX = numpy.linspace(0, array.shape[0]-1, xSize)
    coordsY = numpy.linspace(0, array.shape[1]-1, ySize)

    #If array is complex must do 2 interpolations
    if array.dtype==numpy.complex64 or array.dtype==numpy.complex128:

        realInterpObj = RectBivariateSpline(
                numpy.arange(array.shape[0]),
                numpy.arange(array.shape[1]),
                array.real,
                kx=order, ky=order
        )
        imagInterpObj = RectBivariateSpline(
                numpy.arange(array.shape[0]),
                numpy.arange(array.shape[1]),
                array.imag,
                kx=order, ky=order
        )
    else:

        interpObj = RectBivariateSpline(
                numpy.arange(array.shape[0]),
                numpy.arange(array.shape[1]),
                array,
                kx=order, ky=order
        )

        return interpObj(coordsY,coordsX)

def zoom_rbs(array, newSize, order=3):
    """
    A Class to zoom 2-dimensional arrays using RectBivariateSpline interpolation

    Uses the scipy ``RectBivariateSpline`` interpolation routine to zoom into an array. Can cope with real of complex data. May be slower than above ``zoom``, as RBS routine copies data.

    Parameters:
        array (ndarray): 2-dimensional array to zoom
        newSize (tuple): the new size of the required array
        order (int, optional): Order of interpolation to use. default is 3

    Returns:
        ndarray : zoom array of new size.
    """
    try:
        xSize = int(newSize[0])
        ySize = int(newSize[1])

    except IndexError:
        xSize = ySize = int(newSize)

    coordsX = numpy.linspace(0, array.shape[0]-1, xSize)
    coordsY = numpy.linspace(0, array.shape[1]-1, ySize)

    #If array is complex must do 2 interpolations
    if array.dtype==numpy.complex64 or array.dtype==numpy.complex128:
        realInterpObj = RectBivariateSpline(
                numpy.arange(array.shape[0]), numpy.arange(array.shape[1]),
                array.real, kx=order, ky=order)
        imagInterpObj = RectBivariateSpline(
                numpy.arange(array.shape[0]), numpy.arange(array.shape[1]),
                array.imag, kx=order, ky=order)

        return (realInterpObj(coordsY,coordsX)
                            + 1j*imagInterpObj(coordsY,coordsX))

    else:

        interpObj = RectBivariateSpline(   numpy.arange(array.shape[0]),
                numpy.arange(array.shape[1]), array, kx=order, ky=order)


        return interpObj(coordsY,coordsX)


def binImgs(data, n):
    '''
    Bins one or more images down by the given factor
    bins. n must be a factor of data.shape, who knows what happens
    otherwise......

    Parameters:
        data (ndimage): 2 or 3d array of image(s) to bin
        n (int): binning factor

    Returns:
        binned image(s): ndarray
    '''
    shape = numpy.array( data.shape )

    n = int(numpy.round(n))

    if len(data.shape)==2:
        shape[-1]/=n
        binnedImgTmp = numpy.zeros( shape, dtype=data.dtype )
        for i in range(n):
            binnedImgTmp += data[:,i::n]
        shape[-2]/=n
        binnedImg = numpy.zeros( shape, dtype=data.dtype )
        for i in range(n):
            binnedImg += binnedImgTmp[i::n,:]

        return binnedImg
    else:
        shape[-1]/=n
        binnedImgTmp = numpy.zeros ( shape, dtype=data.dtype )
        for i in range(n):
            binnedImgTmp += data[...,i::n]

        shape[-2] /= n
        binnedImg = numpy.zeros( shape, dtype=data.dtype )
        for i in range(n):
            binnedImg += binnedImgTmp[...,i::n,:]

        return binnedImg


def zoomWithMissingData(data, newSize,
                        method='linear',
                        non_valid_value=numpy.nan):
    '''
    Zoom 2-dimensional or 3D arrays using griddata interpolation.
    This allows interpolation over unstructured data, e.g. interpolating values
    inside a pupil but excluding everything outside.
    See also DM.CustomShapes.

    Note that it can be time consuming, particularly on 3D data

    Parameters
    ----------
    data : ndArray
        2d or 3d array. If 3d array, interpolate by slices of the first dim.
    newSize : tuple
        2 value for the new array (or new slices) size.
    method: str
        'linear', 'cubic', 'nearest'
    non_valid_value: float
        typically, NaN or 0. value in the array that are not valid for the
        interpolation.

    Returns
    -------
    arr : ndarray
        of dimension (newSize[0], newSize[1]) or
        (data.shape[0], newSize[0], newSize[1])
    '''
    if len(data.shape) == 3:
        arr = data[0, :, :]
    else:
        assert len(data.shape) == 2
        arr = data

    Nx = arr.shape[0]
    Ny = arr.shape[1]
    coordX = (numpy.arange(Nx) - Nx / 2. + 0.5) / (Nx / 2.)
    coordY = (numpy.arange(Ny) - Ny / 2. + 0.5) / (Ny / 2.)
    Nx = newSize[0]
    Ny = newSize[1]
    ncoordX = (numpy.arange(Nx) - Nx / 2. + 0.5) / (Nx / 2.)
    ncoordY = (numpy.arange(Ny) - Ny / 2. + 0.5) / (Ny / 2.)

    x, y = numpy.meshgrid(coordX, coordY)
    xnew, ynew = numpy.meshgrid(ncoordX, ncoordY)

    if len(data.shape) == 2:
        idx = ~(arr == non_valid_value)
        znew = griddata((x[idx], y[idx]), arr[idx], (xnew, ynew),
                        method=method)
        return znew
    elif len(data.shape) == 3:
        narr = numpy.zeros((data.shape[0], newSize[0], newSize[1]))
        for i in range(data.shape[0]):
            arr = data[i, :, :]
            idx = ~(arr == non_valid_value)
            znew = griddata((x[idx], y[idx]), arr[idx], (xnew, ynew),
                            method=method)
            narr[i, :, :] = znew
        return narr
