#Copyright Durham University and Andrew Reeves
#2014

# This file is part of pyAOS.

#     pyAOS is free software: you can redistribute it and/or modify
#     it under the terms of the GNU General Public License as published by
#     the Free Software Foundation, either version 3 of the License, or
#     (at your option) any later version.

#     pyAOS is distributed in the hope that it will be useful,
#     but WITHOUT ANY WARRANTY; without even the implied warranty of
#     MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
#     GNU General Public License for more details.

#     You should have received a copy of the GNU General Public License
#     along with pyAOS.  If not, see <http://www.gnu.org/licenses/>.
'''
A module containing useful functions used throughout pyAOS

:Author:
    Andrew Reeves
'''

import numpy

from scipy.interpolate import interp2d,RectBivariateSpline
#a lookup dict for interp2d order (expressed as 'kind')
INTERP_KIND = {1: 'linear', 3:'cubic', 5:'quintic'}

from . import AOFFT

#xrange now just "range" in python3. 
#Following code means fastest implementation used in 2 and 3
try:
    xrange
except NameError:
    xrange = range

def convolve(img1, img2, mode="pyfftw", fftw_FLAGS=("FFTW_MEASURE",),
                 threads=0):
    '''
    Convolves two, 2-dimensional arrays

    Uses the AOFFT library to do fast convolution of 2, 2-dimensional numpy ndarrays. The FFT mode, and some parameters can be set in the arguments.

    Parameters:
        img1 (ndarray): 1st array to be convolved
        img2 (ndarray): 2nd array to be convolved
        mode (string, optional): The fft mode used, defaults to fftw
        fftw_FLAGS (tuple, optional): flags for fftw, defaults to ("FFTW_MEASURE",)
        threads (int, optional): Number of threads used if mode is fftw
    
    Returns:
        ndarray : The convolved 2-dimensional array

    '''
    #Check arrays are same size
    if img1.shape!=img2.shape:
        raise ValueError("Arrays must have same dimensions")
    
    #Initialise FFT objects
    fFFT = AOFFT.FFT(img1.shape, axes=(0,1), mode=mode, dtype="complex64",
                direction="FORWARD", fftw_FLAGS=fftw_FLAGS, THREADS=threads) 
    iFFT = AOFFT.FFT(img1.shape, axes=(0,1), mode=mode, dtype="complex64",
                  direction="BACKWARD", fftw_FLAGS=fftw_FLAGS, THREADS=threads)      
    #backward FFT arrays
    iFFT.inputData[:] = img1
    iImg1 = iFFT().copy()  
    iFFT.inputData[:] = img2
    iImg2 = iFFT()
    
    #Do convolution
    iImg1 *= iImg2
    
    #do forward FFT
    fFFT.inputData[:] = iImg1
    return numpy.fft.fftshift(fFFT().real)
    
    
def circle(radius, size, centre_offset=(0,0)):
    """
    Create a 2-dimensional array equal to 1 in a circle and 0 outside

    Parameters:
        radius (float): The radius in pixels of the circle
        size (int): The size of the the array for the circle
        centre_offset (tuple): The coords of the centre of the circle

    Returns:
        ndarray : The circle array 
    """
    size = int(numpy.round(size))

    coords = numpy.linspace(-size/2.,size/2.,size)
    x,y = numpy.meshgrid(coords,coords)
    x-=centre_offset[0]
    y-=centre_offset[1]

    radius+=0.5

    mask = x*x + y*y <= radius*radius
    
    C = numpy.zeros((size, size))
    C[mask] = 1
    return C

def gaussian2d(size, width, amplitude=1., cent=None):
    '''
    Generates 2D gaussian distribution


    Args:
        size (tuple, float): Dimensions of Array to place gaussian
        width (tuple, float): Width of distribution. 
                                Accepts tuple for x and y values.
        amplitude (float): Amplitude of guassian distribution
        cent (tuple): Centre of distribution on grid.
    '''

    try:
        xSize = size[0]
        ySize = size[1]
    except (TypeError, IndexError):
        xSize = ySize = size

    try:
        xWidth = float(width[0])
        yWidth = float(width[1])
    except (TypeError, IndexError):
        xWidth = yWidth = float(width)

    if not cent:
        xCent = size[0]/2.
        yCent = size[1]/2.
    else:
        xCent = cent[0]
        yCent = cent[1]

    X,Y = numpy.meshgrid(range(0, xSize), range(0, ySize))
    
    image = amplitude * numpy.exp(
            -(((xCent-X)/xWidth)**2 + ((yCent-Y)/yWidth)**2)/2)

    return image



#Interpolation
###############################################

def zoom(array, newSize, order=3):
    """
    A Class to zoom 2-dimensional arrays using interpolation

    Uses the scipy `Interp2d` interpolation routine to zoom into an array. Can cope with real of complex data.

    Parameters:
        array (ndarray): 2-dimensional array to zoom
        newSize (tuple): the new size of the required array
        order (int, optional): Order of interpolation to use. default is 3

    Returns:
        ndarray : zoom array of new size.
    """
   
    if order not in INTERP_KIND:
       raise ValueError("Order can either be 1, 3, or 5 only")

    try:
        xSize = newSize[0]
        ySize = newSize[1]

    except (IndexError, TypeError):
        xSize = ySize = newSize

    coordsX = numpy.linspace(0, array.shape[0]-1, xSize)
    coordsY = numpy.linspace(0, array.shape[1]-1, ySize)

    #If array is complex must do 2 interpolations
    if array.dtype==numpy.complex64 or array.dtype==numpy.complex128:

        realInterpObj = interp2d(   numpy.arange(array.shape[0]),
                numpy.arange(array.shape[1]), array.real, copy=False, 
                kind=INTERP_KIND[order])
        imagInterpObj = interp2d(   numpy.arange(array.shape[0]),
                numpy.arange(array.shape[1]), array.imag, copy=False,
                kind=INTERP_KIND[order])                 
        return (realInterpObj(coordsY,coordsX) 
                            + 1j*imagInterpObj(coordsY,coordsX))

        

    else:

        interpObj = interp2d(   numpy.arange(array.shape[0]),
                numpy.arange(array.shape[1]), array, copy=False,
                kind=INTERP_KIND[order])

        #return numpy.flipud(numpy.rot90(interpObj(coordsY,coordsX)))
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
        xSize = newSize[0]
        ySize = newSize[1]

    except IndexError:
        xSize = ySize = newSize

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


def interp1d_numpy(array, coords):
    """
    A Numpy only inplementation array of 1d interpolation

    Parameters:
        array (ndarray): The 1d array to be interpolated
        coords (ndarray): An array of coords to return values

    Returns:
        ndarray: The interpolated array
    """ 
    intCoords = coords.astype("int")
    arrayInt = array[intCoords] 
    arrayInt1 = array[(intCoords+1).clip(0,array.shape[0]-1)]
    grad = arrayInt1 - arrayInt

    rem = coords - intCoords

    interpArray = arrayInt + grad*rem

    return interpArray


def interp2d_numpy(array, xCoords, yCoords):
    """
    A Numpy only inplementation array of 2d linear interpolation

    Parameters:
        array (ndarray): The 1d array to be interpolated
        xCoords (ndarray): An array of coords to return values
        yCoords (ndarray): An array of coords to return values

    Returns:
        ndarray: The interpolated array
    """ 

    #xCoords, yCoords = numpy.meshgrid(yCoords, xCoords)

    
    xIntCoords = xCoords.astype("int")
    yIntCoords = yCoords.astype("int")

    arrayInt = array[xIntCoords, yIntCoords]

    xGrad = array[
            (xIntCoords+1).clip(0, array.shape[0]-1), yIntCoords] - arrayInt

    yGrad = array[
            xIntCoords, (yIntCoords+1).clip(0, array.shape[1]-1)] - arrayInt

    interpArray = arrayInt + xGrad*(xCoords-xIntCoords) + yGrad*(yCoords-yIntCoords)

    return numpy.flipud(numpy.rot90(interpArray.clip(array.min(), array.max())))


#######################
#WFS Functions
######################

def findActiveSubaps(subaps, mask, threshold):
    '''
    Finds the subapertures which are "seen" be through the
    pupil function. Returns the coords of those subaps
    
    Parameters:
        subaps (int): The number of subaps in x (assumes square)
        mask (ndarray): A pupil mask, where is transparent when 1, and opaque when 0
        threshold (float): The mean value across a subap to make it "active"
        
    Returns:
        ndarray: An array of active subap coords
    '''

    subapCoords = []
    xSpacing = mask.shape[0]/float(subaps)
    ySpacing = mask.shape[1]/float(subaps)

    for x in range(subaps):
        for y in range(subaps):
            subap = mask[   
                    numpy.round(x*xSpacing): numpy.round((x+1)*xSpacing),
                    numpy.round(y*ySpacing): numpy.round((y+1)*ySpacing) 
                    ]

            if subap.mean() >= threshold:
                subapCoords.append( [x*xSpacing, y*ySpacing])

    subapCoords = numpy.array( subapCoords )
    return subapCoords

def binImgs(data, n):
    '''
    Bins one or more images down by the given factor
    bins. n must be a factor of data.shape, who knows what happens
    otherwise......
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

def simpleCentroid(img,threshold_frac=0):
    '''
    Centroids an image, or an array of images.
    Centroids over the last 2 dimensions.
    Sets all values under "threshold_frac*max_value" to zero before centroiding
    '''
    if threshold_frac!=0:
        if len(img.shape)==2:
            img = numpy.where(img>threshold_frac*img.max(), img, 0 )
        else:
            img_temp = (img.T - threshold_frac*img.max(-1).max(-1)).T
            zero_coords = numpy.where(img_temp<0)
            img[zero_coords] = 0

    if len(img.shape)==2:
        y_cent,x_cent = numpy.indices(img.shape)
        y_centroid = (y_cent*img).sum()/img.sum()
        x_centroid = (x_cent*img).sum()/img.sum()

    else:
        y_cent, x_cent = numpy.indices((img.shape[-2],img.shape[-1]))
        y_centroid = (y_cent*img).sum(-1).sum(-1)/img.sum(-1).sum(-1)
        x_centroid = (x_cent*img).sum(-1).sum(-1)/img.sum(-1).sum(-1)

    y_centroid+=0.5
    x_centroid+=0.5

    return numpy.array([y_centroid,x_centroid])

def brtPxlCentroid(img, nPxls):
    """
    Centroids using brightest Pixel Algorithm
    (A. G. Basden et al,  MNRAS, 2011)

     Finds the nPxlsth brightest pixel, subtracts that value from frame, 
    sets anything below 0 to 0, and finally takes centroid
    """

    if len(img.shape)==2:
        pxlValue = numpy.sort(img.flatten())[-nPxls]
        img-=pxlValue
        img.clip(0, img.max())

    elif len(img.shape)==3:
        pxlValues = numpy.sort(
                        img.reshape(img.shape[0], img.shape[-1]*img.shape[-2])
                        )[:,-nPxls]
        img[:]  = (img.T - pxlValues).T
        img.clip(0, img.max(), out=img)

    return simpleCentroid(img)

def quadCell(img):
    
    xSum = img.sum(-2)
    ySum = img.sum(-1)
    
    xCent = xSum[...,1] - xSum[...,0]
    yCent = ySum[...,1] - ySum[...,0]
    
    return numpy.array([xCent, yCent])

def zernike(j, N):
    """
     Creates the Zernike polynomial with mode index j, 
     where j = 1 corresponds to piston.

     Args: 
        j (int): The noll j number of the zernike mode
        N (int): The diameter of the zernike more in pixels
     Returns:
        ndarray: The Zernike mode
     """
    n,m = zernIndex(j);

    coords = numpy.linspace(-1,1,N)
    X,Y = numpy.meshgrid(coords,coords)
    R = numpy.sqrt(X**2 + Y**2)
    theta = numpy.arctan2(Y,X)

    if m==0:
        Z = numpy.sqrt(n+1)*zernikeRadialFunc(n,0,R);
    else:
        if j%2==0: # j is even
                Z = numpy.sqrt(2*(n+1))*zernikeRadialFunc(n,m,R) * numpy.cos(m*theta)
        else:   #i is odd
                Z = numpy.sqrt(2*(n+1))*zernikeRadialFunc(n,m,R) * numpy.sin(m*theta)


    return Z*circle(N/2., N)



def zernikeRadialFunc(n, m, r):
    """
    Fucntion to calculate the Zernike radial function
    """

    R = numpy.zeros(r.shape)
    for i in xrange(0,int((n-m)/2)+1):

        R += r**(n-2*i) * (((-1)**(i))*numpy.math.factorial(n-i)) / ( numpy.math.factorial(i) * numpy.math.factorial(0.5*(n+m)-i) * numpy.math.factorial(0.5*(n-m)-i) )

    return R



def zernIndex(j,sign=0):
    """
    returns the [n,m] list giving the radial order n and azimutal order
    of the zernike polynomial of index j
    if sign is set, will also return a 1 for cos, -1 for sine or 0 when m==0.
    """
    n = int((-1.+numpy.sqrt(8*(j-1)+1))/2.)
    p = (j-(n*(n+1))/2.)
    k = n%2
    m = int((p+k)/2.)*2 - k
    if sign==0:
        return [n,m]
    else:#determine whether is sine or cos term.
        if m!=0:
            if j%2==0:
                s=1
            else:
                s=-1

        else:
            s=0
        return [n,m,s]



def zernikeArray(J, N):
    """
    Creates an array of Zernike Polynomials
    
    Parameters:
        maxJ (int or list): Max Zernike polynomial to create, or list of zernikes J indices to create
        N (int): size of created arrays

    Returns:
        ndarray: array of Zerkike Polynomials
    """
    #If list, make those Zernikes
    try:
        nJ = len(J)
        Zs = numpy.empty((nJ, N, N))
        for i in xrange(nJ):
            Zs[i] = zernike(J[i], N)

    #Else, cast to int and create up to that number
    except TypeError:

        maxJ = int(numpy.round(J))
        N = int(numpy.round(N))

        Zs = numpy.empty((maxJ, N, N))

        for j in xrange(1,maxJ+1):
            Zs[j-1] = zernike(j, N)

    return Zs

