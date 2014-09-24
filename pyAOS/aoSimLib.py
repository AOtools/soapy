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
A module containing functions used in AoSim
'''

import numpy
from scipy.interpolate import interp2d,RectBivariateSpline
from . import AOFFT


def convolve(img1, img2, mode="pyfftw", fftw_FLAGS=("FFTW_MEASURE",),
                 threads=0):
    '''
    Convolves 2, 2 dimensional arrays
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
    
    
def circle(radius, diameter, centre_offset=(0,0)):
    
    diameter = int(numpy.round(diameter))

    coords = numpy.linspace(-diameter/2.,diameter/2.,diameter)
    x,y = numpy.meshgrid(coords,coords)
    x-=centre_offset[0]
    y-=centre_offset[1]

    mask = x*x + y*y <= radius*radius+0.5
    
    C = numpy.zeros((diameter, diameter))
    C[mask] = 1
    return C

def circleOld(radius, diameter, centre_offset=(0,0)):
    '''
    creates an array of size (diameter,diameter) of zeros
    with a circle shape of ones with radius <radius>
    Uses a fast algorithm which splits circle into quads
    '''
    #make sure parameters are int
    diameter = int(numpy.round(diameter))

    #Create mesh of radius quadrant
    quad_coords = numpy.linspace(0,radius,radius)
    X,Y = numpy.meshgrid(quad_coords,quad_coords)
    quad_r2_mesh = X**2 + Y**2

    #Find the points inside this quad less than(or equal to) the radius
    r2 = radius**2
    r_coords = numpy.where( quad_r2_mesh <= r2)

    #Set these points to 1
    circleQuad = numpy.zeros( (radius, radius) )
    circleQuad[r_coords] = 1.

    #Create an array of zeros,
    #then fits flipped quads into it in the right place
    circle = numpy.zeros((diameter, diameter))
    circle[diameter/2.+centre_offset[0]: diameter/2.+centre_offset[0]+radius,
           diameter/2.+centre_offset[1]: diameter/2.+centre_offset[1]+radius] \
                   = circleQuad
    circle[diameter/2.+centre_offset[0]: diameter/2.+centre_offset[0]+radius,
           diameter/2.+centre_offset[1]-radius: diameter/2.+centre_offset[1]] \
                   = numpy.fliplr(circleQuad)
    circle[diameter/2.+centre_offset[0]-radius: diameter/2.+centre_offset[0],
           diameter/2.+centre_offset[1]: diameter/2.+centre_offset[1]+radius] \
                   = numpy.flipud(circleQuad)
    circle[diameter/2.+centre_offset[0]-radius: diameter/2.+centre_offset[0],
           diameter/2.+centre_offset[1]-radius: diameter/2.+centre_offset[1]] \
                   = numpy.flipud(numpy.fliplr(circleQuad))

    return circle

#Interpolation
###############################################

def zoom(array, newSize, order=3):

    try:
        xSize = newSize[0]
        ySize = newSize[1]

    except IndexError:
        xSize = ySize = newSize

    coordsX = numpy.linspace(0, array.shape[0]-1, xSize)
    coordsY = numpy.linspace(0, array.shape[1]-1, ySize)

    #If array is complex must do 2 interpolations
    if array.dtype==numpy.complex64 or array.dtype==numpy.complex128:
        realInterpObj = RectBivariateSpline(   numpy.arange(array.shape[0]),
                numpy.arange(array.shape[1]), array.real, kx=order, ky=order)
        imagInterpObj = RectBivariateSpline(   numpy.arange(array.shape[0]),
                numpy.arange(array.shape[1]), array.imag, kx=order, ky=order)
                            
        return realInterpObj(coordsX,coordsY) \
                            + 1j*imagInterpObj(coordsX,coordsY)
            
    else:

        interpObj = RectBivariateSpline(   numpy.arange(array.shape[0]),
                numpy.arange(array.shape[1]), array, kx=order, ky=order)

        return interpObj(coordsX,coordsY)




class InterpZoom(object):

    def __init__(self, size, newSize):

        self.coordsX = numpy.arange(size[0])
        self.coordsY = numpy.arange(size[1])

        self.newCoordsX = numpy.linspace(0,size[0],newSize[0])
        self.newCoordsY = numpy.linspace(0,size[1],newSize[1])

    def __call__(self, array):

        interpObj = interp2d(self.coordsY,self.coordsX, array, copy=False)

        return interpObj(self.newCoordsX, self.newCoordsY)


#######################
#WFS Functions
######################

def findActiveSubaps(subaps,mask,threshold):
    '''
    Finds the subapertures which are "seen" be through the
    pupil function. Returns the coords of those subaps
    '''

    subapCoords = []
    xSpacing = mask.shape[0]/float(subaps)
    ySpacing = mask.shape[1]/float(subaps)

    for x in range(subaps):
        for y in range(subaps):
            subap = mask[   x*xSpacing:(x+1)*xSpacing,
                            y*ySpacing:(y+1)*ySpacing ]

            if subap.mean() >= threshold:
                subapCoords.append( [x*xSpacing, y*ySpacing])

    subapCoords = numpy.array( subapCoords )
    return subapCoords

def binImgs(data,n):
    '''
    Bins one or more images down by the given factor
    bins. n must be a factor of data.shape, who knows what happens
    otherwise......
    '''
    shape = numpy.array( data.shape )
    
    n = int(numpy.round(n))

    if len(data.shape)==2:
        shape[-1]/=n
        binnedImgTmp = numpy.zeros( shape )
        for i in range(n):
            binnedImgTmp += data[:,i::n]
        shape[-2]/=n
        binnedImg = numpy.zeros( shape )
        for i in range(n):
            binnedImg += binnedImgTmp[i::n,:]

        return binnedImg
    else:
        shape[-1]/=n
        binnedImgTmp = numpy.zeros ( shape )
        for i in range(n):
            binnedImgTmp += data[...,i::n]

        shape[-2] /= n
        binnedImg = numpy.zeros( shape )
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




def zernike(j, N):
    """
     Creates the Zernike polynomial with mode index j, where i = 1 corresponds to piston  
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
    for i in xrange(0,((n-m)/2)+1):

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
    
    Args:
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

