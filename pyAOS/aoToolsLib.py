import numpy
import circle
import zerns
import scipy.optimize

def temporalPowerSpectrum(slopes):
    '''
    Calculates the temoral Power Spectrum for a set of WFS slopes

    INPUT:
        slopes: numpy array - shape (frames, slopes)
    OUTPUT:
        power Spectrum:numpy array - shape (frames/2)
    '''
    print slopes.shape
    frames,subaps = slopes.shape
    print slopes.shape
    fft = numpy.fft.fft( slopes,axis=0)

    PS1 = abs(fft)**2
    PS1 = PS1[:frames/2]

    PSmean = PS1.mean(1)
    PSerror = PS1.var(1)/(frames/2)

    return PSmean, PSerror

def fitTemporalPowerSpectrum(PS,dataRange=None):#,errors=None):
    '''
    Fits a temporal power spectrum, returning the gradient and intercept
    of a straight line fit on a logarithmic plot.

    INPUT:
        temporal Power Spectrum - numpy array of shape ( frames/2 )
    OUTPUT:
        Gradient - float
        Intercept - float
        ChiSquaredMin - float
    '''
    if range!=None:
        XLower = dataRange[0]
        XUpper = dataRange[1]
    else:
        XLower = 1
        XUpper = PS.shape[0]+1

    X = numpy.arange(XLower,XUpper,dtype="float")
#    if errors != None:
#        A0 = 11/3.
#        B0 = PS[0]
#        result = scipy.optimize.minimize(logModelData, (A0,B0),
#                                         args=(X,PS,errors), tol=0.001)
#        A,B = result["x"]
#        chi2 = result["fun"]

   # if errors ==None:

    A = numpy.vstack([numpy.log10(X), numpy.ones(len(X))]).T
    PS = PS[XLower:XUpper]
    res = numpy.linalg.lstsq(A, numpy.log10(PS) )
    A = res[0][0];B = res[0][1]
    print res

    return A,B#,chi2


def logModelData(params,X,data,errors):
    '''
    generates model data for a "log" plot.
    INPUT:
        A: parameter A - float -  the power of the variable X
        B: parameter B - float - the Amplitude of the function
        X: X values - numpy array of size (Values)
    '''
    A,B = params
    print params
    Y = B * 10**(A)

    residuals = (Y-data)

    ChiComponents = (residuals/errors)**2.

    Chi2 = ChiComponents.sum()
    print(Chi2)
    return Chi2

def r0(slopes, theta, wvl, d):
    '''
    Calculates r0 from the centroid variance of all subaps.
    INPUT:
        slopes -- numpy array shape (frames,subaps*2)
        theta -- field of view of each pxl in radians - float
        wvl -- wavelength - float
        d -- subap diameter

    OUTPUT:
        r0, r0error - floats
    '''

    slopeAngles = slopes*theta
    slopeVariance = slopeAngles.var(0)

    r0 = ((0.162 * wvl**2 * d**(-1./3)) / (slopeVariance)) **(3./5.)

    r0mean = r0.mean()
    r0error = r0.var(0)/r0.shape[0]

    return r0mean, r0error

def remapSubaps(slopes, subaps, subapPositions):
    '''
    Remaps the slopes back onto positions on the WFS.
    INPUT:
        slopes -- numpy array shape (frames,subaps*2)
        subaps -- number of subaps per WFS
        subapPositions -- a numpy array of sub-aperture positions,
                            shape (slopes,2)
    OUTPUT:
        slopes -- numpy array shape (frames, 2, subaps**2)
    '''
    frames, nSlopes = slopes.shape

    remapedSlopes = numpy.zeros( (frames,2,subaps,subaps) )
    slopes = slopes.copy().reshape(frames,2 ,nSlopes/2.)

    for subap in xrange(len(subapPositions)):
        x = subapPositions[subap][0]
        y = subapPositions[subap][1]
        remapedSlopes[:,:,x,y] = slopes[:, :, subap]

    return remapedSlopes

def spatialPowerSpectrum(slopes2d):
    '''
    Calculated a spatial Power Spectrum of an array of slopes
    slopes must be in 2d format before they are used here.
    '''
    fft = numpy.fft.fftshift( numpy.fft.fft2( slopes2d, axes=(2,3)), axes=(2,3))

    PS2d = numpy.abs(fft)**2

    PS2dmean = PS2d.mean(0).mean(0)
    radAvg = radialAvg(PS2dmean)
    return radAvg


def radialAvg(array):
    '''
    Returns a radial Average of a given Array
    INPUT:
        2d square numpy Array
    OUPUT:
        numpy array of shape (xSize/2)

    '''
    xShape,yShape=array.shape
    avg = numpy.empty(xShape/2)

    for i in range(xShape/2):
        ring = circle.circleGen(i+1,xShape)
        avg[i] = (ring*array).mean()/ring.sum()
    return avg

def getActiveSubaps(mask, subaps, threshold):
    '''
    Returns a list of subap coordinates for active subapertured,
    given a pupil mask and "fill threshold"
    INPUT:
        mask - 2d numpy of arbitrary size (Should be divisible by "subaps"
        subaps - number of required subaps in 1 dimension (int)
        threshold - if mean of subap in "mask" is larger than this,
                    append to subap List (float)
    OUTPUT:
        2d numpy array of slopes coords, shape (subaps,coords)

    '''
    activeSubaps = []
    pxlsPerSubap = mask.shape/subaps
    for x in xrange(subaps):
        for y in xrange(subaps):
            subap = mask[x+pxlsPerSubap:(x+1)+pxlsPerSubap,
                         y+pxlsPerSubap:(y+1)+pxlsPerSubap]
            if subap.mean() > threshold:
                activeSubaps.append([x,y])
    return getActiveSubaps

def calcCn2(r0, wvl):
    '''
    Calculated the Cn2 value for a given value of r0 and wavelength
    INPUT:
        r0 - float
        wvl - float
    Return:
        float
    '''

    A = r0**(-5./3)
    B = (wvl**2)/(4*(numpy.pi**2) * 0.423)

    return A*B


def totalR0(r0Array):
    '''
    returns the combined value of r0 for a list
    turbulence layer r0s
    INPUT:
        r0Array - list or numpy Array, shape (layers)
    OUTPUT:
        r0 - float
    '''
    r0Array = numpy.array(r0Array)
    return ((r0Array**(-3./5)).sum())**(-5./3)

