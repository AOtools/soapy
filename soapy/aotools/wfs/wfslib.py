"""
A library of functions which may be of use to analyse WFS data
"""

import numpy

def r0fromSlopes(slopes, wavelength, subapDiam):
    """
    Measures the value of R0 from a set of WFS slopes.
    
    Uses the equation in Saint Jaques, 1998, PhD Thesis, Appendix A to calculate the value of atmospheric seeing parameter, r0, that would result in the variance of the given slopes.

    Parameters:
        slopes (ndarray): A 3-d set of slopes in radians, of shape (dimension, nSubaps, nFrames)
        wavelength (float): The wavelegnth of the light observed
        subapDiam (float) The diameter of each sub-aperture

    Returns:
        float: An estimate of r0 for that dataset.

    """
    slopeVar = slopes.var(axis=(-1))

    r0 = ((0.162*(wavelength**2) * subapDiam**(-1./3)) / slopeVar)**(3./5)
    
    r0 = r0.mean()

    return r0


def slopeVarfromR0(r0, wavelength, subapDiam):
    """Returns the expected slope variance for a given r0 ValueError

    Uses the equation in Saint Jaques, 1998, PhD Thesis, Appendix A to calculate the slope variance resulting from a value of r0.    

    """

    slope_var = 0.162 * (wavelength**2) * r0**(-5./3) * subapDiam**(-1./3)

    return slope_var


def findActiveSubaps(subaps, mask, threshold, returnFill=False):
    '''
    Finds the subapertures which are "seen" be through the
    pupil function. Returns the coords of those subaps

    Parameters:
        subaps (int): The number of subaps in x (assumes square)
        mask (ndarray): A pupil mask, where is transparent when 1, and opaque when 0
        threshold (float): The mean value across a subap to make it "active"
        returnFill (optional, bool): Return an array of fill-factors

    Returns:
        ndarray: An array of active subap coords
    '''

    subapCoords = []
    xSpacing = mask.shape[0]/float(subaps)
    ySpacing = mask.shape[1]/float(subaps)

    if returnFill:
        fills = []

    for x in range(subaps):
        for y in range(subaps):
            subap = mask[
                    int(numpy.round(x*xSpacing)): int(numpy.round((x+1)*xSpacing)),
                    int(numpy.round(y*ySpacing)): int(numpy.round((y+1)*ySpacing))
                    ]

            if subap.mean() >= threshold:
                subapCoords.append( [x*xSpacing, y*ySpacing])
                if returnFill:
                    fills.append(subap.mean())

    subapCoords = numpy.array( subapCoords )

    if returnFill:
        return subapCoords, numpy.array(fills)
    else:
        return subapCoords


def computeFillFactor(mask, subapPos, subapSpacing):

    fills = numpy.zeros(len(subapPos))
    for i, (x, y) in enumerate(subapPos):
        x1 = int(round(x))
        x2 = int(round(x + subapSpacing))
        y1 = int(round(y))
        y2 = int(round(y + subapSpacing))
        fills[i] = mask[x1:x2, y1:y2].mean()

    return fills