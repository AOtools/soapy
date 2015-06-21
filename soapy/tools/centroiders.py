import numpy

def centreOfGravity(img, threshold=0, **kwargs):
    '''
    Centroids an image, or an array of images.
    Centroids over the last 2 dimensions.
    Sets all values under "threshold*max_value" to zero before centroiding

    Parameters:
        img (ndarray): 2d or greater rank array of imgs to centroid
        threshold (float): Percentage of max value under which pixels set to 0

    Returns:
        ndarray: Array of centroid values

    '''
    if threshold!=0:
        if len(img.shape)==2:
            img = numpy.where(img>threshold*img.max(), img, 0 )
        else:
            img_temp = (img.T - threshold*img.max(-1).max(-1)).T
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


def brightestPxl(img, threshold, **kwargs):
    """
    Centroids using brightest Pixel Algorithm
    (A. G. Basden et al,  MNRAS, 2011)

    Finds the nPxlsth brightest pixel, subtracts that value from frame, 
    sets anything below 0 to 0, and finally takes centroid.

    Parameters:
        img (ndarray): 2d or greater rank array of imgs to centroid
        threshold (float): Percentage of pixels to use for centroid

    Returns:
        ndarray: Array of centroid values
    """
    
    nPxls = threshold*img.shape[-1]*img.shape[-2]

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

    return centreOfGravity(img)


def corrConvolve(x, y):
    '''
    2D convolution using FFT, use to generate cross-correlations.

    Args:
        x (array): subap image
        y (array): reference image
    Returns:
        ndarray: cross-correlation of x and y
    '''

    fr = numpy.fft.fft2(x)
    fr2 = numpy.fft.fft2(y[::-1, ::-1])
    m, n = fr.shape

    cc = (numpy.fft.ifft2(fr*fr2)).real
    cc = numpy.roll(cc, -m/2+1, axis=0)
    cc = numpy.roll(cc, -n/2+1, axis=1)

    return cc


def correlation(im, threshold, ref):
    '''
    Correlation Centroider, currently only works for 3d im shape.
    Performs a simple thresholded COM on the correlation.

    Args:
        im: subap images (t, y, x)
        threshold_fac: threshold for COM (0=all pixels, 1=brightest pixel)
        ref: reference image (t, y, x)
    Returns:
        ndarray: centroids of im, given as x, y
    '''

    nt, ny, nx = im.shape

    cents = numpy.zeros((2, nt))

    for frame in range(nt):
        # Correlate frame with reference image
        corr = corrConvolve(im[frame], ref[frame])
        # Find brightest pixel.
        index_y, index_x = numpy.unravel_index(corr.argmax(),
                                               corr.shape)

        # Apply threshold
        corr -= corr.min()
        mx = corr.max()
        bg = threshold*mx
        corr = numpy.clip(corr, bg, mx) - bg

        # Centroid
        s_y, s_x = corr.shape
        XRAMP = numpy.arange(s_x)
        YRAMP = numpy.arange(s_y)
        XRAMP.shape = (1, s_x)
        YRAMP.shape = (s_y,  1)

        si = corr.sum()
        if si == 0:
            si = 1
        cx = (corr * XRAMP).sum() / si
        cy = (corr * YRAMP).sum() / si

        cents[:, frame] = cx, cy

    return cents


def quadCell(img, **kwargs):
    """
    Centroider to be used for 2x2 images. 

    Parameters:
        img: 2d or greater rank array, where centroiding performed over last 2 dimensions

    Returns:
        ndarray: Array of centroid values
    """

    xSum = img.sum(-2)
    ySum = img.sum(-1)
    
    xCent = xSum[...,1] - xSum[...,0]
    yCent = ySum[...,1] - ySum[...,0]
    
    return numpy.array([xCent, yCent])

