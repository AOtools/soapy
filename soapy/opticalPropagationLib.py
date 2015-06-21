'''
A library of optical propagation methods.

Many extracted from the book by Schmidt, 2010: Numerical Methods
of optical proagation
'''

import numpy
from . import aoSimLib

def ft2(g, delta, padFactor=1):
    padFactor = int(padFactor)
    G = numpy.fft.fftshift(
            numpy.fft.fft2(
                    numpy.fft.fftshift(g),
                    s=(padFactor*g.shape[0],padFactor*g.shape[1])
                            )
                             ) * delta**2
    return G

def ift2(G, delta_f, padFactor=1):
    N = G.shape[0]
    g = numpy.fft.ifftshift(
            numpy.fft.ifft2(
                    numpy.fft.ifftshift(G),
                    s=( padFactor*G.shape[0], padFactor*G.shape[1])
                            )
                            ) * (N * delta_f)*2

    return g


def angularSpectrum(Uin,wvl,d1,d2,z,):
    N = Uin.shape[0] #Assumes Uin is square.
    k = 2*numpy.pi/wvl     #optical wavevector

    (x1,y1) = numpy.meshgrid(d1*numpy.arange(-N/2,N/2),
                             d1*numpy.arange(-N/2,N/2))
    r1sq = (x1**2 + y1**2) + 1e-10

    #Spatial Frequencies (of source plane)
    df1 = 1. / (N*d1)
    fX,fY = numpy.meshgrid(df1*numpy.arange(-N/2,N/2),
                           df1*numpy.arange(-N/2,N/2))
    fsq = fX**2 + fY**2

    #Scaling Param
    m = float(d2)/d1

    #Observation Plane Co-ords
    x2,y2 = numpy.meshgrid( d2*numpy.arange(-N/2,N/2),
                            d2*numpy.arange(-N/2,N/2) )
    r2sq = x2**2 + y2**2

    #Quadratic phase factors
    Q1 = numpy.exp( 1j * k/2. * (1-m)/z * r1sq)

    Q2 = numpy.exp(-1j * numpy.pi**2 * 2 * z/m/k*fsq)

    Q3 = numpy.exp(1j * k/2. * (m-1)/(m*z) * r2sq)

    #Compute propagated field
    Uout = Q3 * ift2( Q2 * ft2(Q1 * Uin/m,d1), df1)
    return Uout


def oneStepFresnel(Uin,wvl,d1,z):

    N = Uin.shape[0]    #Assume square grid
    k = 2*numpy.pi/wvl  #optical wavevector

    #Source plane coordinates
    x1,y1 = numpy.meshgrid( numpy.arange(-N/2.,N/2.) * d1,
                            numpy.arange(-N/2.,N/2.) * d1)
    #observation plane coordinates
    d2 = wvl*z/(N*d1)
    x2,y2 = numpy.meshgrid( numpy.arange(-N/2.,N/2.) * d2,
                            numpy.arange(-N/2.,N/2.) * d2 )

    #evaluate Fresnel-Kirchoff integral
    A = 1/(1j*wvl*z)
    B = numpy.exp( 1j * k/(2*z) * (x2**2 + y2**2))
    C = ft2(Uin *numpy.exp(1j * k/(2*z) * (x1**2+y1**2)), d1)

    Uout = A*B*C

    return Uout

def twoStepFresnel(Uin,wvl,d1,d2,Dz):
    N = Uin.shape[0] #Number of grid points
    k = 2*numpy.pi/wvl #optical wavevector

    #source plane coordinates
    x1,y1 = numpy.meshgrid( numpy.arange(-N/2,N/2) * d1,
                            numpy.arange(-N/2.,N/2.) * d1 )

    #magnification
    m = float(d2)/d1

    #intermediate plane
    Dz1  = Dz / (1-m) #propagation distance
    d1a = wvl * abs(Dz1) / (N*d1) #coordinates
    x1a,y1a = numpy.meshgrid( numpy.arange( -N/2.,N/2.) * d1a,
                              numpy.arange( -N/2.,N/2.) * d1a )

    #Evaluate Fresnel-Kirchhoff integral
    A = 1./(1j * wvl * Dz1)
    B = numpy.exp(1j * k/(2*Dz1) * (x1a**2 + y1a**2) )
    C = ft2(Uin * numpy.exp(1j * k/(2*Dz1) * (x1**2 + y1**2)), d1)
    Uitm = A*B*C
    #Observation plane
    Dz2 = Dz - Dz1

    #coordinates
    x2,y2 = numpy.meshgrid( numpy.arange(-N/2., N/2.) * d2,
                            numpy.arange(-N/2., N/2.) * d2 )

    #Evaluate the Fresnel diffraction integral
    A = 1. / (1j * wvl * Dz2)
    B = numpy.exp( 1j * k/(2 * Dz2) * (x2**2 + y2**2) )
    C = ft2(Uitm * numpy.exp( 1j * k/(2*Dz2) * (x1a**2 + y1a**2)), d1a)
    Uout = A*B*C

    return Uout

def lensAgainst( Uin, wvl, d1, f):
    '''
    Propagates from the pupil plane to the focal
    plane for an object placed against (and just before)
    a lens
    '''

    N = Uin.shape[0] #Assume square grid
    k = 2*numpy.pi/wvl  #Optical Wavevector

    #Observation plane coordinates
    fX = numpy.arange( -N/2.,N/2.)/(N*d1)

    #Observation plane coordinates
    x2,y2 = numpy.meshgrid(wvl * f * fX, wvl * f * fX)
    del(fX)

    #Evaluate the Fresnel-Kirchoff integral but with the quadratic
    #phase factor inside cancelled by the phase of the lens
    Uout = numpy.exp( 1j*k/(2*f) * (x2**2 + y2**2) )/ (1j*wvl*f) * ft2( Uin, d1)

    return Uout

