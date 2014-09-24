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

#import FITS
import pyfits
import numpy
import time
import random
import scipy.fftpack as fft
from . import AOFFT, logger
import scipy.interpolate

global log


try:
    xrange
except NameError:
    xrange = range

class atmos:
    def __init__(self, simConfig, atmosConfig):
        '''
        New class to replace old rubbish phase screen module.
        Can Generate new screens or load some from file.
        allows different r0 to be set for each screen.

        Returns the first sim scrns sized phase screen
        '''
        self.scrnSize = simConfig.scrnSize
        self.windDirs = atmosConfig.windDirs
        self.windSpeeds = atmosConfig.windSpeeds
        self.pxlScale = simConfig.pxlScale
        self.wholeScrnSize = atmosConfig.wholeScrnSize
        self.scrnNo = atmosConfig.scrnNo
        self.r0 = atmosConfig.r0
        self.looptime = simConfig.loopTime

        self.log = logger.Logger()

        atmosConfig.scrnStrengths /= (
                            atmosConfig.scrnStrengths[:self.scrnNo].sum())

        self.scrnStrengths = ( ((self.r0**(-5./3.))
                                *atmosConfig.scrnStrengths)**(-3./5.) )

        if not atmosConfig.scrnNames:
            new = True
        else:
            new = False

        # #Assume r0 calculated for 550nm.
        # self.wvl = 550e-9

        self.scrnPos = {}
        self.wholeScrns = {}

        scrns={}

        scrnSize = int(round(self.scrnSize))

        #If required, generate some new Kolmogorov phase screens
        if new==True:
            self.log.info("Generating Phase Screens")
            for i in xrange(self.scrnNo):
                #self.wholeScrns[i]=phscrn(self.wholeScrnSize,
                #                          self.pxlScale*self.scrnStrengths[i]
                #                                             ).astype("float32")
                self.log.info("Generate Phase Screen {0}  with r0: {1}, size: {2}, delta: {3}".format(i,self.scrnStrengths[i], self.wholeScrnSize,1./self.pxlScale))
                self.wholeScrns[i] = ft_sh_phase_screen(
                            self.scrnStrengths[i], 
                            self.wholeScrnSize, 1./self.pxlScale, 30., 0.01)

                scrns[i] = self.wholeScrns[i][:scrnSize,:scrnSize]
        #Otherwise, load some others from FITS file
        else:
            self.log.info("Loading Phase Screens")

            for i in xrange(self.scrnNo):
                fitsHDU = pyfits.open(self.screenNames[i])[0]
                self.wholeScrns[i] = fitsHDU.data.astype("float32")
                # self.wholeScrns[i] = (FITS.Read(screenNames[i])[1]
#                                        ).astype("float32")
                scrns[i] = self.wholeScrns[i][:scrnSize,:scrnSize]

                #Do the loaded scrns tell us how strong they are?
                #Theyre r0 must be in pixels! label: "R0"
                #If so, we can scale them to the desired r0
                try:

                    r0 = float(fitsHDU.header["R0"])
                    self.wholeScrns[i] *=( (self.scrnStrengths[i]/
                                            (r0/self.pxlScale))**(-5./6.))

                except KeyError:
                    self.log.info("no r0 info found in screen header - will assume its ok as it is")



            if self.wholeScrnSize!=self.wholeScrns[i].shape[0]:
                self.log.info("Requested phase screen has different size to that input in config file.")

            self.wholeScrnSize = self.wholeScrns[i].shape[0]
            if self.wholeScrnSize < self.scrnSize:
                raise Exception("required scrn size larger than phase screen")



        #Set the initial starting point of the screen,
        #If windspeed is negative, starts from the
        #far-end of the screen to avoid rolling straight away
        windDirs=numpy.array(self.windDirs,dtype="float32") * numpy.pi/180.0
        windV=(self.windSpeeds * numpy.array([numpy.cos(windDirs),
                                              numpy.sin(windDirs)])).T #This is velocity in metres per second
        windV *= self.looptime   #Now metres per looptime
        windV *= self.pxlScale   #Now pxls per looptime.....ideal!
        self.windV = windV

        #Sets initial phase screen pos
        #If velocity is negative in either direction, set the starting point
        #to the end of the screen to avoid rolling too early.
        for i in xrange(self.scrnNo):
            self.scrnPos[i] = numpy.array([0,0])

            if windV[i,0] < 0:
                self.scrnPos[i][0] = self.wholeScrns[i].shape[0] - self.scrnSize

            if windV[i,1] < 0:
                self.scrnPos[i][1] = self.wholeScrns[i].shape[1] - self.scrnSize

        self.windV = windV

        #Init scipy interpolation objects which hold the phase screen data
        self.interpScrns = {}
        self.xCoords = {}
        self.yCoords = {}
        for i in xrange(self.scrnNo):
            self.interpScrns[i] = scipy.interpolate.RectBivariateSpline(
                                        numpy.arange(self.wholeScrnSize),
                                        numpy.arange(self.wholeScrnSize),
                                        self.wholeScrns[i])#, copy=True)
            self.xCoords[i] = numpy.arange(self.scrnSize) + self.scrnPos[i][0]
            self.yCoords[i] = numpy.arange(self.scrnSize) + self.scrnPos[i][1]


    def moveScrns(self):

        scrns={}
        for i in self.wholeScrns:

            #Deals with what happens when the window on the screen
            #reaches the edge - rolls it round and starts again.
            #X direction
            if (self.scrnPos[i][0] + self.scrnSize) >= self.wholeScrnSize:
                self.log.debug("pos > scrnSize: rolling phase screen X")
                self.wholeScrns[i] = numpy.roll(self.wholeScrns[i],
                                                int(-self.scrnPos[i][0]),axis=0)
                self.scrnPos[i][0] = 0
                #and update the coords...
                self.xCoords[i] = numpy.arange(self.scrnSize)
                self.interpScrns[i] = scipy.interpolate.RectBivariateSpline(
                                            numpy.arange(self.wholeScrnSize),
                                            numpy.arange(self.wholeScrnSize),
                                            self.wholeScrns[i])

            if self.scrnPos[i][0] < 0:
                self.log.debug("pos < 0: rolling phase screen X")

                self.wholeScrns[i] = numpy.roll(self.wholeScrns[i],
                                            int(self.wholeScrnSize-self.scrnPos[i][0]-self.scrnSize),axis=0)
                self.scrnPos[i][0] = self.wholeScrnSize-self.scrnSize
                self.xCoords[i] = numpy.arange(self.scrnSize)+self.scrnPos[i][0]
                self.interpScrns[i] = scipy.interpolate.RectBivariateSpline(
                                            numpy.arange(self.wholeScrnSize),
                                            numpy.arange(self.wholeScrnSize),
                                            self.wholeScrns[i])
            #Y direction
            if (self.scrnPos[i][1] + self.scrnSize) >= self.wholeScrnSize:
                self.log.debug("pos > scrnSize: rolling Phase Screen Y")
                self.wholeScrns[i] = numpy.roll(self.wholeScrns[i],
                                                int(-self.scrnPos[i][1]),axis=1)
                self.scrnPos[i][1] = 0
                self.yCoords[i] = numpy.arange(self.scrnSize)
                self.interpScrns[i] = scipy.interpolate.RectBivariateSpline(
                                            numpy.arange(self.wholeScrnSize),
                                            numpy.arange(self.wholeScrnSize),
                                            self.wholeScrns[i])
            if self.scrnPos[i][1] < 0:
                log.debug("pos < 0: rolling Phase Screen Y")

                self.wholeScrns[i] = numpy.roll(self.wholeScrns[i],
                    int(self.wholeScrnSize-self.scrnPos[i][1]-self.scrnSize),
                                                                axis=1)
                self.scrnPos[i][1] = self.wholeScrnSize-self.scrnSize
                self.yCoords[i] = numpy.arange(self.scrnSize)+self.scrnPos[i][1]
                self.interpScrns[i] = scipy.interpolate.RectBivariateSpline(
                                            numpy.arange(self.wholeScrnSize),
                                            numpy.arange(self.wholeScrnSize),
                                            self.wholeScrns[i])


            scrns[i] = self.interpScrns[i](self.xCoords[i],self.yCoords[i])

            #Move window coordinates.
            self.scrnPos[i] = self.scrnPos[i]+self.windV[i]
            self.xCoords[i] += self.windV[i][0]
            self.yCoords[i] += self.windV[i][1]

        return scrns

    def randomScrns(self, subharmonics=True, L0=30., l0=0.01):

        scrns = {}
        for i in xrange(self.scrnNo):
            if subharmonics:
                scrns[i] = ft_sh_phase_screen(self.scrnStrengths[i],
                         self.scrnSize, (self.pxlScale**(-1.)), L0, l0)
        return scrns

#Kolmogorov Phase Screen generation code, shamlessly copied from Dr. Tim Butterly,
#CfAI, Durham University.

def powspec(x,y):                                       # 2D Kolmogorov PSD function
    r=numpy.sqrt( (x+.000001)**2 + ((y)+.000001)**2 )
    amp=r**(-11./3.)
    return amp


def phscrn(n,r0):

    mx0=numpy.fromfunction(powspec,(n/2+1,n/2+1))    # Create PSD array
    mx0=numpy.sqrt(mx0)
    mx0[0,0]=0.
    mx0=mx0.astype(numpy.float32)

    mx1=numpy.zeros((n,n),numpy.float32)
    mx1[:n/2+1,:n/2+1] = mx0[:n/2+1,:n/2+1]
    mx1[:n/2,n/2+1:n] = mx0[:n/2,-2:-n/2-1:-1]
    mx1[n/2+1:n,:n/2] = mx0[-2:-n/2-1:-1,:n/2]
    mx1[n/2:n,n/2:n] = mx0[-1:-n/2-1:-1,-1:-n/2-1:-1]
    del mx0

    R = random.SystemRandom(time.time())
    seed = int(R.random()*10000)
    numpy.random.seed(seed)

    mx2=numpy.zeros((n,n),numpy.complex64) # Random complex variable, normal distbn
    mx2.real=numpy.random.normal(0.,1.,(n,n))

    mx2.imag=numpy.random.normal(0.,1.,(n,n))
    mx2=mx2.astype(numpy.complex64)

    mx3=numpy.zeros((n,n),numpy.complex64)        # Convert to Hermitian array
    mx3[0:n/2+1,:] = mx2[0:n/2+1,:]
    mx3[n/2+1:n,1:n] = numpy.conjugate(mx2[-n/2-1:-n:-1,-1:-n:-1])
    mx3[0,n/2+1:n]=numpy.conjugate(mx2[0,-n/2-1:-n:-1])
    mx3[n/2+1:n,0]=numpy.conjugate(mx2[-n/2-1:-n:-1,0])
    del mx2

    mx4=mx1*mx3
    del mx1                                                    # Times by PSD and do FFT
    phs=(numpy.fft.fft2(mx4)).real

    const=(0.1517/(numpy.sqrt(2.))) * ((float(n)/r0)**(5./6.))   # Use factor 0.1517 (=sqrt(0.023) as per Brent, with sqrt(2)
    #print const
    phs=phs*const                                   # probably because we use a 2 sided PSD and he doesn't
    del mx4
    phs=phs.astype(numpy.float32)

    return phs


def ft_sh_phase_screen(r0, N, delta, L0, l0):
    '''
    Creates a random phase screen with Von Karmen statistics with added
    sub-harmonics to augment tip-tilt modes
    
    Args:
    r0 (float): r0 parameter of scrn in metres
    N (int): Size of phase scrn in pxls
    delta (float): size in Metres of each pxl
    L0 (float): Size of outer-scale in metres
    l0 (float): inner scale in metres


    Returns:
    
    ndarray: numpy array representing phase screen
    '''
    R = random.SystemRandom(time.time())
    seed = int(R.random()*100000)
    numpy.random.seed(seed)

    D = N*delta
    # high-frequency screen from FFT method
    phs_hi = ft_phase_screen(r0, N, delta, L0, l0)

    # spatial grid [m]
    coords = numpy.arange(-N/2,N/2)*delta
    x, y = numpy.meshgrid(coords,coords)

    # initialize low-freq screen
    phs_lo = numpy.zeros(phs_hi.shape)

    # loop over frequency grids with spacing 1/(3^p*L)
    for p in xrange(1,4):
        # setup the PSD
        del_f = 1 / (3**p*D) #frequency grid spacing [1/m]
        fx = numpy.arange(-1,2) * del_f

        # frequency grid [1/m]
        fx, fy = numpy.meshgrid(fx,fx)
        f = numpy.sqrt(fx**2 +  fy**2) # polar grid

        fm = 5.92/l0/(2*numpy.pi) # inner scale frequency [1/m]
        f0 = 1./L0;


        # outer scale frequency [1/m]
        # modified von Karman atmospheric phase PSD
        PSD_phi = (0.023*r0**(-5./3)
                    * numpy.exp(-1*(f/fm)**2) / ((f**2 + f0**2)**(11./6)) )
        PSD_phi[1,1] = 0

        # random draws of Fourier coefficients
        cn = ( (numpy.random.normal(size=(3,3))
            + 1j*numpy.random.normal(size=(3,3)) )
                        * numpy.sqrt(PSD_phi)*del_f )
        SH = numpy.zeros((N,N),dtype="complex")
        # loop over frequencies on this grid
        for i in xrange(0,2):
            for j in xrange(0,2):

                SH += cn[i,j] * numpy.exp(1j*2*numpy.pi*(fx[i,j]*x+fy[i,j]*y))

        phs_lo = phs_lo + SH
        # accumulate subharmonics

    phs_lo = phs_lo.real - phs_lo.real.mean()

    phs = phs_lo+phs_hi

    return phs

def ft2(g,delta):
    G = fft.fftshift(fft.fft2(fft.fftshift(g))) * delta**2
    return G


def ift2(G,delta_f):

    N = G.shape[0]
    g = fft.ifftshift( fft.ifft2( fft.fftshift(G))) * (N * delta_f)**2

    return g

def ft_phase_screen(r0, N, delta, L0, l0):

    delta = float(delta)
    r0 = float(r0)
    L0 = float(L0)
    l0 = float(l0)

    R = random.SystemRandom(time.time())
    seed = int(R.random()*100000)
    numpy.random.seed(seed)


    #print(N*delta)

    del_f = 1./(N*delta)
    #print(del_f)

    fx = numpy.arange(-N/2.,N/2.) * del_f
    #print(fx.max())
    (fx,fy) = numpy.meshgrid(fx,fx)
    f = numpy.sqrt(fx**2 + fy**2)

    #print(f.max())
   #print(f.min())

    fm = 5.92/l0/(2*numpy.pi)
    f0 = 1./L0

    #print(fm,f0)
    PSD_phi  = (0.023*r0**(-5./3.) * numpy.exp(-1*((f/fm)**2)) /
                ( ( (f**2) + (f0**2) )**(11./6) ) )

    PSD_phi[(N/2),(N/2)] = 0


    cn = ( (numpy.random.normal(size=(N,N)) + 1j* numpy.random.normal(size=(N,N)) )
                * numpy.sqrt(PSD_phi)*del_f )

    #make sure everything that isn't needed is deleted.
    #probably not required....
    del(fx)
    del(fy)
    del(f)
    del(PSD_phi)

    phs = ift2(cn,1).real
    del(cn)

    return phs
