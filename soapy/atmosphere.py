#Copyright Durham University and Andrew Reeves
#2014

# This file is part of soapy.

#     soapy is free software: you can redistribute it and/or modify
#     it under the terms of the GNU General Public License as published by
#     the Free Software Foundation, either version 3 of the License, or
#     (at your option) any later version.

#     soapy is distributed in the hope that it will be useful,
#     but WITHOUT ANY WARRANTY; without even the implied warranty of
#     MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
#     GNU General Public License for more details.

#     You should have received a copy of the GNU General Public License
#     along with soapy.  If not, see <http://www.gnu.org/licenses/>.

"""

The Soapy module used to simulate the atmosphere.

This module contains an ``atmos`` object, which can be used to create or load a specified number of phase screens corresponding to atmospheric turbulence layers. The layers can then be moved with the ``moveScrns`` method, at a specified wind velocity and direction, where the screen is interpolated if it does not fall on an integer number of pixels. Alternatively, random screens with the same statistics as the global phase screens can be generated using the ``randomScrns`` method.

The module also contains a number of functions used to create the phase screens, many of these are ported from the book `Numerical Simulation of Optical Propagation`, Schmidt, 2010. It is possible to create a number of phase screens using the :py:func:`makePhaseScreens` function  which are saved to file in a format which can be read by the simulation.

Examples:

    To get the configuration objects::

        from soapy import confParse, atmosphere

        config = confParse.loadSoapyConfig("configfile.yaml")

    Initialise the amosphere (creating or loading phase screens)::

        atmosphere = atmosphere.atmos(config)

    Run the atmosphere for 10 time steps::

        for i in range(10):
            phaseScrns = atmosphere.moveScrns()

    or create 10 sets of random screens::

        for i in range(10):
            randomPhaseScrns = atmosphere.randomScrns()

"""


import numpy
import time
import random
import scipy.fftpack as fft
from . import AOFFT, logger
import scipy.interpolate
#from multiprocessing import Pool
import os


#Use either pyfits or astropy for fits file handling
try:
    from astropy.io import fits
except ImportError:
    try:
        import pyfits as fits
    except ImportError:
        raise ImportError("soapy requires either pyfits or astropy")

try:
    xrange
except NameError:
    xrange = range

class atmos(object):
    '''
    Class to simulate atmosphere above an AO system.

    On initialisation of the object, new phase screens can be created, or others loaded from ``.fits`` file. The atmosphere is created with parameters given in ``ConfigObj.sim`` and ``ConfigObj.atmos``. These are soapy configuration objects, which can be created by the :ref:``confParse`` module, or could be created manually. If created manually, check the :ref: ``confParse`` section to see which attributes the configuration objects must contain.

    If loaded from file, the screens should have a header with the parameter ``R0`` specifying the r0 fried parameter of the screen in pixels.

    The method ``moveScrns`` can be called on each iteration of the AO system to move the scrns forward by one time step. The size of this is defined by parameters given in

    The method ``randomScrns`` returns a set of random phase screens with the smame statistics as the ``atmos`` object.

    Parameters:
        soapyConfig(ConfigObj): The Soapy config object
    '''

    def __init__(self, soapyConfig, mpPool=None):

        self.simConfig = soapyConfig.sim
        self.config =  soapyConfig.atmos

        self.mpPool = mpPool

        self.scrnSize = self.simConfig.scrnSize
        self.windDirs = self.config.windDirs
        self.windSpeeds = self.config.windSpeeds
        self.pxlScale = self.simConfig.pxlScale
        self.wholeScrnSize = self.config.wholeScrnSize
        self.scrnNo = self.config.scrnNo
        self.r0 = self.config.r0
        self.looptime = self.simConfig.loopTime

        self.config.scrnStrengths = numpy.array(self.config.scrnStrengths,
                dtype="float32")

        self.config.normScrnStrengths = self.config.scrnStrengths/(
                            self.config.scrnStrengths[:self.scrnNo].sum())
        self.config.scrnHeights = self.config.scrnHeights[
                    :self.config.scrnNo]

        self.scrnStrengths = ( ((self.r0**(-5./3.))
                                *self.config.normScrnStrengths)**(-3./5.) )
        # #Assume r0 calculated for 550nm.
        # self.wvl = 550e-9

        # Computes tau0, the AO time constant (Roddier 1981), at current wind speed
        vBar53 = (self.windSpeeds[:self.scrnNo]**(5./3.) * self.config.normScrnStrengths[:self.scrnNo]).sum() ** (3./5.)
        tau0 = 0.314 * self.r0 / vBar53

        # If tau0 specified
        if self.config.tau0:
            print("Scaling wind speeds for tau0 from {0:.2f} ms to requested {1:.2f} ms...".format(tau0*1e3, self.config.tau0*1e3))
            # Scale wind speeds
            self.windSpeeds = self.windSpeeds*tau0/self.config.tau0
            # Computes tau0 at scaled wind speeds
            vBar53 = (self.windSpeeds[:self.scrnNo]**(5./3.) * self.config.normScrnStrengths[:self.scrnNo]).sum() ** (3./5.)
            tau0 = 0.314 * self.r0 / vBar53

        ## Print turbulence summary
        logger.info("Turbulence summary @ 500 nm:")
        logger.info('| r0 = {0:.2f} m ({1:.2f}" seeing)'.format(self.r0, numpy.degrees(0.5e-6/self.r0)*3600.0))
        logger.info("| Vbar_5/3 = {0:.2f} m/s".format(vBar53))
        logger.info("| tau0 = {0:.2f} ms".format(tau0*1e3))

        self.scrnPos = {}
        self.wholeScrns = {}

        scrns={}

        scrnSize = int(round(self.scrnSize))

        # The whole screens will be kept at this value, and then scaled to the
        # correct r0 before being sent to the simulation
        self.wholeScrnR0 = 1.

        # If required, generate some new Kolmogorov phase screens
        if not self.config.scrnNames:
            logger.info("Generating Phase Screens")
            for i in xrange(self.scrnNo):

                logger.info("Generate Phase Screen {0}  with r0: {1:.2f}, size: {2}".format(i,self.scrnStrengths[i], self.wholeScrnSize))
                if self.config.subHarmonics:
                    self.wholeScrns[i] = ft_sh_phase_screen(
                            self.wholeScrnR0,
                            self.wholeScrnSize, 1./self.pxlScale,
                            self.config.L0[i], 0.01)
                else:
                    self.wholeScrns[i] = ft_phase_screen(
                            self.wholeScrnR0,
                            self.wholeScrnSize, 1./self.pxlScale,
                            self.config.L0[i], 0.01)

                scrns[i] = self.wholeScrns[i][:scrnSize,:scrnSize]

        # Otherwise, load some others from FITS file
        else:
            logger.info("Loading Phase Screens")

            for i in xrange(self.scrnNo):
                fitsHDU = fits.open(self.config.scrnNames[i])
                scrnHDU = fitsHDU[0]
                self.wholeScrns[i] = scrnHDU.data.astype("float32")

                scrns[i] = self.wholeScrns[i][:scrnSize,:scrnSize]

                # Do the loaded scrns tell us how strong they are?
                # Theyre r0 must be in pixels! label: "R0"
                # If so, we can scale them to the desired r0
                try:
                    r0 = float(scrnHDU.header["R0"])
                    r0_metres = r0/self.pxlScale
                    self.wholeScrns[i] *=(
                                 (self.wholeScrnR0/r0_metres)**(-5./6.)
                                         )

                except KeyError:
                    logger.warning("no r0 info found in screen header - will assume its ok as it is")
                    
            # close fits HDU now we're done with it
            fitsHDU.close()
                   
            if self.wholeScrnSize!=self.wholeScrns[i].shape[0]:
                logger.warning("Requested phase screen has different size to that input in config file....loading anyway")

            self.wholeScrnSize = self.wholeScrns[i].shape[0]
            if self.wholeScrnSize < self.scrnSize:
                raise Exception("required scrn size larger than phase screen")
            
            
        # However we made the phase screen, turn it into meters for ease of
        # use
#        for s in range(self.scrnNo):
#            self.wholeScrns[s] *= (500e-9/(2*numpy.pi))
#
        # Set the initial starting point of the screen,
        # If windspeed is negative, starts from the
        # far-end of the screen to avoid rolling straight away
        windDirs= numpy.array(self.windDirs,dtype="float32") * numpy.pi/180.0
        windV=(self.windSpeeds * numpy.array([numpy.cos(windDirs),
                                              numpy.sin(windDirs)])).T #This is velocity in metres per second
        windV *= self.looptime   #Now metres per looptime
        windV *= self.pxlScale   #Now pxls per looptime.....ideal!
        self.windV = windV

        # Sets initial phase screen pos
        # If velocity is negative in either direction, set the starting point
        # to the end of the screen to avoid rolling too early.
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
            self.xCoords[i] = numpy.arange(self.scrnSize).astype('float') + self.scrnPos[i][0]
            self.yCoords[i] = numpy.arange(self.scrnSize).astype('float') + self.scrnPos[i][1]


    def saveScrns(self, DIR):
        """
        Saves the currently loaded phase screens to file,
        saving the r0 value in the fits header (in units of pixels).

        Args:
            DIR (string): The directory to save the screens
        """

        for scrn in range(self.scrnNo):
            logger.info("Write Sreen {}....".format(scrn))
            hdu = fits.PrimaryHDU(self.wholeScrns[scrn])
            hdu.header["R0"] = self.wholeScrnR0 * self.pxlScale
            hdulist = fits.HDUList([hdu])
            hdulist.writeto(DIR+"/scrn{}.fits".format(scrn))
            hdulist.close()

            logger.info("Done!")


    def moveScrns(self):
        """
        Moves the phase screens one time-step, defined by the atmosphere object parameters.

        Returns:
            dict : a dictionary containing the new set of phase screens
        """

        # If random screens are required:
        if self.config.randomScrns:
            return self.randomScrns(subHarmonics=self.config.subHarmonics)

        # Other wise proceed with translating large phase screens
        scrns={}

        for i in self.wholeScrns:

            #Deals with what happens when the window on the screen
            #reaches the edge - rolls it round and starts again.
            #X direction
            if (self.scrnPos[i][0] + self.scrnSize) >= self.wholeScrnSize:
                logger.debug("pos > scrnSize: rolling phase screen X")
                self.wholeScrns[i] = numpy.roll(self.wholeScrns[i],
                                                int(-self.scrnPos[i][0]),axis=0)
                self.scrnPos[i][0] = 0
                #and update the coords...
                self.xCoords[i] = numpy.arange(self.scrnSize).astype('float')
                self.interpScrns[i] = scipy.interpolate.RectBivariateSpline(
                                            numpy.arange(self.wholeScrnSize),
                                            numpy.arange(self.wholeScrnSize),
                                            self.wholeScrns[i])

            if self.scrnPos[i][0] < 0:
                logger.debug("pos < 0: rolling phase screen X")

                self.wholeScrns[i] = numpy.roll(self.wholeScrns[i],
                                            int(self.wholeScrnSize-self.scrnPos[i][0]-self.scrnSize),axis=0)
                self.scrnPos[i][0] = self.wholeScrnSize-self.scrnSize
                self.xCoords[i] = numpy.arange(self.scrnSize).astype('float')+self.scrnPos[i][0]
                self.interpScrns[i] = scipy.interpolate.RectBivariateSpline(
                                            numpy.arange(self.wholeScrnSize),
                                            numpy.arange(self.wholeScrnSize),
                                            self.wholeScrns[i])
            #Y direction
            if (self.scrnPos[i][1] + self.scrnSize) >= self.wholeScrnSize:
                logger.debug("pos > scrnSize: rolling Phase Screen Y")
                self.wholeScrns[i] = numpy.roll(self.wholeScrns[i],
                                                int(-self.scrnPos[i][1]),axis=1)
                self.scrnPos[i][1] = 0
                self.yCoords[i] = numpy.arange(self.scrnSize).astype('float')
                self.interpScrns[i] = scipy.interpolate.RectBivariateSpline(
                                            numpy.arange(self.wholeScrnSize),
                                            numpy.arange(self.wholeScrnSize),
                                            self.wholeScrns[i])
            if self.scrnPos[i][1] < 0:
                logger.debug("pos < 0: rolling Phase Screen Y")

                self.wholeScrns[i] = numpy.roll(self.wholeScrns[i],
                    int(self.wholeScrnSize-self.scrnPos[i][1]-self.scrnSize),
                                                                axis=1)
                self.scrnPos[i][1] = self.wholeScrnSize-self.scrnSize
                self.yCoords[i] = numpy.arange(self.scrnSize).astype('float')+self.scrnPos[i][1]
                self.interpScrns[i] = scipy.interpolate.RectBivariateSpline(
                                            numpy.arange(self.wholeScrnSize),
                                            numpy.arange(self.wholeScrnSize),
                                            self.wholeScrns[i])


            scrns[i] = self.interpScrns[i](self.xCoords[i],self.yCoords[i])

            # Move window coordinates.
            self.scrnPos[i] = self.scrnPos[i] + self.windV[i]
            self.xCoords[i] += self.windV[i][0].astype('float')
            self.yCoords[i] += self.windV[i][1].astype('float')

            #remove piston from phase screens
            scrns[i] -= scrns[i].mean()

            # Calculate the r0 of each screen
            self.config.normScrnStrengths = (
                    self.config.scrnStrengths/
                        self.config.scrnStrengths[:self.scrnNo].sum())
            self.scrnStrengths = ( ((self.r0**(-5./3.))
                        *self.config.normScrnStrengths)**(-3./5.))
            # Finally, scale for r0 and turn to nm
            scrns[i] *= (self.scrnStrengths[i]/self.wholeScrnR0)**(-5./6.)
            scrns[i] *= (500/(2*numpy.pi))

        return scrns

    def randomScrns(self, subHarmonics=True, l0=0.01):
        """
        Generated random phase screens defined by the atmosphere
        object parameters.

        Returns:
            dict : a dictionary containing the new set of phase screens
        """
        scrns = {}
        args = []

        #If no MP pool, just make screen sequentialy.
        if self.mpPool == None:
            for i in xrange(self.scrnNo):
                if subHarmonics:
                    scrns[i] = ft_sh_phase_screen(
                            self.scrnStrengths[i], self.scrnSize,
                            (self.pxlScale**(-1.)), self.config.L0[i],
                            0.01)
                else:
                    scrns[i] = ft_phase_screen(
                            self.scrnStrengths[i], self.scrnSize,
                            (self.pxlScale**(-1.)), self.config.L0[i],
                            0.01)
                # convert to nm
                scrns[i] *= (500./(2*numpy.pi))

        #If there is a pool, use it
        else:
            if subHarmonics:
                for i in range(self.scrnNo):
                    args.append(
                            (ft_sh_phase_screen,  self.scrnStrengths[i],
                            self.scrnSize, (self.pxlScale**(-1.)),
                            self.atmosConfig.L0[i], l0))

            else:
                for i in range(self.scrnNo):
                    args.append(
                            (ft_phase_screen,  self.scrnStrengths[i],
                            self.scrnSize, (self.pxlScale**(-1.)),
                            self.atmosConfig.L0[i], l0))

            # Do the calculation using the multi-process pool, put into dict
            s = self.mpPool.map(mpWrap, args)
            for i in range(self.scrnNo):
                scrns[i] = s[i]
                # convert to mn
                scrns[i] *= (500./(2*numpy.pi))

        return scrns


def pool_ft_sh_phase_screen(args):
    """
    A helper function for multi-processing of phase screen creation.
    """

    return ft_sh_phase_screen(*args)


def makePhaseScreens(
        nScrns, r0, N, pxlScale, L0, l0, returnScrns=True, DIR=None, SH=False):
    """
    Creates and saves a set of phase screens to be used by the simulation.

    Creates ``nScrns`` phase screens, with the required parameters, then saves
    them to the directory specified by ``DIR``. Each screen is given a FITS
    header with its value of r0, which will be scaled by on simulation when
    its loaded.

    Parameters:
        nScrns (int): The number of screens to make.
        r0 (float): r0 value of the phase screens in metres.
        N (int): Number of elements across each screen.
        pxlScale (float): Size of each element in metres.
        L0 (float): Outer scale of each screen.
        l0 (float): Inner scale of each screen.
        returnScrns (bool, optional): Whether to return a list of screens. True by default, but if screens are very large, False might be preferred so they aren't kept in memory if saving to disk.
        DIR (str, optional): The directory to save the screens.
        SH (bool, optional): If True, add sub-harmonics to screens for more
                accurate power spectra, though screens no-longer periodic.

    Returns:
        list: A list containing all the screens.
    """

    #Make directory if it doesnt exist already
    if DIR:
        if not os.path.isdir(DIR):
            os.makedirs(DIR)

    #An empty container to put our screens in
    if returnScrns:
        scrns = []

    #Now loop over and create all the screens (Currently with the same params)
    for i in range(nScrns):
        if SH:
            scrn = ft_sh_phase_screen(r0, N, pxlScale, L0, l0)
        else:
            scrn = ft_phase_screen(r0, N, pxlScale, L0, l0)

        if returnScrns:
            scrns.append(scrn)

        #If given a directory, save them too!
        if DIR!=None:
            hdu = fits.PrimaryHDU(scrn)
            hdu.header["R0"] = str(r0/pxlScale)
            hdu.writeto(DIR+"/scrn{}.fits".format(i))

    if returnScrns:
        return scrns


def ft_sh_phase_screen(r0, N, delta, L0, l0, FFT=None):
    '''
    Creates a random phase screen with Von Karmen statistics with added
    sub-harmonics to augment tip-tilt modes.
    (Schmidt 2010)

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
    phs_hi = ft_phase_screen(r0, N, delta, L0, l0, FFT)

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


def ift2(G, delta_f ,FFT=None):
    """
    Wrapper for inverse fourier transform

    Parameters:
        G: data to transform
        delta_f: pixel seperation
        FFT (FFT object, optional): An accelerated FFT object
    """

    N = G.shape[0]

    if FFT:
        g = AOFFT.ftShift2d( FFT( AOFFT.ftShift2d(G) ) ) * (N * delta_f)**2
    else:
        g = fft.ifftshift( fft.ifft2( fft.fftshift(G) ) ) * (N * delta_f)**2

    return g

def ft_phase_screen(r0, N, delta, L0, l0, FFT=None):
    '''
    Creates a random phase screen with Von Karmen statistics.
    (Schmidt 2010)

    Parameters:
        r0 (float): r0 parameter of scrn in metres
        N (int): Size of phase scrn in pxls
        delta (float): size in Metres of each pxl
        L0 (float): Size of outer-scale in metres
        l0 (float): inner scale in metres

    Returns:
        ndarray: numpy array representing phase screen
    '''
    delta = float(delta)
    r0 = float(r0)
    L0 = float(L0)
    l0 = float(l0)

    R = random.SystemRandom(time.time())
    seed = int(R.random()*100000)
    numpy.random.seed(seed)

    del_f = 1./(N*delta)

    fx = numpy.arange(-N/2.,N/2.) * del_f

    (fx,fy) = numpy.meshgrid(fx,fx)
    f = numpy.sqrt(fx**2 + fy**2)

    fm = 5.92/l0/(2*numpy.pi)
    f0 = 1./L0

    PSD_phi  = (0.023*r0**(-5./3.) * numpy.exp(-1*((f/fm)**2)) /
                ( ( (f**2) + (f0**2) )**(11./6) ) )

    PSD_phi[(N/2),(N/2)] = 0

    cn = ( (numpy.random.normal(size=(N,N)) + 1j* numpy.random.normal(size=(N,N)) )
                * numpy.sqrt(PSD_phi)*del_f )

    phs = ift2(cn,1, FFT).real

    return phs


def mpWrap(args):
    """
    A helper to run python MP pool functions

    Using the native python MP pool.map, it is not possible to pass multiple
    args. If all args are passed to this helper function as a tuple, with the
    function itself as the first item, it will run it with the unpacked args
    and return the result.

    Parameters:
        args (tuple): A tuple, with the first item the desired function and the rest the function args

    Returns:
        Whatever the function would have returned
    """

    func = args[0]
    args = args[1:]
    return func(*args)
