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

import os
import random
import time

import numpy
import scipy.fftpack as fft
import scipy.interpolate

from . import AOFFT, logger, numbalib
from aotools.turbulence import infinitephasescreen, phasescreen
# Use either pyfits or astropy for fits file handling
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
    def __init__(self, soapyConfig):

        self.simConfig = soapyConfig.sim
        self.config =  soapyConfig.atmos

        self.scrn_size = self.simConfig.scrnSize
        self.windDirs = self.config.windDirs
        self.windSpeeds = self.config.windSpeeds
        self.pixel_scale = 1./self.simConfig.pxlScale
        self.wholeScrnSize = self.config.wholeScrnSize
        self.scrnNo = self.config.scrnNo
        self.r0 = self.config.r0
        self.L0s = self.config.L0
        self.looptime = self.simConfig.loopTime

        if self.config.randomSeed is not None:
            numpy.random.seed(self.config.randomSeed)

        self.config.scrnStrengths = numpy.array(self.config.scrnStrengths,
                dtype="float32")

        self.config.normScrnStrengths = self.config.scrnStrengths/(
                            self.config.scrnStrengths[:self.scrnNo].sum())
        self.config.scrnHeights = self.config.scrnHeights[
                    :self.config.scrnNo]

        self.scrnStrengths = ( ((self.r0**(-5./3.))
                                *self.config.normScrnStrengths)**(-3./5.) )

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


        scrnSize = int(round(self.scrn_size))

        self.scrns = numpy.zeros((self.scrnNo, self.scrn_size, self.scrn_size))


        # The whole screens will be kept at this value, and then scaled to the
        # correct r0 before being sent to the simulation
        self.wholeScrnR0 = 1.

        # If required, generate some new Kolmogorov phase screens
        if self.config.infinite:
            self.infinite_phase_screens = []
            for layer in range(self.config.scrnNo):
                logger.info("Initialise Infinite Phase Screen {}".format(layer+1))
                phase_screen = InfinitePhaseScreen(
                        self.scrn_size, self.pixel_scale, self.scrnStrengths[layer],
                        self.L0s[layer], self.windSpeeds[layer], self.looptime, self.windDirs[layer])
                self.infinite_phase_screens.append(phase_screen)

        else:
            if not self.config.scrnNames:
                logger.info("Generating Phase Screens")
                for i in xrange(self.scrnNo):

                    logger.info("Generate Phase Screen {0}  with r0: {1:.2f}, size: {2}".format(i,self.scrnStrengths[i], self.wholeScrnSize))
                    if self.config.subHarmonics:
                        self.wholeScrns[i] = phasescreen.ft_sh_phase_screen(
                                self.wholeScrnR0,
                                self.wholeScrnSize, self.pixel_scale,
                                self.config.L0[i], 0.01)
                    else:
                        self.wholeScrns[i] = phasescreen.ft_phase_screen(
                                self.wholeScrnR0,
                                self.wholeScrnSize, self.pixel_scale,
                                self.config.L0[i], 0.01)

                    self.scrns[i] = self.wholeScrns[i][:scrnSize,:scrnSize]

            # Otherwise, load some others from FITS file
            else:
                logger.info("Loading Phase Screens")

                for i in xrange(self.scrnNo):
                    logger.info("Load screen {}...".format(i))
                    fitsHDU = fits.open(self.config.scrnNames[i])
                    scrnHDU = fitsHDU[0]
                    self.wholeScrns[i] = scrnHDU.data.astype("float32")

                    self.scrns[i] = self.wholeScrns[i][:scrnSize,:scrnSize]

                    # Do the loaded scrns tell us how strong they are?
                    # Theyre r0 must be in pixels! label: "R0"
                    # If so, we can scale them to the desired r0
                    try:
                        r0 = float(scrnHDU.header["R0"])
                        r0_metres = r0 * self.pixel_scale
                        self.wholeScrns[i] *=(
                                     (self.wholeScrnR0/r0_metres)**(-5./6.)
                                             )

                    except KeyError:
                        logger.warning("no r0 info found in screen header - will assume its ok as it is")
                    
                    # close fits HDU now we're done with it
                    fitsHDU.close()

                    logger.info("Load screen {}...Done".format(i))


                if self.wholeScrnSize!=self.wholeScrns[i].shape[0]:
                    logger.warning("Requested phase screen has different size to that input in config file....loading anyway")

                self.wholeScrnSize = self.wholeScrns[i].shape[0]
                if self.wholeScrnSize < self.scrn_size:
                    raise Exception("required scrn size larger than phase screen")
            
            
            # However we made the phase screen, turn it into meters for ease of
            # use
    #        for s in range(self.scrnNo):
    #            self.wholeScrns[s] *= (500e-9/(2*numpy.pi))
    #
            # Set the initial starting point of the screen,
            # If windspeed is negative, starts from the
            # far-end of the screen to avoid rolling straight away
            windDirs = numpy.array(self.windDirs,dtype="float32") * numpy.pi/180.0
            windV = (self.windSpeeds * numpy.array([numpy.cos(windDirs),
                                                    numpy.sin(windDirs)])).T #This is velocity in metres per second
            windV *= self.looptime   #Now metres per looptime
            windV /= self.pixel_scale   #Now pxls per looptime.....ideal!
            self.windV = windV

            # Sets initial phase screen pos
            # If velocity is negative in either direction, set the starting point
            # to the end of the screen to avoid rolling too early.
            for i in xrange(self.scrnNo):
                self.scrnPos[i] = numpy.array([0,0])

                if windV[i,0] < 0:
                    self.scrnPos[i][0] = self.wholeScrns[i].shape[0] - self.scrn_size

                if windV[i,1] < 0:
                    self.scrnPos[i][1] = self.wholeScrns[i].shape[1] - self.scrn_size

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
                self.xCoords[i] = numpy.arange(self.scrn_size).astype('float') + self.scrnPos[i][0]
                self.yCoords[i] = numpy.arange(self.scrn_size).astype('float') + self.scrnPos[i][1]


    def saveScrns(self, DIR):
        """
        Saves the currently loaded phase screens to file,
        saving the r0 value in the fits header (in units of pixels). 
        Saved phase data is in radians @500nm

        Args:
            DIR (string): The directory to save the screens
        """

        for scrn in range(self.scrnNo):
            logger.info("Write Sreen {}....".format(scrn))
            hdu = fits.PrimaryHDU(self.wholeScrns[scrn])
            hdu.header["R0"] = self.wholeScrnR0 / self.pixel_scale
            hdulist = fits.HDUList([hdu])
            hdulist.writeto(DIR+"/scrn{}.fits".format(scrn))
            hdulist.close()

            logger.info("Done!")


    def moveScrns(self):
        """
        Moves the phase screens one time-step, defined by the atmosphere object parameters.
        
        Returned phase is in units of nana-meters

        Returns:
            dict : a dictionary containing the new set of phase screens
        """

        # If random screens are required:
        if self.config.randomScrns:
            return self.randomScrns(subHarmonics=self.config.subHarmonics)

        if self.config.infinite:
            for layer_n in range(self.scrnNo):
                self.scrns[layer_n] = self.infinite_phase_screens[layer_n].move_screen()

            # Convert to nm
            self.scrns *= (500/(2*numpy.pi))
            return self.scrns

        # Other wise proceed with translating large phase screens
        for i in self.wholeScrns:

            # Deals with what happens when the window on the screen
            # reaches the edge - rolls it round and starts again.
            # X direction
            if (self.scrnPos[i][0] + self.scrn_size) >= self.wholeScrnSize:
                logger.debug("pos > scrn_size: rolling phase screen X")
                self.wholeScrns[i] = numpy.roll(self.wholeScrns[i],
                                                int(-self.scrnPos[i][0]),axis=0)
                self.scrnPos[i][0] = 0
                # and update the coords...
                self.xCoords[i] = numpy.arange(self.scrn_size).astype('float')
                self.interpScrns[i] = scipy.interpolate.RectBivariateSpline(
                                            numpy.arange(self.wholeScrnSize),
                                            numpy.arange(self.wholeScrnSize),
                                            self.wholeScrns[i])

            if self.scrnPos[i][0] < 0:
                logger.debug("pos < 0: rolling phase screen X")

                self.wholeScrns[i] = numpy.roll(self.wholeScrns[i],
                                                int(self.wholeScrnSize - self.scrnPos[i][0] - self.scrn_size), axis=0)
                self.scrnPos[i][0] = self.wholeScrnSize-self.scrn_size
                self.xCoords[i] = numpy.arange(self.scrn_size).astype('float') + self.scrnPos[i][0]
                self.interpScrns[i] = scipy.interpolate.RectBivariateSpline(
                                            numpy.arange(self.wholeScrnSize),
                                            numpy.arange(self.wholeScrnSize),
                                            self.wholeScrns[i])
            # Y direction
            if (self.scrnPos[i][1] + self.scrn_size) >= self.wholeScrnSize:
                logger.debug("pos > scrn_size: rolling Phase Screen Y")
                self.wholeScrns[i] = numpy.roll(self.wholeScrns[i],
                                                int(-self.scrnPos[i][1]),axis=1)
                self.scrnPos[i][1] = 0
                self.yCoords[i] = numpy.arange(self.scrn_size).astype('float')
                self.interpScrns[i] = scipy.interpolate.RectBivariateSpline(
                                            numpy.arange(self.wholeScrnSize),
                                            numpy.arange(self.wholeScrnSize),
                                            self.wholeScrns[i])
            if self.scrnPos[i][1] < 0:
                logger.debug("pos < 0: rolling Phase Screen Y")

                self.wholeScrns[i] = numpy.roll(self.wholeScrns[i],
                                                int(self.wholeScrnSize - self.scrnPos[i][1] - self.scrn_size),
                                                axis=1)
                self.scrnPos[i][1] = self.wholeScrnSize-self.scrn_size
                self.yCoords[i] = numpy.arange(self.scrn_size).astype('float') + self.scrnPos[i][1]
                self.interpScrns[i] = scipy.interpolate.RectBivariateSpline(
                                            numpy.arange(self.wholeScrnSize),
                                            numpy.arange(self.wholeScrnSize),
                                            self.wholeScrns[i])

            self.scrns[i] = self.interpScrns[i](self.xCoords[i], self.yCoords[i])

            # Move window coordinates.
            self.scrnPos[i] = self.scrnPos[i] + self.windV[i]
            self.xCoords[i] += self.windV[i][0].astype('float')
            self.yCoords[i] += self.windV[i][1].astype('float')

            # Calculate the required r0 of each screen from config
            self.config.normScrnStrengths = (
                    self.config.scrnStrengths/
                        self.config.scrnStrengths[:self.scrnNo].sum())
            self.scrnStrengths = ( ((self.r0**(-5./3.))
                        *self.config.normScrnStrengths)**(-3./5.))
            # Finally, scale for r0 and turn to nm
            self.scrns[i] *= (self.scrnStrengths[i]/self.wholeScrnR0)**(-5./6.)
            self.scrns[i] *= (500/(2*numpy.pi))

        return self.scrns

    def randomScrns(self, subHarmonics=True, l0=0.01):
        """
        Generated random phase screens defined by the atmosphere object parameters.

        Returned phase is in units of nana-meters

        Returns:
            dict : a dictionary containing the new set of phase screens
        """

        for i in xrange(self.scrnNo):
            if subHarmonics:
                self.scrns[i] = phasescreen.ft_sh_phase_screen(
                        self.scrnStrengths[i], self.scrn_size,
                        self.pixel_scale, self.config.L0[i], l0)
            else:
                self.scrns[i] = phasescreen.ft_phase_screen(
                        self.scrnStrengths[i], self.scrn_size,
                        self.pixel_scale, self.config.L0[i], l0)

            # Turn to nm
            self.scrns[i] *= (500./(2*numpy.pi))

        return self.scrns


def pool_ft_sh_phase_screen(args):
    """
    A helper function for multi-processing of phase screen creation.
    """

    return phasescreen.ft_sh_phase_screen(*args)


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
            scrn = phasescreen.ft_sh_phase_screen(r0, N, pxlScale, L0, l0)
        else:
            scrn = phasescreen.ft_phase_screen(r0, N, pxlScale, L0, l0)

        if returnScrns:
            scrns.append(scrn)

        #If given a directory, save them too!
        if DIR!=None:
            hdu = fits.PrimaryHDU(scrn)
            hdu.header["R0"] = str(r0/pxlScale)
            hdu.writeto(DIR+"/scrn{}.fits".format(i))

    if returnScrns:
        return scrns


class InfinitePhaseScreen(infinitephasescreen.PhaseScreenVonKarman):
    def __init__(
            self, nx_size, pixel_scale, r0, L0, wind_speed,
            time_step, wind_direction, random_seed=None, n_columns=2):

        if wind_direction not in (0, 90, 180, 270):
            # Have to make screne bigger to cope with rotaation
            self.nx_output_size = nx_size
            nx_size = int(numpy.ceil(2 * 2**0.5 * nx_size))
        else:
            self.nx_output_size = nx_size

        super(InfinitePhaseScreen, self).__init__(nx_size, pixel_scale, r0, L0, random_seed, n_columns)

        self.wind_speed = wind_speed
        self.time_step = time_step
        self.wind_direction = wind_direction

        self.n_move_pixels = (self.wind_speed * self.time_step) / self.pixel_scale

        # Integer number of pixels that must be added on each iteration
        self.int_move_pixels = int(self.n_move_pixels)

        # the remainder that must be interpolated
        self.float_position = 0

        # The coordinates to use to interpolate - will add on a float  less that 1
        self.interp_coords = numpy.arange(1, self.nx_size+1)

        self.output_screen = numpy.zeros((self.nx_size, self.nx_size))
        self.output_rotation_screen = numpy.zeros((self.nx_output_size, self.nx_output_size))

    def move_screen(self):

        n_new_rows = self.int_move_pixels

        self.float_position += (self.n_move_pixels - self.int_move_pixels)
        if self.float_position >= 1:
            n_new_rows += 1
            self.float_position -= 1
        # print("New rows: {}, float_position: {}".format(n_new_rows, self.float_position))

        for i in range(n_new_rows):
            # print("Get row: {}".format(i))
            new_row = self.get_new_row()
            # print("append row: {}".format(i))
            self._scrn = numpy.append(new_row, self._scrn, axis=0)

        self._scrn = self._scrn[:self.stencil_length, :self.nx_size]

        numbalib.bilinear_interp(
                self._scrn, self.interp_coords - self.float_position, self.interp_coords, self.output_screen)

        self.rotate_screen()

        return self.output_rotation_screen

    def rotate_screen(self):

        if self.wind_direction == 0:
            self.output_rotation_screen = self.output_screen[:self.nx_output_size, :self.nx_output_size]
            return self.output_rotation_screen

        elif self.wind_direction == 90:
            self.output_rotation_screen = numpy.rot90(
                    self.output_screen[:self.nx_output_size, :self.nx_output_size])
            return self.output_rotation_screen

        elif self.wind_direction == 180:
            self.output_rotation_screen = numpy.flipud(
                    self.output_screen[:self.nx_output_size, :self.nx_output_size])
            return self.output_rotation_screen

        elif self.wind_direction == 270:
            self.output_rotation_screen = numpy.rot90(
                self.output_screen[:self.nx_output_size, :self.nx_output_size],  k=3)
            return self.output_rotation_screen

        else:
            numbalib.rotate(
                    self.output_screen, self.output_rotation_screen,
                self.wind_direction*numpy.pi/180)
            return self.output_rotation_screen


    def calc_seperations(self):
        """
        Calculates the seperations between the phase points in the stencil and the new phase vector
        """
        positions = numpy.append(self.stencil_positions, self.X_positions, axis=0)
        self.seperations = numpy.zeros((len(positions), len(positions)))

        calculate_seperations(positions, self.seperations)

        # for i, (x1, y1) in enumerate(positions):
        #     for j, (x2, y2) in enumerate(positions):
        #
        #         delta_x = x2 - x1
        #         delta_y = y2 - y1
        #
        #         delta_r = numpy.sqrt(delta_x**2 + delta_y**2)
        #
        #         self.seperations[i, j] = delta_r

import numba
@numba.jit(nopython=True)
def calculate_seperations(positions, seperations):
    for i in range(positions.shape[0]):
        (x1, y1) = positions[i]
        for j in range(positions.shape[0]):
            (x2, y2) = positions[j]
            delta_x = x2 - x1
            delta_y = y2 - y1

            delta_r = numpy.sqrt(delta_x ** 2 + delta_y ** 2)

            seperations[i, j] = delta_r