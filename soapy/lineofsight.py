"""
A generalised module to provide phase or the EField through a "Line Of Sight"

Line of Sight Object
====================
The module contains a 'lineOfSight' object, which calculates the resulting phase or complex amplitude from propogating through the atmosphere in a given
direction. This can be done using either geometric propagation, where phase is simply summed for each layer, or physical propagation, where the phase is propagated between layers using an angular spectrum propagation method. Light can propogate either up or down.

The Object takes a 'config' as an argument, which is likely to be the same config object as the module using it (WFSs, ScienceCams, or LGSs). It should contain paramters required, such as the observation direction and light wavelength. The `config` also determines whether to use physical or geometric propagation through the 'propagationMode' parameter.

Examples::

    from soapy import confParse, lineofsight

    # Initialise a soapy conifuration file
    config = confParse.loadSoapyConfig('conf/sh_8x8.py')

    # Can make a 'LineOfSight' for WFSs
    los = lineofsight.LineOfSight(config.wfss[0], config)

    # Get resulting complex amplitude through line of sight
    EField = los.frame(some_phase_screens)

"""

import numpy

from aotools import opticalpropagation

from . import logger, interp
from . import numbalib

DTYPE = numpy.float32
CDTYPE = numpy.complex64

# Python3 compatability
try:
    xrange
except NameError:
    xrange = range

RAD2ASEC = 206264.849159
ASEC2RAD = 1./RAD2ASEC

class LineOfSight(object):
    """
    A "Line of sight" through a number of turbulence layers in the atmosphere, observing ing a given direction.

    Parameters:
        config: The soapy config for the line of sight
        simConfig: The soapy simulation config object
        propagation_direction (str, optional): Direction of light propagation, either `"up"` or `"down"`
        outPxlScale (float, optional): The EField pixel scale required at the output (m/pxl)
        nOutPxls (int, optional): Number of pixels to return in EFIeld
        mask (ndarray, optional): Mask to apply at the *beginning* of propagation
        metaPupilPos (list, dict, optional): A list or dictionary of the meta pupil position at each turbulence layer height ub metres. If None, works it out from GS position.
    """
    def __init__(
            self, config, soapyConfig,
            propagation_direction="down", out_pixel_scale=None,
            nx_out_pixels=None, mask=None, metaPupilPos=None):

        self.config = config
        self.soapy_config = soapyConfig
        self.pupil_size = self.soapy_config.sim.pupilSize
        self.phase_pixel_scale = 1./self.soapy_config.sim.pxlScale
        self.sim_size = self.soapy_config.sim.simSize
        self.wavelength = self.config.wavelength
        self.telescope_diameter = self.soapy_config.tel.telDiam
        self.propagation_direction = propagation_direction

        self.source_altitude = self.height

        self.nx_scrn_size = self.soapy_config.sim.scrnSize
        self.n_layers = self.soapy_config.atmos.scrnNo
        self.layer_altitudes = self.soapy_config.atmos.scrnHeights

        self.n_dm = self.soapy_config.sim.nDM
        self.dm_altitudes = numpy.array([self.soapy_config.dms[dm_n].altitude for dm_n in range(self.n_dm)])

        # If the line of sight has a launch position, must include in calculations! Conver to metres from centre of telescope
        try:
            self.launch_position = self.config.launchPosition * self.telescope_diameter/2.
        except AttributeError:
            self.launch_position = numpy.array([0, 0])

        self.simConfig = soapyConfig.sim
        self.atmosConfig = soapyConfig.atmos

        self.mask = mask

        self.calcInitParams(out_pixel_scale, nx_out_pixels)

        self.allocDataArrays()

        # Can be set to use other values as metapupil position
        self.metaPupilPos = metaPupilPos


    # Some attributes for compatability between WFS and others
    @property
    def height(self):
        try:
            return self.config.height
        except AttributeError:
            return self.config.GSHeight

    @height.setter
    def height(self, height):
        try:
            self.config.height
            self.config.height = height
        except AttributeError:
            self.config.GSHeight
            self.config.GSHeight = height

    @property
    def position(self):
        try:
            return self.config.position
        except AttributeError:
            return self.config.GSPosition

    @position.setter
    def position(self, position):
        try:
            self.config.position
            self.config.position = position
        except AttributeError:
            self.config.GSPosition
            self.config.GSPosition = position

############################################################
# Initialisation routines


    def calcInitParams(self, out_pixel_scale=None, nx_out_pixels=None):
        """
        Calculates some parameters required later

        Parameters:
            outPxlScale (float): Pixel scale of required phase/EField (metres/pxl)
            nOutPxls (int): Size of output array in pixels
        """
        logger.debug("Calculate LOS Init PArams!")
        # Convert phase deviation to radians at wfs wavelength.
        # (currently in nm remember...?)
        self.phs2Rad = 2*numpy.pi/(self.wavelength * 10**9)

        # Get the size of the phase required by the system
        self.in_pixel_scale = self.phase_pixel_scale

        if out_pixel_scale is None:
            self.out_pixel_scale = self.phase_pixel_scale
        else:
            self.out_pixel_scale = out_pixel_scale

        if nx_out_pixels is None:
            self.nx_out_pixels = self.simConfig.simSize
        else:
            self.nx_out_pixels = nx_out_pixels

        if self.mask is not None:
            self.outMask = interp.zoom(
                    self.mask, self.nx_out_pixels)#.round()
        else:
            self.outMask = None

        self.output_phase_diameter = self.nx_out_pixels * self.out_pixel_scale

        # Calculate coords of phase at each altitude
        self.layer_metapupil_coords = numpy.zeros((self.n_layers, 2, self.nx_out_pixels))
        for i in range(self.n_layers):
            x1, x2, y1, y2 = self.calculate_altitude_coords(self.layer_altitudes[i])
            self.layer_metapupil_coords[i, 0] = numpy.linspace(x1, x2-1, self.nx_out_pixels)
            self.layer_metapupil_coords[i, 1] = numpy.linspace(y1, y2-1, self.nx_out_pixels)

        # ensure coordinates aren't out of bounds for interpolation
        self.layer_metapupil_coords = self.layer_metapupil_coords.clip(0, self.nx_scrn_size - 1.000000001)

        # Calculate coords of phase at each DM altitude
        self.dm_metapupil_coords = numpy.zeros((self.n_dm, 2, self.nx_out_pixels))
        for i in range(self.n_dm):
            x1, x2, y1, y2 = self.calculate_altitude_coords(self.dm_altitudes[i])
            self.dm_metapupil_coords[i, 0] = numpy.linspace(x1, x2-1, self.nx_out_pixels)
            self.dm_metapupil_coords[i, 1] = numpy.linspace(y1, y2-1, self.nx_out_pixels)
        self.dm_metapupil_coords = self.dm_metapupil_coords.clip(0, self.nx_scrn_size - 1.000000001)

        self.radii = None
        
        # Relavent phase  centred on the line of sight direction
        self.phase_screens = numpy.zeros((self.n_layers, self.nx_out_pixels, self.nx_out_pixels))
        # Buffer for corection across fulll FOV
        self.correction_screens = numpy.zeros((self.n_dm, self.nx_out_pixels, self.nx_out_pixels))
        # Relavent correction centred on the line of sight direction
        self.phase_correction = numpy.zeros((self.nx_out_pixels, self.nx_out_pixels))

        self.allocDataArrays()


    def calculate_altitude_coords(self, layer_altitude):
        """
        Calculate the co-ordinates of vertices of fo the meta-pupil at altitude given a guide star
        direction and source altitude

        Paramters:
            layer_altitude (float): Altitude of phase layer
        """
        direction_radians = ASEC2RAD * numpy.array(self.position)

        centre = (direction_radians * layer_altitude)

        # If propagating up must account for launch position
        if self.propagation_direction == "up":
            if self.source_altitude == 0:
                centre += self.launch_position
            else:
                centre += self.launch_position * (1 - layer_altitude/self.source_altitude)

        if self.source_altitude != 0:
            meta_pupil_size = self.output_phase_diameter * (1 - layer_altitude / self.source_altitude)
        else:
            meta_pupil_size = self.output_phase_diameter

        x1 = ((centre[0] - meta_pupil_size / 2.) / self.in_pixel_scale) + self.nx_scrn_size / 2.
        x2 = ((centre[0] + meta_pupil_size / 2.) / self.in_pixel_scale) + self.nx_scrn_size / 2.
        y1 = ((centre[1] - meta_pupil_size / 2.) / self.in_pixel_scale) + self.nx_scrn_size / 2.
        y2 = ((centre[1] + meta_pupil_size / 2.) / self.in_pixel_scale) + self.nx_scrn_size / 2.

        logger.debug("Layer Altitude: {}".format(layer_altitude))
        logger.debug("Coords: x1: {}, x2: {}, y1: {},  y2: {}".format(x1, x2, y1, y2))

        return x1, x2, y1, y2


    def allocDataArrays(self):
        """
        Allocate the data arrays the LOS will require

        Determines and allocates the various arrays the LOS will require to
        avoid having to re-alloc memory during the running of the LOS and
        keep it fast. This includes arrays for phase
        and the E-Field across the LOS
        """
        self.phase = numpy.zeros([self.nx_out_pixels] * 2, dtype=DTYPE)
        self.EField = numpy.ones([self.nx_out_pixels] * 2, dtype=CDTYPE)
        self.residual = numpy.zeros([self.nx_out_pixels] * 2, dtype=DTYPE)

######################################################

    def zeroData(self, **kwargs):
        """
        Sets the phase and complex amp data to zero
        """
        self.EField[:] = 1
        self.phase[:] = 0
        self.phase_screens[:] = 0
        self.correction_screens[:] = 0


    def makePhase(self, radii=None, apos=None):
        """
        Generates the required phase or EField. Uses difference approach depending on whether propagation is geometric or physical
        (makePhaseGeometric or makePhasePhys respectively)

        Parameters:
            radii (dict, optional): Radii of each meta pupil of each screen height in pixels. If not given uses pupil radius.
            apos (ndarray, optional):  The angular position of the GS in radians. If not set, will use the config position
        """
        for i in range(self.scrns.shape[0]):
            numbalib.bilinear_interp(
                self.scrns[i], self.layer_metapupil_coords[i, 0], self.layer_metapupil_coords[i, 1],
                self.phase_screens[i], bounds_check=False)

        # Check if geometric or physical
        if self.config.propagationMode == "Physical":
            return self.makePhasePhys(radii)
        else:
            return self.makePhaseGeometric(radii)


    def makePhaseGeometric(self, radii=None, apos=None):
        '''
        Creates the total phase along line of sight offset by a given angle using a geometric ray tracing approach

        Parameters:
            radii (dict, optional): Radii of each meta pupil of each screen height in pixels. If not given uses pupil radius.
            apos (ndarray, optional):  The angular position of the GS in radians. If not set, will use the config position
        '''


        self.atmos_phase = self.phase_screens.sum(0)

        # Convert phase to radians
        self.atmos_phase *= self.phs2Rad

        # Change sign if propagating up
        # if self.propagation_direction == 'up':
        #     self.atmos_phase *= -1

        self.phase[:] += self.atmos_phase
        self.EField[:] *= numpy.exp(1j*self.atmos_phase)

        return self.EField


    def makePhasePhys(self, radii=None, apos=None):
        '''
        Finds total line of sight complex amplitude by propagating light through phase screens

        Parameters:
            radii (dict, optional): Radii of each meta pupil of each screen height in pixels. If not given uses pupil radius.
            apos (ndarray, optional):  The angular position of the GS in radians. If not set, will use the config position
        '''

        self.EField[:] = physical_atmosphere_propagation(
            self.phase_screens, self.outMask, self.layer_altitudes, self.source_altitude,
            self.wavelength, self.out_pixel_scale,
            propagation_direction=self.propagation_direction,
            input_efield=self.EField)

        return self.EField


    def performCorrection(self, correction):
        """
        Corrects the aberrated line of sight with some given correction phase
        
        Parameters:
            correction (list or ndarray): either 2-d array describing correction, or list of correction arrays
        """
        for i in range(correction.shape[0]):
            numbalib.bilinear_interp(
                correction[i], self.dm_metapupil_coords[i, 0], self.dm_metapupil_coords[i, 1],
                self.correction_screens[i], bounds_check=False)

        self.correction_screens.sum(0, out=self.phase_correction)

        self.phase_correction *= self.phs2Rad
        # Correct EField
        self.EField *= numpy.exp(-1j * self.phase_correction)

        # Also correct phase in case its required
        self.residual = (self.phase - self.phase_correction) / self.phs2Rad

        self.phase = self.residual * self.phs2Rad


    def frame(self, scrns=None, correction=None):
        '''
        Runs one frame through a line of sight

        Finds the phase or complex amplitude through line of sight for a
        single simulation frame, with a given set of phase screens and
        some optional correction. 
        If scrns is ``None``, then light is propagated with no phase.

        Parameters:
            scrns (list): A list or dict containing the phase screens
            correction (ndarray, optional): The correction term to take from the phase screens before the WFS is run.
            read (bool, optional): Should the WFS be read out? if False, then WFS image is calculated but slopes not calculated. defaults to True.

        Returns:
            ndarray: WFS Measurements
        '''

        self.zeroData()

        # If we propagate up, must do correction first!
        if (self.propagation_direction == "up") and (correction is not None):
            self.performCorrection(correction)

        # Now do propagation through atmospheric turbulence
        if scrns is not None:
            if scrns.ndim==2:
                scrns.shape = 1, scrns.shape[0], scrns.shape[1]
            self.scrns = scrns
        else: # If no scrns, just assume no turbulence
            self.scrns = numpy.zeros(
                    (self.n_layers, self.nx_scrn_size, self.nx_scrn_size))

        self.makePhase(self.radii)


        self.residual = self.phase
        # If propagating down, do correction last
        if (self.propagation_direction == "down") and (correction is not None):
            self.performCorrection(correction)

        return self.residual


def physical_atmosphere_propagation(
            phase_screens, output_mask, layer_altitudes, source_altitude,
            wavelength, output_pixel_scale, 
            propagation_direction="up", input_efield=None):
    '''
    Finds total line of sight complex amplitude by propagating light through phase screens

    If the source altitude is infinity (denoted as 0), then the result of the propagation is
    the

    Parameters:
        
    '''

    scrnNo = len(phase_screens)
    z_total = 0
    scrnRange = range(0, scrnNo)

    nx_output_pixels = phase_screens[0].shape[0]

    phs2Rad = 2 * numpy.pi / (wavelength * 10 ** 9)

    EFieldBuf = input_efield
    if input_efield is None:
        EFieldBuf = numpy.exp(
                1j*numpy.zeros((nx_output_pixels,) * 2)).astype(CDTYPE)

    # Get initial up/down dependent params
    if propagation_direction == "up":
        ht = 0
        ht_final = source_altitude
        # if ht_final==0:
        #     raise ValueError("Can't propagate up to infinity")
        scrnAlts = layer_altitudes
        # If propagating up from telescope, apply mask to the EField
        EFieldBuf *= output_mask
        logger.debug("Create EField Buf of mask")

    else:
        ht = layer_altitudes[scrnNo-1]
        ht_final = 0
        scrnAlts = layer_altitudes[::-1]
        phase_screens = phase_screens[::-1]
        logger.debug("Create EField Buf of zero phase")

    # Propagate to first phase screen (if not already there)
    if ht!=scrnAlts[0]:
        logger.debug("propagate to first phase screen")
        z = abs(scrnAlts[0] - ht)
        z_total += z
        EFieldBuf[:] = opticalpropagation.angularSpectrum(
                    EFieldBuf, wavelength,
                    output_pixel_scale, output_pixel_scale, z)

    # Go through and propagate between phase screens
    for i in scrnRange:

        phase = phase_screens[i]
        # print("Got phase")

        # Convert phase to radians
        phase *= phs2Rad

        # Apply phase to EField
        EFieldBuf *= numpy.exp(1j*phase)

        # Change sign if propagating up
        # if propagation_direction == 'up':
        #     phase *= -1
        # print("Get distance")
        # Get propagation distance for this layer
        if i==(scrnNo-1):
            if ht_final == 0:
                # if the final height is infinity, don't propagate any more!
                continue
            else:
                z = abs(ht_final - ht) - z_total
        else:
            z = abs(scrnAlts[i+1] - scrnAlts[i])

        # Update total distance counter
        z_total += z

        # print("Make EField")


        # Do ASP for last layer to next
        EFieldBuf[:] = opticalpropagation.angularSpectrum(
                    EFieldBuf, wavelength,
                    output_pixel_scale, output_pixel_scale, z)

        # logger.debug("Propagation: {}, {} m. Total: {}".format(i, z, z_total))

    return EFieldBuf
