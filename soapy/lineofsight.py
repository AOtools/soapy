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
from scipy.interpolate import interp2d

from . import aoSimLib, logger, opticalPropagationLib

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
        propagationDirection (str, optional): Direction of light propagation, either `"up"` or `"down"`
        outPxlScale (float, optional): The EField pixel scale required at the output (m/pxl)
        nOutPxls (int, optional): Number of pixels to return in EFIeld
        mask (ndarray, optional): Mask to apply at the *beginning* of propagation
        metaPupilPos (list, dict, optional): A list or dictionary of the meta pupil position at each turbulence layer height ub metres. If None, works it out from GS position.
    """
    def __init__(
            self, config, soapyConfig,
            propagationDirection="down", outPxlScale=None,
            nOutPxls=None, mask=None, metaPupilPos=None):

        self.config = config
        self.simConfig = soapyConfig.sim
        self.atmosConfig = soapyConfig.atmos
        self.soapyConfig = soapyConfig

        self.mask = mask

        self.calcInitParams(outPxlScale, nOutPxls)

        self.propagationDirection = propagationDirection

        # If GS not at infinity, find meta-pupil radii for each layer
        if self.height!=0:
            self.radii = self.findMetaPupilSizes(self.height)
        else:
            self.radii = None

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
    def calcInitParams(self, outPxlScale=None, nOutPxls=None):
        """
        Calculates some parameters required later

        Parameters:
            outPxlScale (float): Pixel scale of required phase/EField (metres/pxl)
            nOutPxls (int): Size of output array in pixels
        """
        # Convert phase deviation to radians at wfs wavelength.
        # (in nm remember...?)
        self.phs2Rad = 2*numpy.pi/(self.config.wavelength * 10**9)

        self.telDiam = float(self.simConfig.pupilSize) / self.simConfig.pxlScale

        # Get the size of the phase required by the system
        self.inPxlScale = self.simConfig.pxlScale**-1

        if outPxlScale is None:
            self.outPxlScale = self.simConfig.pxlScale**-1
        else:
            self.outPxlScale = outPxlScale

        if nOutPxls is None:
            self.nOutPxls = self.simConfig.simSize
        else:
            self.nOutPxls = nOutPxls

        if self.mask is not None:
            self.outMask = aoSimLib.zoom(
                    self.mask, self.nOutPxls).round()


    def allocDataArrays(self):
        """
        Allocate the data arrays the LOS will require

        Determines and allocates the various arrays the LOS will require to
        avoid having to re-alloc memory during the running of the LOS and
        keep it fast. This includes arrays for phase
        and the E-Field across the LOS
        """
        self.phase = numpy.zeros([self.nOutPxls]*2, dtype=DTYPE)
        self.EField = numpy.zeros([self.nOutPxls]*2, dtype=CDTYPE)


    def findMetaPupilSizes(self, GSHeight):
        '''
        Evaluates the sizes of the effective metePupils
        at each screen height - if a GS of finite height is used.

        Parameters:
            GSHeight (float): The height of the GS in metres

        Returns:
            dict : A dictionary containing the radii of a meta-pupil at each screen height in phase pixels
        '''

        radii = {}
        for i in xrange(self.atmosConfig.scrnNo):
            radii[i] = self.simConfig.pxlScale * self.calcMetaPupilSize(
                        self.atmosConfig.scrnHeights[i], GSHeight)

        return radii


    def calcMetaPupilSize(self, scrnHeight, GSHeight):
        """
        Calculates the radius of a meta pupil at a altitude layer, of a GS at a give altitude
        
        Parameters:
            scrnHeight (float): Altitude of meta-pupil
            GSHeight (float): Altitude of guide star
        
        Returns:
            float: Radius of metapupil in metres
          
        """
        # If GS at infinity, radius is telescope radius
        if self.height == 0:
            return self.soapyConfig.tel.telDiam/2.
        
        # If scrn is above LGS, radius is 0
        if scrnHeight >= GSHeight:
            return 0
        
        # Find radius of metaPupil geometrically (fraction of pupil at
        # Ground Layer)
        radius = (self.soapyConfig.tel.telDiam/2.) * (1-(float(scrnHeight)/GSHeight))


        
        return radius
        
    #############################################################
    # Phase stacking routines for a WFS frame
    def getMetaPupilPos(self, height, apos=None):
        '''
        Finds the centre of a metapupil at a given height,
        when offset by a given angle in arsecs, in metres from the central position

        Parameters:
            height (float): Height of the layer in metres
            apos (ndarray, optional):  The angular position of the GS in asec. If not set, will use the config position

        Returns:
            ndarray: The position of the centre of the metapupil in metres
        '''
        # if no pos given, use system pos and convert into radians
        if not numpy.any(apos):
            pos = (numpy.array(self.position)).astype('float')

        pos *= ASEC2RAD

        # Position of centre of GS metapupil off axis at required height
        GSCent = (numpy.tan(pos) * height)

        return GSCent

    def getMetaPupilPhase(
            self, scrn, height, radius=None,  apos=None, pos=None):
        '''
        Returns the phase across a metaPupil at some height and angular
        offset in arcsec. Interpolates phase to size of the pupil if cone
        effect is required

        Parameters:
            scrn (ndarray): An array representing the phase screen
            height (float): Height of the phase screen
            radius (float, optional): Radius of the meta-pupil. If not set, will use system pupil size.
            apos (ndarray, optional): X, Y angular position of the guide star in asecs, otherwise will use that set in config or 'pos'
            pos (ndarray, optional): X, Y central position of the metapupil in metres. If None, then config used to calculate it from config pos, or 'apos'.

        Return:
            ndarray: The meta pupil at the specified height
        '''

        # If the radius is 0, then 0 phase is returned
        if radius == 0:
            return numpy.zeros((simSize, simSize))

        if apos is not None:
            GSCent = self.getMetaPupilPos(height, apos) * self.simConfig.pxlScale
        elif pos is not None:
            GSCent = pos * self.simConfig.pxlScale
        else:
            GSCent = self.getMetaPupilPos(height) * self.simConfig.pxlScale

        # Find the size in metres of phase that is required
        self.phaseRadius = (self.outPxlScale * self.nOutPxls)/2.
        # And a corresponding coordinate on the phase screen
        self.phaseCoord = int(round(self.phaseRadius * self.simConfig.pxlScale))

        logger.debug('phaseCoord:{}'.format(self.phaseCoord))
        # The sizes of the phase screen
        scrnX, scrnY = scrn.shape

        # If the GS is not at infinity, take into account cone effect
        if radius != None:
            fact = float(2*radius)/self.simConfig.pupilSize
        else:
            fact = 1

        simSize = self.simConfig.simSize
        x1 = scrnX/2. + GSCent[0] - fact * self.phaseCoord
        x2 = scrnX/2. + GSCent[0] + fact * self.phaseCoord
        y1 = scrnY/2. + GSCent[1] - fact * self.phaseCoord
        y2 = scrnY/2. + GSCent[1] + fact * self.phaseCoord

        logger.debug("LoS Scrn Coords - ({0}:{1}, {2}:{3})".format(
                x1,x2,y1,y2))
        if ( x1 < 0 or x2 > scrnX or y1 < 0 or y2 > scrnY):
            logger.warning("GS separation requires larger screen size. \nheight: {3}, GSCent: {0}, \nscrnSize: {1}, phaseCoord, {8}, simSize: {2}, fact: {9}\nx1: {4},x2: {5}, y1: {6}, y2: {7}".format(
                    GSCent, scrn.shape, simSize, height, x1, x2, y1, y2, self.phaseCoord, fact))
            raise ValueError("Requested phase exceeds phase screen size. See log warnings.")

        xCoords = numpy.linspace(x1, x2-1, self.nOutPxls)
        yCoords = numpy.linspace(y1, y2-1, self.nOutPxls)
        scrnCoords = numpy.arange(scrnX)
        interpObj = interp2d(
                scrnCoords, scrnCoords, scrn, copy=False)
        metaPupil = interpObj(xCoords, yCoords)

        self.metaPupil = metaPupil
        return metaPupil
######################################################

    def zeroData(self, **kwargs):
        """
        Sets the phase and complex amp data to zero
        """
        self.EField[:] = 0
        self.phase[:] = 0

    def makePhase(self, radii=None, apos=None):
        """
        Generates the required phase or EField. Uses difference approach depending on whether propagation is geometric or physical
        (makePhaseGeometric or makePhasePhys respectively)

        Parameters:
            radii (dict, optional): Radii of each meta pupil of each screen height in pixels. If not given uses pupil radius.
            apos (ndarray, optional):  The angular position of the GS in radians. If not set, will use the config position
        """
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

        for i in range(len(self.scrns)):
            logger.debug("Layer: {}".format(i))
            if radii is None:
                radius = None
            else:
                radius = radii[i]

            if self.metaPupilPos is None:
                pos = None
            else:
                pos = self.metaPupilPos[i]

            phase = self.getMetaPupilPhase(
                    self.scrns[i], self.atmosConfig.scrnHeights[i],
                    pos=pos, radius=radius)

            self.phase += phase

        # Convert phase to radians
        self.phase *= self.phs2Rad

        # Change sign if propagating up
        if self.propagationDirection == 'up':
            self.phase *= -1

        self.EField[:] = numpy.exp(1j*self.phase)

        return self.EField

    def makePhasePhys(self, radii=None, apos=None):
        '''
        Finds total line of sight complex amplitude by propagating light through phase screens

        Parameters:
            radii (dict, optional): Radii of each meta pupil of each screen height in pixels. If not given uses pupil radius.
            apos (ndarray, optional):  The angular position of the GS in radians. If not set, will use the config position
        '''

        scrnNo = len(self.scrns)
        z_total = 0
        scrnRange = range(0, scrnNo)

        # Get initial up/down dependent params
        if self.propagationDirection == "up":
            ht = 0
            ht_final = self.config.height
            if ht_final==0:
                raise ValueError("Can't propagate up to infinity")
            scrnAlts = self.atmosConfig.scrnHeights

            self.EFieldBuf = self.outMask.copy().astype(CDTYPE)
            logger.debug("Create EField Buf of mask")

        else:
            ht = self.atmosConfig.scrnHeights[scrnNo-1]
            ht_final = 0
            scrnRange = scrnRange[::-1]
            scrnAlts = self.atmosConfig.scrnHeights[::-1]
            self.EFieldBuf = numpy.exp(
                    1j*numpy.zeros((self.nOutPxls,)*2)).astype(CDTYPE)
            logger.debug("Create EField Buf of zero phase")

        # Propagate to first phase screen (if not already there)
        if ht!=scrnAlts[0]:
            logger.debug("propagate to first phase screen")
            z = abs(scrnAlts[0] - ht)
            self.EFieldBuf[:] = opticalPropagationLib.angularSpectrum(
                        self.EFieldBuf, self.config.wavelength,
                        self.outPxlScale, self.outPxlScale, z)

        # Go through and propagate between phase screens
        for i in scrnRange:
            # Check optional radii and position
            if radii is None:
                radius = None
            else:
                radius = radii[i]

            if self.metaPupilPos is None:
                pos = None
            else:
                pos = self.metaPupilPos[i]

            # Get phase for this layer
            phase = self.getMetaPupilPhase(
                    self.scrns[i],
                    self.atmosConfig.scrnHeights[i], radius=radius, pos=pos)

            # Convert phase to radians
            phase *= self.phs2Rad

            # Change sign if propagating up
            if self.propagationDirection == 'up':
                self.phase *= -1

            # Get propagation distance for this layer
            if i==(scrnNo-1):
                z = abs(ht_final - ht) - z_total
            else:
                z = abs(scrnAlts[i+1] - scrnAlts[i])

            # Update total distance counter
            z_total += z

            # Apply phase to EField
            self.EFieldBuf *= numpy.exp(1j*phase)

            # Do ASP for last layer to next
            self.EFieldBuf[:] = opticalPropagationLib.angularSpectrum(
                        self.EFieldBuf, self.config.wavelength,
                        self.outPxlScale, self.outPxlScale, z)

            logger.debug("Propagation: {}, {} m. Total: {}".format(i, z, z_total))

            self.EField[:] = self.EFieldBuf

        return self.EField

    def performCorrection(self, correction):
        """
        Corrects the aberrated line of sight with some given correction phase
        
        Parameters:
            correction (list or ndarray): either 2-d array describing correction, or list of correction arrays
        """
        # If just an arary, put in list
        if isinstance(correction, numpy.ndarray):
            correction = [correction]
        
        for corr in correction:
            # If correction is a standard ndarray, assume at ground
            if hasattr(corr, "altitude"):
                altitude = corr.altitude
            else:
                altitude = 0
                   
            # Cut out the bit of the correction we need
            metaPupilRadius = self.calcMetaPupilSize(
                        altitude, self.height) * self.simConfig.pxlScale
            corr = self.getMetaPupilPhase(corr, altitude, radius=metaPupilRadius)
            
            # Correct EField
            self.EField *= numpy.exp(-1j * corr * self.phs2Rad)
            
            # self.phase -= corr * self.phs2Rad
            
            # Also correct phase in case its required
            self.residual = self.phase/self.phs2Rad - corr
           
            self.phase = self.residual * self.phs2Rad

    def frame(self, scrns=None, correction=None):
        '''
        Runs one frame through a line of sight

        Finds the phase or complex amplitude through line of sight for a
        single simulation frame, with a given set of phase screens and
        some optional correction.

        Parameters:
            scrns (list): A list or dict containing the phase screens
            correction (ndarray, optional): The correction term to take from the phase screens before the WFS is run.
            read (bool, optional): Should the WFS be read out? if False, then WFS image is calculated but slopes not calculated. defaults to True.

        Returns:
            ndarray: WFS Measurements
        '''

        self.zeroData()

        if scrns is not None:
        
            #If scrns is not dict or list, assume array and put in list
            t = type(scrns)
            if t != dict and t != list:
                scrns = [scrns]
            self.scrns = scrns

            self.makePhase(self.radii)
        
        self.residual = self.phase        
        if correction is not None:
            self.performCorrection(correction)

        return self.residual
