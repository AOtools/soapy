"""
A generalised 'line of sight' object, which calculates the resulting phase
or complex amplitude from propogating through the atmosphere in a given
direction.
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

class LineOfSight(object):
    """
    A "Line of sight" through a number of turbulence layers in the atmosphere, observing ing a given direction.

    Parameters:
        config: The soapy config for the line of sight
        simConfig: The soapy simulation config
        atmosConfig: The soapy atmosphere config
        propagationDirection (str): Direction of light propagation, either `"up"` or `"down"`
        outPxlScale (float): The EField pixel scale required at the output (m/pxl)
        mask (ndarray): Mask to apply at the *beginning* of propagation
    """
    def __init__(
            self, config, simConfig, atmosConfig,
            propagationDirection="down", outPxlScale=None,
            nOutPxls=None, mask=None):

        self.config = config
        self.simConfig = simConfig
        self.atmosConfig = atmosConfig

        self.mask = mask

        self.calcInitParams(outPxlScale, nOutPxls)

        self.propagationDirection = propagationDirection

        # If GS not at infinity, find meta-pupil radii for each layer
        if self.height!=0:
            self.radii = self.findMetaPupilSize(self.height)
        else:
            self.radii = None

        self.allocDataArrays()


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

        print("self.outPxlScale: {}".format(self.outPxlScale))
        print("self.mask: {}".format(self.mask))
        print("self.nOutPxls:{}".format(self.nOutPxls))
        if self.mask is not None:
            self.outMask = aoSimLib.zoom(
                    self.mask, self.nOutPxls).round()


    def allocDataArrays(self):
        """
        Allocate the data arrays the WFS will require

        Determines and allocates the various arrays the WFS will require to
        avoid having to re-alloc memory during the running of the WFS and
        keep it fast. This includes arrays for phase
        and the E-Field across the WFS
        """
        print("allocDataArrays - phaseSize:{}".format(self.nOutPxls))
        self.phase = numpy.zeros([self.nOutPxls]*2, dtype=DTYPE)
        self.EField = numpy.zeros([self.nOutPxls]*2, dtype=CDTYPE)


    def findMetaPupilSize(self, GSHeight):
        '''
        Evaluates the sizes of the effective metePupils
        at each screen height if an GS of finite height is used.

        Parameters:
            GSHeight (float): The height of the GS in metres

        Returns:
            dict : A dictionary containing the radii of a meta-pupil at each screen height
        '''

        radii={}

        for i in xrange(self.atmosConfig.scrnNo):
            #Find radius of metaPupil geometrically (fraction of pupil at
            # Ground Layer)
            radius = (self.simConfig.pupilSize/2.) * (
                    1-(float(self.atmosConfig.scrnHeights[i])/GSHeight))
            radii[i]= radius

            #If scrn is above LGS, radius is 0
            if self.atmosConfig.scrnHeights[i]>=GSHeight:
                radii[i]=0

        return radii


 #############################################################
#Phase stacking routines for a WFS frame

    def getMetaPupilPos(self, height, pos=None):
        '''
        Finds the centre of a metapupil at a given height,
        when offset by a given angle in arsecs, in metres from the ()

        Arguments:
            height (float): Height of the layer in metres
            pos (tuple, optional):  The angular position of the GS in radians.
                                    If not set, will use the WFS position

        Returns:
            ndarray: The position of the centre of the metapupil in metres
        '''
        #if no pos given, use system pos and convert into radians
        if not numpy.any(pos):
            pos = (numpy.array(self.position)
                    *numpy.pi/(3600.0*180.0) )

        #Position of centre of GS metapupil off axis at required height
        GSCent = (numpy.tan(pos) * height)

        return GSCent

    def getMetaPupilPhase(
            self, scrn, height, radius=None, simSize=None, pos=None):
        '''
        Returns the phase across a metaPupil at some height and angular
        offset in arcsec. Interpolates phase to size of the pupil if cone
        effect is required

        Parameters:
            scrn (ndarray): An array representing the phase screen
            height (float): Height of the phase screen
            radius (float, optional): Radius of the meta-pupil. If not set, will use system pupil size.
            pos (tuple, optional): Angular position of guide star. If not set will use system position.

        Return:
            ndarray: The meta pupil at the specified height
        '''

        # If the radius is 0, then 0 phase is returned
        if radius==0:
            return numpy.zeros((simSize, simSize))


        GSCent = self.getMetaPupilPos(height, pos) * self.simConfig.pxlScale

        scrnX, scrnY = scrn.shape
        # If the GS is not at infinity, take into account cone effect
        if radius!=None:
            fact = float(2*radius)/self.simConfig.pupilSize
        else:
            fact = 1
        simSize = self.simConfig.simSize
        x1 = scrnX/2. + GSCent[0] - fact*simSize/2.0
        x2 = scrnX/2. + GSCent[0] + fact*simSize/2.0
        y1 = scrnY/2. + GSCent[1] - fact*simSize/2.0
        y2 = scrnY/2. + GSCent[1] + fact*simSize/2.0

        logger.debug("LoS Scrn Coords - ({0}:{1}, {2}:{3})".format(
                x1,x2,y1,y2))

        if ( x1 < 0 or x2 > scrnX or y1 < 0 or y2 > scrnY):
            raise ValueError(
                    "GS separation requires larger screen size. \nheight: {3}, GSCent: {0}, scrnSize: {1}, simSize: {2}".format(
                            GSCent, scrn.shape, simSize, height) )

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
        self.EField[:] = 0
        self.phase[:] = 0

    def makePhase(self, radii=None, pos=None):
        # Check if geometric or physical
        if self.config.propagationMode == "physical":
            return self.makePhasePhys(radii, pos)
            # if self.propagationDirection == "up":
            #     return self.makePhasePhysUp(radii, pos)
            # else:
            #     return self.makePhasePhysDown(radii, pos)
        else:
            return self.makePhaseGeometric(radii, pos)

    def makePhaseGeometric(self, radii=None, pos=None):
        '''
        Creates the total phase on a wavefront sensor which
        is offset by a given angle

        Parameters
            radii (dict, optional): Radii of each meta pupil of each screen height in pixels. If not given uses pupil radius.
            pos (dict, optional): Position of GS in pixels. If not given uses GS position
        '''

        for i in range(len(self.scrns)):
            logger.debug("Layer: {}".format(i))
            if radii:
                phase = self.getMetaPupilPhase(
                            self.scrns[i], self.atmosConfig.scrnHeights[i],
                            radius=radii[i], pos=pos)
            else:
                phase = self.getMetaPupilPhase(
                            self.scrns[i], self.atmosConfig.scrnHeights[i],
                            pos=pos)

            self.phase += phase

        # Convert phase to radians
        self.phase *= self.phs2Rad

        self.EField[:] = numpy.exp(1j*self.phase)

        return self.EField

    def makePhasePhys(self, radii=None, pos=None):
        '''
        Finds total WFS complex amplitude by propagating light through
        phase scrns

        Parameters
            radii (dict, optional): Radii of each meta pupil of each screen height in pixels. If not given uses pupil radius.
            pos (dict, optional): Position of GS in pixels. If not given uses GS position.
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

        print("ScrnRange:{}".format(scrnRange))
        print("ScrnAlts: {}".format(scrnAlts))
        # Go through and propagate between phase screens
        for i in scrnRange:
            # Get phase for this layer
            if radii:
                radius = radii[i]
            else:
                radius = None
            phase = self.getMetaPupilPhase(
                    self.scrns[i],
                    self.atmosConfig.scrnHeights[i], radius=radius, pos=pos)

            # Get propagation distance for this layer
            if i==(scrnNo-1):
                z = abs(ht_final - ht) - z_total
            else:
                z = abs(scrnAlts[i+1] - scrnAlts[i])

            # Update total distance counter
            z_total += z

            # Apply phase to EField
            self.EFieldBuf *= numpy.exp(1j*phase*self.phs2Rad)

            # Do ASP for last layer to next
            self.EFieldBuf[:] = opticalPropagationLib.angularSpectrum(
                        self.EFieldBuf, self.config.wavelength,
                        self.outPxlScale, self.outPxlScale, z)

            logger.debug("Propagation: {}, {} m. Total: {}".format(i, z, z_total))

            self.EField[:] = self.EFieldBuf

        return self.EField

    def frame(self, scrns, correction=None):
        '''
        Runs one WFS frame

        Runs a single frame of the WFS with a given set of phase screens and
        some optional correction. If elongation is set, will run the phase
        calculating and focal plane making methods multiple times for a few
        different heights of LGS, then sum these onto a ``wfsDetectorPlane``.

        Parameters:
            scrns (list): A list or dict containing the phase screens
            correction (ndarray, optional): The correction term to take from the phase screens before the WFS is run.
            read (bool, optional): Should the WFS be read out? if False, then WFS image is calculated but slopes not calculated. defaults to True.

        Returns:
            ndarray: WFS Measurements
        '''

        #If scrns is not dict or list, assume array and put in list
        t = type(scrns)
        if t != dict and t != list:
            scrns = [scrns]
        self.scrns = scrns

        self.zeroData()
        self.makePhase(self.radii)

        if numpy.any(correction):
            correction = aoSimLib.zoom(correction, self.nOutPxls)
            self.EField *= numpy.exp(-1j*correction*self.phs2Rad)
            self.residual = self.phase/self.phs2Rad- correction
        else:
            self.residual = self.phase

        return self.residual
