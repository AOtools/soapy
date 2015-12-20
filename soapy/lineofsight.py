"""
A generalised 'line of sight' object, which calculates the resulting phase
or complex amplitude from propogating through the atmosphere in a given
direction.
"""

import numpy
from scipy.interpolate import interp2d

from . import aoSimLib, logger

DTYPE = numpy.float32
CDTYPE = numpy.complex64

class LineOfSight(object):
    def __init__(self, phaseSize=None, propagationDirection="down"):

        self.phaseSize = phaseSize
        # If GS not at infinity, find meta-pupil radii for each layer
        if self.height!=0:
            self.radii = self.findMetaPupilSize(self.height)
        else:
            self.radii = None

        self.allocDataArrays()

        # Check if geometric or physical
        if self.config.propagationMode=="physical":
            if propagationDirection == "up":
                self.makePhase = self.makePhasePhysUp
            else:
                self.makePhase = self.makePhasePhysDown
        else:
            self.makePhase = self.makePhaseGeometric

    # Some attributes for compatability
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

############################################################
# Initialisation routines
    def calcInitParams(self, phaseSize=None):
        print("LOS CIP")
        print("PhaseSize:{}".format(phaseSize))
        self.telDiam = self.simConfig.pupilSize/self.simConfig.pxlScale

        # Convert phase deviation to radians at wfs wavelength.
        # (in nm remember...?)
        self.phs2Rad = 2*numpy.pi/(self.config.wavelength * 10**9)


        # Get the size of the phase required by the system
        if phaseSize==None:
            self.phaseSize = self.simConfig.simSize
        else:
            self.phaseSize = phaseSize

        print("self.phaseSize:{}".format(self.phaseSize))
        # These are the coordinates of the sub-scrn to cut from the phase scrns
        # For each scrn height they will be edited per
        self.scrnCoords = numpy.arange(self.simConfig.scrnSize)



    def allocDataArrays(self):
        """
        Allocate the data arrays the WFS will require

        Determines and allocates the various arrays the WFS will require to
        avoid having to re-alloc memory during the running of the WFS and
        keep it fast. This includes arrays for phase
        and the E-Field across the WFS
        """
        print("allocDataArrays - phaseSize:{}".format(self.phaseSize))
        self.phase = numpy.zeros([self.phaseSize]*2, dtype=DTYPE)
        self.EField = numpy.zeros([self.phaseSize]*2, dtype=CDTYPE)


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
            pos = (numpy.array(self.config.position)
                    *numpy.pi/(3600.0*180.0) )

        #Position of centre of GS metapupil off axis at required height
        GSCent = (numpy.tan(pos) * height)

        return GSCent

    def getMetaPupilPhase(  self, scrn, height, radius=None, simSize=None,
                            pos=None):
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

        xCoords = numpy.linspace(x1, x2-1, self.phaseSize)
        yCoords = numpy.linspace(y1, y2-1, self.phaseSize)
        interpObj = interp2d(
                self.scrnCoords, self.scrnCoords, scrn, copy=False)
        metaPupil = interpObj(xCoords, yCoords)

        return metaPupil
######################################################

    def zeroData(self, **kwargs):
        self.EField[:] = 0
        self.phase[:] = 0

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

    def makePhasePhysDown(self, radii=None, pos=None):
        '''
        Finds total WFS complex amplitude by propagating light down
        phase scrns

        Parameters
            radii (dict, optional): Radii of each meta pupil of each screen height in pixels. If not given uses pupil radius.
            pos (dict, optional): Position of GS in pixels. If not given uses GS position.
        '''

        scrnNo = len(self.scrns)  #Number of layers
        ht = self.atmosConfig.scrnHeights[scrnNo] #Height of highest layer
        delta = (self.simConfig.pxlScale)**-1. #Grid spacing for propagation

        # Get initial Phase for highest scrn and turn to efield
        if radii:
            radius = radii[-1]
        else:
            radius = None

        phase = self.getMetaPupilPhase(
                        self.scrns[scrnNo], ht, radius=radius,
                        pos=pos)
        self.EField[:] = numpy.exp(1j*phase)

        # Loop through remaining scrns - update ht according
        for i in range(scrnNo-1)[::-1]:
            # Get propagation distance for this layer
            z = self.atmosConfig.scrnHeights[i+1] - self.atmosConfig.scrnHeights[i]
            # Do ASP for last layer to next
            self.EField[:] = angularSpectrum(
                        self.EField, self.wfsConfig.wavelength,
                        self.inPxlScale, self.inPxlScale, z)

            # Get phase for this layer
            if radii:
                radius = radii[i]
            else:
                radius = None

            phase = self.getMetaPupilPhase(
                    self.scrns[i], self.atmosConfig.scrnHeights[i],
                    radius=radii[i], pos=pos)

            # Convert phase to radians
            phase *= self.phs2Rad

            # Add phase from this layer
            self.EField *= numpy.exp(1j*phase)

        #If not already at ground, propagate the rest of the way.
        if self.atmosConfig.scrnHeights[0]!=0:
            self.EField[:] = angularSpectrum(
                    self.EField, self.wfsConfig.wavelength,
                    self.inPxlScale, self.outPxlScale,
                    self.atmosConfig.scrnHeights[0]
                    )

        return self.EField

    def makePhasePhysUp(self, altitude, radii=None, pos=None, mask=None):
        '''
        Finds total WFS complex amplitude by propagating light down
        phase scrns

        Parameters
            radii (dict, optional): Radii of each meta pupil of each screen height in pixels. If not given uses pupil radius.
            pos (dict, optional): Position of GS in pixels. If not given uses GS position.
        '''

        scrnNo = len(self.scrns)  #Number of layers
        delta = (self.simConfig.pxlScale)**-1. #Grid spacing for propagation

        #Get initial Phase for ground layer scrn and turn to efield
        if radii==None:
            radius = None
        else:
            radius = radii[0]

        if scrnHeights[0]==0:
            phase = self.getMetaPupilPhase(
                        self.scrns[0], height=0, pos=pos, radius=radius
                        )*mask
        else:
            phase = mask

        self.EField[:] = numpy.exp(1j*phase)

        ht = 0 # Counter to keep track of how high we are
        #Loop through remaining scrns - update ht according
        for i in range(1, scrnNo):
            # Get propagation distance for this layer
            z = (self.atmosConfig.scrnHeights[i]
                    - self.atmosConfig.scrnHeights[i-1])

            # Do ASP for last layer to next
            # Keep same scaling for input andoutput
            self.EField[:] = angularSpectrum(
                        self.EField, self.wfsConfig.wavelength,
                        self.inPxlScale, self.inPxlScale, z )

            # Get phase for this layer
            if radii:
                radius = radii[i]
            else:
                radii = None

            phase = self.getMetaPupilPhase(
                            self.scrns[i], self.atmosConfig.scrnHeights[i],
                            radius=radius, pos=pos)
            # Convert phase to radians
            phase *= self.phs2Rad

            #Add add phase from this layer
            self.EField *= numpy.exp(1j*phase)

        #If not already at altitude, propagate the rest of the way.
        if self.scrnHeights[-1]!=altitude:
            self.EField[:] = angularSpectrum(
                    self.EField, self.wfsConfig.wavelength,
                    delta, delta, altitude-self.atmosConfig.scrnHeights[-1]
                    )
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
        if t!=dict and t!=list:
            scrns = [scrns]
        self.scrns = scrns

        self.zeroData()
        # self.scrns = {}
        #
        # # Convert phase to radians
        # for i in xrange(len(scrns)):
        #     self.scrns[i] = scrns[i].copy()*self.phs2Rad

        self.makePhase(self.radii)

        if numpy.any(correction):
            correction = aoSimLib.zoom(correction, self.phaseSize)
            self.EField *= numpy.exp(-1j*correction*self.phs2Rad)
            self.residual = self.phase - correction
        else:
            self.residual = self.phase
