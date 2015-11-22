"""
A generalised 'line of sight' object, which calculates the resulting phase
or complex amplitude from propogating through the atmosphere in a given
direction.
"""

import numpy
from . import aoSimLib

class LineOfSight(object):
        def __init__(
                self, simConfig, wfsConfig, atmosConfig, lgsConfig=None,
                mask=None):

            self.simConfig = simConfig
            self.wfsConfig = wfsConfig
            self.atmosConfig = atmosConfig
            self.lgsConfig = lgsConfig

            # If supplied use the mask
            if numpy.any(mask):
                self.mask = mask
            else:
                self.mask = aoSimLib.circle(
                        self.simConfig.pupilSize/2., self.simConfig.simSize,
                        )

            self.iMat = False

            # Set from knowledge of atmosphere module
            self.phsWvl = 500e-9

            self.calcInitParams()

            # If GS not at infinity, find meta-pupil radii for each layer
            if self.wfsConfig.GSHeight != 0:

                self.radii = self.findMetaPupilSize(self.wfsConfig.GSHeight)
            else:
                self.radii = None

            # Choose propagation method
            if wfsConfig.propagationMode == "physical":

                self.makePhase = self.makePhasePhysical
                self.physEField = numpy.zeros(
                    (self.simConfig.pupilSize,)*2, dtype=CDTYPE)
            else:
                self.makePhase = self.makePhaseGeo

            self.allocDataArrays()

            self.calcTiltCorrect()
            self.getStatic()

    ############################################################
    # Initialisation routines
        def calcInitParams(self):

            self.telDiam = self.simConfig.pupilSize/self.simConfig.pxlScale

            # Phase power scaling factor for wfs wavelength
            self.r0Scale = self.phsWvl/self.wfsConfig.wavelength

            # These are the coordinates of the sub-scrn to cut from the phase scrns
            # For each scrn height they will be edited per
            self.scrnCoords = numpy.arange(self.simConfig.scrnSize)

        def initFFTs(self):
            pass

        def allocDataArrays(self):
            """
            Allocate the data arrays the WFS will require

            Determines and allocates the various arrays the WFS will require to
            avoid having to re-alloc memory during the running of the WFS and
            keep it fast. This includes arrays for phase
            and the E-Field across the WFS
            """

            self.wfsPhase = numpy.zeros([self.simConfig.simSize]*2, dtype=DTYPE)
            self.EField = numpy.zeros([self.simConfig.simSize]*2, dtype=CDTYPE)

        def initLGS(self):
            """
            Initialises tithe LGS objects for the WFS

            Creates and initialises the LGS objects if the WFS GS is a LGS. This
            included calculating the phases additions which are required if the
            LGS is elongated based on the depth of the elongation and the launch
            position. Note that if the GS is at infinity, elongation is not possible
            and a warning is logged.
            """

            # Choose the correct LGS object, either with physical or geometric
            # or geometric propagation.
            if self.lgsConfig.uplink:
                if  (self.lgsConfig.propagationMode=="phys" or
                        self.lgsConfig.propagationMode=="physical"):
                    self.LGS = LGS.PhysicalLGS( self.simConfig, self.wfsConfig,
                                                self.lgsConfig, self.atmosConfig
                                                )
                else:
                    self.LGS = LGS.GeometricLGS( self.simConfig, self.wfsConfig,
                                                 self.lgsConfig, self.atmosConfig
                                                 )

            else:
                self.LGS = None

            self.lgsLaunchPos = None
            self.elong = 0
            self.elongLayers = 0
            if self.wfsConfig.lgs:
                self.lgsLaunchPos = self.lgsConfig.launchPosition
                # LGS Elongation##############################
                if (self.wfsConfig.GSHeight!=0 and
                        self.lgsConfig.elongationDepth!=0):
                    self.elong = self.lgsConfig.elongationDepth
                    self.elongLayers = self.lgsConfig.elongationLayers

                    #Get Heights of elong layers
                    self.elongHeights = numpy.linspace(
                        self.wfsConfig.GSHeight-self.elong/2.,
                        self.wfsConfig.GSHeight+self.elong/2.,
                        self.elongLayers
                        )

                    #Calculate the zernikes to add
                    self.elongZs = aoSimLib.zernikeArray([2,3,4], self.simConfig.pupilSize)

                    #Calculate the radii of the metapupii at for different elong
                    #Layer heights
                    #Also calculate the required phase addition for each layer
                    self.elongRadii = {}
                    self.elongPos = {}
                    self.elongPhaseAdditions = numpy.zeros(
                        (self.elongLayers, self.simConfig.simSize,
                        self.simConfig.simSize))
                    for i in xrange(self.elongLayers):
                        self.elongRadii[i] = self.findMetaPupilSize(
                                                    float(self.elongHeights[i]))
                        self.elongPhaseAdditions[i] = self.calcElongPhaseAddition(i)
                        self.elongPos[i] = self.calcElongPos(i)

                #If GS at infinity cant do elongation
                elif (self.wfsConfig.GSHeight==0 and
                        self.lgsConfig.elongationDepth!=0):
                    logger.warning("Not able to implement LGS Elongation as GS at infinity")

        def calcTiltCorrect(self):
            pass

        def getStatic(self):
            self.staticData = None

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


        def calcElongPhaseAddition(self, elongLayer):
            """
            Calculates the phase required to emulate layers on an elongated source

            For each 'elongation layer' a phase addition is calculated which
            accounts for the difference in height from the nominal GS height where
            the WFS is focussed, and accounts for the tilt seen if the LGS is
            launched off-axis.

            Parameters:
                elongLayer (int): The number of the elongation layer

            Returns:
                ndarray: The phase addition required for that layer.
            """

            #Calculate the path difference between the central GS height and the
            #elongation "layer"
            #Define these to make it easier
            h = self.elongHeights[elongLayer]
            dh = h - self.wfsConfig.GSHeight
            H = self.lgsConfig.height
            d = numpy.array(self.lgsLaunchPos).astype('float32') * self.telDiam/2.
            D = self.telDiam
            theta = (d.astype("float")/H) - self.wfsConfig.GSPosition

            #for the focus terms....
            focalPathDiff = (2*numpy.pi/self.wfsConfig.wavelength) * ( (
                ( (self.telDiam/2.)**2 + (h**2) )**0.5\
              - ( (self.telDiam/2.)**2 + (H)**2 )**0.5 ) - dh )

            #For tilt terms.....
            tiltPathDiff = (2*numpy.pi/self.wfsConfig.wavelength) * (
                numpy.sqrt( (dh+H)**2. + ( (dh+H)*theta-d-D/2.)**2 )
                + numpy.sqrt( H**2 + (D/2. - d + H*theta)**2 )
                - numpy.sqrt( H**2 + (H*theta - d - D/2.)**2)
                - numpy.sqrt( (dh+H)**2 + (D/2. - d + (dh+H)*theta )**2 )    )


            phaseAddition = numpy.zeros(
                    (  self.simConfig.pupilSize, self.simConfig.pupilSize) )

            phaseAddition +=( (self.elongZs[2]/self.elongZs[2].max())
                                 * focalPathDiff )
            #X,Y tilt
            phaseAddition += ( (self.elongZs[0]/self.elongZs[0].max())
                                *tiltPathDiff[0] )
            phaseAddition += ( (self.elongZs[1]/self.elongZs[1].max())
                                *tiltPathDiff[1])

            pad = ((self.simConfig.simPad,)*2, (self.simConfig.simPad,)*2)
            phaseAddition = numpy.pad(phaseAddition, pad, mode="constant")

            return phaseAddition

        def calcElongPos(self, elongLayer):
            """
            Calculates the difference in GS position for each elongation layer
            only makes a difference if LGS launched off-axis

            Parameters:
                elongLayer (int): which elongation layer

            Returns:
                float: The effect position of that layer GS
            """

            h = self.elongHeights[elongLayer]       #height of elonglayer
            dh = h-self.wfsConfig.GSHeight          #delta height from GS Height
            H = self.wfsConfig.GSHeight               #Height of GS

            #Position of launch in m
            xl = numpy.array(self.lgsLaunchPos) * self.telDiam/2.

            #GS Pos in radians
            GSPos=numpy.array(self.wfsConfig.GSPosition)*numpy.pi/(3600.0*180.0)

            #difference in angular Pos for that height layer in rads
            theta_n = GSPos - ((dh*xl)/ (H*(H+dh)))

            return theta_n

    #############################################################

    #############################################################
    #Phase stacking routines for a WFS frame

        def getMetaPupilPos(self, height, GSPos=None):
            '''
            Finds the centre of a metapupil at a given height,
            when offset by a given angle in arsecs, in metres from the ()

            Arguments:
                height (float): Height of the layer in metres
                GSPos (tuple, optional):  The angular position of the GS in radians.
                                        If not set, will use the WFS position

            Returns:
                ndarray: The position of the centre of the metapupil in metres
            '''
            #if no GSPos given, use system pos and convert into radians
            if not numpy.any(GSPos):
                GSPos = (   numpy.array(self.wfsConfig.GSPosition)
                            *numpy.pi/(3600.0*180.0) )

            #Position of centre of GS metapupil off axis at required height
            GSCent = (numpy.tan(GSPos) * height)

            return GSCent

        def getMetaPupilPhase(  self, scrn, height, radius=None, simSize=None,
                                GSPos=None):
            '''
            Returns the phase across a metaPupil at some height and angular
            offset in arcsec. Interpolates phase to size of the pupil if cone
            effect is required

            Parameters:
                scrn (ndarray): An array representing the phase screen
                height (float): Height of the phase screen
                radius (float, optional): Radius of the meta-pupil. If not set, will use system pupil size.
                simSize (ndarray, optional): Size of screen to return. If not set, will use system pupil size.
                GSPos (tuple, optional): Angular position of guide star. If not set will use system position.

            Return:
                ndarray: The meta pupil at the specified height
            '''

            #If no size of metapupil given, use system pupil size
            if not simSize:
                simSize = self.simConfig.simSize

            #If the radius is 0, then 0 phase is returned
            if radius==0:
                return numpy.zeros((simSize, simSize))


            GSCent = self.getMetaPupilPos(height, GSPos) * self.simConfig.pxlScale

            logger.debug("GSCent {}".format(GSCent))
            scrnX, scrnY = scrn.shape
            #If the GS is not at infinity, take into account cone effect
            if self.wfsConfig.GSHeight!=0:
                fact = float(2*radius)/self.simConfig.pupilSize
            else:
                fact=1

            x1 = scrnX/2. + GSCent[0] - fact*simSize/2.0
            x2 = scrnX/2. + GSCent[0] + fact*simSize/2.0
            y1 = scrnY/2. + GSCent[1] - fact*simSize/2.0
            y2 = scrnY/2. + GSCent[1] + fact*simSize/2.0

            logger.debug("WFS Scrn Coords - ({0}:{1}, {2}:{3})".format(
                    x1,x2,y1,y2))

            if ( x1 < 0 or x2 > scrnX or y1 < 0 or y2 > scrnY):
                raise ValueError(
                        "GS separation requires larger screen size. \nheight: {3}, GSCent: {0}, scrnSize: {1}, simSize: {2}".format(
                                GSCent, scrn.shape, simSize, height) )


            if (x1.is_integer() and x2.is_integer()
                    and y1.is_integer() and y2.is_integer()):
                #Old, simple integer based solution
                metaPupil= scrn[ x1:x2, y1:y2]
            else:
                #If points are float, must interpolate. -1 as linspace goes to number
                xCoords = numpy.linspace(x1, x2-1, simSize)
                yCoords = numpy.linspace(y1, y2-1, simSize)
                interpObj = interp2d(
                        self.scrnCoords, self.scrnCoords, scrn, copy=False)
                metaPupil = interpObj(xCoords, yCoords)

            return metaPupil

        def makePhaseGeo(self, radii=None, GSPos=None):
            '''
            Creates the total phase on a wavefront sensor which
            is offset by a given angle

            Parameters
                radii (dict, optional): Radii of each meta pupil of each screen height in pixels. If not given uses pupil radius.
                GSPos (dict, optional): Position of GS in pixels. If not given uses GS position
            '''

            for i in self.scrns:
                logger.debug("Layer: {}".format(i))
                if radii:
                    phase = self.getMetaPupilPhase(
                                self.scrns[i], self.atmosConfig.scrnHeights[i],
                                radius=radii[i], GSPos=GSPos)
                else:
                    phase = self.getMetaPupilPhase(
                                self.scrns[i], self.atmosConfig.scrnHeights[i],
                                GSPos=GSPos)

                self.wfsPhase += phase

            self.EField[:] = numpy.exp(1j*self.wfsPhase)


        def makePhasePhysical(self, radii=None, GSPos=None):
            '''
            Finds total WFS complex amplitude by propagating light down
            phase scrns

            Parameters
                radii (dict, optional): Radii of each meta pupil of each screen height in pixels. If not given uses pupil radius.
                GSPos (dict, optional): Position of GS in pixels. If not given uses GS position.
            '''

            scrnNo = len(self.scrns)-1  #Number of layers (0 indexed)
            ht = self.atmosConfig.scrnHeights[scrnNo] #Height of highest layer
            delta = (self.simConfig.pxlScale)**-1. #Grid spacing for propagation

            #Get initial Phase for highest scrn and turn to efield
            if radii:
                phase1 = self.getMetaPupilPhase(
                            self.scrns[scrnNo], ht, radius=radii[scrnNo],
                            GSPos=GSPos)
                            #pupilSize=2*self.simConfig.pupilSize, GSPos=GSPos )
            else:
                phase1 = self.getMetaPupilPhase(self.scrns[scrnNo], ht,
                            GSPos=GSPos)
                            #pupilSize=2*self.simConfig.pupilSize, GSPos=GSPos)

            self.EField[:] = numpy.exp(1j*phase1)
            #Loop through remaining scrns in reverse order - update ht accordingly
            for i in range(scrnNo)[::-1]:
                #Get propagation distance for this layer
                z = ht - self.atmosConfig.scrnHeights[i]
                ht -= z
                #Do ASP for last layer to next
                self.EField[:] = angularSpectrum(
                            self.EField, self.wfsConfig.wavelength,
                            delta, delta, z )

                # Get phase for this layer
                if radii:
                    phase = self.getMetaPupilPhase(
                                self.scrns[i], self.atmosConfig.scrnHeights[i],
                                radius=radii[i], GSPos=GSPos)
                                # pupilSize=2*self.simConfig.pupilSize)
                else:
                    phase = self.getMetaPupilPhase(
                                self.scrns[i], self.atmosConfig.scrnHeights[i],
                                #pupilSize=2*self.simConfig.pupilSize,
                                GSPos=GSPos)

                #Add add phase from this layer
                self.EField *= numpy.exp(1j*phase)

            #If not already at ground, propagate the rest of the way.
            if self.atmosConfig.scrnHeights[0]!=0:
                self.EField[:] = angularSpectrum(
                        self.EField, self.wfsConfig.wavelength,
                        delta, delta, ht
                        )

            # Multiply EField by aperture
            # self.EField[:] *= self.mask
            # self.EField[:] = self.physEField[
            #                    self.simConfig.pupilSize/2.:
            #                    3*self.simConfig.pupilSize/2.,
            #                    self.simConfig.pupilSize/2.:
            #                    3*self.simConfig.pupilSize/2.] * self.mask

    ######################################################

        def readNoise(self, dPlaneArray):
            dPlaneArray += numpy.random.normal((self.maxFlux/self.wfsConfig.SNR),
            0.1*self.maxFlux/self.wfsConfig.SNR, dPlaneArray.shape).clip(0,self.maxFlux).astype(self.dPlaneType)

        def photonNoise(self):
            pass

        def iMatFrame(self, phs):
            '''
            Runs an iMat frame - essentially gives slopes for given "phs" so
            useful for other stuff too!

            Parameters:
                phs (ndarray):  The phase to apply to the WFS. Should be of shape
                                (simConfig.simSize, simConfig.simSize)
            Returns:
                ndarray: A 1-d array of WFS measurements
            '''
            self.iMat=True
            #Set "removeTT" to false while we take an iMat
            removeTT = self.wfsConfig.removeTT
            self.wfsConfig.removeTT=False

            self.zeroData()
            self.EField[:] =  numpy.exp(1j*phs)#*self.r0Scale)
            self.calcFocalPlane()
            self.makeDetectorPlane()
            self.calculateSlopes()

            self.wfsConfig.removeTT = removeTT
            self.iMat=False

            return self.slopes

        def zeroPhaseData(self):
            self.EField[:] = 0
            self.wfsPhase[:] = 0


        def frame(self, scrns, correction=None, read=True, iMatFrame=False):
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
                iMatFrame (bool, optional): If True, will assume an interaction matrix is being measured. Turns off some AO loop features before running

            Returns:
                ndarray: WFS Measurements
            '''

           #If iMatFrame, turn off unwanted effects
            if iMatFrame:
                self.iMat = True
                removeTT = self.wfsConfig.removeTT
                self.wfsConfig.removeTT = False
                if self.wfsConfig.lgs:
                    elong = self.elong
                self.elong = 0


            #If scrns is not dict or list, assume array and put in list
            t = type(scrns)
            if t!=dict and t!=list:
                scrns = [scrns]

            self.zeroData(detector=read, inter=False)
            self.scrns = {}
            #Scale phase to WFS wvl
            for i in xrange(len(scrns)):
                self.scrns[i] = scrns[i].copy()*self.r0Scale


            #If LGS elongation simulated
            if self.wfsConfig.lgs and self.elong!=0:
                for i in xrange(self.elongLayers):
                    self.zeroPhaseData()

                    self.makePhase(self.elongRadii[i], self.elongPos[i])
                    self.uncorrectedPhase = self.wfsPhase.copy()
                    self.EField *= numpy.exp(1j*self.elongPhaseAdditions[i])
                    if numpy.any(correction):
                        self.EField *= numpy.exp(-1j*correction*self.r0Scale)
                    self.calcFocalPlane(intensity=self.lgsConfig.naProfile[i])

            #If no elongation
            else:
                #If imate frame, dont want to make it off-axis
                if iMatFrame:
                    try:
                        self.EField[:] = numpy.exp(1j*scrns[0])
                    except ValueError:
                        raise ValueError("If iMat Frame, scrn must be ``simSize``")
                else:
                    self.makePhase(self.radii)

                self.uncorrectedPhase = self.wfsPhase.copy()
                if numpy.any(correction):
                    self.EField *= numpy.exp(-1j*correction*self.r0Scale)
                self.calcFocalPlane()

            if read:
                self.makeDetectorPlane()
                self.calculateSlopes()
                self.zeroData(detector=False)

            #Turn back on stuff disabled for iMat
            if iMatFrame:
                self.iMat=False
                self.wfsConfig.removeTT = removeTT
                if self.wfsConfig.lgs:
                    self.elong = elong

            # Check that slopes aint `nan`s. Set to 0 if so
            if numpy.any(numpy.isnan(self.slopes)):
                self.slopes[:] = 0

            return self.slopes
