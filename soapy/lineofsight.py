"""
A generalised 'line of sight' object, which calculates the resulting phase
or complex amplitude from propogating through the atmosphere in a given
direction.
"""

import numpy

from . import aoSimLib, logger

class LineOfSight(object):
        def __init__(
                self, simConfig, wfsConfig, atmosConfig,
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

    ############################################################
    # Initialisation routines
        def calcInitParams(self):

            self.telDiam = self.simConfig.pupilSize/self.simConfig.pxlScale

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

            self.wfsPhase = numpy.zeros([self.simConfig.simSize]*2, dtype=DTYPE)
            self.EField = numpy.zeros([self.simConfig.simSize]*2, dtype=CDTYPE)


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
                pos = (   numpy.array(self.wfsConfig.GSPosition)
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
                simSize (ndarray, optional): Size of screen to return. If not set, will use system pupil size.
                pos (tuple, optional): Angular position of guide star. If not set will use system position.

            Return:
                ndarray: The meta pupil at the specified height
            '''

            #If no size of metapupil given, use system pupil size
            if not simSize:
                simSize = self.simConfig.simSize

            #If the radius is 0, then 0 phase is returned
            if radius==0:
                return numpy.zeros((simSize, simSize))


            GSCent = self.getMetaPupilPos(height, pos) * self.simConfig.pxlScale

            scrnX, scrnY = scrn.shape
            #If the GS is not at infinity, take into account cone effect
            if radius!=None:
                fact = float(2*radius)/self.simConfig.pupilSize
            else:
                fact = 1

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


    ######################################################
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

 


            #If scrns is not dict or list, assume array and put in list
            t = type(scrns)
            if t!=dict and t!=list:
                scrns = [scrns]

            self.zeroData(detector=read, inter=False)
            self.scrns = {}
            #Scale phase to WFS wvl
            for i in xrange(len(scrns)):
                self.scrns[i] = scrns[i].copy()*self.r0Scale

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






class LineOfSight_Geometric(LineOfSight):
        def makePhase(self, radii=None, pos=None):
            '''
            Creates the total phase on a wavefront sensor which
            is offset by a given angle

            Parameters
                radii (dict, optional): Radii of each meta pupil of each screen height in pixels. If not given uses pupil radius.
                pos (dict, optional): Position of GS in pixels. If not given uses GS position
            '''

            for i in self.scrns:
                logger.debug("Layer: {}".format(i))
                if radii:
                    phase = self.getMetaPupilPhase(
                                self.scrns[i], self.atmosConfig.scrnHeights[i],
                                radius=radii[i], pos=pos)
                else:
                    phase = self.getMetaPupilPhase(
                                self.scrns[i], self.atmosConfig.scrnHeights[i],
                                pos=pos)

                self.wfsPhase += phase

            self.EField[:] = numpy.exp(1j*self.wfsPhase)


class LineOfSight_Physical(LineOfSight):

        def __init__(
                self, direction="down", inputPxlScale=None, 
                outputPxlScale=None):
            super(LineOfSight_Physical, self).__init__()

        def makePhase(self, radii=None, pos=None):
            '''
            Finds total WFS complex amplitude by propagating light down
            phase scrns

            Parameters
                radii (dict, optional): Radii of each meta pupil of each screen height in pixels. If not given uses pupil radius.
                pos (dict, optional): Position of GS in pixels. If not given uses GS position.
            '''

            scrnNo = len(self.scrns)-1  #Number of layers (0 indexed)
            ht = self.atmosConfig.scrnHeights[scrnNo] #Height of highest layer
            delta = (self.simConfig.pxlScale)**-1. #Grid spacing for propagation

            #Get initial Phase for highest scrn and turn to efield
            if radii:
                phase1 = self.getMetaPupilPhase(
                            self.scrns[scrnNo], ht, radius=radii[scrnNo],
                            pos=pos)
                            #pupilSize=2*self.simConfig.pupilSize, pos=pos )
            else:
                phase1 = self.getMetaPupilPhase(self.scrns[scrnNo], ht,
                            pos=pos)
                            #pupilSize=2*self.simConfig.pupilSize, pos=pos)

            self.EField[:] = numpy.exp(1j*phase1)
            #Loop through remaining scrns - update ht according
    
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
                                radius=radii[i], pos=pos)
                                # pupilSize=2*self.simConfig.pupilSize)
                else:
                    phase = self.getMetaPupilPhase(
                                self.scrns[i], self.atmosConfig.scrnHeights[i],
                                #pupilSize=2*self.simConfig.pupilSize,
                                pos=pos)

                #Add add phase from this layer
                self.EField *= numpy.exp(1j*phase)

            #If not already at ground, propagate the rest of the way.
            if self.atmosConfig.scrnHeights[0]!=0:
                self.EField[:] = angularSpectrum(
                        self.EField, self.wfsConfig.wavelength,
                        delta, delta, ht
                        )

