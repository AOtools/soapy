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
The Soapy WFS module.


This module contains a number of classes which simulate different adaptive optics wavefront sensor (WFS) types. All wavefront sensor classes can inherit from the base ``WFS`` class. The class provides the methods required to calculate phase over a WFS pointing in a given WFS direction and accounts for Laser Guide Star (LGS) geometry such as cone effect and elongation. This is  If only pupil images (or complex amplitudes) are required, then this class can be used stand-alone.

Example:

    Make configuration objects::

        from soapy import WFS, confParse

        config = confParse.Configurator("config_file.py")
        config.loadSimParams()

    Initialise the wave-front sensor::

        wfs = WFS.WFS(config, 0 mask)

    Set the WFS scrns (these should be made in advance, perhaps by the :py:mod:`soapy.atmosphere` module). Then run the WFS::

        wfs.scrns = phaseScrnList
        wfs.makePhase()

    Now you can view data from the WFS frame::

        frameEField = wfs.EField


A Shack-Hartmann WFS is also included in the module, this contains further methods to make the focal plane, then calculate the slopes to send to the reconstructor.

Example:
    Using the config objects from above...::

        shWfs = WFS.ShackHartmann(config, 0, mask)

    As we are using a full WFS with focal plane making methods, the WFS base classes ``frame`` method can be used to take a frame from the WFS::

        slopes = shWfs.frame(phaseScrnList)

    All the data from that WFS frame is available for inspection. For instance, to obtain the electric field across the WFS and the image seen by the WFS detector::

        EField = shWfs.EField
        wfsDetector = shWfs.wfsDetectorPlane


Adding new WFSs
^^^^^^^^^^^^^^^

New WFS classes should inherit the ``WFS`` class, then create methods which deal with creating the focal plane and making a measurement from it. To make use of the base-classes ``frame`` method, which will run the WFS entirely, the new class must contain the following methods::

    calcFocalPlane(self)
    makeDetectorPlane(self)
    calculateSlopes(self)

The Final ``calculateSlopes`` method must set ``self.slopes`` to be the measurements made by the WFS. If LGS elongation is to be used for the new WFS, create a ``detectorPlane``, which is added to for each LGS elongation propagation. Have a look at the code for the ``Shack-Hartmann`` and experimental ``Pyramid`` WFSs to get some ideas on how to do this.


:Author:
    Andrew Reeves
"""

import numpy
import numpy.random
try:
    from astropy.io import fits
except ImportError:
    try:
        import pyfits as fits
    except ImportError:
        raise ImportError("PyAOS requires either pyfits or astropy")

from .. import AOFFT, aoSimLib, LGS, logger, lineofsight
from ..tools import centroiders
from ..opticalPropagationLib import angularSpectrum

# xrange now just "range" in python3.
# Following code means fastest implementation used in 2 and 3
try:
    xrange
except NameError:
    xrange = range

# The data type of data arrays (complex and real respectively)
CDTYPE = numpy.complex64
DTYPE = numpy.float32

RAD2ASEC = 206264.849159
ASEC2RAD = 1./RAD2ASEC

class WFS(object):
    '''
    A WFS class.

    This is a base class which contains methods to initialise the WFS,
    and calculate the phase across the WFSs input aperture, given the WFS
    guide star geometry.

    Parameters:
        soapyConfig (ConfigObj): The soapy configuration object
        nWfs (int): The ID number of this WFS
        mask (ndarray, optional): An array or size (simConfig.simSize, simConfig.simSize) which is 1 at the telescope aperture and 0 else-where.
    '''

    def __init__(
            self, soapyConfig, nWfs=0, mask=None):

        self.soapyConfig = soapyConfig
        self.config = self.wfsConfig = soapyConfig.wfss[nWfs] # For compatability
        self.simConfig = soapyConfig.sim
        self.telConfig = soapyConfig.tel
        self.atmosConfig = soapyConfig.atmos
        self.lgsConfig = self.config.lgs

        # If supplied use the mask
        if numpy.any(mask):
            self.mask = mask
        # Else we'll just make a circle
        else:
            self.mask = aoSimLib.circle(
                    self.simConfig.pupilSize/2., self.simConfig.simSize,
                    )

        self.iMat = False

        # Init the line of sight
        self.initLos()

        self.calcInitParams()
        # If GS not at infinity, find meta-pupil radii for each layer
        if self.config.GSHeight != 0:
            self.radii = self.los.findMetaPupilSizes(self.config.GSHeight)
        else:
            self.radii = None

        # Init LGS, FFTs and allocate some data arrays
        self.initFFTs()
        if self.lgsConfig and self.config.lgs:
            self.initLGS()

        self.allocDataArrays()

        self.calcTiltCorrect()
        self.getStatic()

############################################################
# Initialisation routines

    def setMask(self, mask):
        # If supplied use the mask
        if numpy.any(mask):
            self.mask = mask
        else:
            self.mask = aoSimLib.circle(
                    self.simConfig.pupilSize/2., self.simConfig.simSize,
                    )


    def calcInitParams(self, phaseSize=None):
        self.los.calcInitParams(nOutPxls=phaseSize)

    def initFFTs(self):
        pass

    def allocDataArrays(self):
        pass

    def initLos(self):
        """
        Initialises the ``LineOfSight`` object, which gets the phase or EField in a given direction through turbulence.
        """
        self.los = lineofsight.LineOfSight(
                self.config, self.soapyConfig,
                propagationDirection="down")


    def initLGS(self):
        """
        Initialises the LGS objects for the WFS

        Creates and initialises the LGS objects if the WFS GS is a LGS. This
        included calculating the phases additions which are required if the
        LGS is elongated based on the depth of the elongation and the launch
        position. Note that if the GS is at infinity, elongation is not possible
        and a warning is logged.
        """

        # Choose the correct LGS object, either with physical or geometric
        # or geometric propagation.
        if self.lgsConfig.uplink:
            lgsObj = eval("LGS.LGS_{}".format(self.lgsConfig.propagationMode))
            self.lgs = lgsObj(self.config, self.soapyConfig)
        else:
            self.lgs = None

        self.lgsLaunchPos = None
        self.elong = 0
        self.elongLayers = 0
        if self.config.lgs:
            self.lgsLaunchPos = self.lgsConfig.launchPosition
            # LGS Elongation##############################
            if (self.config.GSHeight!=0 and
                    self.lgsConfig.elongationDepth!=0):
                self.elong = self.lgsConfig.elongationDepth
                self.elongLayers = self.lgsConfig.elongationLayers

                # Get Heights of elong layers
                self.elongHeights = numpy.linspace(
                    self.config.GSHeight-self.elong/2.,
                    self.config.GSHeight+self.elong/2.,
                    self.elongLayers
                    )

                # Calculate the zernikes to add
                self.elongZs = aoSimLib.zernikeArray([2,3,4], self.simConfig.pupilSize)

                # Calculate the radii of the metapupii at for different elong
                # Layer heights
                # Also calculate the required phase addition for each layer
                self.elongRadii = {}
                self.elongPos = {}
                self.elongPhaseAdditions = numpy.zeros(
                    (self.elongLayers, self.los.nOutPxls, self.los.nOutPxls))
                for i in xrange(self.elongLayers):
                    self.elongRadii[i] = self.los.findMetaPupilSizes(
                                                float(self.elongHeights[i]))
                    self.elongPhaseAdditions[i] = self.calcElongPhaseAddition(i)
                    self.elongPos[i] = self.calcElongPos(i)

                # self.los.metaPupilPos = self.elongPos

                logger.debug(
                        'Elong Meta Pupil Pos: {}'.format(self.los.metaPupilPos))
            # If GS at infinity cant do elongation
            elif (self.config.GSHeight==0 and
                    self.lgsConfig.elongationDepth!=0):
                logger.warning("Not able to implement LGS Elongation as GS at infinity")

    def calcTiltCorrect(self):
        pass

    def getStatic(self):
        self.staticData = None

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

        # Calculate the path difference between the central GS height and the
        # elongation "layer"
        # Define these to make it easier
        h = self.elongHeights[elongLayer]
        dh = h - self.config.GSHeight
        H = float(self.lgsConfig.height)
        d = numpy.array(self.lgsLaunchPos).astype('float32') * self.los.telDiam/2.
        D = self.los.telDiam
        theta = (d.astype("float")/H) - self.config.GSPosition


        # for the focus terms....
        focalPathDiff = (2*numpy.pi/self.wfsConfig.wavelength) * ((
            ((self.los.telDiam/2.)**2 + (h**2) )**0.5\
          - ((self.los.telDiam/2.)**2 + (H)**2 )**0.5 ) - dh)

        # For tilt terms.....
        tiltPathDiff = (2*numpy.pi/self.wfsConfig.wavelength) * (
                numpy.sqrt( (dh+H)**2. + ( (dh+H)*theta-d-D/2.)**2 )
                + numpy.sqrt( H**2 + (D/2. - d + H*theta)**2 )
                - numpy.sqrt( H**2 + (H*theta - d - D/2.)**2)
                - numpy.sqrt( (dh+H)**2 + (D/2. - d + (dh+H)*theta )**2))


        phaseAddition = numpy.zeros(
                    (self.simConfig.pupilSize, self.simConfig.pupilSize))

        phaseAddition +=((self.elongZs[2]/self.elongZs[2].max())
                             * focalPathDiff )
        # X,Y tilt
        phaseAddition += ((self.elongZs[0]/self.elongZs[0].max())
                            *tiltPathDiff[0] )
        phaseAddition += ((self.elongZs[1]/self.elongZs[1].max()) *tiltPathDiff[1])

        # Pad from pupilSize to simSize
        pad = ((self.simConfig.simPad,)*2, (self.simConfig.simPad,)*2)
        phaseAddition = numpy.pad(phaseAddition, pad, mode="constant")

        phaseAddition = aoSimLib.zoom_numba(
                phaseAddition, (self.los.nOutPxls,)*2,
                threads=self.simConfig.threads)

        return phaseAddition

    def calcElongPos(self, elongLayer):
        """
        Calculates the difference in GS position for each elongation layer
        only makes a difference if LGS launched off-axis

        Parameters:
            elongLayer (int): which elongation layer

        Returns:
            float: The effective position of that layer GS on the simulation phase grid
        """

        h = self.elongHeights[elongLayer]       # height of elonglayer
        dh = h - self.config.GSHeight          # delta height from GS Height
        H = float(self.config.GSHeight)            # Height of GS

        # Position of launch in m
        xl = numpy.array(self.lgsLaunchPos) * self.los.telDiam/2.

        # GS Pos in radians
        GSPos = numpy.array(self.config.GSPosition) * RAD2ASEC

        # difference in angular Pos for that height layer in rads
        theta_n = GSPos - ((dh*xl)/ (H*(H+dh)))

        # metres from on-axis point of each elongation point
        elongPos = (GSPos + theta_n) * RAD2ASEC
        return elongPos

    def zeroPhaseData(self):
        self.los.EField[:] = 0
        self.los.phase[:] = 0


    def makeElongationFrame(self, correction=None):
        """
        Find the focal plane resulting from an elongated guide star, such as LGS.

        Runs the phase stacking and propagation routines multiple times with different GS heights, positions and/or aberrations to simulation the effect of a number of points in an elongation guide star.
        """
        # Loop over the elongation layers
        for i in xrange(self.elongLayers):
            logger.debug('Elong layer: {}'.format(i))
            # Reset the phase propagation routines (not the detector though)
            self.zeroData(FP=False)

            # Find the phase from that elongation layer (with different cone effect radii and potentially angular position)
            self.los.makePhase(self.elongRadii[i], apos=self.elongPos[i])

            # Make a copy of the uncorrectedPhase for plotting
            self.uncorrectedPhase = self.los.phase.copy()/self.los.phs2Rad

            # Add the effect of the defocus and possibly tilt
            self.los.EField *= numpy.exp(1j*self.elongPhaseAdditions[i])
            self.los.phase += self.elongPhaseAdditions[i]

            # Apply any correction
            if correction is not None:
                self.los.performCorrection(correction)

            # Add onto the focal plane with that layers intensity
            self.calcFocalPlane(intensity=self.lgsConfig.naProfile[i])

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
            removeTT = self.config.removeTT
            self.config.removeTT = False
            photonNoise = self.config.photonNoise
            self.config.photonNoise = False
            eReadNoise = self.config.eReadNoise
            self.config.eReadNoise = 0

        self.zeroData(detector=read, FP=False)

        self.los.frame(scrns)

        # If LGS elongation simulated
        if self.config.lgs and self.elong!=0:
            self.makeElongationFrame(correction)

        # If no elongation
        else:
            # If imat frame, dont want to make it off-axis

            # if iMatFrame:
            #     try:
            #         iMatPhase = aoSimLib.zoom(scrns, self.los.nOutPxls, order=1)
            #         self.los.EField[:] = numpy.exp(1j*iMatPhase*self.los.phs2Rad)
            #     except ValueError:
            #         raise ValueError("If iMat Frame, scrn must be ``simSize``")
            # else:
            self.los.makePhase(self.radii)

            self.uncorrectedPhase = self.los.phase.copy()/self.los.phs2Rad
            if correction is not None:
                self.los.performCorrection(correction)
                
            self.calcFocalPlane()

        if read:
            self.makeDetectorPlane()
            self.calculateSlopes()
            self.zeroData(detector=False)

        # Turn back on stuff disabled for iMat
        if iMatFrame:
            self.iMat=False
            self.config.removeTT = removeTT
            self.config.photonNoise = photonNoise
            self.config.eReadNoise = eReadNoise

        # Check that slopes aint `nan`s. Set to 0 if so
        if numpy.any(numpy.isnan(self.slopes)):
            self.slopes[:] = 0

        return self.slopes

    def addPhotonNoise(self):
        """
        Add photon noise to ``wfsDetectorPlane`` using ``numpy.random.poisson``
        """
        self.wfsDetectorPlane = numpy.random.poisson(
                self.wfsDetectorPlane).astype(DTYPE)


    def addReadNoise(self):
        """
        Adds read noise to ``wfsDetectorPlane using ``numpy.random.normal``.
        This generates a normal (guassian) distribution of random numbers to
        add to the detector. Any CCD bias is assumed to have been removed, so
        the distribution is centred around 0. The width of the distribution
        is determined by the value `eReadNoise` set in the WFS configuration.
        """
        self.wfsDetectorPlane += numpy.random.normal(
                0, self.config.eReadNoise, self.wfsDetectorPlane.shape
                )

    def calcFocalPlane(self, intensity=None):
        pass

    def makeDetectorPlane(self):
        pass

    def LGSUplink(self):
        pass

    def calculateSlopes(self):
        self.slopes = self.los.EField

    def zeroData(self, detector=True, FP=True):
        self.zeroPhaseData()


    @property
    def EField(self):
        return self.los.EField

    @EField.setter
    def EField(self, EField):
        self.los.EField = EField
