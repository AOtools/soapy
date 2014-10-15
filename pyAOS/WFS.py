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

"""
The pyAOS Wavefront Sensor module.

^^^^^^^^^
WFS Class
^^^^^^^^^

This module contains a number of classes which simulate different adaptive optics wavefront sensor (WFS) types. All wavefront sensor classes can inherit from the base ``WFS`` class. The class provides the methods required to calculate phase over a WFS pointing in a given WFS direction and accounts for Laser Guide Star (LGS) geometry such as cone effect and elongation. This is  If only pupil images (or complex amplitudes) are required, then this class can be used stand-alone.

Example:

    Make configuration objects::
    
        from pyAOS import WFS, confParse
    
        config = confParse.Configurator("config_file.py")
        config.readFile()
        config.loadSimParams()
        config.calcParams()
    
    Initialise the wave-front sensor::
    
        wfs = WFS.WFS(config.sim, config.wfs[0], config.atmos, config.lgs[0], mask)
    
    Set the WFS scrns (these should be made in advance, perhaps by the atmosphere module). Then run the WFS::
    
        wfs.scrns = phaseScrnList
        wfs.makePhase()
    
    Now you can view data from the WFS frame::
    
        frameEField = wfs.EField
    
^^^^^^^^^^^^^^^^^^
Shack-Hartmann WFS
^^^^^^^^^^^^^^^^^^

A Shack-Hartmann WFS is also included in the module, this contains further methods to make the focal plane, then calculate the slopes to send to the reconstructor.

Example:
    Using the config objects from above...::
        
        shWfs = WFS.ShackHartmann(config.sim, config.wfs[0], config.atmos, config.lgs[0], mask)
        
    As we are using a full WFS with focal plane making methods, the WFS base classes ``frame`` method can be used to take a frame from the WFS::
        
        slopes = shWfs.frame(phaseScrnList)

    All the data from that WFS frame is available for inspection. For instance, to obtain the electric field across the WFS and the image seen by the WFS detector::
        
        EField = shWfs.EField            
        wfsDetector = shWfs.wfsDetectorPlane

        
^^^^^^^^^^^^^^^
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
import scipy.ndimage
import scipy.optimize

from . import AOFFT, aoSimLib, LGS, logger
from .opticalPropagationLib import angularSpectrum

#xrange now just "range" in python3. 
#Following code means fastest implementation used in 2 and 3
try:
    xrange
except NameError:
    xrange = range

#The data type of data arrays
CDTYPE = numpy.complex64
DTYPE = numpy.float32

class WFS(object):
    ''' A  WFS class.

        This is a base class which contains methods to initialise the WFS,
        and calculate the phase across the WFSs input aperture, given the WFS
        guide star geometry.
        
        Parameters:
            simConfig (confObj): The simulation configuration object
            wfsConfig (confObj): The WFS configuration object
            atmosConfig (confObj): The atmosphere configuration object
            lgsConfig (confObj): The Laser Guide Star configuration object
            mask (ndarray): An array or size (simConfig.pupilSize, simConfig.pupilSize) which is 1 at the telescope aperture and 0 else-where.
    '''

    def __init__(self, simConfig, wfsConfig, atmosConfig, lgsConfig, mask):

        self.simConfig = simConfig
        self.wfsConfig = wfsConfig
        self.atmosConfig = atmosConfig
        self.lgsConfig = lgsConfig

        self.centMethod = "simple"
        self.mask = mask

        self.subapFOVrad = wfsConfig.subapFOV * numpy.pi / (180. * 3600)

        self.telDiam = simConfig.pupilSize/simConfig.pxlScale
        self.subapDiam = self.telDiam/self.wfsConfig.subaps

        #######################################################
        #LGS Initialisation
        if wfsConfig.lgs:     
            if  (self.lgsConfig.propagationMode=="phys" or 
                    self.lgsConfig.propagationMode=="physical"):
                LGSClass = LGS.PhysicalLGS         
            else:
                LGSClass = LGS.GeometricLGS
                        
            self.LGS = LGSClass( self.simConfig, wfsConfig, self.lgsConfig,
                                    self.atmosConfig
                                    )
        else:
            self.LGS = None

        self.lgsLaunchPos = None
        self.elong=0
        self.elongLayers = 0
        if self.LGS:
            self.lgsLaunchPos = self.lgsConfig.launchPosition

            #LGS Elongation##############################
            if (self.wfsConfig.GSHeight!=0 and 
                    self.lgsConfig.elongationDepth!=0):
                self.elong = self.LGS.lgsConfig.elongationDepth
                self.elongLayers = self.LGS.lgsConfig.elongationLayers

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
                self.elongPhaseAdditions = numpy.empty( 
                    (self.elongLayers,self.simConfig.pupilSize, 
                    self.simConfig.pupilSize))
                for i in xrange(self.elongLayers):
                    self.elongRadii[i] = self.findMetaPupilSize(
                                                float(self.elongHeights[i]))
                    self.elongPhaseAdditions[i] = self.calcElongPhaseAddition(i)
                    self.elongPos[i] = self.calcElongPos(i)

            #If GS at infinity cant do elongation
            elif (self.wfsConfig.GSHeight==0 and 
                    self.lgsConfig.elongationDepth!=0):
                logger.warning("Not able to implement lgs Elongation as GS at infinity")
        #Done LGS
        ##############################################

        self.iMat=False

        #Phase power scaling factor for wfs wavelength
        phsWvl = 550e-9
        self.r0Scale = phsWvl/self.wfsConfig.wavelength

        #spacing between subaps in pupil Plane (size "pupilSize")
        self.PPSpacing = float(self.simConfig.pupilSize)/self.wfsConfig.subaps

        #Spacing on the "FOV Plane" - the number of elements required
        #for the correct subap FOV (from way FFT "phase" to "image" works)
        self.subapFOVSpacing = numpy.round(self.subapDiam * self.subapFOVrad / self.wfsConfig.wavelength)

        #make twice as big to double subap FOV
        self.SUBAP_OVERSIZE = 2
        self.detectorPxls = self.wfsConfig.pxlsPerSubap*self.wfsConfig.subaps
        self.subapFOVSpacing *= self.SUBAP_OVERSIZE
        self.wfsConfig.pxlsPerSubap2 = self.SUBAP_OVERSIZE*self.wfsConfig.pxlsPerSubap


        if self.wfsConfig.GSHeight!=0:
            self.radii = self.findMetaPupilSize(self.wfsConfig.GSHeight)
        else:
            self.radii=None

        self.findActiveSubaps()
        #self.scaledMask = numpy.round(scipy.ndimage.zoom(self.mask,
                                     #self.subapFOVSpacing/self.PPSpacing))
        self.scaledMask = aoSimLib.zoom(self.mask,
                                        self.subapFOVSpacing*self.wfsConfig.subaps)
    
        
        #Choose propagation method
        if wfsConfig.propagationMode=="physical":
            self.makePhase = self.makePhasePhysical
            self.physEField = numpy.zeros(
                (2*self.simConfig.pupilSize,)*2, dtype=CDTYPE)
        else:
            self.makePhase = self.makePhaseGeo


        #Initialise FFT classes
        #########################################
        self.subapFFTPadding = self.wfsConfig.pxlsPerSubap2 * self.wfsConfig.subapOversamp
        if self.subapFFTPadding < self.subapFOVSpacing:
            while self.subapFFTPadding<self.subapFOVSpacing:
                self.wfsConfig.subapOversamp+=1
                self.subapFFTPadding\
                        =self.wfsConfig.pxlsPerSubap2*self.wfsConfig.subapOversamp

            logger.warning("requested WFS FFT Padding less than FOV size... Setting oversampling to: %d"%self.wfsConfig.subapOversamp)

        self.FFT = AOFFT.FFT(
                inputSize=(
                self.activeSubaps, self.subapFFTPadding, self.subapFFTPadding),
                axes=(-2,-1), mode="pyfftw",dtype=CDTYPE,
                THREADS=wfsConfig.fftwThreads, 
                fftw_FLAGS=(wfsConfig.fftwFlag,"FFTW_DESTROY_INPUT"))

        if self.lgsConfig.lgsUplink:
            self.iFFT = AOFFT.FFT(
                    inputSize = (self.activeSubaps,
                                        self.subapFFTPadding,
                                        self.subapFFTPadding),
                    axes=(-2,-1), mode="pyfftw",dtype=CDTYPE,
                    THREADS=wfsConfig.fftwThreads, 
                    fftw_FLAGS=(wfsConfig.fftwFlag,"FFTW_DESTROY_INPUT")
                    )

            self.lgs_iFFT = AOFFT.FFT(
                    inputSize = (self.subapFFTPadding,
                                self.subapFFTPadding),
                    axes=(0,1), mode="pyfftw",dtype=CDTYPE,
                    THREADS=wfsConfig.fftwThreads, 
                    fftw_FLAGS=(wfsConfig.fftwFlag,"FFTW_DESTROY_INPUT")
                    )

        ######################################################

        #Tell LGS some WFS stuff
        if self.LGS:
            self.LGS.setWFSParams(self.SUBAP_OVERSIZE*self.subapFOVrad,
                                     self.wfsConfig.subapOversamp, self.subapFFTPadding)
            

        #initialise some arrays which will be useful later
        
        #First make wfs detector array - only needs to be of ints,
        #Find which kind to save memory
        if (wfsConfig.bitDepth==8 or 
                wfsConfig.bitDepth==16 or 
                wfsConfig.bitDepth==32):
            self.dPlaneType = eval("numpy.uint%d"%wfsConfig.bitDepth)
        else:
            self.dPlaneType = numpy.uint32
        self.maxFlux = 0.7 * 2**wfsConfig.bitDepth -1
        self.wfsDetectorPlane = numpy.zeros( (  self.detectorPxls,
                                                self.detectorPxls   ),
                                               dtype = self.dPlaneType )

        self.wfsPhase = numpy.zeros( [self.simConfig.pupilSize]*2, dtype=DTYPE)

        self.subapArrays=numpy.empty((self.activeSubaps,
                                      self.subapFOVSpacing,
                                      self.subapFOVSpacing),
                                     dtype=CDTYPE)
        self.binnedFPSubapArrays = numpy.zeros( (self.activeSubaps,
                                                self.wfsConfig.pxlsPerSubap2,
                                                self.wfsConfig.pxlsPerSubap2),
                                                dtype=DTYPE)
        self.FPSubapArrays = numpy.zeros((self.activeSubaps,
                                          self.subapFFTPadding,
                                          self.subapFFTPadding),dtype=DTYPE)
        self.EField = numpy.zeros(
                (self.simConfig.pupilSize,)*2, dtype=CDTYPE)



############################################################
#Initialisation routines

    def findMetaPupilSize(self, GSHeight):
        '''
        Evaluates the sizes of the effective metePupils
        at each screen height if an GS of finite height is used.
        
        Args:
            
        
        '''

        radii={}

        for i in xrange(self.atmosConfig.scrnNo):
            #Find radius of metaPupil geometrically (fraction of pupil at
            # Ground Layer)
            radius = (self.telDiam/2.) * (
                                1-(float(self.atmosConfig.scrnHeights[i])/GSHeight))
            radii[i]= numpy.round(radius * self.simConfig.pxlScale)

            #If scrn is above LGS, radius is 0
            if self.atmosConfig.scrnHeights[i]>=GSHeight:
                radii[i]=0

        return radii

    def findActiveSubaps(self):
        '''
        Finds the subapertures which are not empty space
        determined if mean of subap coords of the mask is above threshold
        '''

        self.subapCoords = aoSimLib.findActiveSubaps(self.wfsConfig.subaps,self.mask, self.wfsConfig.subapThreshold)
        self.activeSubaps = self.subapCoords.shape[0]

        #When scaled to pxl sizes, need to scale subap coordinates too!
        self.detectorSubapCoords = numpy.round(
            self.subapCoords*(self.detectorPxls/float(self.simConfig.pupilSize) ) )


    def calcElongPhaseAddition(self,elongLayer):

        #Calculate the path difference between the central GS height and the
        #elongation "layer"
        #Define these to make it easier
        h = self.elongHeights[elongLayer]
        dh = h-self.wfsConfig.GSHeight
        H = self.lgsConfig.height
        d = numpy.array(self.lgsLaunchPos) * self.telDiam/2.
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


        phaseAddition = numpy.zeros( (  self.simConfig.pupilSize, self.simConfig.pupilSize) )

        phaseAddition +=( (self.elongZs[2]/self.elongZs[2].max())
                             * focalPathDiff )
        #X,Y tilt
        phaseAddition += ( (self.elongZs[0]/self.elongZs[0].max())
                            *tiltPathDiff[0] )
        phaseAddition += ( (self.elongZs[1]/self.elongZs[1].max())
                            *tiltPathDiff[1])


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
        H = self.lgsConfig.height               #Height of GS

        #Position of launch in m
        xl = numpy.array(self.lgsLaunchPos) * self.telDiam/2.  

        #GS Pos in radians
        GSPos=numpy.array(self.wfsConfig.GSPosition)*numpy.pi/(3600.0*180.0)


        theta_n = GSPos - ((dh*xl)/ (H*(H+dh)))

        return theta_n

#############################################################

#############################################################
#Phase stacking routines for a WFS frame

    def getMetaPupilPos(self, height, GSPos=None):
        '''
        Finds the centre of a metapupil at a given height, 
        when offset by a given angle in arcsecs, in units of phase pixels
        
        Arguments:
            height (float): Height of the layer in metres
            GSPos (tuple, optional):  The angular position of the GS. If not set, will use the WFS position
            
        Returns:
            ndarray: The position of the centre of the metapupil in units of phase pixels
        '''
        #if no GSPos given, use system pos and convert into radians
        if not numpy.any(GSPos):
            GSPos = (   numpy.array(self.wfsConfig.GSPosition)
                        *numpy.pi/(3600.0*180.0) )

        #Position of centre of GS metapupil off axis at required height
        GSCent = (numpy.tan(GSPos) * height).astype("int")

        return GSCent

    def getMetaPupilPhase(  self, scrn, height, radius=None, pupilSize=None,
                            GSPos=None):
        '''
        Returns the phase across a metaPupil at some height and angular 
        offset in arcsec. Interpolates phase to size of the pupil if cone 
        effect is required
        
        Parameters:
            scrn (ndarray): An array representing the phase screen
            height (float): Height of the phase screen
            radius (radius, optional): Radius of the meta-pupil. If not set, will use system pupil size.
            pupilSize (ndarray, optional): Size of screen to return. If not set, will use system pupil size.
            GSPos (tuple, optional): Angular position of guide star. If not set will use system position.
            
        Return:
            ndarray: The meta pupil at the specified height
        '''
        #If no size of metapupil given, use system pupil size
        if not pupilSize:
            pupilSize = self.simConfig.pupilSize

        GSCent = self.getMetaPupilPos(height, GSPos) * self.simConfig.pxlScale
        scrnX,scrnY=scrn.shape

        #Check screen is big enough to get a pupil from
        if ( (scrnX/2. + GSCent[0] - pupilSize/2.0) < 0 
                or (scrnX/2. + GSCent[0] - pupilSize/2.0) > scrnX 
                or scrnX/2. + GSCent[0] - pupilSize/2.0 < 0 
                or (scrnY/2. + GSCent[1] - pupilSize/2.0) > scrnY):
 
            raise ValueError( "GS seperation requires larger screen size" )

        metaPupil=scrn[ 
                int(round(scrnX/2. + GSCent[0] - pupilSize/2.0)):
                int(round(scrnX/2. + GSCent[0] + pupilSize/2.0)),
                int(round(scrnY/2. + GSCent[1] - pupilSize/2.0)):
                int(round(scrnY/2. + GSCent[1] + pupilSize/2.0))
                ]

        if self.wfsConfig.GSHeight!=0:
            if radius!=0:
                metaPupil = aoSimLib.zoom(
                        metaPupil[
                            int(round(pupilSize/2. - radius)):
                            int(round(pupilSize/2. + radius)),
                            int(round(pupilSize/2. - radius)):
                            int(round(pupilSize/2. + radius))],
                        (pupilSize, pupilSize)
                        )

        return metaPupil

    def makePhaseGeo(self, radii=None, GSPos=None):
        '''
        Creates the total phase on a wavefront sensor which 
        is offset by a given angle
        '''
        
        for i in self.scrns:
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
        phase scrns'''

        scrnNo = len(self.scrns)-1
        ht = self.atmosConfig.scrnHeights[scrnNo]
        delta = (self.simConfig.pxlScale)**-1. #Grid spacing
        
        #Get initial Phase for highest scrn and turn to efield
        if radii:
            phase1 = self.getMetaPupilPhase(
                        self.scrns[scrnNo], ht, radius=radii[scrnNo],
                        pupilSize=2*self.simConfig.pupilSize, GSPos=GSPos )
        else:
            phase1 = self.getMetaPupilPhase(self.scrns[scrnNo], ht,
                        pupilSize=2*self.simConfig.pupilSize, GSPos=GSPos)
        
        self.physEField[:] = numpy.exp(1j*phase1)
        #Loop through remaining scrns in reverse order - update ht accordingly
        for i in range(scrnNo)[::-1]:
            #Get propagation distance for this layer
            z = ht - self.atmosConfig.scrnHeights[i]
            ht -= z
            
            #Do ASP for last layer to next
            self.physEField[:] = angularSpectrum(
                        self.physEField, self.wfsConfig.wavelength, 
                        delta, delta, z )
            
            #Get phase for this layer
            if radii:
                phase = self.getMetaPupilPhase(
                            self.scrns[i], self.atmosConfig.scrnHeights[i],
                            radius=radii[i], GSPos=GSPos,
                            pupilSize=2*self.simConfig.pupilSize)
            else:
                phase = self.getMetaPupilPhase(
                            self.scrns[i], self.atmosConfig.scrnHeights[i],
                            pupilSize=2*self.simConfig.pupilSize,
                            GSPos=GSPos)
            
            #Add add phase from this layer
            self.physEField *= numpy.exp(1j*phase)
        
        #If not already at ground, propagate the rest of the way.
        if self.atmosConfig.scrnHeights[0]!=0:
            self.physEField[:] = angularSpectrum(
                    self.physEField, self.wfsConfig.wavelength, 
                    delta, delta, ht
                    )

        #Multiply EField by aperture
        self.EField[:] = self.physEField[
                            self.simConfig.pupilSize/2.:
                            3*self.simConfig.pupilSize/2.,
                            self.simConfig.pupilSize/2.:
                            3*self.simConfig.pupilSize/2.] * self.mask

######################################################

    def readNoise(self, dPlaneArray):
        dPlaneArray += numpy.random.normal( (self.maxFlux/self.wfsConfig.SNR),
        0.1*self.maxFlux/self.wfsConfig.SNR, dPlaneArray.shape).clip(0,self.maxFlux).astype(self.dPlaneType)


    def photonNoise(self):
        pass


    def iMatFrame(self,phs):
        '''
        Runs an iMat frame - essentially gives slopes for given "phs" so
        useful for other stuff too!
        
        Parameters:
            phs (ndarray):  The phase to apply to the WFS. Should be of shape
                            (simConfig.pupilSize, simConfig.pupilSize)
        Returns:
            ndarray: A 1-d array of WFS measurements
        '''
        self.iMat=True
        #Set "removeTT" to false while we take an iMat
        removeTT = self.wfsConfig.removeTT
        self.wfsConfig.removeTT=False

        self.zeroData()
        self.EField[:] =  numpy.exp(1j*phs)
        self.calcFocalPlane()
        self.makeDetectorPlane()
        self.calculateSlopes()
        
        self.wfsConfig.removeTT = removeTT
        self.iMat=False
        
        return self.slopes

    def zeroPhaseData(self):
        self.EField[:] = 0
        self.wfsPhase[:] = 0


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
            
        Returns:
            ndarray: WFS Measurements
        '''

        self.scrns = {}
        #Scale phase to WFS wvl
        for i in xrange(len(scrns)):
            self.scrns[i] = scrns[i].copy()*self.r0Scale

        self.zeroData()

        #If no elongation
        if self.elong==0:
            self.makePhase(self.radii)
            self.uncorrectedPhase = self.wfsPhase.copy()
            if numpy.any(correction):
                self.EField *= numpy.exp(-1j*correction)
            self.calcFocalPlane()


        #If LGS elongation simulated
        if self.elong!=0:
            for i in xrange(self.elongLayers):
                self.zeroPhaseData()

                self.makePhase(self.elongRadii[i], self.elongPos[i])
                self.uncorrectedPhase = self.wfsPhase
                self.EField *= numpy.exp(1j*self.elongPhaseAdditions[i])
                if numpy.any(correction):
                    self.EField *= numpy.exp(-1j*correction)
                self.calcFocalPlane()

        self.makeDetectorPlane()
        self.calculateSlopes()

        return self.slopes

    def photoNoise(self):
        pass

    def calcFocalPlane(self):
        pass

    def makeDetectorPlane(self):
        pass

    def LGSUplink(self):
        pass

    def calculateSlopes(self):
        self.slopes = numpy.zeros(2*self.activeSubaps)

    def zeroData(self):
        self.zeroPhaseData()


class ShackHartmann(WFS):
    """Class to simulate a Shack-Hartmann WFS"""
    
    def __init__(self, simConfig, wfsConfig, atmosConfig, lgsConfig, mask):
        
        super(ShackHartmann,self).__init__(simConfig, wfsConfig, atmosConfig, lgsConfig, mask)
        
        #Tilt Correction for even number sizes subaps -
        #otherwise the PSF is binned to one corner
        if not self.wfsConfig.pxlsPerSubap%2:
            self.calcTiltCorrect()


    ##################################################################
    #Centre spot in sub-aperture after FFT

    def calcTiltCorrect(self):
        """
        Calculates the required tilt to add to avoid the PSF being centred on 
        only 1 pixel
        """
        #Angle we need to correct for half a pixel
        theta = 2*self.subapFOVrad/ (2*self.subapFFTPadding)

        #Magnitude of tilt required to get that angle
        A = theta*self.subapDiam/(2*self.wfsConfig.wavelength)*2*numpy.pi

        #Create tilt arrays and apply magnitude
        coords = numpy.linspace(-1,1,self.subapFOVSpacing)
        X,Y = numpy.meshgrid(coords,coords)

        self.tiltFix = -1*A*(X+Y)

    def oneSubap(self,phs):
        '''
        Processes one subaperture only, with given phase
        '''
        EField = numpy.exp(1j*phs)
        FP = numpy.abs(numpy.fft.fftshift(
            numpy.fft.fft2(EField * numpy.exp(1j*self.XTilt),
                s=(self.subapFFTPadding,self.subapFFTPadding))))**2

        FPDetector = aoSimLib.binImgs(FP,self.wfsConfig.subapOversamp)

        slope = aoSimLib.simpleCentroid(FPDetector, 
                    self.wfsConfig.centThreshold)
        slope -= self.wfsConfig.pxlsPerSubap2/2.
        return slope
#######################################################################


    def zeroData(self):
        self.zeroPhaseData()
        self.wfsDetectorPlane[:] = 0
        self.FPSubapArrays[:] = 0

    def photonNoise(self):

        '''Photon Noise'''
        raise NotImplementedError



    def calcFocalPlane(self):
        '''
        Calculates the wfs focal plane, given the phase across the WFS
        '''

        #Scale phase (EField) to correct size for FOV
        self.scaledEField = aoSimLib.zoom( self.EField, 
                self.wfsConfig.subaps*self.subapFOVSpacing) * self.scaledMask 

        #create an array of individual subap EFields
        for i in xrange(self.activeSubaps):
            x,y = numpy.round(self.subapCoords[i] *
                                     self.subapFOVSpacing/self.PPSpacing)
            self.subapArrays[i] = self.scaledEField[
                                    int(x):
                                    int(x+self.subapFOVSpacing) ,
                                    int(y):
                                    int(y+self.subapFOVSpacing)]

        #do the fft to all subaps at the same time
        # and convert into intensity
        self.FFT.inputData[:] = 0
        self.FFT.inputData[:,:int(round(self.subapFOVSpacing))
                        ,:int(round(self.subapFOVSpacing))] \
                = self.subapArrays*numpy.exp(1j*(self.tiltFix))

        self.FPSubapArrays += numpy.abs(AOFFT.ftShift2d(self.FFT()))**2


    def makeDetectorPlane(self):
        '''
        if required, will convolve final psf with LGS psf, then bins
        psf down to detector size. Finally puts back into wfsFocalPlane Array
        in correct order.
        '''

        #If required, convolve with LGS PSF
        if self.LGS and self.lgsConfig.lgsUplink and self.iMat!=True:
            self.LGSUplink()

        #bins back down to correct size and then
        #fits them back in to a focal plane array
        self.binnedFPSubapArrays[:] = aoSimLib.binImgs(self.FPSubapArrays,
                                                        self.wfsConfig.subapOversamp)

        self.binnedFPSubapArrays[:] = self.maxFlux* (self.binnedFPSubapArrays.T/
                                            self.binnedFPSubapArrays.max((1,2))
                                                                         ).T

        for i in xrange(self.activeSubaps):

            x,y=self.detectorSubapCoords[i]

            #Set default position to put arrays into (2 subap FOV)
            x1 = int(round(x-self.wfsConfig.pxlsPerSubap/2.))
            x2 = int(round(x+self.wfsConfig.pxlsPerSubap*3./2))
            y1 = int(round(y-self.wfsConfig.pxlsPerSubap/2.))
            y2 = int(round(y+self.wfsConfig.pxlsPerSubap*3./2.))

            #Set defualt size of input array (i.e. all of it)
            x1_fp = int(0)
            x2_fp = int(round(self.wfsConfig.pxlsPerSubap2))
            y1_fp = int(round(0))
            y2_fp = int(round(self.wfsConfig.pxlsPerSubap2))

            #If at the edge of the field, may only fit a fraction in 
            if x==0:
                x1 = int(0)
                x1_fp = int(round(self.wfsConfig.pxlsPerSubap/2.))

            elif x==(self.detectorPxls-self.wfsConfig.pxlsPerSubap):
                x2 = int(round(self.detectorPxls))
                x2_fp = int(round(-1*self.wfsConfig.pxlsPerSubap/2.))

            if y==0:
                y1 = int(0)
                y1_fp = int(round(self.wfsConfig.pxlsPerSubap/2.))

            elif y==(self.detectorPxls-self.wfsConfig.pxlsPerSubap):
                y2 = int(self.detectorPxls)
                y2_fp = int(round(-1*self.wfsConfig.pxlsPerSubap/2.))

            self.wfsDetectorPlane[x1:x2, y1:y2] += (
                    self.binnedFPSubapArrays[i, x1_fp:x2_fp, y1_fp:y2_fp] )

        if self.wfsConfig.SNR and self.iMat!=True:

            self.photonNoise()
            self.readNoise(self.wfsDetectorPlane)


    def LGSUplink(self):
        '''
        A method to deal with convolving the LGS PSF
        with the subap focal plane.
        '''

        self.LGS.LGSPSF(self.scrns)

        self.lgs_iFFT.inputData[:] = self.LGS.PSF
        self.iFFTLGSPSF = self.lgs_iFFT()

        self.iFFT.inputData[:] = self.FPSubapArrays
        self.iFFTFPSubapsArray = self.iFFT()

        #Do convolution
        self.iFFTFPSubapsArray *= self.iFFTLGSPSF

        #back to Focal Plane.
        self.FFT.inputData[:] = self.iFFTFPSubapsArray
        self.FPSubapArrays[:] = AOFFT.ftShift2d( self.FFT() ).real


    def calculateSlopes(self):
        '''
        returns wfs slopes from wfsFocalPlane
        '''

        #Sort out FP into subaps
        self.centSubapArrays = numpy.empty( (self.activeSubaps,  self.wfsConfig.pxlsPerSubap,
                                                        self.wfsConfig.pxlsPerSubap) )

        for i in xrange(self.activeSubaps):
            x,y = self.detectorSubapCoords[i]
            x = int(x)
            y = int(y)
            self.centSubapArrays[i] = self.wfsDetectorPlane[ x:x+self.wfsConfig.pxlsPerSubap,
                                                    y:y+self.wfsConfig.pxlsPerSubap ]

        if self.wfsConfig.centMethod=="brightestPxl":
            slopes = aoSimLib.brtPxlCentroid(
                    self.centSubapArrays, (self.wfsConfig.centThreshold*
                                (self.wfsConfig.pxlsPerSubap**2))
                                            )
        else:
            slopes=aoSimLib.simpleCentroid(
                    self.centSubapArrays, self.wfsConfig.centThreshold
                     )


        #shift slopes relative to subap centre
        slopes-=self.wfsConfig.pxlsPerSubap/2.0
        
        if self.wfsConfig.removeTT==True:
            slopes = (slopes.T - slopes.mean(1)).T

        self.slopes = slopes.reshape(self.activeSubaps*2)
        
        if self.wfsConfig.angleEquivNoise and not self.iMat:
            pxlEquivNoise = (
                    self.wfsConfig.angleEquivNoise * 
                    float(self.wfsConfig.pxlsPerSubap)
                    /self.wfsConfig.subapFOV )
            self.slopes += numpy.random.normal( 0, pxlEquivNoise, 
                                                2*self.activeSubaps)

        return self.slopes


class Pyramid(WFS):

    def __init__(self, simConfig, wfsConfig, atmosConfig, lgsConfig, mask):
        
 
        super(Pyramid,self).__init__(simConfig, wfsConfig, atmosConfig, lgsConfig, mask)
        
        self.FOV_OVERSAMP = 4
        self.wfsConfig = wfsConfig
        self.simConfig = simConfig
        self.atmosConfig = atmosConfig
        self.lgsConfig = lgsConfig
        self.mask = mask

        self.telDiam = simConfig.pupilSize/simConfig.pxlScale

        self.FOVrad = self.wfsConfig.subapFOV * numpy.pi / (180.*3600)
        self.FOVPxlNo = numpy.round( self.telDiam * self.FOVrad/self.wfsConfig.wavelength)

        self.FFT = AOFFT.FFT(   [self.FOV_OVERSAMP*self.FOVPxlNo,]*2, 
                                axes=(0,1), mode="pyfftw", 
                                fftw_FLAGS=("FFTW_DESTROY_INPUT",
                                            wfsConfig.fftwFlag),
                                THREADS=wfsConfig.fftwThreads
                                )

        #Find sizes of detector planes
        while (self.wfsConfig.pxlsPerSubap*self.wfsConfig.subapOversamp
                    < self.FOVPxlNo):
            self.wfsConfig.subapOversamp+=1

        self.paddedDetectorPxls = 2*self.wfsConfig.pxlsPerSubap*self.wfsConfig.subapOversamp
        self.detectorPxls = 2*self.wfsConfig.pxlsPerSubap


        self.iFFTPadding = self.FOV_OVERSAMP*(self.wfsConfig.subapOversamp*self.wfsConfig.pxlsPerSubap)

        self.iFFT = AOFFT.FFT(
                    [4, self.iFFTPadding, self.iFFTPadding],
                    axes=(1,2), mode="pyfftw", THREADS = wfsConfig.fftwThreads,
                    fftw_FLAGS=("FFTW_DESTROY_INPUT", wfsConfig.fftwFlag),
                    direction="BACKWARD"
                    )

        #Allocate arrays 
        self.paddedDetectorPlane = numpy.empty([self.paddedDetectorPxls]*2,
                                                dtype="float32")
        self.wfsDetectorPlane = numpy.empty([self.detectorPxls]*2,
                                            dtype="float32")

        self.activeSubaps = self.wfsConfig.pxlsPerSubap**2

        self.scaledMask = aoSimLib.zoom(self.mask, self.FOVPxlNo)
        
 
        #If not a odd number of pixels, calculate a shift to put 
        # the psf in a central position   
        self.calcTiltCorrect()

    def zeroData(self):
        self.zeroPhaseData()
        self.wfsDetectorPlane[:] = 0
        self.paddedDetectorPlane[:] = 0

    def calcFocalPlane(self):
        '''
        takes the calculated pupil phase, and uses FFT
        to transform to the focal plane, and scales for correct FOV.
        '''
        #Apply tilt fix and scale EField for correct FOV
        self.EField*=numpy.exp(1j*self.tiltFix)
        self.scaledEField = aoSimLib.zoom(self.EField, self.FOVPxlNo)*self.scaledMask

        #Go to the focus 
        self.FFT.inputData[:]=0
        self.FFT.inputData[ :self.FOVPxlNo,
                            :self.FOVPxlNo ] = self.scaledEField
        self.focalPlane = AOFFT.ftShift2d( self.FFT() ).copy()

        #Cut focus into 4
        shapeX,shapeY = self.focalPlane.shape
        self.quads = numpy.empty(   (4,shapeX/2.,shapeX/2.),
                                    dtype=self.focalPlane.dtype)
        n=0
        for x in xrange(2):
            for y in xrange(2):
                self.quads[n] = self.focalPlane[x*shapeX/2 : (x+1)*shapeX/2,
                                                y*shapeX/2 : (y+1)*shapeX/2]
                n+=1

        #Propogate each quadrant back to the pupil plane
        self.iFFT.inputData[:] = 0
        self.iFFT.inputData[:,
                            :0.5*self.FOV_OVERSAMP*self.FOVPxlNo,
                            :0.5*self.FOV_OVERSAMP*self.FOVPxlNo] = self.quads
        self.pupilImages = abs(AOFFT.ftShift2d(self.iFFT()))**2

        size = self.paddedDetectorPxls/2
        pSize = self.iFFTPadding/2.


        #add this onto the padded detector array
        for x in range(2):
            for y in range(2):
                self.paddedDetectorPlane[
                        x*size:(x+1)*size,
                        y*size:(y+1)*size] += self.pupilImages[
                                                2*x+y,
                                                pSize:
                                                pSize+size,
                                                pSize:
                                                pSize+size]

    def makeDetectorPlane(self):
    
        #Bin down to requried pixels
        self.wfsDetectorPlane[:] += aoSimLib.binImgs(
                        self.paddedDetectorPlane,
                        self.wfsConfig.subapOversamp 
                        )

    def calculateSlopes(self):

        xDiff = (self.wfsDetectorPlane[ :self.wfsConfig.pxlsPerSubap,:]-
                    self.wfsDetectorPlane[  self.wfsConfig.pxlsPerSubap:,:]) 
        xSlopes = (xDiff[:,:self.wfsConfig.pxlsPerSubap]
                    +xDiff[:,self.wfsConfig.pxlsPerSubap:])

        yDiff = (self.wfsDetectorPlane[:, :self.wfsConfig.pxlsPerSubap]-
                    self.wfsDetectorPlane[:, self.wfsConfig.pxlsPerSubap:]) 
        ySlopes = (yDiff[:self.wfsConfig.pxlsPerSubap, :]
                    +yDiff[self.wfsConfig.pxlsPerSubap:, :])


        self.slopes = numpy.append(xSlopes.flatten(), ySlopes.flatten())

    #Tilt optimisation
    ################################
    def calcTiltCorrect(self):
        """
        Calculates the required tilt to add to avoid the PSF being centred on 
        only 1 pixel
        """
        if not self.wfsConfig.pxlsPerSubap%2: 
            #Angle we need to correct 
            theta = self.subapFOVrad/ (2*self.FOV_OVERSAMP*self.FOVPxlNo)

            A = theta*self.telDiam/(2*self.wfsConfig.wavelength)*2*numpy.pi

            coords = numpy.linspace(-1,1,self.simConfig.pupilSize)
            X,Y = numpy.meshgrid(coords,coords)

            self.tiltFix = -1*A*(X+Y)
            
        else:
            self.tiltFix = numpy.zeros((self.simConfig.pupilSize,)*2)


def pyramid(phs):

    shapeX,shapeY = phs.shape
    quads = numpy.empty( (4,shapeX/2.,shapeX/2.), dtype="complex64")
    EField = numpy.exp(1j*phs) * aoSimLib.circle(shapeX/2,shapeX)
    FP = numpy.fft.fftshift(numpy.fft.fft2(EField))
    n=0
    for x in xrange(2):
        for y in xrange(2):
            quads[n] = FP[x*shapeX/2 : (x+1)*shapeX/2,
                        y*shapeX/2 : (y+1)*shapeX/2]
            n+=1

    PPupilPlane = abs(numpy.fft.ifft2(quads))**2

    allPupilPlane = numpy.empty( (shapeX,shapeY) )
    allPupilPlane[:shapeX/2.,:shapeY/2] = PPupilPlane[0]
    allPupilPlane[:shapeX/2.:,shapeY/2:] = PPupilPlane[1]
    allPupilPlane[shapeX/2.:,:shapeY/2] = PPupilPlane[2]
    allPupilPlane[shapeX/2.:,shapeY/2:] = PPupilPlane[3]

    return allPupilPlane

