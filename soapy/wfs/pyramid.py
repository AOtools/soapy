import numpy
import numpy.random

from .. import AOFFT, LGS, logger, interp
from . import wfs

# xrange now just "range" in python3.
# Following code means fastest implementation used in 2 and 3
try:
    xrange
except NameError:
    xrange = range

# The data type of data arrays (complex and real respectively)
CDTYPE = numpy.complex64
DTYPE = numpy.float32


class Pyramid(wfs.WFS):
    """
    *Experimental* Pyramid WFS.

    This is an early prototype for a Pyramid WFS. Currently, its at a very early stage. It doesn't oscillate, so performance aint too good at the minute.

    To use, set the wfs parameter ``type'' to ``Pyramid'' type is a list of length number of wfs.
    """
    # oversampling for the first FFT from EField to focus (4 seems ok...)
    FOV_OVERSAMP = 4
    def __init__(self, soapy_config, n_wfs=0, mask=None):

        super(Pyramid, self).__init__(
            soapy_config, n_wfs, mask)
        
#        self.pupilMask()
        #sys.exit(1)
    
    

    def calcInitParams(self):
        print ('Calc init parameters')
        super(Pyramid, self).calcInitParams()
        self.pyrtip =  self.wfsConfig.pyrtt[0]*numpy.pi
        self.pyrtilt =  self.wfsConfig.pyrtt[1]*numpy.pi
        self.nb_modulation =  self.wfsConfig.nb_modulation
        self.r_modulation =  self.wfsConfig.r_modulation
 
        self.FOVrad = self.wfsConfig.subapFOV * numpy.pi / (180. * 3600)

        self.FOVPxlNo = int(numpy.round(self.telescope_diameter *
                                    self.FOVrad/self.wfsConfig.wavelength))

        self.detectorPxls = 2*self.wfsConfig.pxlsPerSubap
        self.scaledMask = interp.zoom(self.mask, self.FOVPxlNo)



        


    def initFFTs(self):
        
        print("init FFTs")

        self.FFT = AOFFT.FFT(   self.los.phase.shape,
                                axes=(0,1), mode="pyfftw",
                                fftw_FLAGS=("FFTW_DESTROY_INPUT",
                                            self.wfsConfig.fftwFlag),
                                THREADS=self.wfsConfig.fftwThreads
                                )


    def allocDataArrays(self):

        super(Pyramid, self).allocDataArrays()
        # Allocate arrays
        # Find sizes of detector planes

        self.paddedDetectorPxls = (2*self.wfsConfig.pxlsPerSubap
                                    *self.wfsConfig.fftOversamp)
        self.paddedDetectorPlane = numpy.zeros([self.paddedDetectorPxls]*2,
                                                dtype=DTYPE)

        self.focalPlane = numpy.zeros(  self.los.phase.shape,
                                        dtype=CDTYPE)

        self.wfsDetectorPlane = numpy.zeros([self.detectorPxls]*2,
                                            dtype=DTYPE)
        

        self.pupilMask()
        self.slopes = numpy.zeros(self.slopeSize)
        
        self.derivmask = numpy.zeros_like(self.wfsDetectorPlane)
        

        self.activeSubaps = self.slopeSize
        while (self.wfsConfig.pxlsPerSubap*self.wfsConfig.fftOversamp
                    < self.FOVPxlNo):
            self.wfsConfig.fftOversamp += 1
        self.n_measurements = self.activeSubaps

    def zeroData(self, detector=True, FP=True):
        """
        Sets data structures in WFS to zero.

        Parameters:
            detector (bool, optional): Zero the detector? default:True
            inter (bool, optional): Zero intermediate arrays? default:True
        """

        self.zeroPhaseData()

        if FP:
            self.paddedDetectorPlane[:] = 0

        if detector:
            self.wfsDetectorPlane[:] = 0
            
            
    def initLos(self):
        """
        Initialises the ``LineOfSight`` object, which gets the phase or EField in a given direction through turbulence.
        """
        self.los = lineofsight.LineOfSight(
                self.config, self.soapy_config,
                propagation_direction="down")

#
    def calcFocalPlane(self):
        '''
        takes the calculated pupil phase, and uses FFT
        to transform to the focal plane, and scales for correct FOV.
        '''
#
        self.pupilEField = self.los.phase*self.mask
        
        #if it is the calibration phase : modulations 
        if self.pupmask:
            self.pupilEField = self.mask
            self.modulation(nb_modulation=4)
        
        self.modulation()
        self.pupilEField = numpy.sum(self.pupilEField*self.phaseModMatrices, axis = 0)



        # Go to the focus
        self.FFT.inputData[:] = 0
        self.FFT.inputData[:] = self.pupilEField
        self.focalPlane[:] = AOFFT.ftShift2d( self.FFT() )
        
        self.pyramidMask()
                
        self.wfsDetectorPlane = fp.ifft2(self.focalPlane* self.pyramid_mask) 
        self.wfsDetectorPlane = numpy.abs(self.wfsDetectorPlane)**2



    def calculateSlopes(self):
        
        """
            Calculate the slopes of the pyramid WFS
            First, application of the pupil mask. 
            Size of the slopes : fixed by the mask
        """
        
        maskedpupil = self.pupil_mask*self.wfsDetectorPlane

        centre_x = numpy.int(numpy.shape(maskedpupil)[1]/2) #This will onlywork considering we are always using an array of size 2*N so the size is always even
        centre_y = numpy.int(numpy.shape(maskedpupil)[0]/2)
        
        quad1 =( maskedpupil [0:centre_x, 0: centre_y]).flatten()
        quad1 = quad1[quad1 !=0]
        quad2 =( maskedpupil [centre_x::, 0: centre_y]).flatten()
        quad2 = quad2[quad2 !=0]
        quad3 =( maskedpupil [0:centre_x, centre_y::]).flatten()
        quad3 = quad3[quad3 !=0]
        quad4 =( maskedpupil [centre_x::, centre_y::]).flatten()
        quad4 = quad4[quad4 !=0]
        
        xSlopes = (quad1 + quad3) - (quad2 + quad4)/ (quad1 + quad2 + quad3 + quad4)
        ySlopes = (quad1 + quad2) - (quad3 + quad4)/ (quad1 + quad2 + quad3 + quad4)
    

        self.slopes = numpy.abs(numpy.append(xSlopes, ySlopes))

    


            
    def pyramidMask(self):
        """
            Creation of the pyramid mask. 
            The mask will split the beam in 4 beams
        """
        
        self.pyramid_mask =  numpy.zeros_like(self.focalPlane, dtype='complex') #This step is necessary otherwise it overwrites the input array...
        centre_x = numpy.int(numpy.shape(self.focalPlane)[1]/2) #This will onlywork considering we are always using an array of size 2*N so the size is always even
        centre_y = numpy.int(numpy.shape(self.focalPlane)[0]/2)
        quadrant_array = numpy.zeros([centre_y, centre_x])
    
        #Create the top-left quadrant of the phase mask
        self.pyramid_mask[0:centre_y, 0:centre_x] = self.phaseMask(quadrant_array, self.pyrtip, self.pyrtilt)
        #Create the top-right quadrant of the phase mask
        self.pyramid_mask[0:centre_y, centre_x:numpy.shape(self.focalPlane)[1]] = self.phaseMask(quadrant_array, -self.pyrtip, self.pyrtilt)
        #Create the bottom-left quadrant of the phase mask
        self.pyramid_mask[centre_y:numpy.shape(self.focalPlane)[0], 0:centre_x] = self.phaseMask(quadrant_array, self.pyrtip, -self.pyrtilt)
        #Create the bottom-right quadrant of the phase mask
        self.pyramid_mask[centre_y:numpy.shape(self.focalPlane)[0], centre_x:numpy.shape(self.focalPlane)[1]] = self.phaseMask(quadrant_array, -self.pyrtip, -self.pyrtilt)
        

    def phaseMask(self, input_array, tip_value, tilt_value):
        '''Create a phase mask of complex exponentials with modulus one and an appropriate phase at each element
        for the given tip and tilt angles (in radians) - only to be used in the focal plane (to construct the pyramid)'''
        tip = numpy.arange(numpy.shape(input_array)[1])*tip_value
        tilt = numpy.arange(numpy.shape(input_array)[0])*tilt_value
        phase_tip_tilt = numpy.zeros((2, numpy.shape(input_array)[0], numpy.shape(input_array)[1]))
        phase_tip_tilt[0,:] = -(tip - tip.mean()) #Tip is about x-axis so need horizontal lines of constant phase, not sure why it needs to be subtracted though - because it's an IFFT perhaps?
        phase_tip_tilt[1,:] = tilt - tilt.mean()
        phase_tip_tilt[1,:] = -phase_tip_tilt[1,:].T
        phase_tip_tilt_tot = phase_tip_tilt.sum(axis=0)
    
        phase_matrix =  numpy.exp(1j*phase_tip_tilt_tot)
    
        return phase_matrix
    
    
    
    def pupilMask(self):
        """
            Creation of pupil mask on the detector plane.
            The function will create a mask on the detector plane
                1. Compute the detector plane with a flat wavefront with 4 modultations
                2. Stack the 4 Detector planes
                3. Get the mask where the intensity>0.1
                
                Is created at the initialization of the wfs.
        """
        #self.los.phase = numpy.ones_like(self.mask)
        self.iMat = True # to have a modulation
        self.pupmask = True
        self.calcFocalPlane()
        self.pupmask = False

         #Check the sizes of the quadrant (in case the matrix isn't square)
        QuadSizeY = numpy.int(numpy.shape(self.wfsDetectorPlane)[0]/2)
        QuadSizeX = numpy.int(numpy.shape(self.wfsDetectorPlane)[1]/2)
    
        #Separate out the top left quadrant (Derivs_tt values don'tmatter as long as they're large enough to move the )
        PupilQuadrant = self.wfsDetectorPlane[0:QuadSizeY, 0:QuadSizeX]
    
        #Take the absolute value and square the amplitude to get the intensity in the detector plane
        PupilQuadAbs = numpy.abs(PupilQuadrant)**2
        #Produce the mask - where the intensity >mean(pupil quad) make the value 1, else 0
        #QuaskMask1 = Top left, QuadMask2 = Top right, QuadMask3 = Bottom Left, QuadMask4 = Bottom Right
        QuadMask1 = numpy.where(PupilQuadAbs>numpy.mean(PupilQuadAbs), 1, 0)
        QuadMask2 = numpy.flip(QuadMask1, axis=1)
        QuadMask3 = numpy.flip(QuadMask1, axis = 0)
        QuadMask4 = numpy.flip(QuadMask3, axis=1)
    
        #Put the quadrants back together to produce the final derivatives mask
        self.pupil_mask = numpy.zeros_like(self.wfsDetectorPlane)
        self.pupil_mask[0:QuadSizeY, 0:QuadSizeX] = QuadMask1
        self.pupil_mask[0:QuadSizeY, QuadSizeX:2*QuadSizeX] = QuadMask2
        self.pupil_mask[QuadSizeY:2*QuadSizeY, 0:QuadSizeX] = QuadMask3
        self.pupil_mask[QuadSizeY:2*QuadSizeY, QuadSizeX:2*QuadSizeX] = QuadMask4
        
        plt.figure()
        plt.imshow(self.pupil_mask)

        self.slopeSize = numpy.count_nonzero(QuadMask1)*2
    
        
        
        
    def modPoints(self, nb_modulation, r):
        """
            Will create a list of tip and tilts according to inputs parameters :
                r = radius of the 
                nb_modulation = number of modulations (min should be 4)
            Output :
                tip, tilt : list of the angles used for the modulation
        """
        theta = numpy.linspace(0, 2*numpy.pi, num = nb_modulation, endpoint=False)
        tip = []
        tilt = []
        for t in theta:
            tip.append(r*numpy.cos(t+(numpy.pi/4))*numpy.pi)
            tilt.append(-r*numpy.sin(t+(numpy.pi/4))*numpy.pi)
            
        return tip, tilt
       
    def modulation (self, nb_modulation =None, r = None):
        
        """
            Create an array of modulation phases screens that have to be multiplied to the input phase screen
            
        """
        if nb_modulation == None:
            nb_modulation = self.nb_modulation
        
        if r == None:
            r = self.r_modulation
            
        #Get the tip ad tilt values
        tip_list, tilt_list = self.modPoints(nb_modulation, r)
        
        self.phaseModMatrices = numpy.zeros((nb_modulation, numpy.shape(self.pupilEField)[0], numpy.shape(self.pupilEField)[1]), dtype=complex)
        
        for i in range (nb_modulation):
            tip = numpy.arange(numpy.shape(self.pupilEField)[1])*tip_list[i]
            tilt = numpy.arange(numpy.shape(self.pupilEField)[0])*tilt_list[i]
            phase_tip_tilt = numpy.zeros((2, numpy.shape(self.pupilEField)[0], numpy.shape(self.pupilEField)[1]))
            phase_tip_tilt[0,:] = (tip - tip.mean()) #Tip is about x-axis so need horizontal lines of constant phase
            phase_tip_tilt[1,:] = tilt - tilt.mean()
            phase_tip_tilt[1,:] = -phase_tip_tilt[1,:].T
            phase_tip_tilt_tot = phase_tip_tilt.sum(axis=0)
        
            self.phaseModMatrices[i] = numpy.exp(1j*phase_tip_tilt_tot)
        