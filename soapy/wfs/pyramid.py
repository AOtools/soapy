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

Module of the Pyramid wavefront sensor (PWS)


The PWS is simulated using following the 'Pyramid' class. It is a 4 faces model
Pupil phase or complex amplitude is recieved from the line of sight on the WFS, via the base WFS class.
A modulated PSF is then constructed according to input parameters (r_modulation (amplitude), nb_modulation (number of modulations))
Then a phase mask representing the pyramid is applied.
Finally slopes are constructed using the gradient of the intensities 4 pupil images on the detector plane

Author : Aurelie Magniez


"""

import numpy 
import numpy.random
from . import wfs
import pyfftw


from .. import LGS, logger, lineofsight, AOFFT, interp

# xrange now just "range" in python3.
# Following code means fastest implementation used in 2 and 3
try:
    xrange
except NameError:
    xrange = range


# The data type of data arrays (complex and real respectively)
CDTYPE = numpy.complex64
DTYPE = numpy.float32

FFT_OVERSAMP = 10
MOD_MASK = 4

def phase_screen_tip_tilt(input_array, tip_value, tilt_value):
    """
    Create a phase mask of complex exponentials with modulus one and an appropriate phase at each element
    for the given tip and tilt angles (in radians)
    
    Param:
        - input_array : use to define the size of the phase screen. Typically, give the phase screen the tip-tilt aberration
                        will be aplied to
        - tip_value, tilt_value
    output
        - phase screen
    
    """
    #Positive tip = moves PSF to right, positive tilt = moves PSF up
    sizeX, sizeY = input_array.shape
    tip = numpy.arange(sizeX)*tip_value
    tilt = numpy.arange(sizeY)*tilt_value
    tilt = tilt-tilt.mean()

    tip_phase = (numpy.ones((sizeX,sizeY), dtype = DTYPE)*tip-tip.mean())
    tilt_phase = numpy.ones((sizeX,sizeY), dtype = DTYPE)*tilt-tilt.mean()
    
    return tip_phase-tilt_phase.T


class Pyramid(wfs.WFS):
    """
       Class to simulate de 4-faces Pyramid wavefront sensor
    """         

        

    def calcInitParams(self):
        """
        Define essentials parameters of the PWS
        Set the different masks used in the simulation

        """
        
        super(Pyramid, self).calcInitParams()
        
        
        # ---- prism param ---
        self.apex =  self.wfsConfig.apex_prism*numpy.pi
        
        # ---- Modulation param ---
        self.nb_modulation =  self.wfsConfig.nb_modulation
        self.amplitude_modulation =  self.wfsConfig.amplitude_modulation
        
        
        # ---- PWS sim param ----
        
        # define the size array of the simulation
        if self.soapy_config.sim.simOversize > 1.2:
            self.size_array = self.sim_size
            self.oversamp = False
            self.phase_size = self.pupil_size
            
            self.fovScale = 2*numpy.pi/self.pupil_size
            
        else : 
            self.size_array = FFT_OVERSAMP * self.pupil_size
            self.oversamp = True
            self.phase_size = self.mask.shape[0]
            
            #remove the edges of the mask due to 1.2 oversampling by default

            self.fovScale = 2*numpy.pi/self.pupil_size

        # ---- Rescaling parameters ---
        self.FOV = self.wfsConfig.FOV/206265 # Convert Fov from arcsec to rad
        
        # New size of the pupil plane
        self.interp_size = int (self.telescope_diameter*self.FOV/self.wavelength)
        if self.interp_size %2 !=0: # To have an even number
            self.interp_size+=1
            
            
        # Final number of pixels (subaps) for the pupil images on the detector plane
        self.nx_subaps = self.wfsConfig.nxSubaps
        
        if self.interp_size > self.nx_subaps:
            logger.warning(" The interpolation will make you loose information :\n\
                                telescope_diameter*FOV/wavelength = {} should be superior to nxSubaps = {}"\
                                .format(self.interp_size, self.nx_subaps))
        
        # Detector size, to have the right number of subapertures
        self.detector_interp_size = int(self.nx_subaps*self.size_array/self.interp_size)
        if self.detector_interp_size %2 !=0:
            self.detector_interp_size+=1
        
        
        # Final detector size, if the user defined a detector size bigger than the number of used pixels
        # the final size is the used pixel
        ds = self.wfsConfig.detector_size
        if ds and ds < self.detector_interp_size:
            self.detector_size = self.wfsConfig.detector_size
        else :
            self.detector_size = self.detector_interp_size
            logger.warning("WARNING :the user defined a detector size bigger than the number of used pixels")

        # ---- Masks Creation ----
        self.initFFTs()
        self.set_pyramid_mask()
        self.set_centroid_mask()
        self.set_detector_mask()
        self.get_tip_tilt_modulations_points()
        self.set_modulation_matrix()
        
        self.n_measurements = self.SlopeSize
        
        self.los.allocDataArrays()
                

    def initFFTs(self):
        
        """
            Initialisation of the FFT objects, allowing multithreading.
        """
        pyfftw.config.NUM_THREADS = self.threads
        self.fft_input_data = pyfftw.empty_aligned(
                (max(MOD_MASK,self.nb_modulation),self.size_array,self.size_array), dtype=CDTYPE)
        self.fft_output_data = pyfftw.empty_aligned(
                (max(MOD_MASK,self.nb_modulation),self.size_array,self.size_array), dtype=CDTYPE)
        self.FFT = pyfftw.FFTW(
                self.fft_input_data, self.fft_output_data,axes=(-2,-1),
                threads=self.threads, flags=(self.config.fftwFlag, "FFTW_DESTROY_INPUT"))
        
        self.ifft_input_data = pyfftw.empty_aligned(
                (max(MOD_MASK,self.nb_modulation),self.size_array,self.size_array), dtype=CDTYPE)
        self.ifft_output_data = pyfftw.empty_aligned(
                (max(MOD_MASK,self.nb_modulation),self.size_array,self.size_array), dtype=CDTYPE)
        self.iFFT = pyfftw.FFTW(
                self.ifft_input_data, self.ifft_output_data, axes=(-2,-1),
                threads=self.threads, flags=(self.config.fftwFlag, "FFTW_DESTROY_INPUT"),
                direction="FFTW_BACKWARD"
                )


    def set_centroid_mask(self):
        """
        Centroid mask, use to shif of 1 pixel the inputphase screen to center for the fft
        """
        pupil_plane = numpy.zeros((self.phase_size,self.phase_size))
        self.centroidMask = phase_screen_tip_tilt(pupil_plane, -2*numpy.pi/(2*(self.size_array )), 
                                                  2*numpy.pi/(2*(self.size_array)))


    def set_detector_mask(self, threshold = 0.1):
        """
        To use only the illuminated pixel
        We illuminate de 4 faces with a flat phase screen and get the correpondent detector array
        Then we keep the pixel with a value above the threshold
        Also give the slope size

        Parameters
        ----------
        threshold : Float, optional
            Threshold to set the detector maks. The default is 0.1.

        """

        # Get the pupil images on the detector plane
        self.get_tip_tilt_modulations_points(ampl = 4, num_of_mod=4)
        self.set_modulation_matrix(num_of_mod=4)
        self.calcFocalPlane( num_of_mod=4)
        
        # half_size = int(self.size_array/2)
        half_size = int(self.detector_size/2)
        
        # split in 4 and define the mask
        inputArray = numpy.abs(self.wfsDetectorPlane[0:half_size, 0:half_size])**4

        QuadMask1 = numpy.where(inputArray>threshold, 1, 0)
        QuadMask2 = numpy.flip(QuadMask1, axis=1)
        QuadMask3 = numpy.flip(QuadMask1, axis = 0)
        QuadMask4 = numpy.flip(QuadMask3, axis=1)
        

        # Build the mask        
        # self.detector_mask = numpy.zeros((self.size_array, self.size_array))
        self.detector_mask = numpy.zeros((self.detector_size, self.detector_size))
        
        self.detector_mask[0:half_size, 0:half_size] = QuadMask1
        self.detector_mask[0:half_size, half_size:2*half_size] = QuadMask2
        self.detector_mask[half_size:2*half_size, 0:half_size] = QuadMask3
        self.detector_mask[half_size:2*half_size, half_size:2*half_size] = QuadMask4
        
        self.SlopeSize = numpy.count_nonzero(QuadMask1)*2
        
        # Clean the class to be sure it is not use again in that modulation configuration
        del self.modulation_phase_matrix
        del self.wfsDetectorPlane
        del self.tip_mod
        del self.tilt_mod
        

        
    def set_pyramid_mask(self):
        """
        Creation of the pyramid mask that splits the wavefront into 4 pupils images
        The pyramid is considered symetrical ie each face has the same angle.
        
        """

        quad = numpy.zeros((int(self.size_array/2),int(self.size_array/2)))
        quad1 = phase_screen_tip_tilt (quad, -self.apex,  self.apex)
        quad2 = phase_screen_tip_tilt (quad, self.apex,  self.apex)
        quad3 = phase_screen_tip_tilt (quad, -self.apex, -self.apex)
        quad4 = phase_screen_tip_tilt (quad, self.apex, -self.apex)
        
        self.pyramid_mask = numpy.append(numpy.append(quad1,quad2, axis=1),numpy.append(quad3,quad4, axis = 1),axis = 0)
        
        #Complex phase screen
        self.pyramid_mask_phs_screen = numpy.exp(1j*self.pyramid_mask)
        
        
        
    def get_tip_tilt_modulations_points(self, ampl=None, num_of_mod = None):
        """
        Define the modulations according to the parameters

        Parameters
        ----------
        ampl : float, optional. Amplitude of the modulations. If none it is define by the input parameters
                of the simulation
        num_of_mod : float, optional. Number of the modulation. If none it is define by the input parameters
                of the simulation

        """
        
        if ampl==None:
            ampl = self.amplitude_modulation
        if num_of_mod ==None:
            num_of_mod = self.nb_modulation
        
        #Endpoint=Flase prevents double counting of the theta = 0 position
        theta = numpy.linspace(0, 2*numpy.pi, num = num_of_mod, endpoint=False)
        tip = []
        tilt = []
        for t in theta:
            tip.append(ampl*numpy.cos(t+(numpy.pi/4))*self.fovScale)
            tilt.append(-ampl*numpy.sin(t+(numpy.pi/4))*self.fovScale)
            
        self.tip_mod = tip
        self.tilt_mod = tilt
        

    def set_modulation_matrix(self, num_of_mod=None):
        """
        Creation of the modulation matrix. Creates num_modulation tip-tilt aberrated phase screens
        correnpondent of the modulation points define by the function 'tip_tilt_modulation'

        Parameters
        ----------
        num_of_mod : float, optional. Number of the modulation. If none it is define by the input parameters
                of the simulation

        """
        if num_of_mod==None:
            num_of_mod = self.nb_modulation

        #Data allocation
        if self.oversamp:
            self.modulation_phase_matrix = numpy.zeros((num_of_mod, self.phase_size, self.phase_size),
                                                       dtype=DTYPE)
        else:
            self.modulation_phase_matrix = numpy.zeros((num_of_mod, self.phase_size, self.phase_size),
                                                       dtype=DTYPE)
            
            
        # Get the phase screens
        for i in range (num_of_mod):
            self.modulation_phase_matrix[i]=phase_screen_tip_tilt( self.modulation_phase_matrix[0],
                                                                    self.tip_mod[i],self.tilt_mod[i])
            
            
            

    def calcFocalPlane(self, num_of_mod=None):
        """
        Calculates the detector plane corresponding to the phase screen generated by the los class.
        This detector plane consists of an intensity map of the 4 pupil images.
        It takes into account modulation and noise.
                
        Parameters
        ----------
        num_of_mod : float, optional. Number of the modulation. If none it is define by the input parameters
                of the simulation
                
        Returns
        -------
        None.

        """       
        
        if num_of_mod==None:
            num_of_mod = self.nb_modulation
            
            
        # Correction of the phi2rad phase conversion 
        # self.phs2Rad = 2*numpy.pi/(self.wavelength * 10**9)
        if self.iMat :
            phase =  self.los.phase
        else:
            phase = self.los.phase * self.wavelength * 10**9 / (2*numpy.pi)*0.003
        #set phase screen
        if not self.oversamp:
            mid = int(numpy.abs(self.pupil_size-self.size_array)/2)
            mask =  self.mask[mid:-mid,mid:-mid]
            phase_screen = (phase[mid:-mid,mid:-mid] * mask + self.modulation_phase_matrix + self.centroidMask) 
        
        else:
            mask =  self.mask
            phase_screen = (phase * mask + self.modulation_phase_matrix + self.centroidMask) 
            
        
        
        interp_phase = numpy.zeros((num_of_mod,self.interp_size, self.interp_size ), dtype=CDTYPE)
        for n in range(num_of_mod):
            interp_phase[n] = interp.zoom(phase_screen[n], [self.interp_size, self.interp_size], order = 1)
        
        reshaped_mask = interp.zoom(mask, [self.interp_size, self.interp_size], order = 1)
        complex_phase_screen = reshaped_mask*numpy.exp(1j*interp_phase)
        
        # middle = int(self.size_array/2 - self.phase_size/2)
        middle = int(self.size_array/2 - self.interp_size/2)
        self.fft_input_data[:num_of_mod,middle:-middle,middle:-middle] = complex_phase_screen
       
        # ---- Fourier filter with a pyramid mask ----
        self.FFT()
        focal_plane = numpy.fft.fftshift(self.fft_output_data)
        self.ifft_input_data[:,:,:] = focal_plane*self.pyramid_mask_phs_screen
        self.iFFT()
        
        
        DetectorPlane = numpy.sum((numpy.abs(self.ifft_output_data)**2),axis=0)
        # self.wfsDetectorPlane = numpy.sum((numpy.abs(self.ifft_output_data)**2),axis=0)
        DetectorPlane = interp.zoom(DetectorPlane, [self.detector_interp_size, self.detector_interp_size])
        mid = int(numpy.abs(self.detector_interp_size-self.detector_size)/2)
        if mid != 0:
            self.wfsDetectorPlane = DetectorPlane[mid:-mid, mid:-mid]
        else:
            self.wfsDetectorPlane = DetectorPlane

        # ---- Add read noise ----
        if self.config.eReadNoise!=0:
            self.addReadNoise()

            
            
    def calculateSlopes(self):
        """
        Get the detector plane and compute the slopes according to the formula in the Raggazzoni et al. 1996            

        """
        
        #Get each pupil image
        x =numpy.array(numpy.split(self.wfsDetectorPlane*self.detector_mask,2, axis=0))
        quad1,quad2=numpy.split(x[0],2, axis=1)
        quad3,quad4=numpy.split(x[1],2, axis=1)
    
        #Transform in intensity vertor
        quad1 = quad1.flatten()
        quad1 = numpy.abs(quad1[quad1!=0])
        quad2 = quad2.flatten()
        quad2 = numpy.abs(quad2[quad2!=0])
        quad3 = quad3.flatten()
        quad3 = numpy.abs(quad3[quad3!=0])
        quad4 = quad4.flatten()
        quad4 = numpy.abs(quad4[quad4!=0])
    
        #Gradient according to the 2 axis of the detector plane
        self.Yslopes = (quad1+quad2-(quad3+quad4))/ (quad1+quad2 + quad3+quad4)
        self.Xslopes = (quad1+quad3-(quad2+quad4))/ (quad1+quad2 + quad3+quad4)

        self.slopes = numpy.append(self.Yslopes,self.Xslopes)
        

    
    def get_flatphase_slopes(self):
        """
        Use to get theanswer of the PWS of a flat phase, ie the null phase.

        """
        
        self.calcFocalPlane(intensity = 0)
        self.Flats_slope = self.calculateSlopes()
 




    def addPhotonNoise(self):
        """
        Add photon noise to ``wfsDetectorPlane`` using ``numpy.random.poisson``
        """
        self.detector[:] = numpy.random.poisson(self.detector).astype(DTYPE)


    def addReadNoise(self):
        """
        Adds read noise to ``wfsDetectorPlane using ``numpy.random.normal``.
        This generates a normal (guassian) distribution of random numbers to
        add to the detector. Any CCD bias is assumed to have been removed, so
        the distribution is centred around 0. The width of the distribution
        is determined by the value `eReadNoise` set in the WFS configuration.
        """
        self.wfsDetectorPlane += numpy.random.normal(
                0, self.config.eReadNoise, self.wfsDetectorPlane.shape)
        
        
def photons_per_mag(mag, exposureTime):
    """
    Calculates the number of photons per guide star magnitude

    Parameters:
        mag (int): Magnitude of guide star
        mask (ndarray): 2-d pupil mask. 1 if aperture clear, 0 if not
        phase_scale (float): Size of pupil mask pixel in metres
        exposureTime (float): WFS exposure time in seconds
        zeropoint (float): Photometric zeropoint of mag 0 star in photons/metre^2/seconds

    Returns:
        float: photons per WFS frame
    """
    # ZP of telescope
    

    # N photons for mag and exposure time
    n_photons = (10**(-float(mag)/2.5)) * exposureTime

    return n_photons
