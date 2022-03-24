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
from aotools.functions import circle

from .. import AOFFT, LGS, logger
from . import wfs
import pyfftw

# xrange now just "range" in python3.
# Following code means fastest implementation used in 2 and 3
try:
    xrange
except NameError:
    xrange = range

# The data type of data arrays (complex and real respectively)
CDTYPE = numpy.complex64
DTYPE = numpy.float32

# Parameters use for the detector mask
MOD_MASK = 32
AMPL_MASK = 1/numpy.pi

# Deflaut wavelength of Soapy
LAMBDA_0 = 500e-9


def rebin(arr, new_shape):
    shape = (new_shape[0], arr.shape[0] // new_shape[0],
             new_shape[1], arr.shape[1] // new_shape[1])
    return arr.reshape(shape).mean(-1).mean(1)

def calcPhaseScreenTipTilt(input_array, tip_value, tilt_value):
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

    # oversampling for the first FFT from EField to focus (4 seems ok...)
    FOV_OVERSAMP = 4

    def calcInitParams(self):
        """
            Define essentials parameters of the PWS
            Set the different masks used in the simulation

        """
        super(Pyramid, self).calcInitParams()

        # ---- prism param ---
        self.pupil_separation = self.wfsConfig.pupil_separation
        
        # ---- Modulation param ---
        self.nb_modulation =  self.wfsConfig.nb_modulation
        self.amplitude_modulation =  self.wfsConfig.amplitude_modulation / 206265
        
        # ---  Sampling parameters
        self.size_array = self.sim_size
        
         # ---- Detector parameters ----
        self.nx_subaps = self.wfsConfig.nxSubaps
        self.bin = int(self.pupil_size / self.nx_subaps)

        self.detector_size = int(self.size_array / self.bin)
        # self.detector_size = self.wfsConfig.detector_size
        self.detector = self.wfsConfig.detector
        
        self.count=0
        
        # ---- GS Parameters
        self.GSMag = self.wfsConfig.GSMag
        self.FOV = self.wfsConfig.FOV/206265    # Convert Fov from arcsec to rad
        
        # ---- Masks Creation ----
        self.initFFTs()
        self.calcPyramidMask()
        self.calcCentroidMask()
        self.calcDetectorMask()
        self.calcTipTiltModulationsPoints()
        self.calcModulationMatrix()
        self.calcFlatSlopes()
        
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
        self.focal_plane =  pyfftw.empty_aligned(
                (max(MOD_MASK,self.nb_modulation),self.size_array,self.size_array), dtype=CDTYPE)

    def calcCentroidMask(self):
        """
        Centroid mask, use to shif of 1 pixel the inputphase screen to center for the fft
        """
        pupil_plane = numpy.zeros((self.size_array,self.size_array))
        self.centroidMask = calcPhaseScreenTipTilt(pupil_plane, - numpy.pi / (self.size_array ), 
                                                 numpy.pi / (self.size_array))

    def calcDetectorMask(self):
        """
        To use only the illuminated pixel
        We illuminate de 4 faces with a flat phase screen and get the correpondent detector array
        Then we keep the pixel with a value above the threshold
        Also give the slope size

        Parameters
        ----------
        threshold : Float, optional
            Threshold to set the detector maks. The default is 0.7

        """

        # Get the pupil images on the detector plane
    
        half_size = int(self.detector_size/2)
        
        
        submask = circle(int(self.nx_subaps/2),self.nx_subaps)

        
        QuadMask1 = numpy.zeros((half_size,half_size))
        
        begin = int(half_size-(self.nx_subaps+self.pupil_separation/2)+0.5)
        end = int(half_size-(self.pupil_separation/2)+0.5)
        QuadMask1[begin:end,begin:end] = submask

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
        

        
    def calcPyramidMask(self):
        """
        Creation of the pyramid mask that splits the wavefront into 4 pupils images
        The pyramid is considered symetrical ie each face has the same angle.
        
        """
        #Calc Apex angle
        self.apex = ((self.pupil_size/self.nx_subaps)*self.pupil_separation \
                     + self.pupil_size)*(1/self.size_array) * numpy.pi

        quad = numpy.zeros((int(self.size_array/2),int(self.size_array/2)))
        quad1 = calcPhaseScreenTipTilt (quad, -self.apex,  self.apex)
        quad2 = calcPhaseScreenTipTilt (quad, self.apex,  self.apex)
        quad3 = calcPhaseScreenTipTilt (quad, -self.apex, -self.apex)
        quad4 = calcPhaseScreenTipTilt (quad, self.apex, -self.apex)
        
        self.pyramid_mask = numpy.append(numpy.append(quad1,quad2, axis=1),numpy.append(quad3,quad4, axis = 1),axis = 0)
        
        #Complex phase screen
        self.pyramid_mask_phs_screen = numpy.exp(1j*self.pyramid_mask)
        
    def calcTipTiltModulationsPoints(self, ampl=None, num_of_mod = None):
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
            ampl =  self.amplitude_modulation *  (self.telescope_diameter)/(self.wavelength * self.pupil_size) * 2*numpy.pi 
        
        if num_of_mod ==None:
            num_of_mod = self.nb_modulation
        
        #Endpoint=Flase prevents double counting of the theta = 0 position
        theta = numpy.linspace(0, 2*numpy.pi, num = num_of_mod, endpoint=False)
        tip = []
        tilt = []
        for t in theta:
            tip.append(ampl*numpy.cos(t+(numpy.pi/4)))
            tilt.append(-ampl*numpy.sin(t+(numpy.pi/4)))
            
        self.tip_mod = tip
        self.tilt_mod = tilt
        
    def calcModulationMatrix(self, num_of_mod=None):
        
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


        self.modulation_phase_matrix = numpy.zeros((num_of_mod, self.size_array,
                                                    self.size_array),
                                                       dtype=DTYPE)
        # Get the phase screens
        for i in range (num_of_mod):
            self.modulation_phase_matrix[i] = \
                calcPhaseScreenTipTilt( self.modulation_phase_matrix[0],
                                       self.tip_mod[i],self.tilt_mod[i])
                
        
    def calcFocalPlane(self, num_of_mod=None, intensity=1, maskd = False):
        """
        

        Parameters
        ----------
        num_of_mod : float, optional. Number of the modulation. If none it is define by
        the input parameters of the simulation
        intensity : TYPE, optional
            DESCRIPTION. The default is 1.

        Returns
        -------
        None.

        """       
        
        
        if num_of_mod==None:
            num_of_mod = self.nb_modulation
            
                
        if maskd :
            phase  = self.mask
        elif self.iMat :
            phase =  self.los.phase 
            # plt.figure()
            # plt.imshow(self.los.phase)
            # plt.colorbar()
            # plt.savefig(str(self.count)+'.png')
            # self.count+=1
            # plt.close()
        else:
            phase = self.los.phase     

            
        #set phase screen
        phase_screen = (phase  + self.modulation_phase_matrix + self.centroidMask)\
                                    * intensity * self.mask
        complex_phase_screen = self.mask * numpy.exp(1j * phase_screen)
        

        
       
        # ---- Fourier filter with a pyramid mask ----
        self.fft_input_data[:num_of_mod,:,:] = complex_phase_screen
        self.FFT()
        coord = int(self.size_array/4)
        self.focal_plane= numpy.fft.fftshift(self.fft_output_data)
        # self.focal_plane[:,coord:-coord,coord:-coord] = numpy.fft.fftshift(self.fft_output_data)[:,coord:-coord,coord:-coord]
        self.ifft_input_data[:,:,:] = self.focal_plane*self.pyramid_mask_phs_screen
        self.iFFT()
        self.wfsDetectorPlane  = numpy.sum((numpy.abs(self.ifft_output_data)**2),axis=0)
        
        if self.bin != 1:
            new_shape = int(self.size_array/self.bin)
            self.wfsDetectorPlane = rebin(self.wfsDetectorPlane, [new_shape,new_shape])
        if not maskd :
            self.wfsDetectorPlane /= num_of_mod
            
        # ---- Add read noise ----
        if self.config.eReadNoise!=0:
            self.addReadNoise()


    def calculateSlopes(self,flat = False):
        """
        Get the detector plane and compute the slopes according to the formula in the Raggazzoni et al. 1996            

        """
        
        #Get each pupil image
        x = numpy.array(numpy.split(self.wfsDetectorPlane*self.detector_mask,2, axis=0))
        quad1,quad2=numpy.split(x[0],2, axis=1)
        quad3,quad4=numpy.split(x[1],2, axis=1)
    
        #Transform in intensity vertor
        # I use the mask because if there is a 0 in the 
        x_d = numpy.array(numpy.split(self.detector_mask,2, axis=0))
        quad_d1,quad_d2=numpy.split(x_d[0],2, axis=1)
        quad_d3,quad_d4=numpy.split(x_d[1],2, axis=1)
    
        quad1 = quad1.flatten()
        quad_d1 = quad_d1.flatten()
        quad1 = numpy.abs(quad1[quad_d1!=0])
        
        quad2 = quad2.flatten()
        quad_d2 = quad_d2.flatten()
        quad2 = numpy.abs(quad2[quad_d2!=0])
        
        quad3 = quad3.flatten()
        quad_d3 = quad_d3.flatten()
        quad3 = numpy.abs(quad3[quad_d3!=0])
        
        quad4 = quad4.flatten()
        quad_d4 = quad_d4.flatten()
        quad4 = numpy.abs(quad4[quad_d4!=0])


        #Gradient according to the 2 axis of the detector plane
        sum_int = numpy.sum(( quad1, quad2, quad3, quad4 ))
        self.Xslopes = (quad1 + quad2 - (quad3 + quad4)) / sum_int
        self.Yslopes = (quad1 + quad3 - (quad2 + quad4)) / sum_int

        if flat :
            self.slopes = numpy.append(self.Yslopes, self.Xslopes)
        else :
            self.slopes = numpy.append(self.Yslopes, self.Xslopes) - self.flat_slopes

        
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
    
    def calcFlatSlopes(self):
        self.calcFocalPlane( maskd = True)
        self.calculateSlopes(flat = True)
        self.flat_slopes = self.slopes
        
        
        
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
