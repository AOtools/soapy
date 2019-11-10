"""
A 'Zernike WFS' that can magically detect the Zernikes in the phase. 

Simply keeps an array of Zernikes that will be dotted wiht hte phase 
to calculate the Zernike coefficients. An imaginary simulated WFS that 
doesnt represent a physical device, just useful for testing purposes.
"""
import numpy

import aotools

from . import base


class Zernike(base.WFS):

    def calcInitParams(self, phaseSize=None):
        # Call the case WFS methoc
        super(Zernike, self).calcInitParams(phaseSize=phaseSize)

        # Generate an array of Zernikes to use in the loop
        self.n_zerns = self.config.nxSubaps

        self.n_measurements = self.n_zerns

        # Make Zernike array
        self.zernike_array = aotools.zernikeArray(self.n_zerns, self.pupil_size)
        self.zernike_array.shape = self.n_zerns, self.pupil_size**2

        mask_sum = self.zernike_array[0].sum()
        
        # Scale to be 1rad in nm ## SHOULDNT NEED THIS AS PHASE ALREADY IN RAD
        # self.zernike_array *=  ((2 * numpy.pi) / (10e9*self.config.wavelength))
        
        # Normalise for hte number of points in the Zernike anlaysis
        self.zernike_array /= mask_sum


    def allocDataArrays(self):
        self.wfsDetectorPlane = numpy.zeros(
                (self.pupil_size, self.pupil_size),
                dtype=base.DTYPE
                )

    def integrateDetectorPlane(self):
        # Cut out the phase from the oversized data
        sim_pad = (self.sim_size - self.pupil_size) // 2
        self.pupil_phase = self.los.phase[sim_pad: -sim_pad, sim_pad: -sim_pad]

        self.wfsDetectorPlane += self.pupil_phase

    def readDetectorPlane(self):
        self.detector_data = self.wfsDetectorPlane.copy()

    def calculateSlopes(self):


        z_coeffs = self.zernike_array.dot(self.detector_data.flatten())

        self.slopes = z_coeffs
        


    def zeroData(self, detector=True, FP=True):
        super(Zernike, self).zeroData(detector, FP)
        if detector:
            self.wfsDetectorPlane[:] = 0
