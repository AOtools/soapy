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
The module simulating Deformable Mirrors in pyAOS
"""
import numpy
from scipy.ndimage.interpolation import rotate

from . import aoSimLib, logger


try:
    xrange
except NameError:
    xrange = range

class DM:
    def __init__ (self, simConfig, dmConfig, wfss, mask):

        self.simConfig = simConfig
        self.dmConfig = dmConfig
        self.wfss = wfss
        self.mask = mask
        self.acts = self.getActiveActs()
        self.wvl = wfss[0].wfsConfig.wavelength
       
        self.actCoeffs = numpy.zeros( (self.acts) )
       
        #find the total number of WFS subaps, and make imat
        #placeholder
        self.totalSubaps = 0
        self.wfs = wfss[self.dmConfig.wfs]
        self.totalSubaps = self.wfs.activeSubaps
        
    def getActiveActs(self):
        """
        Method returning the total number of actuators used by the DM - May be overwritten in DM classes

        Returns:
            int: number of active DM actuators
        """
        return self.dmConfig.dmActs

    def makeIMat(self, callback=None, progressCallback=None ):
       '''
       makes IMat
       '''
       self.makeIMatShapes()

       if self.dmConfig.rotation:
           self.iMatShapes = rotate(    
                   self.iMatShapes, self.dmConfig.rotation,
                   order=self.dmConfig.interpOrder, axes=(-2,-1)
                   )
           rotShape = self.iMatShapes.shape
           self.iMatShapes = self.iMatShapes[:,
                   rotShape[1]/2. - self.simConfig.pupilSize/2.:
                   rotShape[1]/2. + self.simConfig.pupilSize/2.,
                   rotShape[2]/2. - self.simConfig.pupilSize/2.:
                   rotShape[2]/2. + self.simConfig.pupilSize/2.
                   ]
       

       iMat = numpy.zeros( (self.iMatShapes.shape[0],2*self.totalSubaps) )

       subap=0

       for i in xrange(self.iMatShapes.shape[0]):
           iMat[i,subap:subap+(2*self.wfs.activeSubaps)] =\
                   self.wfs.iMatFrame( self.iMatShapes[i])

           logger.debug("DM IMat act: %i"%i)

           self.dmShape = self.iMatShapes[i]
       
           if callback!=None:

               callback() 
       
           logger.statusMessage(i, self.iMatShapes.shape[0],
                    "Generating {} Actuator DM iMat".format(self.acts))

       self.iMat = iMat
       return iMat

    def dmFrame ( self, dmCommands, closed=False):
        '''
        Uses interaction matrix to calculate the final DM shape
        '''

        self.newActCoeffs = dmCommands
        
        #If loop is closed, only add residual measurements onto old
        #actuator values
        #if closed:
            #self.newActCoeffs += self.actCoeffs
        self.actCoeffs = (self.dmConfig.gain*self.newActCoeffs 
                    + (1-self.dmConfig.gain)*self.actCoeffs)

        #else:
        #    self.actCoeffs = (self.dmConfig.gain * self.newActCoeffs)\
        #        + ( (1-self.dmConfig.gain) * self.actCoeffs)
        
        self.dmShape = (self.iMatShapes.T*self.actCoeffs.T).T.sum(0)
        
        #Remove any piston term from DM
        self.dmShape-=self.dmShape.mean()
        self.dmShape*=self.mask
        return self.dmShape


class Zernike(DM):

    def makeIMatShapes(self):
        '''
        Creates all the DM shapes which are required for creating the
        interaction Matrix
        '''

        shapes = self.dmConfig.iMatValue*aoSimLib.zernikeArray(
                        int(self.acts+3),int(self.simConfig.pupilSize))[3:]

        self.iMatShapes = shapes*self.mask


class Piezo(DM):

    def getActiveActs(self):
        activeActs = []
        xActs = int(numpy.round(numpy.sqrt(self.dmConfig.dmActs)))
        spcing = self.mask.shape[0]/float(xActs)

        for x in xrange(xActs):
            for y in xrange(xActs):
                if self.mask[x*spcing:(x+1)*spcing,y*spcing:(y+1)*spcing].sum() > 0:
                    activeActs.append([x,y])
        self.activeActs = numpy.array(activeActs)
        self.xActs = xActs
        return self.activeActs.shape[0]


    def makeIMatShapes(self):

        shapes = numpy.zeros( (
                self.acts, self.simConfig.pupilSize, self.simConfig.pupilSize) )

        for i in xrange(self.acts):
            x,y = self.activeActs[i]

            shape = numpy.zeros( (self.xActs,self.xActs) )
            shape[x,y] = 1

            shapes[i] = self.dmConfig.iMatValue * aoSimLib.zoom(shape,
                    (self.simConfig.pupilSize,self.simConfig.pupilSize),
                    order=self.dmConfig.interpOrder)
        self.iMatShapes = (shapes * self.mask) #*self.wvl

class GaussStack(Piezo):

    def makeIMatShapes(self):
        shapes = numpy.zeros((
                self.acts, self.simConfig.pupilSize, self.simConfig.pupilSize))
    
        actSpacing = self.simConfig.pupilSize/(numpy.sqrt(self.dmConfig.dmActs)-1)
        width = actSpacing/2.

        for i in xrange(self.acts):
            x,y = self.activeActs[i]*actSpacing
            shapes[i] = aoSimLib.gaussian2d(
                    self.simConfig.pupilSize, width, cent = (x,y))
        
        self.iMatShapes = shapes

            

        
class TT(DM):

    def getActiveActs(self):
        return 2


    def makeIMatShapes(self):

        coords = self.dmConfig.iMatValue*numpy.linspace(
                    -1,1,self.simConfig.pupilSize)
        self.iMatShapes = numpy.array(numpy.meshgrid(coords,coords))*self.mask


    # def makeIMat(self, callback=None, progressCallback=None ):
   #      '''
   #      makes IMat
   #      '''
   #      self.makeIMatShapes()
   #
   #      if self.dmConfig.rotation:
   #          self.iMatShapes = rotate(
   #                  self.iMatShapes, self.dmConfig.rotation,
   #                  order=self.dmConfig.interpOrder, axes=(-2,-1))
   #          rotShape = self.iMatShapes.shape
   #          self.iMatShapes = self.iMatShapes[:,
   #                  rotShape[1]/2. - self.simConfig.pupilSize/2.:
   #                  rotShape[1]/2. + self.simConfig.pupilSize/2.,
   #                  rotShape[2]/2. - self.simConfig.pupilSize/2.:
   #                  rotShape[2]/2. + self.simConfig.pupilSize/2.
   #                  ]
   #
   #      iMat = numpy.zeros( (2,2) )
   #
   #      slopesToTT = numpy.zeros((self.wfs.activeSubaps*2, 2))
   #      slopesToTT[:self.wfs.activeSubaps, 0] = 1./self.wfs.activeSubaps
   #      slopesToTT[self.wfs.activeSubaps:, 1] = 1./self.wfs.activeSubaps
   #
   #      for i in xrange(self.iMatShapes.shape[0]):
   #
   #          slopes = self.wfs.iMatFrame(self.iMatShapes[i])
   #          iMat[i,:] = slopes.reshape(2,self.wfs.activeSubaps).mean(1)
   #
   #          logger.debug("DM IMat act: %i"%i)
   #
   #          self.dmShape = self.iMatShapes[i]
   #
   #          if callback!=None:
   #              callback()
   #
   #          logger.statusMessage(i, self.iMatShapes.shape[0],
   #                  "Generating {} Actuator DM iMat".format(self.acts))
   #
   #
   #      self.iMat = iMat.dot(numpy.linalg.pinv(slopesToTT))
   # return self.iMat


class TT1:

    def __init__(self, pupilSize, wfs, mask):
        self.pupilSize = pupilSize
        self.mask = mask
        self.dmCommands = numpy.zeros(2)
        self.wfs = wfs
        self.wvl = wfs.wfsConfig.wavelength
        
        self.makeIMatShapes()

    def getActiveActs(self):
        return 2

    def makeIMatShapes(self):

        coords = numpy.linspace(-1, 1, self.pupilSize)

        X,Y = numpy.meshgrid( coords, coords ) 

        self.iMatShapes = 30* numpy.array( [X*self.mask,Y*self.mask] )# * self.wvl

    def makeIMat(self, callback=None, progressCallback=None):

        iMat = numpy.empty((2,2))

        for i in xrange(2):
            self.dmShape = self.iMatShapes[i]
            slopes = self.wfs.iMatFrame(self.iMatShapes[i]
                                            ).reshape(2,
                                            self.wfs.activeSubaps)
            iMat[i] = slopes.mean(1)

            if callback !=None:
                callback()

        self.iMat = iMat
        self.controlMatrix = numpy.linalg.pinv(iMat)
        return iMat

    def dmFrame(self, slopes, gain, closed=False):

        #Get new commands from slopes
        meanSlopes = slopes.reshape(2,self.wfs.activeSubaps).mean(1)
        self.newDmCommands =  self.controlMatrix.dot(meanSlopes)

        if closed:
            #if closed loop update old commands
            self.newDmCommands += self.dmCommands
          
        #apply gain
        self.dmCommands = (gain * self.newDmCommands)\
                                + ( (1-gain) * self.dmCommands)

        #Finally use commands to calculate dm shape
        self.dmShape = (self.iMatShapes.T * self.dmCommands).T.sum(0)

        #remove piston, and apply mask.
        self.dmShape -= self.dmShape.mean()
        self.dmShape *= self.mask

        return self.dmShape
