import numpy
from . import zerns, aoSimLib
import logging
import sys
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
        for wfs in wfss:
            self.totalSubaps += self.wfss[wfs].activeSubaps
           
    def getActiveActs(self):
        """
        Method returning the total number of actuators used by the DM - May be overwritten in DM classes

        Returns:
            int: number of active DM actuators
        """
        return self.acts


    def makeIMat(self,wfsList=[0], callback=None, progressCallback=None ):
       '''
       makes IMat
       '''
       self.makeIMatShapes()

       iMat = numpy.zeros( (self.iMatShapes.shape[0],2*self.totalSubaps) )

       subap=0
       for wfs in wfsList:
           for i in xrange(self.iMatShapes.shape[0]):
               iMat[i,subap:subap+(2*self.wfss[wfs].activeSubaps)] =\
                       self.wfss[wfs].iMatFrame(
                                           self.iMatShapes[i])#/
                                            #self.wfss[wfs].waveLength)

               logging.debug("DM IMat act: %i"%i)

               self.dmShape = self.iMatShapes[i]
               
               if callback!=None:

                   callback() 
               
               L = logging.getLogger()
               if L.level<=30:
                   sys.stdout.write("\rPoking actuator %d    "%i)
                   sys.stdout.flush()
                   
               if progressCallback!=None:
                   progressCallback(i, self.iMatShapes.shape[0],
                       "Generating %d Actuator DM iMat"%self.acts)
                
               
       self.iMat = iMat
       return iMat



    def dmFrame ( self, dmCommands, gain, closed=False):
        '''
        Uses interaction matrix to calculate the final DM shape
        '''

        self.newActCoeffs = dmCommands
        
        #If loop is closed, only add residual measurements onto old
        #actuator values

        if closed:
            self.newActCoeffs += self.actCoeffs
        
        self.actCoeffs = (gain * self.newActCoeffs)\
                            + ( (1-gain) * self.actCoeffs)


        self.dmShape = (self.iMatShapes.T*self.actCoeffs.T).T.sum(0)
        
        #Remove any piston term from DM
        self.dmShape-=self.dmShape.mean()
        self.dmShape*=self.mask
        return self.dmShape


class zernike(DM):


    def makeIMatShapes(self):
        '''
        Creates all the DM shapes which are required for creating the
        interaction Matrix
        '''

        shapes = zerns.zernikeArray(int(self.acts+2),int(self.simConfig.pupilSize))[2:]

        self.iMatShapes = shapes*self.mask


class peizo(DM):


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

        shapes = numpy.zeros( (self.acts, self.simConfig.pupilSize, self.simConfig.pupilSize) )
        zoomFactor = float(self.simConfig.pupilSize)/float(self.xActs)

        for i in xrange(self.acts):
            x,y = self.activeActs[i]

            shape = numpy.zeros( (self.xActs,self.xActs) )
            shape[x,y] = 1

            shapes[i] = 5 * aoSimLib.zoom(shape,
                    (self.simConfig.pupilSize,self.simConfig.pupilSize), order=1)
        self.iMatShapes = (shapes * self.mask) #*self.wvl



class TT:

    def __init__(self,pupilSize, wfs, mask):
        self.simConfig.pupilSize = pupilSize
        self.mask = mask
        self.dmCommands = numpy.zeros(2)
        self.wfs = wfs
        
        self.wvl = wfs.waveLength
        
        self.makeIMatShapes()

    def makeIMatShapes(self):

        coords = numpy.arange(-1, 1, 2./self.simConfig.pupilSize) + 1./self.simConfig.pupilSize

        X,Y = numpy.meshgrid( coords, coords ) 

        self.iMatShapes = numpy.array( [X*self.mask,Y*self.mask] )# * self.wvl

    def makeIMat(self, callback=None, progressCallback=None):

        iMat = numpy.empty((2,2))

        for i in xrange(2):
            self.dmShape = self.iMatShapes[i]
            slopes = self.wfs.iMatFrame(self.iMatShapes[i]
                                            #/self.wfs.waveLength
                                            ).reshape(2,
                                            self.wfs.activeSubaps)
            iMat[i] = slopes.mean(1)

            if callback !=None:
                callback()
            if progressCallback!=None:
                progressCallback(i, 2, "Tip-Tilt Mirror")

        self.iMat = iMat
        self.controlMatrix = numpy.linalg.inv(iMat)
        return iMat

    def dmFrame(self, slopes, gain, closed = True):

        #Get new commands from slopes
        meanSlopes = slopes.reshape(2,self.wfs.activeSubaps).mean(1)
        self.newDmCommands =  self.controlMatrix.dot(meanSlopes)

        if closed:
            #if closed loop update old commands
            self.newDmCommands += self.dmCommands
          
        #leaky box gain

        self.dmCommands = (gain * self.newDmCommands)\
                                + ( (1-gain) * self.dmCommands)

        #Finally use commands to calculate dm shape
        self.dmShape = (self.iMatShapes.T * self.dmCommands).T.sum(0)

        #remove piston, and apply mask.
        self.dmShape -= self.dmShape.mean()
        self.dmShape *= self.mask

        return self.dmShape






