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

import numpy
import scipy
import pyfits
from . import logger
import traceback
import sys
import time

#xrange now just "range" in python3. 
#Following code means fastest implementation used in 2 and 3
try:
    xrange
except NameError:
    xrange = range

class Reconstructor:
    def __init__( self, simConfig, dms, wfss, atmos, runWfsFunc=None):
                
        self.dms = dms
        self.wfss = wfss
        self.simConfig = simConfig
        self.atmos = atmos

        self.learnIters = self.simConfig.learnIters

        self.dmActs = []
        self.dmConds = []
        self.dmTypes = []
        for dm in xrange(self.simConfig.nDM):
            self.dmActs.append(self.dms[dm].dmConfig.dmActs)
            self.dmConds.append(self.dms[dm].dmConfig.dmCond)
            self.dmTypes.append(self.dms[dm].dmConfig.dmType)

        self.dmConds = numpy.array(self.dmConds)
        self.dmActs = numpy.array(self.dmActs)

        #2 functions used in case reconstructor requires more WFS data.
        #i.e. learn and apply
        self.runWfs = runWfsFunc
        if self.simConfig.learnAtmos == "random":
            self.moveScrns = atmos.randomScrns
        else:
            self.moveScrns = atmos.moveScrns
        self.wfss = wfss
                
        self.initControlMatrix()

        self.Trecon = 0

    def initControlMatrix(self):

        self.controlMatrix = numpy.zeros((self.simConfig.totalSlopes, self.simConfig.totalActs))
        self.controlShape = (self.simConfig.totalSlopes, self.simConfig.totalActs)

    def saveCMat(self):
        filename = self.simConfig.filePrefix+"/cMat.fits"
        
        cMatHDU = pyfits.PrimaryHDU(self.controlMatrix)
        cMatHDU.header["DMNO"] = self.simConfig.nDM
        cMatHDU.header["DMACTS"] = "%s"%list(self.dmActs)
        cMatHDU.header["DMTYPE"]  = "%s"%list(self.dmTypes)
        cMatHDU.header["DMCOND"]  = "%s"%list(self.dmConds)
        
        cMatHDU.writeto(filename, clobber=True)

    def loadCMat(self):
        
        filename=self.simConfig.filePrefix+"/cMat.fits"
        
        cMatHDU = pyfits.open(filename)[0]
        cMatHDU.verify("fix")
        header = cMatHDU.header
        
        
        try:
            dmActs = dmTypes = dmConds = None
            
            
            dmNo = int(header["DMNO"])
            exec("dmActs = numpy.array(%s)"%cMatHDU.header["DMACTS"])
            exec("dmTypes = %s"%header["DMTYPE"])
            exec("dmConds = numpy.array(%s)"%header["DMCOND"])
            
            if not numpy.all(dmConds==self.dmConds):
                raise Exception("DM conditioning Parameter changed - will make new control matrix")
            if not numpy.all(dmActs==self.dmActs) or dmTypes!=self.dmTypes or dmNo!=dmNo:
                logger.warning("loaded control matrix may not be compatibile with \
                                the current simulation. Will try anyway....")
                                
            #cMat = cMatFile[1]
            cMat = cMatHDU.data
            
        except KeyError:
            logger.warning("loaded control matrix header has not created by this ao sim. Will load anyway.....")
            #cMat = cMatFile[1]
            cMat = cMatHDU.data
            
        if cMat.shape != self.controlShape:
            logger.warning("designated control matrix does not match the expected shape")
            raise Exception
        else:
            self.controlMatrix = cMat
    
    def saveIMat(self):

        for dm in xrange(self.simConfig.nDM):
            filenameIMat = self.simConfig.filePrefix+"/iMat_dm%d.fits"%dm
            filenameShapes = self.simConfig.filePrefix+"/dmShapes_dm%d.fits"%dm
            
            pyfits.PrimaryHDU(self.dms[dm].iMat).writeto(filenameIMat,
                                                        clobber=True)
            pyfits.PrimaryHDU(self.dms[dm].iMatShapes).writeto(filenameShapes,
                                                        clobber=True)
            
                            
    def loadIMat(self):
        
        for dm in xrange(self.simConfig.nDM):
            filenameIMat = self.simConfig.filePrefix+"/iMat_dm%d.fits"%dm
            filenameShapes = self.simConfig.filePrefix+"/dmShapes_dm%d.fits"%dm
            
            iMat = pyfits.open(filenameIMat)[0].data
            iMatShapes = pyfits.open(filenameShapes)[0].data
            
            if iMat.shape!=(self.dms[dm].acts,2*self.dms[dm].totalSubaps):
                logger.warning("interaction matrix does not match required required size.")
                raise Exception
            if iMatShapes.shape[-1]!=self.dms[dm].simConfig.pupilSize:
                logger.warning("loaded DM shapes are not same size as current pupil.")
                raise Exception
            else:
                self.dms[dm].iMat = iMat
                self.dms[dm].iMatShapes = iMatShapes
    
        
    def makeIMat(self,callback, progressCallback):
 
        for dm in xrange(self.simConfig.nDM):
            logger.info("Creating Interaction Matrix on DM %d..."%dm)
            self.dms[dm].makeIMat(callback=callback, 
                                        progressCallback=progressCallback)

        
    def makeCMat(self,loadIMat=True, loadCMat=True, callback=None, 
                        progressCallback=None):
        
        if loadIMat:
            try:
                self.loadIMat()
                logger.info("Interaction Matrices loaded successfully")
            except:
                traceback.print_exc()
                logger.warning("Load Interaction Matrices failed - will create new one.")
                self.makeIMat(callback=callback,    
                         progressCallback=progressCallback)
                self.saveIMat()
                logger.info("Interaction Matrices Done")
                
        else:
            
            self.makeIMat(callback=callback, progressCallback=progressCallback)
            logger.info("Interaction Matrices Done")
        
        if loadCMat:
            try:
                self.loadCMat()
                logger.info("Command Matrix Loaded Successfully")
            except:
                traceback.print_exc()
                logger.warning("Load Command Matrix failed - will create new one")
                
                self.calcCMat(callback, progressCallback)
                self.saveCMat()
                logger.info("Command Matrix Generated!")
        else:
            logger.info("Creating Command Matrix")
            self.calcCMat(callback, progressCallback)
            logger.info("Command Matrix Generated!")
       

    def reconstruct(self,slopes):
        t=time.time()

        dmCommands = self.controlMatrix.T.dot(slopes)

        self.Trecon += time.time()-t
        return dmCommands

            
class WooferTweeter(Reconstructor):
    '''
    Reconstructs a 2 DM system, where 1 DM is of low order, high stroke
    and the other has a higher, but low stroke.
    
    Reconstructs dm commands for each DM, then removes the low order 
    component from the high order commands by propagating back to the 
    slopes corresponding to the lower order DM shape, and propagating 
    to the high order DM shape.
    '''
    
    def calcCMat(self,callback=None, progressCallback=None):
        '''
        Creates control Matrix. 
        Assumes that DM 0 (or 1 if TT used) is low order, 
        and DM 1 (or 2 if TT used) is high order.
        '''

        if self.simConfig.nDM==1:
            logger.warning("Woofer Tweeter Reconstruction not valid for 1 dm.")
            return None
        acts = 0
        dmCMats = []
        for dm in xrange(self.simConfig.nDM):
            dmIMat = self.dms[dm].iMat
            
            if dmIMat.shape[0]==dmIMat.shape[1]:
                dmCMat = numpy.linalg.pinv(dmIMat)
            else:
                dmCMat = numpy.linalg.pinv(
                                    dmIMat, self.dms[dm].dmConfig.dmCond)
            
            if dm != self.simConfig.nDM-1:
                self.controlMatrix[:,acts:acts+self.dms[dm].acts] = dmCMat
                acts+=self.dms[dm].acts
            
            dmCMats.append(dmCMat)
            
        #This is the matrix which converts from Low order DM commands
        #to high order DM commands, via slopes
        lowToHighTransform = self.dms[self.simConfig.nDM-2].iMat.T.dot( dmCMats[-2].T )

        highOrderCMat = dmCMats[-1].T.dot( 
                numpy.identity(self.simConfig.totalSlopes)-lowToHighTransform)
                            
        self.controlMatrix[:,acts:acts+self.dms[self.simConfig.nDM-1].acts] = highOrderCMat.T
 


class MVM(Reconstructor):


    def calcCMat(self,callback=None, progressCallback=None):
        '''
        Uses DM object makeIMat methods, then inverts each to create a 
        control matrix
        '''
        acts = 0
        for dm in xrange(self.simConfig.nDM):

            dmIMat = self.dms[dm].iMat

            if dmIMat.shape[0]==dmIMat.shape[1]:
                dmCMat = numpy.linalg.pinv(dmIMat)
            else:
                dmCMat = scipy.linalg.pinv( dmIMat, 
                                            self.dms[dm].dmConfig.dmCond)

            self.controlMatrix[:,acts:acts+self.dms[dm].acts] = dmCMat
            acts += self.dms[dm].acts

    def reconstruct(self, slopes):
        #If theres a TT mirror, remove the TT from the slopes and get TT commands

        if self.dms[0].dmConfig.dmType=="TT":
            ttMean = slopes.reshape(2, self.wfss[0].activeSubaps).mean(1)
            ttCommands = self.controlMatrix[:,:2].T.dot(slopes)
            slopes[:self.wfss[0].activeSubaps] -= ttMean[0]
            slopes[self.wfss[0].activeSubaps:] -= ttMean[1]

            #get dm commands for the calculated on axis slopes
            dmCommands = self.controlMatrix[:,2:].T.dot(slopes)

            return numpy.append(ttCommands, dmCommands)

        #get dm commands for the calculated on axis slopes
        dmCommands = self.controlMatrix.T.dot(slopes)
        return dmCommands


class LearnAndApply(Reconstructor):
    '''
    Class to perform a simply learn and apply algorithm, where
    "learn" slopes are recorded, and an interaction matrix between off-axis 
    and on-axis WFS is computed from these slopes. 
    
    Assumes that on-axis sensor is WFS 0
    '''

    def saveCMat(self):
        cMatFilename = self.simConfig.filePrefix+"/cMat.fits"
        tomoMatFilename = self.simConfig.filePrefix+"/tomoMat.fits"

        cMatHDU = pyfits.PrimaryHDU(self.controlMatrix)
        cMatHDU.header["DMNO"] = self.simConfig.nDM
        cMatHDU.header["DMACTS"] = "%s"%list(self.dmActs)
        cMatHDU.header["DMTYPE"]  = "%s"%list(self.dmTypes)
        cMatHDU.header["DMCOND"]  = "%s"%list(self.dmConds)
        
        tomoMatHDU = pyfits.PrimaryHDU(self.tomoRecon)

        tomoMatHDU.writeto(tomoMatFilename, clobber=True)
        cMatHDU.writeto(cMatFilename, clobber=True)

    def loadCMat(self):
            
        super(LearnAndApply, self).loadCMat()

        #Load tomo reconstructor
        tomoFilename = self.simConfig.filePrefix+"/tomoMat.fits"
        tomoMat = pyfits.getdata(tomoFilename)

        #And check its the right size
        if tomoMat.shape != (2*self.wfss[0].activeSubaps, self.simConfig.totalSlopes - 2*self.wfss[0].activeSubaps):
            logger.warning("Loaded Tomo matrix not the expected shape - gonna make a new one..." )
            raise Exception
        else:
            self.tomoRecon = tomoMat


    def initControlMatrix(self):

        self.controlShape = (2*self.wfss[0].activeSubaps, self.simConfig.totalActs)
        self.controlMatrix = numpy.empty( self.controlShape )
    

    def learn(self,callback=None, progressCallback=None):
        '''
        Takes "self.learnFrames" WFS frames, and computes the tomographic
        reconstructor for the system. This method uses the "truth" sensor, and
        assumes that this is WFS0
        '''

        self.learnSlopes = numpy.empty( (self.learnIters,self.simConfig.totalSlopes) )
        for i in xrange(self.learnIters):
            self.learnIter=i            

            scrns = self.moveScrns()
            self.learnSlopes[i] = self.runWfs(scrns)
            
            sys.stdout.write("\rLearn Frame: {}".format(i))
            sys.stdout.flush()
            
            if callback!=None:
                callback()
            if progressCallback!=None:
               progressCallback("Performing Learn", i, self.learnIters ) 
            
        if self.simConfig.saveLearn:
            #FITS.Write(self.learnSlopes,self.simConfig.filePrefix+"/learn.fits")
            pyfits.PrimaryHDU(self.learnSlopes).writeto(
                            self.simConfig.filePrefix+"/learn.fits",clobber=True )


    def calcCMat(self,callback=None, progressCallback=None):
        '''
        Uses the slopes recorded in the "learn" and DM interaction matrices
        to create a CMat.
        '''

        logger.info("Performing Learn....")
        self.learn(callback, progressCallback)
        logger.info("Done. Creating Tomographic Reconstructor...")
        
        if progressCallback!=None:
            progressCallback(1,1, "Calculating Covariance Matrices")
        
        self.covMat = numpy.cov(self.learnSlopes.T)
        Conoff = self.covMat[   :2*self.wfss[0].activeSubaps,
                                2*self.wfss[0].activeSubaps:     ]
        Coffoff = self.covMat[  2*self.wfss[0].activeSubaps:,
                                2*self.wfss[0].activeSubaps:    ]
        
        logger.info("Inverting offoff Covariance Matrix")
        iCoffoff = numpy.linalg.pinv(Coffoff)
        
        self.tomoRecon = Conoff.dot(iCoffoff)
        logger.info("Done. \nCreating full reconstructor....")
        
        #Same code as in "MVM" class to create dm-slopes reconstructor.
        acts = 0
        for dm in xrange(self.simConfig.nDM):
            dmIMat = self.dms[dm].iMat
            
            if dmIMat.shape[0]==dmIMat.shape[1]:
                dmCMat = numpy.inv(dmIMat)
            else:
                dmCMat = numpy.linalg.pinv(dmIMat, self.dms[dm].dmConfig.dmCond)
            
            self.controlMatrix[:,acts:acts+self.dms[dm].acts] = dmCMat
            acts += self.dms[dm].acts
        


        #self.controlMatrix = (self.controlMatrix.T.dot(self.tomoRecon)).T
        logger.info("Done.")
        

    def reconstruct(self,slopes):
        """
        Determine DM commands using previously made 
        reconstructor from slopes. 
        Args:
            slopes (ndarray): array of slopes to reconstruct from
        Returns:
            ndarray: array to comands to be sent to DM 
        """

        #Retreive pseudo on-axis slopes from tomo reconstructor
        slopes = self.tomoRecon.dot(slopes[2*self.wfss[0].activeSubaps:])

        if self.dms[0].dmConfig.dmType=="TT":
            ttMean = slopes.reshape(2, self.wfss[0].activeSubaps).mean(1)
            ttCommands = self.controlMatrix[:,:2].T.dot(slopes)
            slopes[:self.wfss[0].activeSubaps] -= ttMean[0]
            slopes[self.wfss[0].activeSubaps:] -= ttMean[1]

            #get dm commands for the calculated on axis slopes
            dmCommands = self.controlMatrix[:,2:].T.dot(slopes)

            return numpy.append(ttCommands, dmCommands)

        #get dm commands for the calculated on axis slopes
        dmCommands = self.controlMatrix.T.dot(slopes)
        return dmCommands

            


#####################################
#Experimental....
#####################################

class ANN(Reconstructor):
    """
    Reconstructs using a neural net
    Assumes on axis slopes are WFS 0

    Net must be set by setting ``sim.recon.net = net`` before loop is run 
    net object must have a ``run`` method, which accepts slopes and returns
    on Axis slopes
    """

    def calcCMat(self, callback=None, progressCallback=None):

        nSlopes = self.wfss[0].activeSubaps*2 

        self.controlShape = (nSlopes, self.simConfig.totalActs)
        self.controlMatrix = numpy.zeros((nSlopes, self.simConfig.totalActs))
        acts = 0
        for dm in xrange(self.simConfig.nDM):
            dmIMat = self.dms[dm].iMat
            
            if dmIMat.shape[0]==dmIMat.shape[1]:
                dmCMat = numpy.inv(dmIMat)
            else:
                dmCMat = numpy.linalg.pinv(dmIMat, self.dmCond[dm])
            
            self.controlMatrix[:,acts:acts+self.dms[dm].acts] = dmCMat
            acts += self.dms[dm].acts

    def reconstruct(self, slopes):
        """
        Determine DM commands using previously made 
        reconstructor from slopes. Uses Artificial Neural Network.
        Args:
            slopes (ndarray): array of slopes to reconstruct from
        Returns:
            ndarray: array to comands to be sent to DM 
        """
        t=time.time()
        offSlopes = slopes[self.wfss[0].activeSubaps*2:]
        onSlopes = self.net.run(offSlopes)
        dmCommands = self.controlMatrix.T.dot(onSlopes)

        self.Trecon += time.time()-t
        return dmCommands


class LearnAndApplyLGS(Reconstructor):
    '''
    Class to perform a simply learn and apply algorithm, where
    "learn" slopes are recorded, and an interaction matrix between off-axis 
    and on-axis WFS is computed from these slopes. 
    
    Assumes that on-axis sensor is WFS 0
    
    Only works for 1 DM at present
    
    This is a specific implementation designed to test LGS tip tilt 
    prediction. It uses the last 10 frames of off-axis slopes to help predict
    the on-axis slopes.
    '''
    
    def learn(self):
        '''
        Takes "self.learnFrames" WFS frames, and computes the tomographic
        reconstructor for the system. This method uses the "truth" sensor, and
        assumes that this is WFS0
        '''
        FRAMES = 10
        
        onAxisSize = self.wfss[0].activeSubaps*2
        offAxisSize = self.simConfig.totalSlopes - self.wfss[0].activeSubaps*2
        
        self.onSlopes = numpy.empty( (self.learnIters+FRAMES,onAxisSize) )
        self.offSlopes = numpy.empty( (self.learnIters+FRAMES,
                                                FRAMES*offAxisSize) )
        self.slopesBuffer = numpy.empty( (FRAMES,offAxisSize) )
        
        for i in xrange(self.learnIters+FRAMES):
            self.learnIter = i
            logger.debug("Learn Iteration %i",i)
            scrns = self.moveScrns()
            slopes = self.runWfs(scrns)
            
            self.onSlopes[i] = slopes[:onAxisSize]            
            self.slopesBuffer[0] = slopes[onAxisSize:]
            self.offSlopes[i] = self.slopesBuffer.flatten()
            self.slopesBuffer = numpy.roll(self.slopesBuffer,1,axis=0)

        self.offSlopes = self.offSlopes[FRAMES:]
        self.onSlopes = self.onSlopes[FRAMES:]

    def calcCMat(self):
        '''
        Uses the slopes recorded in the "learn" and DM interaction matrices
        to create a CMat.
        '''
        self.controlMatrix = numpy.empty( (2*self.wfss[0].activeSubaps,
                                            self.simConfig.totalActs) )
        logger.info("Performing Learn....")
        self.learn()
        logger.info("Done. Creating Tomographic Reconstructor...")
        onAxisSlopes = self.onSlopes
        iOffAxisSlopes = numpy.linalg.pinv(self.offSlopes)
        
        tomoRecon = onAxisSlopes.T.dot(iOffAxisSlopes.T)
        logger.info("Done. Creating full reconstructor....")
        
        
        #Same code as in "MVM" class to create dm-slopes reconstructor.
        acts = 0
        for dm in xrange(self.simConfig.nDM):
            dmIMat = self.dms[dm].iMat
            
            if dmIMat.shape[0]==dmIMat.shape[1]:
                dmCMat = numpy.inv(dmIMat)
            else:
                dmCMat = numpy.linalg.pinv(dmIMat, self.dmCond[dm])
            
            self.controlMatrix[:,acts:acts+self.dms[dm].acts] = dmCMat
            acts += self.dms[dm].acts

        
        self.controlMatrix = (self.controlMatrix.T.dot(tomoRecon)).T
        logger.info("Done.")
        
    def reconstruct(self,slopes):
        
        logger.debug("LA Reconstruction - slopes Shape: %s"%slopes[2*self.wfss[0].activeSubaps:].shape)
        logger.debug("LA Reconstruction - Reconstructor Shape: %s,%s"%self.controlMatrix.shape)

        self.slopesBuffer[0] = slopes[self.wfss[0].activeSubaps*2:]
        
        dmCommands = self.controlMatrix.T.dot(
                            self.slopesBuffer.flatten())
        self.slopesBuffer = numpy.roll(self.slopesBuffer,1,0)
        
        return dmCommands
        
