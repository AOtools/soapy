#! /usr/bin/env python
'''
The <main> module to run AoSim.
It reads all parameters from a config file, initilises
the AO objects, then finally runs a choice of loops.
'''


#sim imports
from . import atmosphere
from . import WFS
from . import DM
from . import LGS
from . import RECON
from . import SCI
from . import confParse
from . import aoSimLib

#standard python imports
import numpy
import datetime
import sys
import logging
import os
import time
import traceback
from multiprocessing import Process, Queue
from argparse import ArgumentParser
import pyfits
import shutil

#xrange now just "range" in python3.
#Following code means fastest implementation used in 2 and 3
try:
    xrange
except NameError:
    xrange = range


class Sim(object):
    """
    The pyAOS Simulation class.

    This class holds all configuration information, data and control
    methods of the simulation. It contains high level methods dealing with
    initialising all component objects, making reconstructor control
    matrices, running the loop and saving data after the loop has run.

    Can be sub-classed and the 'aoloop' method overwritten for different loops
    to be used

    Args:
        configFile (string): The filename of the AO configuration file
    """

    def __init__(self, configFile=None):
        if not configFile:
            configFile = "conf/testConf.py"

        self.configFile = configFile
        self.readParams()

    def readParams(self):
        '''
        Reads configuration file parameters

        Calls the radParams function in confParse to read, parse and if required
        set reasonable defaults to AO parameters
        '''

        confParse.readParams(self, self.configFile)

    def aoinit(self):
        '''
        Initialises all simulation objects.

        Initialises and passes relevant data to sim objects. This does important pre-run tasks, such as creating or loading phase screens, determining WFS geometry, setting propagation modes and pre-allocating data arrays used later in the simulation.
        '''
        #Read params if they haven't been read before
        try:
            self.pupilSize
        except:
            #self.readParams()
            confParse.readParams(self, self.configFile)

        if self.loggingMode == 0:
            logging.basicConfig(level=logging.WARNING)

        elif self.loggingMode == 1:
            logging.basicConfig(level=logging.INFO)

        elif self.loggingMode == 2:
            logging.basicConfig(level=logging.DEBUG)

        #calculate some params from read ones
        #calculated
        self.aoloop = eval("self."+self.aoloopMode)
        self.pxlScale = self.pupilSize/float(self.telDiam)  # pxls per Metre
        self.scrnSize = numpy.ceil(
                            2*self.pxlScale*self.scrnHeights.max()
                            * self.GSPositions.max() * numpy.pi/(3600.*180))\
                            + self.pupilSize


        #Init Pupil Mask
        logging.info("Creating mask...")
        if self.mask == "circle":
            self.mask = aoSimLib.circle(self.pupilSize/2.,self.pupilSize)
            if self.obs!=None:
                self.mask -= aoSimLib.circle(self.obs*self.pxlScale/2., self.pupilSize)



        self.initAtmos()   

        #Find if WFSs should each have own process
        if self.mp_wfs:
            self.wfs_fftw_threads=1
            self.runWfs = self.runWfs_MP
        else:
            self.wfs_fftw_threads=None
            self.runWfs = self.runWfs_noMP

        #Init LGS module
        logging.info("Initialising LGS..")
        self.LGSs = {}
        for i in xrange(self.GSNo):
            if self.LGSUplink[i] != 0:
                print("propMode:%s"%self.lgsPropMode)
                if self.lgsPropMode[i]=="phys" or self.lgsPropMode[i]=="physical":
                    LGS.LGS = LGS.PhysicalLGS
                    print("Physical LGS")
                else:
                    LGS.LGS = LGS.GeometricLGS
                    print("Geometric LGS")
                self.LGSs[i] = LGS.LGS( self.pupilSize, self.GSPositions[i],
                                        self.pxlScale, self.scrnHeights,
                                        self.lgsHeight[i], self.LGSPupilSize,
                                        self.scrns, self.LGSFFTPadding[i],
                                        self.LGSWL[i],
                                        self.subapFOV[i],
                                        self.pxlsPerSubap[i],
                                        self.subapOversamp[i], self.atmos.wvl,
                                        self.wfs_fftw_threads,
                                        self.lgsPyfftw_FLAG)
            else:
                self.LGSs[i] = None

        #init WFSs
        logging.info("Initialising WFSs....")
        self.wfss = {}
        self.totalSubaps = 0

        for wfs in xrange(self.GSNo):
            self.wfss[wfs]=WFS.ShackHartmannWfs(
                self.subaps[wfs], self.pupilSize, self.GSPositions[wfs],
                self.pxlScale, self.scrnHeights, self.mask,
                self.subapThreshold, self.GSHeight[wfs], self.pxlsPerSubap[wfs],
                        self.subapOversamp[wfs], self.subapFOV[wfs],
                        self.waveLengths[wfs], self.telDiam,  self.LGSs[wfs],
                        self.elongation, self.elongLayers,
                        self.wfsFftProcs, self.wfsPyfftw_FLAG, self.wfs_fftw_threads,
                        self.lgsLaunchPos[wfs],
                        self.atmos.wvl, self.removeTT[wfs], self.wfsSNR[i],
                        self.angleEquivNoise[i], self.wfsBitDepth[i], self.wfsPropMode[wfs],
                        self.scrns )

            self.totalSubaps += self.wfss[wfs].activeSubaps

            logging.info("WFS %i: %i active sub-apertures",wfs,
                     len(self.wfss[wfs].subapCoords))


        #init DMs
        logging.info("Initialising DMs...")
        if self.tipTilt:
            self.TT = DM.TT(self.pupilSize, self.wfss[0], self.mask)

        self.dms = {}
        self.dmActCommands = {}
        self.totalActs = 0
        for dm in xrange(self.dmNo):
            dmObj = eval( "DM."+self.dmTypes[dm])
            self.dms[dm] = dmObj(self.pupilSize,self.dmActs[dm],
                                 self.wfss,self.mask, self.dmCond[dm])
            self.dmActCommands[dm] = numpy.empty( (self.nIters, self.dms[dm].acts) )
            self.totalActs+=self.dms[dm].acts

            logging.info("DM %d: %d active actuators"%(dm,self.dms[dm].acts))
        logging.info("%d total DM Actuators"%self.totalActs)


        #init Reconstructor
        logging.info("Initialising Reconstructor...")
        reconObj = eval("RECON."+self.reconstructor)
        self.recon = reconObj(self.dms, self.wfss, 2*self.totalSubaps,
                            self.dmNo, self.dmTypes, self.dmActs, self.dmCond, 
                            self.totalActs, self.runWfs, self.atmos,
                            self.learnAtmos, self.learnIters, self.filePrefix,
                            self.saveLearn)

        #init Science Cameras
        logging.info("Initialising Science Cams...")
        self.sciCams = {}
        self.sciImgs = {}
        self.sciImgNo=0
        for sci in xrange(self.sciNo):
            self.sciCams[sci] = SCI.scienceCam(self.pupilSize, self.pxlScale,
                                               self.sciPos[sci],
                                                self.scrnHeights, self.mask,
                                                self.sciFOV[sci],
                                                self.sciWvls[sci],
                                                self.telDiam, self.sciPxls[sci],
                                                self.sciOversamp[sci],
                                                self.atmos.wvl)
            self.sciImgs[sci] = numpy.zeros( [self.sciPxls[sci]]*2 )


        #Init data storage
        logging.info("Initialise Data Storage...")
        self.initSaveData()

        self.iters=0

        #Init performance tracking
        self.Twfs = 0
        self.Tlgs = 0
        self.Tdm = 0
        self.Tsci = 0
        self.Trecon = 0
        self.Timat = 0
        self.Tatmos = 0


        logging.info("Initialisation Complete!")

    def initAtmos(self):
        """
        Helper function to handle atmosphere initialisation
        """
        logging.info("Initialising Atmosphere...")
        self.atmos = atmosphere.atmos()
        self.scrns = self.atmos.loadScrns(  self.scrnNo, self.scrnNames,
                                            self.scrnSize, self.windSpeeds,
                                            self.windDirs, self.scrnStrengths,
                                            self.pxlScale, self.newScreens,
                                            self.wholeScrnSize, self.loopTime,
                                            self.r0)

    def makeIMat(self,forceNew=False, progressCallback=None):
        """
        Creates interaction and control matrices for simulation reconstruction
        
        Makes and inverts Interaction matrices for each DM in turn to
        create a DM control Matrix for each DM.
        Each DM's control Matrix is independent of the others,
        so care must be taken so DM correction modes do not "overlap".
        Some reconstruction modes may require WFS frames to be taken for the 
        creation of a control matrix. Depending on set parameters, 
        can load previous control and interaction matrices.
        
        Args:
            forceNew (bool): if true, will force making of new iMats and cMats, otherwise will attempt to load previously made matrices from same filePrefix
            progressCallback (func): function called to report progress of interaction matrix construction
        """
        t = time.time()
        logging.info("Making interaction Matrices...")
        if self.tipTilt:
            logging.info("Generating Tip Tilt IMat")
            self.TT.makeIMat(self.addToGuiQueue,
                                    progressCallback=progressCallback)

        if forceNew:
            loadIMat=False
            loadCMat=False
        else:
            if self.filePrefix==None:
                loadIMat=False
                loadCMat=False
            else:
                loadIMat=True
                loadCMat=True

        self.recon.makeCMat(loadIMat=loadIMat,loadCMat=loadCMat,
                callback=self.addToGuiQueue, progressCallback=progressCallback)
        self.Timat+= time.time()-t

    def runWfs_noMP(self, scrns = None, dmShape=None, wfsList=None):
        """
        Runs all WFSs
        
        Runs a single frame for each WFS in wfsList, passing the given phase screens and optional dmShape (if WFS in closed loop). If LGSs are present it will also deals with LGS propagation. Finally, the slopes from all WFSs are returned.
        
        Args:
            scrns (list): List of phase screens passing over telescope
            dmShape (np array): 2-dimensional array of the total corrector shape
            wfsList (list): A list of the WFSs to be run
            
        Returns:
            array: The slope data return from the WFS frame (may not be actual slopes if WFS other than SH used)
            """
        t_wfs = time.time()
        if scrns != None:
            self.scrns=scrns

        slopes = numpy.zeros( (self.totalSubaps*2))
        s = 0
        if wfsList==None:
            wfsList=range(self.GSNo)

        for wfs in wfsList:

            if self.LGSUplink[wfs]!=0:
                tlgs=time.time()
                self.LGSs[wfs].LGSPSF(self.scrns)
                self.Tlgs +=time.time()-tlgs

            #self.wfss[wfs].frame(self.scrns)
            slopes[s:s+self.wfss[wfs].activeSubaps*2] = \
                    self.wfss[wfs].frame(self.scrns,dmShape)
            s += self.wfss[wfs].activeSubaps*2

        self.Twfs+=time.time()-t_wfs
        return slopes

    def runWfs_MP(self, scrns=None, dmShape=None, wfsList=None):
        """
        Runs all WFSs using multiprocessing
        
        Runs a single frame for each WFS in wfsList, passing the given phase screens and optional dmShape (if WFS in closed loop). If LGSs are present it will also deals with LGS propagation. Finally, the slopes from all WFSs are returned. Each WFS is allocated a seperate process to complete the frame, giving a significant increase in speed, expecially for computationally heavy WFSs.
        
        Args:
            scrns (list): List of phase screens passing over telescope
            dmShape (np array): 2-dimensional array of the total corrector shape
            wfsList (list): A list of the WFSs to be run
            
        Returns:
            ndarray: The slope data return from the WFS frame (may not be actual slopes if WFS other than SH used)
        """
        t_wfs = time.time()
        if scrns != None:
            self.scrns=scrns

        slopes = numpy.zeros( (self.totalSubaps*2))
        s = 0
        if wfsList==None:
            wfsList=range(self.GSNo)

        wfsProcs = []
        wfsQueues = []

        for wfs in xrange(self.GSNo):

            wfsQueues.append(Queue())
            wfsProcs.append(Process(target=multiWfs,
                    args=[ self.scrns,self.wfss[wfs],self.LGSs[wfs],dmShape,
                                                        wfsQueues[wfs]]))
        for wfs in xrange(self.GSNo):
            wfsProcs[wfs].start()

        for wfs in xrange(self.GSNo):

            (slopes[s:s+self.wfss[wfs].activeSubaps*2],
                    self.wfss[wfs].wfsDetectorPlane,
                    self.wfss[wfs].uncorrectedPhase,
                    lgsPsf) = \
                        wfsQueues[wfs].get()

            if lgsPsf!=None:
                self.LGSs[wfs].psf1 = lgsPsf

            wfsProcs[wfs].join()
            s += self.wfss[wfs].activeSubaps*2

        self.Twfs+=time.time()-t_wfs
        return slopes



    def runTipTilt(self,slopes,closed=True):
        """
        Runs a single frame of the Tip-Tilt Mirror

        Uses a previously created interaction matrix to calculate the correction required for the given slopes from a tip-tilt (TT) mirror. Returns the phase of the TT mirror and also the now TT corrected slopes. If no TT mirror is present in the system, then an array of zeros is returns and the slopes are unchanged.

        Args:
            slopes (ndarray): An array of WFS slope values
            closed (bool, optional): if True, TT acts in closed loop and the mirror shape is only updated from previous shape based on set TT gain value
        Returns:
            ndArray: New shape of the TT mirror
            ndArray: TT corrected slopes
        """

        if self.tipTilt:
            #calculate the shape of TT Mirror
            TTShape = self.TT.dmFrame(slopes,self.ttGain,closed)

            #Then remove TT signal from slopes
            xySlopes = slopes.reshape(2,self.TT.wfs.activeSubaps)
            xySlopes = (xySlopes.T - xySlopes.mean(1)).T
            slopes = xySlopes.reshape(self.TT.wfs.activeSubaps*2)
        else:
            TTShape = numpy.zeros( [self.pupilSize]*2 )

        return TTShape,slopes


    def runDM(self,dmCommands,closed=True):
        """
        Runs a single frame of the deformable mirrors

        Calculates the total combined shape of all deformable mirrors (DMs), given an array of DM commands. DM commands correspond to shapes generated during the making of interaction matrices, the final DM shape for each DM is a combination of these. The DM commands will have already been calculated by the systems reconstructor.

        Args:
            dmCommands (ndarray): an array of dm commands corresponding to dm shapes
            closed (bool): if True, indicates to DM that slopes are residual errors from previous frame, if False, slopes correspond to total phase error over pupil.
        Returns:
            ndArray: the combined DM shape
        """
        t = time.time()
        dmShape = numpy.zeros( [self.pupilSize]*2 )
        act=0
        for dm in xrange(self.dmNo):
            dmShape += self.dms[dm].dmFrame(
                    dmCommands[act:act+self.dms[dm].acts],
                    self.gain, closed)
            act+=self.dms[dm].acts

        self.Tdm += time.time()-t
        return dmShape


    def runSciCams(self,dmShape=None):
        """
        Runs a single frame of the science Cameras

        Calculates the image recorded by all science cameras in the system for the current phase over the telescope one frame. If a dmShape is present (which it usually will be in AO!) this correction is applied to the science phase before the image is calculated.

        Args:
            dmShape (ndarray, optional): An array of the combined system DM shape to correct the science path. If not given science cameras are in open loop.
        """
        t = time.time()

        for sci in xrange( self.sciNo ):
            self.sciImgs[sci] +=self.sciCams[sci].frame(self.scrns,dmShape)
            self.sciImgNo +=1

            self.sciCams[sci].longExpStrehl = self.sciImgs[sci].max()/(
                                    self.sciImgNo*self.sciCams[sci].psfMax)

        self.Tsci +=time.time()-t

    def open(self, progressCallback=None):
        '''
        Main AO Loop - loop open

        Runs a WFS iteration, reconstructs the phase, runs DMs and finally the science cameras. Also makes some nice output to the console and can add data to the Queue for the GUI if it has been requested. Repeats for nIters. Runs sim Open loop, i.e., the WFSs are executed before the DM with no DM information being sent back to the WFSs.

        Args:
            progressCallback (func): a function which is called after every iteration, can be used to output status information
        '''
        
        self.iters=1
        self.correct=1
        self.go = True

        for i in xrange(self.nIters):
            if self.go:
                logging.debug("Iteration %i",i)

                #Run Loop...
                self.slopes = self.runWfs()
                self.dmCommands = self.recon.reconstruct(self.slopes)
                dmShape = self.runDM(self.dmCommands,closed=False)
                self.runSciCams(dmShape)

                #get next phase screens
                t = time.time()
                self.scrns = self.atmos.moveScrns()
                self.Tatmos = time.time()-t

                #Save Data
                self.storeData(i)

                self.iters = i


                sys.stdout.write("\rLoop Frame: %d    "%i)
                sys.stdout.flush()
                
                if progressCallback!=None:
                    progressCallback(i, self.nIters, "Open AO loop" )
                self.addToGuiQueue()
            else:
                break

        self.saveData()
        self.finishUp()


    def closed(self, progressCallback=None):
        '''
        Main AO Loop - closed open

        Runs a WFS iteration, reconstructs the phase, runs DMs and finally the science cameras. Also makes some nice output to the console and can add data to the Queue for the GUI if it has been requested. Repeats for nIters. Runs sim Closed loop, i.e., the DM shape is computed before WFS slopes, and feedback from the DM is given to the WFSs.

        Args:
            progressCallback (func): a function which is called after every iteration, can be used to output status information
        '''
        self.iters=1
        self.correct=1
        self.go = True
        self.slopes = numpy.zeros( ( 2*self.totalSubaps) )
        self.dmShapes = numpy.zeros( (self.nIters, self.pupilSize, self.pupilSize))

        #pylab.figure()


        for i in xrange(self.nIters):
            if self.go:
                logging.debug("Iteration %i",i)


                ttShape,self.slopes = self.runTipTilt(self.slopes)
                self.dmCommands = self.recon.reconstruct(self.slopes)
                dmShape = self.runDM(self.dmCommands) + ttShape
                self.runSciCams(dmShape)
                self.slopes = self.runWfs(dmShape=dmShape)

                #Useful to track progress externally
                self.iters = i

                #Get next phase screens from atmos object.
                t = time.time()
                self.scrns = self.atmos.moveScrns()
                self.Tatmos += time.time()-t
                #saveData
                self.storeData(i)

                sys.stdout.write("\rLoop Frame: %d    "%i)
                sys.stdout.flush()

                self.addToGuiQueue()
                if progressCallback!=None:
                    progressCallback(i, self.nIters, "Closed AO loop" )

            else:
                #Stop sim if "go"!=True
                break

        #Finally save data after loop is over.
        self.saveData()
        self.finishUp()

    def finishUp(self):
        """
        Prints a message to the console giving timing data. Used on sim end.
        """
        print("\n\nTime moving atmosphere: %0.2f"%self.Tatmos)
        print("Time making IMats and CMats: %0.2f"%self.Timat)
        print("Time in WFS: %0.2f"%self.Twfs)
        print ("\t of which time spent in LGS: %0.2f"%self.Tlgs)
        print("Time in Reconstruction: %0.2f"%self.recon.Trecon)
        print("Time in DM: %0.2f"%self.Tdm)
        print("Time making science image: %0.2f"%self.Tsci)
        
        # if self.longStrehl:
#    print("\n\nLong Exposure Strehl Rate: %0.2f"%self.longStrehl[-1])

    def initSaveData(self):
        '''
        Initialise data structures used for data saving.

        Initialise the data structures which will be used to store data which will be saved or analysed once the simulation has ended. If the ``filePrefix = None``, no data is saved, other wise a directory called ``filePrefix`` is created, and data from simulation runs are saved in a time-stamped directory inside this.
        '''

        if self.filePrefix!=None:
            self.path = self.filePrefix +"/"+self.timeStamp()
            try:
                os.mkdir( self.path )
            except OSError:

                os.mkdir(self.filePrefix)
                os.mkdir(self.path)

            #Init WFS FP Saving
            if self.saveWfsFrames:
                os.mkdir(self.path+"/wfsFPFrames/")


            shutil.copyfile( self.configFile, self.path+"/conf.py" )

        #Init Strehl Saving
        if self.sciNo>0:
            self.instStrehl = numpy.zeros( (self.sciNo, self.nIters) )
            self.longStrehl = numpy.zeros( (self.sciNo, self.nIters) )

        #Init science residual phase saving
        self.sciPhase = []
        if self.saveSciRes and self.sciNo>0:
            for sci in xrange(self.sciNo):
                self.sciPhase.append(
                    numpy.empty( (self.nIters, self.pupilSize, self.pupilSize)))

        #Init WFS slopes data saving
        if self.saveSlopes:
            self.allSlopes = numpy.empty( (self.nIters, 2*self.totalSubaps) )
        else:
            self.allSlopes = None

        #Init DM Command Data saving
        if self.saveDmCommands:
            ttActs = 0
            if self.tipTilt:
                ttActs = 2
            self.allDmCommands = numpy.empty( (self.nIters, ttActs+self.totalActs))

        else:
            self.allDmCommands = None

        #Init LGS PSF Saving
        if self.saveLgsPsf:
            self.lgsPsfs = numpy.empty( (self.nIters, numpy.sum(self.LGSUplink),
                                self.pupilSize,self.pupilSize) )
                                # self.pxlsPerSubap[0]*self.subapOversamp[0],
#                                 self.pxlsPerSubap[0]*self.subapOversamp[0]) )

        else:
            self.lgsPsfs = None




    def storeData(self,i):
        """
        Stores data from each frame in an appropriate data structure.

        Called on each frame to store the simulation data into various data structures corresponding to different data sources in the system. 

        Args:
            i(int): The system iteration number
        """
        if self.saveSlopes:
            self.allSlopes[i] = self.slopes

        if self.saveDmCommands:
            act=0
            if self.tipTilt:
                self.allDmCommands[i,:2] = self.TT.dmCommands
                act=2
            self.allDmCommands[i,act:] = self.dmCommands

        #Quick bodge to save lgs psfs as images
        #FITS.Write(self.LGSs[0].psf1,self.path+"/lgsPsf_frame-%s.fits"%i)
        if self.saveLgsPsf:
            lgs=0
            for wfs in xrange(self.GSNo):
                if self.LGSUplink[wfs]!=0:
                    self.lgsPsfs[i,lgs] = self.LGSs[wfs].PSF
                    lgs+=1

        if self.sciNo>0:
            for sci in xrange(self.sciNo):
                self.instStrehl[sci,i] = self.sciCams[sci].instStrehl
                self.longStrehl[sci,i] = self.sciCams[sci].longExpStrehl
            
            if self.saveSciRes:
                for sci in xrange(self.sciNo):
                    self.sciPhase[sci][i] = self.sciCams[sci].residual

        if self.filePrefix!=None:
            if self.saveWfsFrames:
                for wfs in xrange(self.GSNo):
                    pyfits.writeto(
                        self.path+"/wfsFPFrames/wfs-%d_frame-%d.fits"%(wfs,i),
                        self.wfss[wfs].wfsDetectorPlane     )
                    # FITS.Write(
#                         self.wfss[wfs].wfsDetectorPlane,
#   self.path+"/wfsFPFrames/wfs-%d_frame-%d.fits"%(wfs,i))
                

    def saveData(self):
        """
        Saves all recorded data to disk

        Called once simulation has ended to save the data recorded during the simulation to disk in the directories created during initialisation.
        """

        if self.filePrefix!=None:

            if self.saveSlopes:
                pyfits.writeto(self.path+"/slopes.fits", self.allSlopes)

            if self.saveDmCommands:
                pyfits.writeto(self.path+"/dmCommands.fits", self.allDmCommands)

            if self.saveLgsPsf:
                pyfits.writeto(self.path+"/lgsPsf.fits", self.lgsPsfs)


            if self.saveStrehl:
                pyfits.writeto(self.path+"/instStrehl.fits", self.instStrehl)
                pyfits.writeto(self.path+"/longStrehl.fits", self.longStrehl)  

            if self.saveSciRes:
                for i in xrange(self.sciNo):
                    pyfits.writeto(self.path+"/sciResidual_%02d.fits"%i,
                            self.sciPhase[i])

            if self.saveSciPsf:
                for i in xrange(self.sciNo):
                    pyfits.writeto(self.path+"/sciPsf_%02d.fits"%i,
                                        self.sciImgs[i] )
                    

    def timeStamp(self):
        """
        Returns a formatted timestamp
        
        Returns:
            string: nicely formatted timestamp of current time.
        """

        return datetime.datetime.now().strftime("%Y-%m-%d-%H-%M-%S")


    def printOutput(self,label,iter,strehl=False):
        """
        Prints simulation information  to the console

        Called on each iteration to print information about the current simulation, such as current strehl ratio, to the console. Still under development
        Args:
            label(str): Simulation Name
            iter(int): simulation frame number
            strehl(float, optional): current strehl ration if science cameras are present to record it.
        """
        
        string = "%s Frame: %d      \n"%(label,iter)
        if strehl:
            string += "Strehl:      %2.2f   "%(sim.longStrehl[iter])
            string += "inst Strehl  %2.2f   "%(sim.instStrehl[iter])
            
        sys.stdout.write(string)
        sys.stdout.flush()
            
        

    def addToGuiQueue(self):
        """
        Adds data to a Queue object provided by the pyAOS GUI.

        The pyAOS GUI doesn't need to plot every frame from the simulation. When it wants a frame, it will request if by setting ``waitingPlot = True``. As this function is called on every iteration, data is passed to the GUI only if ``waitingPlot = True``. This allows efficient and abstracted interaction between the GUI and the simulation
        """

        if self.guiQueue!=None:
            if self.waitingPlot:
                guiPut = []
                wfsFocalPlane = {}
                wfsPhase = {}
                lgsPsf = {}
                for i in xrange(self.GSNo):
                    wfsFocalPlane[i] = self.wfss[i].wfsDetectorPlane.copy().astype("float32")
                    try:
                        wfsPhase[i] = self.wfss[i].uncorrectedPhase.copy()
                    except AttributeError:
                        wfsPhase[i] = None
                        pass

                    try:
                        lgsPsf[i] = self.LGSs[i].psf1.copy()
                    except AttributeError:
                        lgsPsf[i] = None
                        pass

                try:
                    ttShape = self.TT.dmShape
                except AttributeError:
                    ttShape = None

                dmShape = {}
                for i in xrange(self.dmNo):
                    try:
                        dmShape[i] = self.dms[i].dmShape.copy()
                    except AttributeError:
                        dmShape[i] = None

                sciImg = {}
                residual = {}
                instSciImg = {}
                for i in xrange(self.sciNo):
                    try:
                        sciImg[i] = self.sciImgs[i].copy()
                    except AttributeError:
                        sciImg[i] = None
                    try:
                        instSciImg[i] = self.sciCams[i].focalPlane.copy()
                    except AttributeError:
                        instSciImg[i] = None

                    try:
                        residual[i] = self.sciCams[i].residual.copy()
                    except AttributeError:
                        residual[i] = None

                guiPut = {  "wfsFocalPlane":wfsFocalPlane,
                            "wfsPhase":     wfsPhase,
                            "lgsPsf":       lgsPsf,
                            "ttShape":      ttShape,
                            "dmShape":      dmShape,
                            "sciImg":       sciImg,
                            "instSciImg":   instSciImg,
                            "residual":     residual }

                self.guiLock.lock()
                try:
                    self.guiQueue.put_nowait(guiPut)
                except:
                    self.guiLock.unlock()
                    traceback.print_exc()
                self.guiLock.unlock()

                self.waitingPlot = False


#Functions used by MP stuff
def multiWfs(scrns, wfsObj, lgsObj, dmShape, queue):
    """
    Function to run the WFS in multiprocessing mode.

    Function is called by each of the new WFS processes spawned to run each WFS. Does the same job as the sim runWfs_noMP method of running LGS, then getting slopes from each WFS.

    Args:
        scrns (list): list of the phase screens over the WFS 
        wfsObj (WFS object): the WFS object being run
        lgsObj (LGS object): the LGS object corresponding to this WFS 
        dmShape (ndArray):  shape of system DMs for WFS phase correction
        queue (Queue object): a multiprocessing Queue object used to pass data back to host process.
    """

    lgsPsf = None
    if lgsObj!=None:
        lgsObj.LGSPSF(scrns)
        lgsPsf = lgsObj.psf1


    slopes = wfsObj.frame(scrns, dmShape)

    queue.put([slopes,wfsObj.wfsDetectorPlane, wfsObj.uncorrectedPhase, lgsPsf])

if __name__ == "__main__":


    parser = ArgumentParser()
    parser.add_argument("configFile",nargs="?",action="store")
    args = parser.parse_args()
    if args.configFile != None:
        confFile = args.configFile
    else:
        confFile = "conf/testConf.py"


    sim = Sim(confFile)
    print("AOInit...") 
    sim.aoinit()
    sim.makeIMat()
    sim.aoloop()


