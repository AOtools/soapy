#! /usr/bin/env python

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


'''
The main Soapy Simulation module

This module contains the ``Sim`` class, which can be used to run an end-to-end simulation. Initally, a configuration file is read, the system is initialised, interaction and command matrices calculated and finally a loop run. The simulation outputs some information to the console during the simulation.

    The ``Sim`` class holds all configuration information and data from the simulation.

Examples:

    To initialise the class::

        import soapy
        sim = soapy.Sim("sh_8x8_4.2m.py")

    Configuration information has now been loaded, and can be accessed through the ``config`` attribute of the ``sim`` class. In fact, each sub-module of the system has a configuration object accessed through this config attribute::

        print(sim.config.sim.pupilSize)
        sim.config.wfss[0].pxlsPerSubap = 10

    Next, the system is initialised, this entails calculating various parameters in the system sub-modules, so must be done after changing some simulation parameters::

        sim.aoinit()

    DM Interation and command matrices are calculated now. If ``sim.config.sim.simName`` is not ``None``, then these matrices will be saved in ``data/simName`` (data will be saved here also in a time-stamped directory)::

        sim.makeIMat()


    Finally, the loop is run with the command::

        sim.aoloop()

    Some output will be printed to the console. After the loop has finished, data specified to be saved in the config file will be saved to ``data/simName`` (if it is not set to ``None``). Data can also be accessed from the simulation class, e.g. ``sim.allSlopes``, ``sim.longStrehl``


:Author:
    Andrew Reeves

'''
from ._version import get_versions
__version__ = get_versions()['version']
del get_versions

# standard python imports
import datetime
import os
import time
import traceback
from multiprocessing import Process, Queue
from argparse import ArgumentParser
import shutil
import importlib
import threading

import numpy
#Use pyfits or astropy for fits file handling
try:
    from astropy.io import fits
except ImportError:
    try:
        import pyfits as fits
    except ImportError:
        raise ImportError("soapy requires either pyfits or astropy")

import aotools

#sim imports
from . import atmosphere, logger, wfs, DM, reconstruction, scienceinstrument, confParse, interp

import shutil

#xrange now just "range" in python3.
#Following code means fastest implementation used in 2 and 3
try:
    xrange
except NameError:
    xrange = range

class Sim(object):
    """
    The soapy Simulation class.

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

        self.readParams(configFile)
        #logger.info("Loaded config file successfully!")

        self.guiQueue = None
        self.go = False
        self._sim_running = False


    def readParams(self, configFile=None):
        """
        Reads configuration file parameters

        Calls the radParams function in confParse to read, parse and if required
        set reasonable defaults to AO parameters
        """

        if configFile:
            self.configFile = configFile

        logger.info("Loading configuration file...")
        self.config = confParse.loadSoapyConfig(self.configFile)
        logger.info("Loading configuration file... success!")


    def setLoggingLevel(self, level):
        """
        sets which messages are printed from logger.

        if logging level is set to 0, nothing is printed. if set to 1, only
        warnings are printed. if set to 2, warnings and info is printed. if set
        to 3 detailed debugging info is printed.

        parameters:
            level (int): the desired logging level
        """
        logger.setLoggingLevel(level)


    def aoinit(self):
        '''
        Initialises all simulation objects.

        Initialises and passes relevant data to sim objects. 
        This does important pre-run tasks, such as creating or 
        loading phase screens, determining WFS geometry, 
        setting propagation modes and pre-allocating data arrays 
        used later in the simulation.
        '''
        # Read params if they haven't been read before
        try:
            self.config.sim.pupilSize
        except:
            self.readParams()

        logger.setLoggingLevel(self.config.sim.verbosity)
        logger.setLoggingFile(self.config.sim.logfile)
        logger.info("Starting Sim: {}".format(self.getTimeStamp()))

        # Calculate some params from read ones
        self.config.calcParams()

        # Init Pupil Mask
        logger.info("Creating mask...")
        self.mask = make_mask(self.config)

        self.atmos = atmosphere.atmos(self.config)

        # Find if WFSs should each have own process
        if self.config.sim.wfsMP:
            logger.info("Setting fftwThreads to 1 as WFS MP")
            for nwfs in xrange(self.config.sim.nGS):
                self.config.wfss[nwfs].fftwThreads = 1
            self.runWfs = self.runWfs_MP
        else:
            self.runWfs = self.runWfs_noMP

        # Init WFSs
        logger.info("Initialising WFSs....")
        self.wfss = {}
        self.config.sim.totalWfsData = 0
        self.wfsFrameNo = numpy.zeros(self.config.sim.nGS)
        for nwfs in xrange(self.config.sim.nGS):
            try:
                if self.config.wfss[nwfs].loadModule:
                    wfs_lib = importlib.import_module(self.config.wfsss[nwfs].loadModule)
                else:
                    wfs_lib = wfs
                wfsClass = getattr(wfs_lib, self.config.wfss[nwfs].type)
            except AttributeError:
                raise confParse.ConfigurationError(
                        "No WFS of type {} found.".format(
                                self.config.wfss[nwfs].type))

            self.wfss[nwfs] = wfsClass(
                    self.config, n_wfs=nwfs, mask=self.mask)

            self.config.wfss[nwfs].dataStart = self.config.sim.totalWfsData
            self.config.sim.totalWfsData += self.wfss[nwfs].n_measurements

            logger.info("WFS {0}: {1} measurements".format(nwfs,
                     self.wfss[nwfs].n_measurements))

        # Init DMs
        logger.info("Initialising {0} DMs...".format(self.config.sim.nDM))
        self.dms = {}
        self.dmActCommands = {}
        self.config.sim.totalActs = 0
        self.dmShape = numpy.zeros([self.config.sim.simSize]*2)
        self.dmAct1 = []
        for dm in xrange(self.config.sim.nDM):
            self.dmAct1.append(self.config.sim.totalActs)
            try:
                if self.config.dms[dm].loadModule:
                    dm_lib = importlib.import_module(self.config.dms[dm].loadModule)
                else:
                    dm_lib = DM
                dmObj = getattr(dm_lib, self.config.dms[dm].type)
            except AttributeError:
                raise confParse.ConfigurationError("No DM of type {} found".format(self.config.dms[dm].type))

            self.dms[dm] = dmObj(
                    self.config, n_dm=dm, wfss=self.wfss,
                    mask=self.mask
                    )

            self.dmActCommands[dm] = numpy.empty(
                    (self.config.sim.nIters, self.dms[dm].n_acts))

            self.dmAct1.append(self.config.sim.totalActs)
            self.config.sim.totalActs += self.dms[dm].n_acts

            logger.info("DM %d: %d active actuators"%(dm,self.dms[dm].n_acts))
        logger.info("%d total DM Actuators"%self.config.sim.totalActs)


        # Init Reconstructor
        logger.info("Initialising Reconstructor...")
        try:
            if self.config.recon.loadModule:
                recon_lib = importlib.import_module(self.config.recon.loadModule)
            else:
                recon_lib = reconstruction
            reconObj = getattr(recon_lib, self.config.recon.type)
        except AttributeError:
            raise confParse.ConfigurationError("No reconstructor of type {} found.".format(self.config.recon.type))
        self.recon = reconObj(
                self.config, self.dms, self.wfss, self.atmos,
                self.runWfs
                )

        # Init Science Cameras
        logger.info("Initialising {0} Science Cams...".format(self.config.sim.nSci))
        self.sciCams = {}
        self.sciImgs = {}
        self.sciImgNo=0
        for nSci in xrange(self.config.sim.nSci):
            try:
                if self.config.scis[nSci].loadModule:
                    sci_lib = importlib.import_module(self.config.scis[nSci].loadModule)
                else:
                    sci_lib = scienceinstrument
                sciObj = getattr(sci_lib, self.config.scis[nSci].type)
            except AttributeError:
                raise confParse.ConfigurationError("No science camera of type {} found".format(self.config.scis[nSci].type))
            self.sciCams[nSci] = sciObj(
                        self.config, nSci=nSci, mask=self.mask
                        )

            self.sciImgs[nSci] = numpy.zeros( [self.config.scis[nSci].pxls]*2 )

        # Init data storage
        logger.info("Initialise Data Storage...")
        self.initSaveData()

        # Init simulation
        #Circular buffers to hold loop iteration correction data
        self.slopes = numpy.zeros((self.config.sim.totalWfsData))
        self.closed_correction = numpy.zeros((
                self.config.sim.nDM, self.config.sim.scrnSize, self.config.sim.scrnSize
                ))
        self.open_correction = self.closed_correction.copy()
        self.dmCommands = numpy.zeros(self.config.sim.totalActs)
        self.buffer = DelayBuffer()
        self.iters = 0

        # Init performance tracking
        self.Twfs = 0
        self.Tlgs = 0
        self.Tdm = 0
        self.Tsci = 0
        self.Trecon = 0
        self.Timat = 0
        self.Tatmos = 0

        logger.info("Initialisation Complete!")


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
            forceNew (bool): if true, will force making of new iMats and cMats, otherwise will attempt to load previously made matrices from same simName
            progressCallback (func): function called to report progress of interaction matrix construction
        """
        t = time.time()
        logger.info("Making interaction Matrices...")

        if forceNew:
            loadIMat=False
            loadCMat=False
        else:
            if self.config.sim.simName==None:
                loadIMat=False
                loadCMat=False
            else:
                loadIMat=True
                loadCMat=True

        self.recon.makeCMat(loadIMat=loadIMat,loadCMat=loadCMat,
                callback=self.addToGuiQueue, progressCallback=progressCallback)


        # Now know valid actuators for each DM, can get the index of the each DM in the command vector
        self.dmAct1 = []
        self.config.sim.totalActs = 0
        for dm in self.dms.values():
            self.dmAct1.append(self.config.sim.totalActs)
            self.config.sim.totalActs += dm.n_valid_actuators

        self.dmCommands = numpy.zeros(self.config.sim.totalActs)

        self.Timat += time.time() - t


    def runWfs_noMP(self, scrns = None, dmShape=None, wfsList=None,
                    loopIter=None):
        """
        Runs all WFSs

        Runs a single frame for each WFS in wfsList, passing the given phase screens and optional dmShape (if WFS in closed loop). The WFSs are only read out if the wfs frame time co-incides with the WFS frame rate, else old slopes are provided. If iter is not given, then all WFSs are run and read out. If LGSs are present it will also deals with LGS propagation. Finally, the slopes from all WFSs are returned.

        Args:
            scrns (list): List of phase screens passing over telescope
            dmShape (ndarray, optional): 2-dim array of the total corrector shape
            wfsList (list, optional): A list of the WFSs to be run
            loopIter (int, optional): The loop iteration number

        Returns:
            ndarray: The slope data return from the WFS frame (may not be actual slopes if WFS other than SH used)
            """
        t_wfs = time.time()
        if scrns is not None:
            self.scrns=scrns

        if wfsList==None:
            wfsList=range(self.config.sim.nGS)

        slopesSize = 0
        for nwfs in wfsList:
            slopesSize+=self.wfss[nwfs].n_measurements
        slopes = numpy.zeros( (slopesSize) )

        s = 0
        for nwfs in wfsList:
            #check if due to read out WFS
            if (int(float(self.config.sim.loopTime*(loopIter+1))
                    /self.config.wfss[nwfs].exposureTime)
                                    != self.wfsFrameNo[nwfs]):
                self.wfsFrameNo[nwfs]+=1
                read=True
            else:
                read=False

            slopes[s:s+self.wfss[nwfs].n_measurements] = \
                    self.wfss[nwfs].frame(self.scrns, dmShape, read=read)
            s += self.wfss[nwfs].n_measurements

        self.Twfs+=time.time()-t_wfs
        return slopes


    def runWfs_MP(self, scrns=None, dmShape=None, wfsList=None, loopIter=None):
        """
        Runs all WFSs using multiprocessing

        Runs a single frame for each WFS in wfsList, passing the given phase
        screens and optional dmShape (if WFS in closed loop). If LGSs are
        present it will also deals with LGS propagation. Finally, the slopes
        from all WFSs are returned. Each WFS is allocated a separate process
        to complete the frame, giving a significant increase in speed,
        especially for computationally heavy WFSs.

        Args:
            scrns (list): List of phase screens passing over telescope
            dmShape (ndarray, optional): 2-dimensional array of the total corrector shape
            wfsList (list, optional): A list of the WFSs to be run, if not set, runs all WFSs
            loopIter (int, optional): The loop iteration number

        Returns:
            ndarray: The slope data return from the WFS frame (may not be actual slopes if WFS other than SH used)
        """
        t_wfs = time.time()
        if scrns is not None:
            self.scrns=scrns
        if wfsList==None:
            wfsList=range(self.config.sim.nGS)

        slopesSize = 0
        for nwfs in wfsList:
            slopesSize+=self.wfss[nwfs].n_measurements
        slopes = numpy.zeros( (slopesSize) )

        wfsProcs = []
        wfsQueues = []
        s = 0
        for proc in xrange(len(wfsList)):
            nwfs = wfsList[proc]
            # check if due to read out WFS
            if loopIter:
                read=False
                if (int(float(self.config.sim.loopTime*(loopIter+1))
                        /self.config.wfss[nwfs].exposureTime)
                                        != self.wfsFrameNo[nwfs]):
                    self.wfsFrameNo[nwfs]+=1
                    read = True
            else:
                read = True

            wfsQueues.append(Queue())
            wfsProcs.append(Process(target=multiWfs,
                    args=[  self.scrns, self.wfss[nwfs], dmShape, read,
                            wfsQueues[proc]])
                    )
            wfsProcs[proc].daemon = True
            wfsProcs[proc].start()

        for proc in xrange(len(wfsList)):
            nwfs = wfsList[proc]

            (slopes[s:s+self.wfss[nwfs].n_measurements],
                    self.wfss[nwfs].wfsDetectorPlane,
                    self.wfss[nwfs].uncorrectedPhase,
                    lgsPsf) = wfsQueues[proc].get()

            if numpy.any(lgsPsf)!=None:
                self.wfss[nwfs].LGS.psf1 = lgsPsf

            wfsProcs[proc].join()
            s += self.wfss[nwfs].n_measurements

        self.Twfs+=time.time()-t_wfs
        return slopes


    def runDM(self, dmCommands, closed=True):
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
        self.dmShapes = []
        if closed:
            correction_buffer = self.closed_correction
        else:
            correction_buffer = self.open_correction

        for dm in xrange(self.config.sim.nDM):
            if self.config.dms[dm].closed == closed:
                correction_buffer[dm] = self.dms[dm].dmFrame(
                        dmCommands[ self.dmAct1[dm]:
                                    self.dmAct1[dm]+self.dms[dm].n_valid_actuators])

        self.Tdm += time.time() - t
        return correction_buffer


    def runSciCams(self, dmShape=None):
        """
        Runs a single frame of the science Cameras

        Calculates the image recorded by all science cameras in the system for the current phase over the telescope one frame. If a dmShape is present (which it usually will be in AO!) this correction is applied to the science phase before the image is calculated.

        Args:
            correction (list or ndarray, optional): An array of the combined system DM shape to correct the science path. If not given science cameras are in open loop.
        """
        t = time.time()

        self.sciImgNo +=1
        for sci in xrange(self.config.sim.nSci):
            self.sciImgs[sci] += self.sciCams[sci].frame(self.scrns, dmShape)

            # Normalise long exposure psf
            #self.sciImgs[sci] /= self.sciImgs[sci].sum()
            self.sciCams[sci].longExpStrehl = (
                    self.sciImgs[sci].max()/
                    self.sciImgs[sci].sum()/
                    self.sciCams[sci].psfMax)

        self.Tsci +=time.time()-t


    def loopFrame(self):
        """
        Runs a single from of the entire AO system.

        Moves the atmosphere, runs the WFSs, finds the corrective DM shape and finally runs the science cameras. This can be called over and over to form the "loop"
        """
        # Get next phase screens
        t = time.time()
        self.scrns = self.atmos.moveScrns()
        self.Tatmos += time.time()-t

        # Run Loop...
        ########################################

        # Get dmCommands from reconstructor
        t_recon = time.time()
        if self.config.sim.nDM:
            self.dmCommands[:] = self.recon.reconstruct(self.slopes)
        self.Trecon += (time.time() - t_recon)

        # Delay the dmCommands if loopDelay is configured
        self.dmCommands = self.buffer.delay(self.dmCommands, self.config.sim.loopDelay)

        # Get dmShape from closed loop DMs
        self.closed_correction = self.runDM(
                self.dmCommands, closed=True)

        # Run WFS, with closed loop DM shape applied
        self.slopes = self.runWfs(dmShape=self.closed_correction,
                                  loopIter=self.iters)

        # Get DM shape for open loop DMs
        self.open_correction = self.runDM(self.dmCommands,
                                          closed=False)

        # Pass whole combined DM shapes to science target
        self.combinedCorrection = self.open_correction + self.closed_correction

        self.runSciCams(self.combinedCorrection)

        # Save Data
        i = self.iters % self.config.sim.nIters # If sim is run continuously in loop, overwrite oldest data in buffer
        self.storeData(i)

        self.printOutput(self.iters, strehl=True)

        self.addToGuiQueue()

        self.iters += 1


    def aoloop(self):
        """
        Main AO Loop

        Runs a WFS iteration, reconstructs the phase, runs DMs and finally the science cameras. Also makes some nice output to the console and can add data to the Queue for the GUI if it has been requested. Repeats for nIters.
        """

        self.go = True
        try:
            while self.iters < self.config.sim.nIters:
                if self.go:
                    self.loopFrame()
                else:
                    break
        except KeyboardInterrupt:
            self.go = False
            logger.info("\nSim exited by user\n")


        # Finally save data after loop is over.
        self.saveData()
        self.finishUp()


    def start_aoloop_thread(self):
        """
        Run the simulation continuously in a thread

        The Simulation will loop continuously as long as it is required. The data buffers for
        simulation are limited however to the size given by sim.config.nIters. Once this is 
        full, the oldest data will be overwritten.
        """
        if self._sim_running is False:
            self._loop_thread = threading.Thread(
                    target=self._aoloop_thread, daemon=True)

            self._sim_running = True
            self._loop_thread.start()


    def stop_aoloop_thread(self):
        """
        Stops the AO loop if its running continuously in a thread.

        Stops the simulation after the current iteration and joins the loop thread.
        Will save the data buffers to disk if configured to do so and v
        the output summary
        """

        if self._sim_running:
            # signal for thread to stop
            self._sim_running = False
            # wait for thread to finish
            self._loop_thread.join()

            # save data and finish up
            self.saveData()
            self.finishUp()
            

    def _aoloop_thread(self):
        """
        Runs the AO Loop as a while loop to be used in a thread
        """
        while self._sim_running:
            self.loopFrame()


    def reset_loop(self):
        """
        Resets parameters in the system to zero, to restart an AO run wihtout reinitialising
        """
        self.iters = 0

        self.slopes[:] = 0
        self.dmCommands[:] = 0

        if self.config.sim.saveSlopes:
            self.allSlopes[:] = 0

        if self.config.sim.saveDmCommands:
            self.allDmCommands[:] = 0

    
        self.recon.reset()

        if self.config.sim.nSci > 0:
            self.longStrehl[:] = 0
            self.ee50d[:] = 0
            for sci in self.sciImgs.values(): sci[:] = 0
        
        for dm in self.dms.values(): dm.reset()


    def finishUp(self):
        """
        Prints a message to the console giving timing data. Used on sim end.
        """
        print('\n')
        iter_num = self.iters % self.config.sim.nIters -1
        if hasattr(self, "longStrehl") and (self.longStrehl is not None):
            for sci_n in range(self.config.sim.nSci):
                print("Science Camera {}: Long Exposure Strehl Ratio: {:0.2f}".
                      format(sci_n, self.longStrehl[sci_n][iter_num]))
                if hasattr(self, "ee50d") and (self.ee50d is not None):
                    print("                  EE50 diameter [mas]: {:0.0f}".
                          format(self.ee50d[sci_n] * 1000))

        print("\n\nTime moving atmosphere: %0.2f"%self.Tatmos)
        print("Time making IMats and CMats: %0.2f"%self.Timat)
        print("Time in WFS: %0.2f"%self.Twfs)
        print ("\t of which time spent in : %0.2f"%self.Tlgs)
        print("Time in Reconstruction: %0.2f"%self.recon.Trecon)
        print("Time in DM: %0.2f"%self.Tdm)
        print("Time making science image: %0.2f"%self.Tsci)
        print("\n")


    def initSaveData(self):
        '''
        Initialise data structures used for data saving.

        Initialise the data structures which will be used to store data which will be saved or analysed once the simulation has ended. If the ``simName = None``, no data is saved, other wise a directory called ``simName`` is created, and data from simulation runs are saved in a time-stamped directory inside this.
        '''

        # Initialise the FITS header to use. Store in `config.sim`
        self.config.sim.saveHeader = self.makeSaveHeader()

        if self.config.sim.simName!=None:
            self.path = self.config.sim.simName +"/"+self.timeStamp
            # make sure a different directory used by sleeping
            time.sleep(1)
            try:
                os.mkdir(self.path)
            except OSError:
                os.mkdir(self.config.sim.simName)
                os.mkdir(self.path)

            #Init WFS FP Saving
            if self.config.sim.saveWfsFrames:
                os.mkdir(self.path+"/wfsFPFrames/")
            
            # Copy the config file to the save directory so you can 
            # remember what the parameters where
            if isinstance(self.config, confParse.YAML_Configurator):
                fname = "conf.yaml"
            else:
                fname = "conf.py"
            shutil.copyfile(self.configFile, os.path.join(self.path, fname))

        # Init Strehl Saving
        if self.config.sim.nSci>0:
            self.instStrehl = numpy.zeros(
                    (self.config.sim.nSci, self.config.sim.nIters) )
            self.longStrehl = numpy.zeros(
                    (self.config.sim.nSci, self.config.sim.nIters) )
            self.ee50d = numpy.zeros((self.config.sim.nSci))

            # Init science WFE saving
            self.WFE = numpy.zeros(
                        (self.config.sim.nSci, self.config.sim.nIters)
                        )

        #Init science residual phase saving
        self.sciPhase = []
        if self.config.sim.saveSciRes and self.config.sim.nSci>0:
            for sci in xrange(self.config.sim.nSci):
                self.sciPhase.append(
                    numpy.empty(
                            (self.config.sim.nIters, self.config.sim.simSize,
                            self.config.sim.simSize)))



        #Init WFS slopes data saving
        if self.config.sim.saveSlopes:
            self.allSlopes = numpy.zeros(
                    (self.config.sim.nIters, self.config.sim.totalWfsData) )
        else:
            self.allSlopes = None

        #Init DM Command Data saving
        if self.config.sim.saveDmCommands:
            ttActs = 0

            self.allDmCommands = numpy.zeros( (self.config.sim.nIters, ttActs+self.config.sim.totalActs))

        else:
            self.allDmCommands = None

        #Init LGS PSF Saving
        if self.config.sim.saveLgsPsf:
            self.lgsPsfs = []
            for lgs in xrange(self.config.sim.nGS):
                if self.config.wfss[lgs].lgs and self.config.wfss[lgs].lgs.uplink:
                    self.lgsPsfs.append(
                            numpy.empty((self.config.sim.nIters,
                            self.wfss[lgs].lgs.nOutPxls,
                            self.wfss[lgs].lgs.nOutPxls))
                            )
            self.lgsPsfs = numpy.array(self.lgsPsfs)

        else:
            self.lgsPsfs = None

        #Init Instantaneous PSF saving
        if self.config.sim.nSci>0 and self.config.sim.saveInstPsf==True:
            self.sciImgsInst = {}

            for sci in xrange(self.config.sim.nSci):
                self.sciImgsInst[sci] = numpy.zeros([self.config.sim.nIters,self.config.scis[sci].pxls,self.config.scis[sci].pxls])


        #Init Instantaneous electric field
        if self.config.sim.nSci>0 and self.config.sim.saveInstScieField==True:
            self.scieFieldInst = {}

            for sci in xrange(self.config.sim.nSci):
                self.scieFieldInst[sci] = numpy.zeros(([self.config.sim.nIters,self.config.scis[sci].pxls,self.config.scis[sci].pxls]), dtype=complex )


    def storeData(self, i):
        """
        Stores data from each frame in an appropriate data structure.

        Called on each frame to store the simulation data into various data 
        structures corresponding to different data sources in the system.

        For some data streams that are very large, data gets saved to disk on 
        each iteration - this also happens here.

        Args:
            i (int): The system iteration number
        """
        if self.config.sim.saveSlopes:
            self.allSlopes[i] = self.slopes

        if self.config.sim.saveDmCommands:
            act=0
            self.allDmCommands[i,act:] = self.dmCommands

        #Quick bodge to save lgs psfs as images
        if self.config.sim.saveLgsPsf:
            lgs=0
            for nwfs in xrange(self.config.sim.nGS):
                if self.config.wfss[nwfs].lgs and self.config.wfss[nwfs].lgs.uplink:
                    self.lgsPsfs[lgs, i] = self.wfss[nwfs].lgs.psf
                    lgs+=1

        if self.config.sim.nSci>0:
            for sci in xrange(self.config.sim.nSci):
                self.instStrehl[sci,i] = self.sciCams[sci].instStrehl
                self.longStrehl[sci,i] = self.sciCams[sci].longExpStrehl

                # Record WFE residual
                self.WFE[sci, i] = self.sciCams[sci].calc_wavefronterror()

            if self.config.sim.saveSciRes:
                for sci in xrange(self.config.sim.nSci):
                    self.sciPhase[sci][i] = self.sciCams[sci].residual

        if self.config.sim.simName!=None:
            if self.config.sim.saveWfsFrames:
                for nwfs in xrange(self.config.sim.nGS):
                    fits.writeto(
                        self.path+"/wfsFPFrames/wfs-%d_frame-%d.fits"%(nwfs,i),
                        self.wfss[nwfs].wfsDetectorPlane,
                        header=self.config.sim.saveHeader)

        # Save Instantaneous PSF
        if self.config.sim.nSci>0 and self.config.sim.saveInstPsf==True:
            for sci in xrange(self.config.sim.nSci):
                self.sciImgsInst[sci][i,:,:] = self.sciCams[sci].detector


        # Save Instantaneous electric field
        if self.config.sim.nSci>0 and self.config.sim.saveInstScieField==True:
            for sci in xrange(self.config.sim.nSci):
                self.scieFieldInst[sci][self.iters,:,:] = self.sciCams[sci].focalPlane_efield


    def saveData(self):
        """
        Saves all recorded data to disk

        Called once simulation has ended to save the data recorded during 
        the simulation to disk in the directories created during initialisation.
        """

        # compute final ee50d
        for sci in range(self.config.sim.nSci):
            pxscale = self.sciCams[sci].fov / self.sciCams[sci].nx_pixels
            ee50d = aotools.encircled_energy(
                self.sciImgs[sci], fraction=0.5, eeDiameter=True) * pxscale
            if ee50d < (self.sciCams[sci].fov / 2):
                self.ee50d[sci] = ee50d
            else:
                logger.info(("\nEE50d computation invalid "
                             "due to small FoV of Science Camera {}\n").
                            format(sci))
                self.ee50d[sci] = None

        if self.config.sim.simName!=None:

            if self.config.sim.saveSlopes:
                fits.writeto(
                        self.path+"/slopes.fits", self.allSlopes,
                        header=self.config.sim.saveHeader, overwrite=True)

            if self.config.sim.saveDmCommands:
                fits.writeto(
                        self.path+"/dmCommands.fits",
                        self.allDmCommands, header=self.config.sim.saveHeader,
                        overwrite=True)

            if self.config.sim.saveLgsPsf:
                fits.writeto(
                        self.path+"/lgsPsf.fits", self.lgsPsfs,
                        header=self.config.sim.saveHeader, overwrite=True)

            if self.config.sim.saveWfe:
                fits.writeto(
                        self.path+"/WFE.fits", self.WFE,
                        header=self.config.sim.saveHeader, overwrite=True)

            if self.config.sim.saveStrehl:
                fits.writeto(
                        self.path+"/instStrehl.fits", self.instStrehl,
                        header=self.config.sim.saveHeader, overwrite=True)
                fits.writeto(
                        self.path+"/longStrehl.fits", self.longStrehl,
                        header=self.config.sim.saveHeader, overwrite=True)

            if self.config.sim.saveSciRes:
                for i in xrange(self.config.sim.nSci):
                    fits.writeto(self.path+"/sciResidual_%02d.fits"%i,
                                self.sciPhase[i],
                                header=self.config.sim.saveHeader,
                                overwrite=True)

            if self.config.sim.saveSciPsf:
                for i in xrange(self.config.sim.nSci):
                    fits.writeto(self.path+"/sciPsf_%02d.fits"%i,
                                        self.sciImgs[i],
                                        header=self.config.sim.saveHeader,
                                        overwrite=True )

            if self.config.sim.saveInstPsf:
                for i in xrange(self.config.sim.nSci):
                    fits.writeto(self.path+"/sciPsfInst_%02d.fits"%i,
                                 self.sciImgsInst[i],
                                 header=self.config.sim.saveHeader,
                                 overwrite=True )

            if self.config.sim.saveInstScieField:
                for i in xrange(self.config.sim.nSci):
                    fits.writeto(self.path+"/scieFieldInst_%02d_real.fits"%i,
                                 self.scieFieldInst[i].real,
                                 header=self.config.sim.saveHeader,
                                 overwrite=True )

            if self.config.sim.saveInstScieField:
                for i in xrange(self.config.sim.nSci):
                    fits.writeto(self.path+"/scieFieldInst_%02d_imag.fits"%i,
                                 self.scieFieldInst[i].imag,
                                 header=self.config.sim.saveHeader,
                                 overwrite=True )

            if self.config.sim.saveCalib:
                shutil.copy(self.config.sim.simName + '/cMat.fits',
                            self.path + "/cMat.fits")
                shutil.copy(self.config.sim.simName + '/iMat.fits',
                            self.path + "/iMat.fits")

            # Creating cubes with the WfsFrames
            if self.config.sim.saveWfsFrames:
                for nwfs in xrange(self.config.sim.nGS):
                    i = 0
                    wfs_first = fits.getdata(
                        self.path + "/wfsFPFrames/wfs-%d_frame-%d.fits" % (nwfs, i))
                    wfs_cube = numpy.zeros((self.iters, wfs_first.shape[0],
                                            wfs_first.shape[1]))
                    wfs_cube[0, :, :] = wfs_first
                    for i in range(1, self.iters):
                        wfs_cube[i, :, :] = fits.getdata(
                            self.path + "/wfsFPFrames/wfs-%d_frame-%d.fits" % (nwfs, i))

                    fits.writeto(self.path + "/wfs_frames_%02d.fits" % (nwfs),
                                 wfs_cube,
                                 header=self.config.sim.saveHeader,
                                 overwrite = True)

    def makeSaveHeader(self):
        """
        Forms a header which can be used to give a header to FITS files saved by the simulation.
        """

        header = fits.Header()
        self.timeStamp = self.getTimeStamp()

        

        # Sim Params
        header["INSTRUME"] = "SOAPY"
        header["SVER"] = __version__
        header["RTCNAME"] = "SOAPY"
        header["RTCVER"] = __version__
        header["TELESCOP"] = "SOAPY"
        header["RUNID"] = self.config.sim.simName
        header["LOOP"] = True
        header["DATE-OBS"] = self.time.strftime("%Y-%m-%dT%H:%M:%S")

        # Tel Params
        header["TELDIAM"] = self.config.tel.telDiam
        header["TELOBS"] = self.config.tel.obsDiam
        header["FR"] = 1./self.config.sim.loopTime

        # DM Params
        header["NBDM"] = self.config.sim.nDM
        header["DMNACTU"] = self.config.sim.totalActs

        dmActs = []
        dmConds = []
        dmTypes = []
        dmGain = []
        for dm in xrange(self.config.sim.nDM):
            dmActs.append(self.dms[dm].dmConfig.nxActuators)
            dmConds.append(self.dms[dm].dmConfig.svdConditioning)
            dmTypes.append(self.dms[dm].dmConfig.type)
            dmGain.append(self.dms[dm].dmConfig.gain)
        header["DMACTS"] = "{}".format(list(dmActs))
        header["DMCOND"] = "{}".format(list(dmConds))
        header["DMTYPE"] = "{}".format(list(dmTypes))
        header["DMGAIN"] = "{}".format(list(dmGain))

        # Atmos Params
        header["NBSCRNS"] = self.config.atmos.scrnNo
        header["SCRNALT"] = str(list(self.config.atmos.scrnHeights))
        header["WINDSPD"] = str(list(self.config.atmos.windSpeeds))

        # WFS Params
        header["NBWFS"] = self.config.sim.nGS
        header["NSUB"] = int(self.config.sim.totalWfsData/2.)
        header["NSLOP"] = self.config.sim.totalWfsData
        wfsPosX = []
        wfsPosY = []
        wfsSubX = []
        wfsSubY = []
        for w in range(self.config.sim.nGS):
            wfsPosX.append(self.config.wfss[w].GSPosition[0])
            wfsPosY.append(self.config.wfss[w].GSPosition[1])
            wfsSubX.append(self.config.wfss[w].nxSubaps)
            wfsSubY.append(self.config.wfss[w].nxSubaps)

            header["PIXARC{:d}".format(w)] = self.config.wfss[w].subapFOV/self.config.wfss[w].pxlsPerSubap

        header["WFSPOSX"] = str(wfsPosX)
        header["WFSPOSY"] = str(wfsPosY)
        header["WFSSUBX"] = str(wfsSubX)
        header["WFSSUBY"] = str(wfsSubY)
        header["NFRAMES"] = self.config.sim.nIters

        return header


    def getTimeStamp(self):
        """
        Returns a formatted timestamp

        Returns:
            string: nicely formatted timestamp of current time.
        """

        self.time = datetime.datetime.now()
        return self.time.strftime("%Y-%m-%d-%H-%M-%S")


    def printOutput(self, iter, strehl=False):
        """
        Prints simulation information  to the console

        Called on each iteration to print information about the current simulation, such as current strehl ratio, to the console. Still under development
        Args:
            label(str): Simulation Name
            iter(int): simulation frame number
            strehl(float, optional): current strehl ration if science cameras are present to record it.
        """
        if self.config.sim.simName:
            string = self.config.sim.simName.split("/")[-1]
        else:
            string = self.config.filename.split("/")[-1].split(".")[0]

        if strehl:
            string += "  Strehl -- "
            for sci in xrange(self.config.sim.nSci):
                string += "sci_{0}: inst {1:.2f}, long {2:.2f} ".format(
                        sci, self.sciCams[sci].instStrehl,
                        self.sciCams[sci].longExpStrehl)

        logger.statusMessage(iter, self.config.sim.nIters, string )


    def addToGuiQueue(self):
        """
        Adds data to a Queue object provided by the soapy GUI.

        The soapy GUI doesn't need to plot every frame from the simulation. 
        When it wants a frame, it will request if by setting 
        ``waitingPlot = True``. As this function is called on
        every iteration, data is passed to the GUI only if 
        ``waitingPlot = True``. 
        This allows efficient and abstracted interaction 
        between the GUI and the simulation
        """
        if self.guiQueue != None:
            if self.waitingPlot:
                guiPut = []
                wfsFocalPlane = {}
                wfsPhase = {}
                lgsPsf = {}
                for i in xrange(self.config.sim.nGS):
                    wfsFocalPlane[i] = self.wfss[i].wfsDetectorPlane.copy().astype("float32")
                    try:
                        wfsPhase[i] = self.wfss[i].uncorrectedPhase
                    except AttributeError:
                        wfsPhase[i] = None
                        pass

                    try:
                        lgsPsf[i] = self.wfss[i].lgs.psf.copy()
                    except AttributeError:
                        lgsPsf[i] = None
                        pass


                dmShape = {}
                for i in xrange(self.config.sim.nDM):
                    try:
                        dmShape[i] = self.dms[i].dm_shape.copy()#*self.mask
                    except AttributeError:
                        dmShape[i] = None

                sciImg = {}
                residual = {}
                instSciImg = {}
                for i in xrange(self.config.sim.nSci):
                    try:
                        sciImg[i] = self.sciImgs[i].copy()
                    except AttributeError:
                        sciImg[i] = None
                    try:
                        instSciImg[i] = self.sciCams[i].detector.copy()
                    except AttributeError:
                        instSciImg[i] = None

                    try:
                        residual[i] = self.sciCams[i].residual
                    except AttributeError:
                        residual[i] = None

                guiPut = {  "wfsFocalPlane":wfsFocalPlane,
                            "wfsPhase":     wfsPhase,
                            "lgsPsf":       lgsPsf,
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


def make_mask(config):
    """
    Generates a Soapy pupil mask

    Parameters:
        config (SoapyConfig): Config object describing Soapy simulation

    Returns:
        ndarray: 2-d pupil mask
    """
    if config.tel.mask == "circle":
        mask = aotools.circle(config.sim.pupilSize / 2.,
                                  config.sim.simSize)
        if config.tel.obsDiam != None:
            mask -= aotools.circle(
                config.tel.obsDiam * config.sim.pxlScale / 2.,
                config.sim.simSize
            )

    elif isinstance(config.tel.mask, str):
        maskHDUList = fits.open(config.tel.mask)
        mask = maskHDUList[0].data.copy()
        maskHDUList.close()
        logger.info('load mask "{}", of size: {}'.format(config.tel.mask, mask.shape))

        if not numpy.array_equal(mask.shape, (config.sim.pupilSize,) * 2):
            # interpolate mask to pupilSize if not that size already
            mask = numpy.round(interp.zoom(mask, config.sim.pupilSize))

    else:
        mask = config.tel.mask.copy()

    # Check its size is compatible. If its the pupil size, pad to sim size
    if (not numpy.array_equal(mask.shape, (config.sim.pupilSize,)*2)
            and not numpy.array_equal(mask.shape, (config.sim.simSize,)*2) ):
        raise ValueError("Mask Shape {} not compatible. Should be either `pupilSize` or `simSize`".format(mask.shape))

    if mask.shape != (config.sim.simSize, )*2:
        mask = numpy.pad(
                mask, config.sim.simPad, mode="constant")

    return mask


# Functions used by MP stuff
def multiWfs(scrns, wfsObj, dmShape, read, queue):
    """
    Function to run the WFS in multiprocessing mode.

    Function is called by each of the new WFS processes spawned to run each WFS. Does the same job as the sim runWfs_noMP method of running LGS, then getting slopes from each WFS.

    Args:
        scrns (list): list of the phase screens over the WFS
        wfsObj (WFS object): the WFS object being run
        dmShape (ndArray):  shape of system DMs for WFS phase correction
        queue (Queue object): a multiprocessing Queue object used to pass data back to host process.
    """

    slopes = wfsObj.frame(scrns, dmShape, read=read)

    if wfsObj.LGS:
        lgsPsf = wfsObj.LGS.psf1
    else:
        lgsPsf = None

    res = [slopes, wfsObj.wfsDetectorPlane, wfsObj.uncorrectedPhase, lgsPsf]

    queue.put(res)


#######################
#Control Functions
######################
class DelayBuffer(list):
    '''
    A delay buffer.

    Each time delay() is called on the buffer, the input value is stored.
    If the buffer is larger than count, the oldest value is removed and returned.
    If the buffer is not yet full, a zero of similar shape as the last input
    is returned.
    '''

    def delay(self, value, count):
        self.append(value)
        if len(self) <= count:
            result = value*0.0
        else:
            for _ in range(len(self)-count):
                result = self.pop(0)
        return result


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
