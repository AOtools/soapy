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

import traceback
import time

import numpy
import scipy

from . import logger

# Use pyfits or astropy for fits file handling
try:
    from astropy.io import fits
except ImportError:
    try:
        import pyfits as fits
    except ImportError:
        raise ImportError("soapy requires either pyfits or astropy")

#xrange now just "range" in python3.
#Following code means fastest implementation used in 2 and 3
try:
    xrange
except NameError:
    xrange = range

class Reconstructor(object):
    """
    Reconstructor that will give DM commands required to correct an AO frame for a given set of WFS measurements
    """
    def __init__(self, soapy_config, dms, wfss, atmos, runWfsFunc=None):


        self.dms = dms
        self.wfss = wfss
        self.sim_config = soapy_config.sim
        self.atmos = atmos

        self.learnIters = self.sim_config.learnIters

        self.dmActs = []
        self.dmConds = []
        self.dmTypes = []
        for dm in xrange(self.sim_config.nDM):
            self.dmActs.append(self.dms[dm].dmConfig.nxActuators)
            self.dmConds.append(self.dms[dm].dmConfig.svdConditioning)
            self.dmTypes.append(self.dms[dm].dmConfig.type)

        self.dmConds = numpy.array(self.dmConds)
        self.dmActs = numpy.array(self.dmActs)

        #2 functions used in case reconstructor requires more WFS data.
        #i.e. learn and apply
        self.runWfs = runWfsFunc
        if self.sim_config.learnAtmos == "random":
            self.moveScrns = atmos.randomScrns
        else:
            self.moveScrns = atmos.moveScrns
        self.wfss = wfss

        self.controlMatrix = numpy.zeros(
            (self.sim_config.totalWfsData, self.sim_config.totalActs))
        self.controlShape = (
            self.sim_config.totalWfsData, self.sim_config.totalActs)

        self.actuator_values = numpy.zeros(self.sim_config.totalActs)

        self.Trecon = 0

        self.find_closed_actuators()

    def find_closed_actuators(self):
        self.closed_actuators = numpy.zeros(self.sim_config.totalActs)
        n_act = 0
        for i_dm, dm in self.dms.items():
            if dm.dmConfig.closed:
                self.closed_actuators[n_act: n_act + dm.n_acts] = 1
            n_act += dm.n_acts

    def saveCMat(self):
        """
        Writes the current control Matrix to FITS file
        """
        filename = self.sim_config.simName+"/cMat.fits"

        fits.writeto(
                filename, self.controlMatrix,
                header=self.sim_config.saveHeader, clobber=True)

    def loadCMat(self):
        """
        Loads a control matrix from file to the reconstructor

        Looks in the standard reconstructor directory for a control matrix and loads the file.
        Also looks at the FITS header and checks that the control matrix is compatible with the current simulation.
        """

        filename = self.sim_config.simName+"/cMat.fits"

        logger.statusMessage(
                    1,  1, "Load Command Matrix")

        cMatHDU = fits.open(filename)[0]
        cMatHDU.verify("fix")
        header = cMatHDU.header

        try:
            # dmActs = dmTypes = dmConds = None
            dmNo = int(header["NBDM"])
            exec("dmActs = numpy.array({})".format(
                    cMatHDU.header["DMACTS"]), globals())
            exec("dmTypes = %s" % header["DMTYPE"], globals())
            exec("dmConds = numpy.array({})".format(
                    cMatHDU.header["DMCOND"]), globals())

            if not numpy.allclose(dmConds, self.dmConds):
                raise IOError("DM conditioning Parameter changed - will make new control matrix")
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
            raise IOError
        else:
            self.controlMatrix = cMat

    def save_interaction_matrix(self):
        """
        Writes the current control Matrix to FITS file
        """
        filename = self.sim_config.simName+"/iMat.fits"

        fits.writeto(
                filename, self.interaction_matrix,
                header=self.sim_config.saveHeader, clobber=True)


    def loadIMat(self):
        acts = 0
        self.interaction_matrix = numpy.empty_like(self.controlMatrix.T)
        for dm in xrange(self.sim_config.nDM):
            logger.statusMessage(
                    dm+1,  self.sim_config.nDM-1, "Load DM Interaction Matrix")
            filenameIMat = self.sim_config.simName+"/iMat_dm%d.fits" % dm
            filenameShapes = self.sim_config.simName+"/dmShapes_dm%d.fits" % dm

            iMat = fits.open(filenameIMat)[0].data

            # See if influence functions are also there...
            try:
                iMatShapes = fits.open(filenameShapes)[0].data
                # Check if loaded influence funcs are the right size
                if iMatShapes.shape[-1] != self.dms[dm].simConfig.simSize:
                    logger.warning(
                            "loaded DM shapes are not same size as current sim."
                            )
                    raise IOError
                self.dms[dm].iMatShapes = iMatShapes

            # If not, assume doesn't need them.
            # May raise an error elsewhere though
            except IOError:
                logger.info("DM Influence functions not found. If the DM doesn't use them, this is ok. If not, set 'forceNew=True' when making IMat")
                pass


            if iMat.shape != (self.dms[dm].n_acts, self.dms[dm].totalWfsMeasurements):
                logger.warning(
                    "interaction matrix does not match required required size."
                    )
                raise IOError

            else:
                self.dms[dm].iMat = iMat
                self.interaction_matrix[acts:acts+self.dms[dm].n_acts] = self.dms[dm].iMat
                acts += self.dms[dm].n_acts



    def makeIMat(self, callback=None):

        self.interaction_matrix = numpy.zeros((self.sim_config.totalActs, self.sim_config.totalWfsData))

        n_acts = 0
        for dm_n, dm in self.dms.items():
            n_wfs_measurments = 0
            for wfs_n, wfs in self.wfss.items():
                logger.info("Creating Interaction Matrix beteen DM %d and WFS %d..." % (dm_n, wfs_n))
                self.interaction_matrix[
                        n_acts: n_acts+dm.n_acts, n_wfs_measurments: n_wfs_measurments+wfs.n_measurements
                        ] = self.make_dm_iMat(dm,  wfs, callback=callback)
                n_wfs_measurments += wfs.n_measurements

            n_acts += dm.n_acts

    def make_dm_iMat(self, dm, wfs, callback=None):
        """
        Makes an interaction matrix for a given DM with a given WFS

        Parameters:
            dm (DM): The Soapy DM for which an interaction matri is required.
            wfs (WFS): The Soapy WFS for which an interaction matrix is required
            callback (func, optional): A function to be called each iteration accepting no arguments
        """

        iMat = numpy.zeros(
            (dm.n_acts, wfs.n_measurements))

        # A vector of DM commands to use when making the iMat
        actCommands = numpy.zeros(dm.n_acts)

        for i in xrange(dm.n_acts):
            # Set vector of iMat commands and phase to 0
            actCommands[:] = 0

            # Except the one we want to make an iMat for!
            actCommands[i] = dm.dmConfig.iMatValue

            # Now get a DM shape for that command
            phase = dm.makeDMFrame(actCommands)
            # Send the DM shape off to the relavent WFS. put result in iMat
            iMat[i] = (
                    -1 * wfs.frame(None, phase_correction=phase)) / dm.dmConfig.iMatValue

            if callback != None:
                callback()

            logger.statusMessage(i+1, dm.n_acts,
                                 "Generating {} Actuator DM iMat".format(dm.n_acts))

        return iMat


    def makeCMat(
            self, loadIMat=True, loadCMat=True, callback=None,
            progressCallback=None):

        if loadIMat:
            try:
                self.loadIMat()
                logger.info("Interaction Matrices loaded successfully")
            except IOError:
                #traceback.print_exc()
                logger.info("Load Interaction Matrices failed - will create new one.")
                self.makeIMat(callback=callback)
                if self.sim_config.simName is not None:
                    self.save_interaction_matrix()
                logger.info("Interaction Matrices Done")

        else:
            self.makeIMat(callback=callback)
            if self.sim_config.simName is not None:
                    self.save_interaction_matrix()
            logger.info("Interaction Matrices Done")

        if loadCMat:
            try:
                self.loadCMat()
                logger.info("Command Matrix Loaded Successfully")
            except IOError:
                #traceback.print_exc()
                logger.warning("Load Command Matrix failed - will create new one")

                self.calcCMat(callback, progressCallback)
                if self.sim_config.simName is not None:
                    self.saveCMat()
                logger.info("Command Matrix Generated!")
        else:
            logger.info("Creating Command Matrix")
            self.calcCMat(callback, progressCallback)
            if self.sim_config.simName is not None:
                    self.saveCMat()
            logger.info("Command Matrix Generated!")

    def apply_gain(self):
        """
        Applies the gains set for each DM to the DM actuator commands.

        """
        self.gain = self.dms[0].dmConfig.gain

        # If loop is closed, only add residual measurements onto old
        # actuator values
        self.actuator_values += (self.dms[0].dmConfig.gain * self.new_actuator_values * self.closed_actuators)

        open_actuator_values = ((self.dms[0].dmConfig.gain * self.new_actuator_values)
                                + ( (1. - self.dms[0].dmConfig.gain) * self.actuator_values) * abs(self.closed_actuators-1))
        self.actuator_values = numpy.where(self.closed_actuators, self.actuator_values, open_actuator_values)

    def reconstruct(self, wfs_measurements):
        t = time.time()

        self.new_actuator_values = self.controlMatrix.T.dot(wfs_measurements)

        self.apply_gain()
        # self.actuator_values = self.new_actuator_values

        self.Trecon += time.time()-t
        return self.actuator_values

    def reset(self):
        self.actuator_values[:] = 0


class MVM(Reconstructor):
    """
    Re-constructor which combines all DM interaction matrices from all DMs and
    WFSs and inverts the resulting matrix to form a global interaction matrix.
    """

    def calcCMat(self, callback=None, progressCallback=None):
        '''
        Uses DM object makeIMat methods, then inverts each to create a
        control matrix
        '''

        logger.info("Invert iMat with cond: {}".format(
                self.dms[0].dmConfig.svdConditioning))
        self.controlMatrix[:] = scipy.linalg.pinv(
                self.interaction_matrix, self.dms[0].dmConfig.svdConditioning
                )


class MVM_SeparateDMs(Reconstructor):
    """
    Re-constructor which treats a each DM Separately.

    Similar to ``MVM`` re-constructor, except each DM has its own control matrix.
    Its is assumed that each DM is "associated" with a different WFS.
    """

    def calcCMat(self,callback=None, progressCallback=None):
        '''
        Uses DM object makeIMat methods, then inverts each to create a
        control matrix
        '''
        acts = 0
        for dm in xrange(self.sim_config.nDM):
            dmIMat = self.dms[dm].iMat
            #Treats each DM iMat seperately
            dmCMat = scipy.linalg.pinv(
                    dmIMat, self.dms[dm].dmConfig.svdConditioning
                    )
            # now put carefully back into one control matrix
            for w, wfs in enumerate(self.dms[dm].wfss):
                self.controlMatrix[
                        wfs.config.dataStart:
                                wfs.config.dataStart + 2*wfs.activeSubaps,
                        acts:acts+self.dms[dm].n_acts] = dmCMat

            acts += self.dms[dm].n_acts

    def reconstruct(self, slopes):
        """
        Returns DM commands given some slopes

        First, if there's a TT mirror, remove the TT from the TT WFS (the 1st
        WFS slopes) and get TT commands to send to the mirror. These slopes may
        then be used to reconstruct commands for others DMs, or this could be
        the responsibility of other WFSs depending on the config file.
        """

        if self.dms[0].dmConfig.type=="TT":
            ttMean = slopes[self.dms[0].wfs.config.dataStart:
                            (self.dms[0].wfs.activeSubaps*2
                                +self.dms[0].wfs.config.dataStart)
                            ].reshape(2,
                                self.dms[0].wfs.activeSubaps).mean(1)
            ttCommands = self.controlMatrix[:,:2].T.dot(slopes)
            slopes[
                    self.dms[0].wfs.config.dataStart:
                    (self.dms[0].wfs.config.dataStart
                        +self.dms[0].wfs.activeSubaps)] -= ttMean[0]
            slopes[
                    self.dms[0].wfs.config.dataStart
                        +self.dms[0].wfs.activeSubaps:
                    self.dms[0].wfs.config.dataStart
                        +2*self.dms[0].wfs.activeSubaps] -= ttMean[1]

            #get dm commands for the calculated on axis slopes
            dmCommands = self.controlMatrix[:,2:].T.dot(slopes)

            return numpy.append(ttCommands, dmCommands)



        #get dm commands for the calculated on axis slopes
        dmCommands = self.controlMatrix.T.dot(slopes)
        return dmCommands




class LearnAndApply(MVM):
    '''
    Class to perform a simply learn and apply algorithm, where
    "learn" slopes are recorded, and an interaction matrix between off-axis
    and on-axis WFS is computed from these slopes.

    Assumes that on-axis sensor is WFS 0
    '''

    def saveCMat(self):
        cMatFilename = self.sim_config.simName+"/cMat.fits"
        tomoMatFilename = self.sim_config.simName+"/tomoMat.fits"

        # cMatHDU = fits.PrimaryHDU(self.controlMatrix)
        # cMatHDU.header["DMNO"] = self.sim_config.nDM
        # cMatHDU.header["DMACTS"] = "%s"%list(self.dmActs)
        # cMatHDU.header["DMTYPE"]  = "%s"%list(self.dmTypes)
        # cMatHDU.header["DMCOND"]  = "%s"%list(self.dmConds)

        # tomoMatHDU = fits.PrimaryHDU(self.tomoRecon)

        # tomoMatHDU.writeto(tomoMatFilename, clobber=True)
        # cMatHDU.writeto(cMatFilename, clobber=True)
        # Commented 8/7/15 to add sim wide header. - apr

        fits.writeto(
                cMatFilename, self.controlMatrix,
                header=self.sim_config.saveHeader, clobber=True
                )

        fits.writeto(
                tomoMatFilename, self.tomoRecon,
                header=self.sim_config.saveHeader, clobber=True
                )

    def loadCMat(self):

        super(LearnAndApply, self).loadCMat()

        #Load tomo reconstructor
        tomoFilename = self.sim_config.simName+"/tomoMat.fits"
        tomoMat = fits.getdata(tomoFilename)

        #And check its the right size
        if tomoMat.shape != (
                2*self.wfss[0].activeSubaps,
                self.sim_config.totalWfsData - 2*self.wfss[0].activeSubaps):
            logger.warning("Loaded Tomo matrix not the expected shape - gonna make a new one..." )
            raise Exception
        else:
            self.tomoRecon = tomoMat


    def initControlMatrix(self):

        self.controlShape = (2*self.wfss[0].activeSubaps, self.sim_config.totalActs)
        self.controlMatrix = numpy.zeros( self.controlShape )


    def learn(self, callback=None, progressCallback=None):
        '''
        Takes "self.learnFrames" WFS frames, and computes the tomographic
        reconstructor for the system. This method uses the "truth" sensor, and
        assumes that this is WFS0
        '''

        self.learnSlopes = numpy.zeros( (self.learnIters,self.sim_config.totalWfsData) )
        for i in xrange(self.learnIters):
            self.learnIter=i

            scrns = self.moveScrns()
            self.learnSlopes[i] = self.runWfs(scrns)


            logger.statusMessage(i+1, self.learnIters, "Performing Learn")
            if callback!=None:
                callback()
            if progressCallback!=None:
               progressCallback("Performing Learn", i, self.learnIters )

        if self.sim_config.saveLearn:
            #FITS.Write(self.learnSlopes,self.sim_config.simName+"/learn.fits")
            fits.writeto(
                    self.sim_config.simName+"/learn.fits",
                    self.learnSlopes, header=self.sim_config.saveHeader,
                    clobber=True )


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
        iCoffoff = numpy.linalg.pinv(Coffoff, rcond=1e-8)

        self.tomoRecon = Conoff.dot(iCoffoff)
        logger.info("Done. \nCreating full reconstructor....")

        #Same code as in "MVM" class to create dm-slopes reconstructor.

        super(LearnAndApply, self).calcCMat(callback, progressCallback)

        #Dont make global reconstructor. Will reconstruct on-axis slopes, then
        #dmcommands explicitly
        #self.controlMatrix = (self.controlMatrix.T.dot(self.tomoRecon)).T
        logger.info("Done.")


    def reconstruct(self, slopes):
        """
        Determine DM commands using previously made
        reconstructor from slopes.
        Args:
            slopes (ndarray): array of slopes to reconstruct from
        Returns:
            ndarray: array of commands to be sent to DM
        """

        #Retreive pseudo on-axis slopes from tomo reconstructor
        slopes = self.tomoRecon.dot(slopes[2*self.wfss[0].activeSubaps:])

        if self.dms[0].dmConfig.type=="TT":
            ttMean = slopes.reshape(2, self.wfss[0].activeSubaps).mean(1)
            ttCommands = self.controlMatrix[:,:2].T.dot(slopes)
            slopes[:self.wfss[0].activeSubaps] -= ttMean[0]
            slopes[self.wfss[0].activeSubaps:] -= ttMean[1]

            #get dm commands for the calculated on axis slopes
            dmCommands = self.controlMatrix[:,2:].T.dot(slopes)

            return numpy.append(ttCommands, dmCommands)

        #get dm commands for the calculated on axis slopes
        dmCommands = super(LearnAndApply, self).reconstruct(slopes)
        #dmCommands = self.controlMatrix.T.dot(slopes)
        return dmCommands


class LearnAndApplyLTAO(LearnAndApply, MVM_SeparateDMs):
    '''
    Class to perform a simply learn and apply algorithm, where
    "learn" slopes are recorded, and an interaction matrix between off-axis
    and on-axis WFS is computed from these slopes.

    This is an ``
    Assumes that on-axis sensor is WFS 1
    '''

    def initControlMatrix(self):

        self.controlShape = (2*(self.wfss[0].activeSubaps+self.wfss[1].activeSubaps), self.sim_config.totalActs)
        self.controlMatrix = numpy.zeros( self.controlShape )


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
        Conoff = self.covMat[
                self.wfss[1].config.dataStart:
                        self.wfss[2].config.dataStart,
                self.wfss[2].config.dataStart:
                ]
        Coffoff = self.covMat[  self.wfss[2].config.dataStart:,
                                self.wfss[2].config.dataStart:    ]

        logger.info("Inverting offoff Covariance Matrix")
        iCoffoff = numpy.linalg.pinv(Coffoff)

        self.tomoRecon = Conoff.dot(iCoffoff)
        logger.info("Done. \nCreating full reconstructor....")

        #Same code as in "MVM" class to create dm-slopes reconstructor.

        MVM_SeparateDMs.calcCMat(self, callback, progressCallback)

        #Dont make global reconstructor. Will reconstruct on-axis slopes, then
        #dmcommands explicitly
        #self.controlMatrix = (self.controlMatrix.T.dot(self.tomoRecon)).T
        logger.info("Done.")

    def reconstruct(self, slopes):
        """
        Determine DM commands using previously made
        reconstructor from slopes.
        Args:
            slopes (ndarray): array of slopes to reconstruct from
        Returns:
            ndarray: array to comands to be sent to DM
        """

        #Retreive pseudo on-axis slopes from tomo reconstructor
        slopes_HO = self.tomoRecon.dot(
                slopes[self.wfss[2].config.dataStart:])

        # Probably should remove TT from these slopes?
        nSubaps = slopes_HO.shape[0]
        slopes_HO[:nSubaps] -= slopes_HO[:nSubaps].mean()
        slopes_HO[nSubaps:] -= slopes_HO[nSubaps:].mean()

        # Final slopes are TT slopes appended to the tomographic High order slopes
        onSlopes = numpy.append(
                slopes[:self.wfss[1].config.dataStart], slopes_HO)

        dmCommands = self.controlMatrix.T.dot(onSlopes)

        #
        # ttCommands = self.controlMatrix[
        #         :self.wfss[1].config.dataStart,:2].T.dot(slopes_TT)
        #
        # hoCommands = self.controlMatrix[
        #         self.wfss[1].config.dataStart:,2:].T.dot(slopes_HO)
        #
        # #if self.dms[0].dmConfig.type=="TT":
        #    ttMean = slopes.reshape(2, self.wfss[0].activeSubaps).mean(1)
        #    ttCommands = self.controlMatrix[:,:2].T.dot(slopes)
        #    slopes[:self.wfss[0].activeSubaps] -= ttMean[0]
        #    slopes[self.wfss[0].activeSubaps:] -= ttMean[1]

        #    #get dm commands for the calculated on axis slopes
        #    dmCommands = self.controlMatrix[:,2:].T.dot(slopes)

        #    return numpy.append(ttCommands, dmCommands)

        #get dm commands for the calculated on axis slopes

       # dmCommands = self.controlMatrix.T.dot(slopes)

        return dmCommands



#####################################
#Experimental....
#####################################
class GLAO_4LGS(MVM):
    """
    Reconstructor of LGS TT prediction algorithm.

    Uses one TT DM and a high order DM. The TT WFS controls the TT DM and
    the second WFS controls the high order DM. The TT WFS and DM are
    assumed to be the first in the system.
    """


    def initControlMatrix(self):

        self.controlShape = (2*self.wfss[0].activeSubaps+2*self.wfss[1].activeSubaps,
                             self.sim_config.totalActs)
        self.controlMatrix = numpy.zeros( self.controlShape )


    def reconstruct(self, slopes):
        """
        Determine DM commands using previously made
        reconstructor from slopes.
        Args:
            slopes (ndarray): array of slopes to reconstruct from
        Returns:
            ndarray: array to commands to be sent to DM
        """

        offSlopes = slopes[self.wfss[2].config.dataStart:]
        meanOffSlopes = offSlopes.reshape(4,self.wfss[2].activeSubaps*2).mean(0)

        meanOffSlopes = self.removeCommonTT(meanOffSlopes, [1])

        slopes = numpy.append(
                slopes[:self.wfss[1].config.dataStart], meanOffSlopes)

        return super(LgsTT, self).reconstruct(slopes)


    def removeCommonTT(self, slopes, wfsList):

        xSlopesShape = numpy.array(slopes.shape)
        xSlopesShape[-1] /= 2.
        xSlopes = numpy.zeros(xSlopesShape)
        ySlopes = numpy.zeros(xSlopesShape)

        for i in range(len(wfsList)):
            wfs = wfsList[i]
            wfsSubaps = self.wfss[wfs].activeSubaps
            xSlopes[..., i*wfsSubaps:(i+1)*wfsSubaps] = slopes[..., i*2*wfsSubaps:i*2*wfsSubaps+wfsSubaps]
            ySlopes[..., i*wfsSubaps:(i+1)*wfsSubaps] = slopes[..., i*2*wfsSubaps+wfsSubaps:i*2*wfsSubaps+2*wfsSubaps]

        xSlopes = (xSlopes.T - xSlopes.mean(-1)).T
        ySlopes = (ySlopes.T - ySlopes.mean(-1)).T

        for i in range(len(wfsList)):
            wfs = wfsList[i]
            wfsSubaps = self.wfss[wfs].activeSubaps

            slopes[..., i*2*wfsSubaps:i*2*wfsSubaps+wfsSubaps] = xSlopes[..., i*wfsSubaps:(i+1)*wfsSubaps]
            slopes[..., i*2*wfsSubaps+wfsSubaps:i*2*wfsSubaps+2*wfsSubaps] = ySlopes[..., i*wfsSubaps:(i+1)*wfsSubaps]

        return slopes

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
        Assumes that DM 0  is low order,
        and DM 1 is high order.
        '''

        if self.sim_config.nDM==1:
            logger.warning("Woofer Tweeter Reconstruction not valid for 1 dm.")
            return None
        acts = 0
        dmCMats = []
        for dm in xrange(self.sim_config.nDM):
            dmIMat = self.dms[dm].iMat

            logger.info("Invert DM {} IMat with conditioning:{}".format(dm,self.dms[dm].dmConfig.svdConditioning))
            if dmIMat.shape[0]==dmIMat.shape[1]:
                dmCMat = numpy.linalg.pinv(dmIMat)
            else:
                dmCMat = numpy.linalg.pinv(
                                    dmIMat, self.dms[dm].dmConfig.svdConditioning)

            #if dm != self.sim_config.nDM-1:
            #    self.controlMatrix[:,acts:acts+self.dms[dm].n_acts] = dmCMat
            #    acts+=self.dms[dm].n_acts

            dmCMats.append(dmCMat)


        self.controlMatrix[:, 0:self.dms[0].n_acts]
        acts = self.dms[0].n_acts
        for dm in range(1, self.sim_config.nDM):

            #This is the matrix which converts from Low order DM commands
            #to high order DM commands, via slopes
            lowToHighTransform = self.dms[dm-1].iMat.T.dot( dmCMats[dm-1] )

            highOrderCMat = dmCMats[dm].T.dot(
                    numpy.identity(self.sim_config.totalWfsData)-lowToHighTransform)

            dmCMats[dm] = highOrderCMat

            self.controlMatrix[:,acts:acts+self.dms[dm].n_acts] = highOrderCMat.T
            acts += self.dms[dm].n_acts


class LgsTT(LearnAndApply):
    """
    Reconstructor of LGS TT prediction algorithm.

    Uses one TT DM and a high order DM. The TT WFS controls the TT DM and
    the second WFS controls the high order DM. The TT WFS and DM are
    assumed to be the first in the system.
    """

    def initControlMatrix(self):

        self.controlShape = (2*self.wfss[0].activeSubaps+2*self.wfss[1].activeSubaps,
                             self.sim_config.totalActs)
        self.controlMatrix = numpy.zeros( self.controlShape )


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

        #Need to remove all *common* TT from off-axis learn slopes
        self.learnSlopes[:, 2*self.wfss[1].activeSubaps:] = self.removeCommonTT(
                self.learnSlopes[:, 2*self.wfss[1].activeSubaps:], [2,3,4,5])

        self.covMat = numpy.cov(self.learnSlopes.T)
        Conoff = self.covMat[   :2*self.wfss[1].activeSubaps,
                                2*self.wfss[1].activeSubaps:     ]
        Coffoff = self.covMat[  2*self.wfss[1].activeSubaps:,
                                2*self.wfss[1].activeSubaps:    ]

        logger.info("Inverting offoff Covariance Matrix")
        iCoffoff = numpy.linalg.pinv(Coffoff)

        self.tomoRecon = Conoff.dot(iCoffoff)
        logger.info("Done. \nCreating full reconstructor....")

        super(LgsTT, self).calcCMat(callback, progressCallback)


    def reconstruct(self, slopes):
        """
        Determine DM commands using previously made
        reconstructor from slopes.
        Args:
            slopes (ndarray): array of slopes to reconstruct from
        Returns:
            ndarray: array to commands to be sent to DM
        """

        #Get off axis slopes and remove *common* TT
        offSlopes = slopes[self.wfss[2].config.dataStart:]
        offSlopes = self.removeCommonTT(offSlopes,[2,3,4,5])

        #Use the tomo matrix to get pseudo on-axis slopes
        psuedoOnSlopes = self.tomoRecon.dot(offSlopes)

        #Combine on-axis slopes with TT measurements
        slopes = numpy.append(
                slopes[:self.wfss[1].config.dataStart], psuedoOnSlopes)

        #Send to command matrices to get dmCommands
        return super(LgsTT, self).reconstruct(slopes)


    def removeCommonTT(self, slopes, wfsList):

        xSlopesShape = numpy.array(slopes.shape)
        xSlopesShape[-1] /= 2.
        xSlopes = numpy.zeros(xSlopesShape)
        ySlopes = numpy.zeros(xSlopesShape)

        for i in range(len(wfsList)):
            wfs = wfsList[i]
            wfsSubaps = self.wfss[wfs].activeSubaps
            xSlopes[..., i*wfsSubaps:(i+1)*wfsSubaps] = slopes[..., i*2*wfsSubaps:i*2*wfsSubaps+wfsSubaps]
            ySlopes[..., i*wfsSubaps:(i+1)*wfsSubaps] = slopes[..., i*2*wfsSubaps+wfsSubaps:i*2*wfsSubaps+2*wfsSubaps]

        xSlopes = (xSlopes.T - xSlopes.mean(-1)).T
        ySlopes = (ySlopes.T - ySlopes.mean(-1)).T

        for i in range(len(wfsList)):
            wfs = wfsList[i]
            wfsSubaps = self.wfss[wfs].activeSubaps

            slopes[..., i*2*wfsSubaps:i*2*wfsSubaps+wfsSubaps] = xSlopes[..., i*wfsSubaps:(i+1)*wfsSubaps]
            slopes[..., i*2*wfsSubaps+wfsSubaps:i*2*wfsSubaps+2*wfsSubaps] = ySlopes[..., i*wfsSubaps:(i+1)*wfsSubaps]

        return slopes

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

        self.controlShape = (nSlopes, self.sim_config.totalActs)
        self.controlMatrix = numpy.zeros((nSlopes, self.sim_config.totalActs))
        acts = 0
        for dm in xrange(self.sim_config.nDM):
            dmIMat = self.dms[dm].iMat

            if dmIMat.shape[0]==dmIMat.shape[1]:
                dmCMat = numpy.inv(dmIMat)
            else:
                dmCMat = numpy.linalg.pinv(dmIMat, self.dmConds[dm])

            self.controlMatrix[:,acts:acts+self.dms[dm].n_acts] = dmCMat
            acts += self.dms[dm].n_acts

    def reconstruct(self, slopes):
        """
        Determine DM commands using previously made
        reconstructor from slopes. Uses Artificial Neural Network.

        Slopes are normalised before being run through the network.

        Args:
            slopes (ndarray): array of slopes to reconstruct from
        Returns:
            ndarray: array to comands to be sent to DM
        """
        t=time.time()
        offSlopes = slopes[self.wfss[0].activeSubaps*2:]/7 # normalise
        onSlopes = self.net.run(offSlopes)*7 # un-normalise
        dmCommands = self.controlMatrix.T.dot(onSlopes)

        self.Trecon += time.time()-t
        return dmCommands
