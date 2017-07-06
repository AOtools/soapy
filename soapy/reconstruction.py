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

        self.soapy_config = soapy_config

        self.dms = dms
        self.wfss = wfss
        self.sim_config = soapy_config.sim
        self.atmos = atmos
        self.config = soapy_config.recon

        self.n_dms = soapy_config.sim.nDM
        self.scrn_size = soapy_config.sim.scrnSize

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

        n_acts = 0
        self.first_acts = []
        for i, dm in self.dms.items():
            self.first_acts.append(n_acts)
            n_acts += dm.n_acts

        n_wfs_measurements = 0
        self.first_measurements = []
        for i, wfs in self.wfss.items():
            self.first_measurements.append(n_wfs_measurements)
            n_wfs_measurements += wfs.n_measurements

        #2 functions used in case reconstructor requires more WFS data.
        #i.e. learn and apply
        self.runWfs = runWfsFunc
        if self.sim_config.learnAtmos == "random":
            self.moveScrns = atmos.randomScrns
        else:
            self.moveScrns = atmos.moveScrns
        self.wfss = wfss

        self.control_matrix = numpy.zeros(
            (self.sim_config.totalWfsData, self.sim_config.totalActs))
        self.controlShape = (
            self.sim_config.totalWfsData, self.sim_config.totalActs)

        self.Trecon = 0

        self.find_closed_actuators()

        self.actuator_values = None

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
                filename, self.control_matrix,
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
            self.control_matrix = cMat

    def save_interaction_matrix(self):
        """
        Writes the current control Matrix to FITS file
        """
        imat_filename = self.sim_config.simName+"/iMat.fits"

        fits.writeto(
                imat_filename, self.interaction_matrix,
                header=self.sim_config.saveHeader, overwrite=True)

        for i in range(self.n_dms):
            valid_acts_filename =  self.sim_config.simName+"/active_acts_dm{}.fits".format(i)
            valid_acts = self.dms[i].valid_actuators
            fits.writeto(valid_acts_filename, valid_acts, header=self.sim_config.saveHeader, overwrite=True)



    def load_interaction_matrix(self):

        filename = self.sim_config.simName+"/iMat.fits"

        imat_header = fits.getheader(filename)
        imat_data = fits.getdata(filename)

        # Check that imat shape is copatibile with curretn sim
        if imat_data.shape != (self.sim_config.totalActs, self.sim_config.totalWfsData):
            logger.warning(
                "interaction matrix does not match required required size."
            )
            raise IOError("interaction matrix does not match required required size.")

        # Load valid actuators
        for i in range(self.n_dms):
            valid_acts_filename =  self.sim_config.simName+"/active_acts_dm{}.fits".format(i)
            valid_acts = fits.getdata(valid_acts_filename)
            self.dms[i].valid_actuators = valid_acts


    # def loadIMat(self):
    #     acts = 0
    #     self.interaction_matrix = numpy.empty_like(self.control_matrix.T)
    #     for dm in xrange(self.sim_config.nDM):
    #         logger.statusMessage(
    #                 dm+1,  self.sim_config.nDM-1, "Load DM Interaction Matrix")
    #         filenameIMat = self.sim_config.simName+"/iMat_dm%d.fits" % dm
    #         filenameShapes = self.sim_config.simName+"/dmShapes_dm%d.fits" % dm
    #
    #         iMat = fits.open(filenameIMat)[0].data
    #
    #         # See if influence functions are also there...
    #         try:
    #             iMatShapes = fits.open(filenameShapes)[0].data
    #             # Check if loaded influence funcs are the right size
    #             if iMatShapes.shape[-1] != self.dms[dm].simConfig.simSize:
    #                 logger.warning(
    #                         "loaded DM shapes are not same size as current sim."
    #                         )
    #                 raise IOError
    #             self.dms[dm].iMatShapes = iMatShapes
    #
    #         # If not, assume doesn't need them.
    #         # May raise an error elsewhere though
    #         except IOError:
    #             logger.info("DM Influence functions not found. If the DM doesn't use them, this is ok. If not, set 'forceNew=True' when making IMat")
    #             pass
    #
    #
    #         if iMat.shape != (self.dms[dm].n_acts, self.dms[dm].totalWfsMeasurements):
    #             logger.warning(
    #                 "interaction matrix does not match required required size."
    #                 )
    #             raise IOError
    #
    #         else:
    #             self.dms[dm].iMat = iMat
    #             self.interaction_matrix[acts:acts+self.dms[dm].n_acts] = self.dms[dm].iMat
    #             acts += self.dms[dm].n_acts



    def makeIMat(self, callback=None):

        self.interaction_matrix = numpy.zeros((self.sim_config.totalActs, self.sim_config.totalWfsData))

        n_acts = 0
        dm_imats = []
        total_valid_actuators = 0
        for dm_n, dm in self.dms.items():
            logger.info("Creating Interaction Matrix for DM %d " % (dm_n))
            dm_imats.append(self.make_dm_iMat(dm, callback=callback))

            total_valid_actuators += dm_imats[dm_n].shape[0]

        self.interaction_matrix = numpy.zeros((total_valid_actuators, self.sim_config.totalWfsData))
        act_n = 0
        for imat in dm_imats:
            self.interaction_matrix[act_n: act_n + imat.shape[0]] = imat
            act_n += imat.shape[0]

    def make_dm_iMat(self, dm, callback=None):
        """
        Makes an interaction matrix for a given DM with a given WFS

        Parameters:
            dm (DM): The Soapy DM for which an interaction matri is required.
            wfs (WFS): The Soapy WFS for which an interaction matrix is required
            callback (func, optional): A function to be called each iteration accepting no arguments
        """

        iMat = numpy.zeros(
            (dm.n_acts, self.sim_config.totalWfsData))

        # A vector of DM commands to use when making the iMat
        actCommands = numpy.zeros(dm.n_acts)

        phase = numpy.zeros((self.n_dms, self.scrn_size, self.scrn_size))
        for i in range(dm.n_acts):
            # Set vector of iMat commands and phase to 0
            actCommands[:] = 0

            # Except the one we want to make an iMat for!
            actCommands[i] = 1 # dm.dmConfig.iMatValue

            # Now get a DM shape for that command
            phase[:] = 0
            phase[dm.n_dm] = dm.dmFrame(actCommands)
            # Send the DM shape off to the relavent WFS. put result in iMat
            n_wfs_measurments = 0
            for wfs_n, wfs in self.wfss.items():
                # turn off wfs noise if set
                if self.config.imat_noise is False:
                    wfs_pnoise = wfs.config.photonNoise
                    wfs.config.photonNoise = False
                    wfs_rnoise = wfs.config.eReadNoise
                    wfs.config.eReadNoise = 0

                iMat[i, n_wfs_measurments: n_wfs_measurments+wfs.n_measurements] = (
                        -1 * wfs.frame(None, phase_correction=phase))# / dm.dmConfig.iMatValue
                n_wfs_measurments += wfs.n_measurements

                # Turn noise back on again if it was turned off
                if self.config.imat_noise is False:
                    wfs.config.photonNoise = wfs_pnoise
                    wfs.config.eReadNoise = wfs_rnoise

            if callback != None:
                callback()

            logger.statusMessage(i+1, dm.n_acts,
                                 "Generating {} Actuator DM iMat".format(dm.n_acts))

        logger.info("Checking for redundant actuators...")
        # Now check tath each actuator actually does something on a WFS.
        # If an act has a <0.1% effect then it will be removed
        # NOTE: THIS SHOULD REALLY BE DONE ON A PER WFS BASIS
        valid_actuators = numpy.zeros((dm.n_acts), dtype="int")
        act_threshold = abs(iMat).max() * 0.001
        for i in range(dm.n_acts):
            if abs(iMat[i]).max() > act_threshold:
                valid_actuators[i] = 1
            else:
                valid_actuators[i] = 0

        dm.valid_actuators = valid_actuators
        n_valid_acts = valid_actuators.sum()
        logger.info("DM {} has {} valid actuators ({} dropped)".format(
                dm.n_dm, n_valid_acts, dm.n_acts - n_valid_acts))

        # Can now make a final interaction matrix with only valid entries
        valid_iMat = numpy.zeros((n_valid_acts, self.sim_config.totalWfsData))
        i_valid_act = 0
        for i in range(dm.n_acts):
            if valid_actuators[i]:
                valid_iMat[i_valid_act] = iMat[i]
                i_valid_act += 1

        return valid_iMat

    def get_dm_imat(self, dm_index, wfs_index):
        """
        Slices and returns the interaction matrix between a given wfs and dm from teh main interaction matrix

        Parameters:
            dm_index (int): Index of required DM
            wfs_index (int): Index of required WFS

        Return:
             ndarray: interaction matrix
        """

        act_n1 = self.first_acts[dm_index]
        act_n2 = act_n1 + self.dms[dm_index].n_acts

        wfs_n1 = self.wfss[wfs_index].config.dataStart
        wfs_n2 = wfs_n1 + self.wfss[wfs_index].n_measurements
        return self.interaction_matrix[act_n1: act_n2, wfs_n1: wfs_n2]


    def makeCMat(
            self, loadIMat=True, loadCMat=True, callback=None,
            progressCallback=None):

        if loadIMat:
            try:
                self.load_interaction_matrix()
                logger.info("Interaction Matrices loaded successfully")
            except:
                tc = traceback.format_exc()
                logger.info("Load Interaction Matrices failed with error: {} - will create new one...".format(tc))
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
            except:
                tc = traceback.format_exc()
                logger.warning("Load Command Matrix failed qith error: {} - will create new one...".format(tc))

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
        Also applies different control law if DM is in "closed" or "open" loop mode
        """
        # Loop through DMs and apply gain
        n_act1 = 0
        for dm_i, dm in self.dms.items():

            n_act2 = n_act1 + dm.n_valid_actuators
            # If loop is closed, only add residual measurements onto old
            # actuator values
            if dm.dmConfig.closed:
                self.actuator_values[n_act1: n_act2] += (dm.dmConfig.gain * self.new_actuator_values[n_act1: n_act2])

            else:
                self.actuator_values[n_act1: n_act2] = ((dm.dmConfig.gain * self.new_actuator_values[n_act1: n_act2])
                                + ( (1. - dm.dmConfig.gain) * self.actuator_values[n_act1: n_act2]) )

            n_act1 += dm.n_valid_actuators


    def reconstruct(self, wfs_measurements):
        t = time.time()

        if self.actuator_values is None:
            self.actuator_values = numpy.zeros((self.sim_config.totalActs))

        self.new_actuator_values = self.control_matrix.T.dot(wfs_measurements)

        self.apply_gain()

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

        logger.info("Invert iMat with conditioning: {:.4f}".format(
                self.config.svdConditioning))
        self.control_matrix = scipy.linalg.pinv(
                self.interaction_matrix, self.config.svdConditioning
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
        for dm_index, dm in self.dms.items():

            n_wfs_measurements = 0
            for wfs in dm.wfss:
                n_wfs_measurements += wfs.n_measurements

            dm_interaction_matrix = numpy.zeros((dm.n_acts, n_wfs_measurements))
            # Get interaction matrices from main matrix
            n_wfs_measurement = 0
            for wfs_index in [dm.dmConfig.wfs]:
                wfs = self.wfss[wfs_index]
                wfs_imat = self.get_dm_imat(dm_index, wfs_index)
                print("DM: {}, WFS: {}".format(dm_index, wfs_index))
                dm_interaction_matrix[:, n_wfs_measurement:n_wfs_measurement + wfs.n_measurements] = wfs_imat

            dm_control_matrx = numpy.linalg.pinv(dm_interaction_matrix, dm.dmConfig.svdConditioning)

            # now put carefully back into one control matrix
            for wfs_index in [dm.dmConfig.wfs]:
                wfs = self.wfss[wfs_index]
                self.control_matrix[
                        wfs.config.dataStart:
                                wfs.config.dataStart + wfs.n_measurements,
                        acts:acts+dm.n_acts] = dm_control_matrx

            acts += dm.n_acts


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

        fits.writeto(
                cMatFilename, self.controlMatrix,
                header=self.sim_config.saveHeader, overwrite=True
                )

        fits.writeto(
                tomoMatFilename, self.tomoRecon,
                header=self.sim_config.saveHeader, overwrite=True
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
                    overwrite=True )


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


        self.control_matrix[:, 0:self.dms[0].n_acts]
        acts = self.dms[0].n_acts
        for dm in range(1, self.sim_config.nDM):

            #This is the matrix which converts from Low order DM commands
            #to high order DM commands, via slopes
            lowToHighTransform = self.dms[dm-1].iMat.T.dot( dmCMats[dm-1] )

            highOrderCMat = dmCMats[dm].T.dot(
                    numpy.identity(self.sim_config.totalWfsData)-lowToHighTransform)

            dmCMats[dm] = highOrderCMat

            self.control_matrix[:, acts:acts + self.dms[dm].n_acts] = highOrderCMat.T
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
