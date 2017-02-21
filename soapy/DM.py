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
The module simulating Deformable Mirrors in Soapy

DMs in Soapy
============
DMs are represented in Soapy by python objects which are initialised at startup
with some configuration parameters given, as well as a list of one or more
WFS objects which can be used to measure an interaction matrix.

Upon creation of an interaction matrix, the object first generations all the
possible independant shapes which the DM may form, known as "influence functions".
Then each influence function is passed to the specified WFS(s) and the response
noted to form an interaction matrix. The interaction matrix may then be used
to forma reconstructor.

During the AO loop, commands corresponding to the required amplitude of each
DM influence function are sent to the :py:meth:`DM.dmFrame` method, which
returns an array representing the DMs shape.

Adding New DMs
==============

New DMs are easy to add into the simulation. At its simplest, the :py:class:`DM`
class is inherited by the new DM class. Only a ``makeIMatShapes` method need be provided, 
which creates the independent influence function the DM can make. The
base class deals with the rest, including making interaction matrices and loop
operation.
"""
import numpy
from scipy.ndimage.interpolation import rotate

from . import logger
from .aotools import interp, circle

try:
    xrange
except NameError:
    xrange = range

class DM(object):
    """
    The base DM class

    This class is intended to be inherited by other DM classes which describe
    real DMs. It provides methods to create DM shapes and then interaction matrices, 
    given a specific WFS or WFSs.

    Parameters:
        soapyConfig (ConfigObj): The soapy configuration object
        nDm (int): The ID number of this DM
        wfss (list, optional): A list of Soapy WFS object with which to record the interaction matrix
        mask (ndarray, optional): An array or size (simConfig.simSize, simConfig.simSize) which is 1 at the telescope aperture and 0 else-where. If None then a circle is generated.
    """

    def __init__ (self, soapyConfig, n_dm=0, wfss=None, mask=None):
        
        self.soapyConfig = soapyConfig
        self.simConfig = soapyConfig.sim
        self.dmConfig = soapyConfig.dms[n_dm]
        self.wfss = wfss
        self.wvl = wfss[0].wavelength
        self.n_dm = n_dm

        # If supplied use the mask
        if numpy.any(mask):
            self.mask = mask
        # Else we'll just make a circle
        else:
            self.mask = circle.circle(
                    self.simConfig.pupilSize/2., self.simConfig.simSize,
                    )

        self.n_acts = self.getActiveActs()
        self.actCoeffs = numpy.zeros( (self.n_acts) )

        # Sort out which WFS(s) observes the DM (for iMat making)
        if self.dmConfig.wfs!=None:
            try:
                # Make sure the specifed WFS actually exists
                self.wfss = [wfss[self.dmConfig.wfs]]
                self.wfs = self.wfss[0]
            except KeyError:
                raise KeyError("DM attached to WFS {}, but that WFS is not specifed in config".format(self.dmConfig.wfs))
        else:
            self.wfss = wfss

        # find the total number of WFS subaps, and make imat
        # placeholder
        self.totalWfsMeasurements = 0
        for nWfs in range(len(self.wfss)):
            self.totalWfsMeasurements += self.wfss[nWfs].n_measurements

        logger.info("Making DM Influence Functions...")
        self.makeIMatShapes()
        if self.dmConfig.rotation:
            self.iMatShapes = rotate(
                self.iMatShapes, self.dmConfig.rotation,
                order=self.dmConfig.interpOrder, axes=(-2, -1)
            )
            rotShape = self.iMatShapes.shape
            self.iMatShapes = self.iMatShapes[:,
                              rotShape[1] / 2. - self.simConfig.simSize / 2.:
                              rotShape[1] / 2. + self.simConfig.simSize / 2.,
                              rotShape[2] / 2. - self.simConfig.simSize / 2.:
                              rotShape[2] / 2. + self.simConfig.simSize / 2.
                              ]

    def getActiveActs(self):
        """
        Method returning the total number of actuators used by the DM - May be overwritten in DM classes

        Returns:
            int: number of active DM actuators
        """
        return self.dmConfig.nxActuators

    def makeIMat(self, callback=None):
        '''
        Makes DM Interation Matrix

        Initially, the DM influence functions are created using the method
        ``makeIMatShapes'', then if a rotation is specified these are rotated.
        Each of the influence functions is passed to the specified ``WFS'' and
        wfs measurements recorded.

        Parameters:
            callback (function): Function to be called on each WFS run

        Returns:
            ndarray: 2-dimensional interaction matrix
        '''


        # Imat value is in microns
        # self.iMatShapes *= (self.dmConfig.iMatValue)

        if self.dmConfig.rotation:
            self.iMatShapes = rotate(
                    self.iMatShapes, self.dmConfig.rotation,
                    order=self.dmConfig.interpOrder, axes=(-2, -1)
                    )
            rotShape = self.iMatShapes.shape
            self.iMatShapes = self.iMatShapes[:,
                    rotShape[1]/2. - self.simConfig.simSize/2.:
                    rotShape[1]/2. + self.simConfig.simSize/2.,
                    rotShape[2]/2. - self.simConfig.simSize/2.:
                    rotShape[2]/2. + self.simConfig.simSize/2.
                    ]

        iMat = numpy.zeros(
                (self.n_acts, self.totalWfsMeasurements))

        # A vector of DM commands to use when making the iMat
        actCommands = numpy.zeros(self.n_acts)
 
        # Blank phase to use whilst making iMat
        phs = numpy.zeros((self.simConfig.simSize, self.simConfig.simSize))
        for i in xrange(self.n_acts):
            subap = 0

            # Set vector of iMat commands to 0...
            actCommands[:] = 0
            # Except the one we want to make an iMat for!
            actCommands[i] = self.dmConfig.iMatValue
            # Now get a DM shape for that command
            self.dmShape = self.makeDMFrame(actCommands)
            for nWfs in range(len(self.wfss)):
                logger.debug("subap: {}".format(subap))

                # Send the DM shape off to the relavent WFS. put result in iMat
                wfs_phs = self.wfss[nWfs].los.frame(self.dmShape)

                iMat[i, subap: subap + (2*self.wfss[nWfs].n_subaps)] = (
                       -1*self.wfss[nWfs].frame(wfs_phs))/self.dmConfig.iMatValue

                if callback != None:
                    callback()

                logger.statusMessage(i, self.n_acts,
                        "Generating {} Actuator DM iMat".format(self.n_acts))

                subap += 2*self.wfss[nWfs].n_subaps

        self.iMat = iMat
        return iMat

    def dmFrame(self, dmCommands):
        '''
        Uses interaction matrix to calculate the final DM shape.

        Given the supplied DM commands, this method will apply a gain and add
        to the previous DM commands. This works differently for open or closed
        loop DMs. Multiplies each of the DM influence functions by the
        corresponding DM command, then sums to create the final DM shape.
        Lastly, the mean value is subtracted to avoid piston terms building up.

        Parameters:
            dmCommands (ndarray): A 1-dimensional vector of the multiplying factor of each DM influence function
            closed (bool, optional): Specifies how to great gain. If ``True'' (closed) then ``dmCommands'' are multiplied by gain and summed with previous commands. If ``False'' (open), then ``dmCommands'' are multiplied by gain, and summed withe previous commands multiplied by (1-gain).

        Returns:
            ndarray: A 2-d array with the DM shape
        '''
        # try:
        # self.newActCoeffs = dmCommands

        # If loop is closed, only add residual measurements onto old
        # actuator values
        # if closed:
        #     self.actCoeffs += self.dmConfig.gain*self.newActCoeffs
        #
        # else:
        #     self.actCoeffs = (self.dmConfig.gain * self.newActCoeffs)\
        #         + ( (1. - self.dmConfig.gain) * self.actCoeffs)

        self.dmShape = self.makeDMFrame(dmCommands)
        # Remove any piston term from DM
        self.dmShape -= self.dmShape.mean()

        return self.dmShape

    def makeDMFrame(self, actCoeffs):
        dmShape = (self.iMatShapes.T*actCoeffs.T).T.sum(0)
        
        # If DM not telescope diameter, must adjust pixel scale
        if self.dmConfig.diameter != self.soapyConfig.tel.telDiam:
            scaledDMSize = (dmShape.shape[0] 
                    * float(self.dmConfig.diameter)/self.soapyConfig.tel.telDiam)
            dmShape = interp.zoom(dmShape, scaledDMSize, order=1)
            
        # Turn into phase pbject with altitude
        dmShape = Phase(dmShape, altitude=self.dmConfig.altitude)
            
        return dmShape

    def reset(self):
        self.dmShape[:] = 0
        self.actCoeffs[:] = 0

class Zernike(DM):
    """
    A DM which corrects using a provided number of Zernike Polynomials
    """

    def makeIMatShapes(self):
        '''
        Creates all the DM shapes which are required for creating the
        interaction Matrix. In this case, this is a number of Zernike Polynomials
        '''

        shapes = circle.zernikeArray(
                int(self.n_acts + 1), int(self.simConfig.pupilSize))[1:]


        pad = self.simConfig.simPad
        self.iMatShapes = numpy.pad(
                shapes, ((0, 0), (pad, pad), (pad, pad)), mode="constant"
                ).astype("float32")

class Piezo(DM):
    """
    A DM emulating a Piezo actuator style stack-array DM.

    This class represents a standard stack-array style DM with push-pull actuators
    behind a continuous phase sheet. The number of actuators is given in the
    configuration file.

    Each influence function is created by started with an N x N grid of zeros,
    where N is the number of actuators in one direction, and setting a single
    value to ``1``, which corresponds with a "pushed" actuator. This grid is then
    interpolated up to the ``pupilSize``, to form the shape of the DM when that
    actuator is activated. This is repeated for all actuators.
    """

    def getActiveActs(self):
        """
        Finds the actuators which will affect phase whithin the pupil to avoid
        reconstructing for redundant actuators.
        """
        activeActs = []
        xActs = self.dmConfig.nxActuators
        self.spcing = self.simConfig.pupilSize/float(xActs)

        for x in xrange(xActs):
            for y in xrange(xActs):
                x1 = int(x*self.spcing+self.simConfig.simPad)
                x2 = int((x+1)*self.spcing+self.simConfig.simPad)
                y1 = int(y*self.spcing+self.simConfig.simPad)
                y2 = int((y+1)*self.spcing+self.simConfig.simPad)
                if self.mask[x1: x2, y1: y2].sum() > 0:
                    activeActs.append([x,y])
        self.activeActs = numpy.array(activeActs)
        self.xActs = xActs
        return self.activeActs.shape[0]


    def makeIMatShapes(self):
        """
        Generate Piezo DM influence functions

        Generates the shape of each actuator on a Piezo stack DM
        (influence functions). These are created by interpolating a grid
        on the size of the number of actuators, with only the 'poked'
        actuator set to 1 and all others set to zero, up to the required
        simulation size. This grid is actually padded with 1 extra actuator
        spacing to avoid strange edge effects.
        """

        #Create a "dmSize" - the pupilSize but with 1 extra actuator on each
        #side
        dmSize =  int(self.simConfig.pupilSize + 2 * numpy.round(self.spcing))

        shapes = numpy.zeros((int(self.n_acts), dmSize, dmSize), dtype="float32")

        for i in xrange(self.n_acts):
            x,y = self.activeActs[i]

            #Add one to avoid the outer padding
            x+=1
            y+=1

            shape = numpy.zeros( (self.xActs+2,self.xActs+2) )
            shape[x,y] = 1

            #Interpolate up to the padded DM size
            shapes[i] = interp.zoom_rbs(shape,
                    (dmSize, dmSize), order=self.dmConfig.interpOrder)

            shapes[i] -= shapes[i].mean()


        if dmSize>self.simConfig.simSize:
            coord = int(round(dmSize/2. - self.simConfig.simSize/2.))
            self.iMatShapes = shapes[:,coord:-coord, coord:-coord].astype("float32")

        else:
            pad = int(round((self.simConfig.simSize - dmSize)/2))
            self.iMatShapes = numpy.pad(
                    shapes, ((0,0), (pad,pad), (pad,pad)), mode="constant"
                    ).astype("float32")

class GaussStack(Piezo):
    """
    A Stack Array DM where each influence function is a 2-D Gaussian shape.

    This class represents a Stack-Array DM, similar to the :py:class:`Piezo` DM,
    where each influence function is a 2-dimensional Gaussian function. Though
    not realistic, it provides a known influence function which can be useful
    for some analysis.
    """
    def makeIMatShapes(self):
        """
        Generates the influence functions for the GaussStack DM.

        Creates a number of Guassian distributions which are centred at points
        across the pupil to act as DM influence functions. The width of the
        guassian is determined from the configuration file.
        """
        shapes = numpy.zeros((
                self.n_acts, self.simConfig.pupilSize, self.simConfig.pupilSize))

        actSpacing = self.simConfig.pupilSize/(self.dmConfig.nxActuators-1)
        width = actSpacing/2.

        for i in xrange(self.n_acts):
            x,y = self.activeActs[i]*actSpacing
            shapes[i] = circle.gaussian2d(
                    self.simConfig.pupilSize, width, cent = (x,y))

        self.iMatShapes = shapes

        pad = self.simConfig.simPad
        self.iMatShapes = numpy.pad(
                self.iMatShapes, ((0,0), (pad,pad), (pad,pad)), mode="constant"
                )

class TT(DM):
    """
    A class representing a tip-tilt mirror.

    This can be used as a tip-tilt mirror, it features two actuators, where each
    influence function is simply a tip and a tilt.

    """

    def getActiveActs(self):
        """
        Returns the number of active actuators on the DM. Always 2 for a TT.
        """
        return 2


    def makeIMatShapes(self):
        """
        Forms the DM influence functions, in this case just a tip and a tilt.
        """
        # Make the TT across the entire sim shape, but want it 1 to -1 across
        # pupil
        padMax = float(self.simConfig.simSize)/self.simConfig.pupilSize

        coords = numpy.linspace(
                    -padMax, padMax, self.simConfig.simSize)
        self.iMatShapes = numpy.array(numpy.meshgrid(coords,coords))


class FastPiezo(Piezo):
    """
    A DM which simulates a Piezo DM. Faster than standard for big simulations as interpolates on each frame.
    """

    def getActiveActs(self):
        acts = super(FastPiezo, self).getActiveActs()
        self.actGrid = numpy.zeros(
                (self.dmConfig.nxActuators, self.dmConfig.nxActuators))

        # DM size is the pupil size, but withe one extra act on each side
        self.dmSize =  self.simConfig.pupilSize + 2 * numpy.round(self.spcing)

        return acts

    def makeDMFrame(self, actCoeffs):

        self.actGrid[(self.activeActs[:,0], self.activeActs[:,1])] = actCoeffs

        # Add space around edge for 1 extra act to avoid edge effects
        actGrid = numpy.pad(self.actGrid, ((1,1), (1,1)), mode="constant")

        # Interpolate to previously determined "dmSize"
        dmShape = interp.zoom_rbs(
                actGrid, self.dmSize, order=self.dmConfig.interpOrder)


        # Now check if "dmSize" bigger or smaller than "simSize".
        # Crop or pad as appropriate
        if self.dmSize>self.simConfig.simSize:
            coord = int(round(self.dmSize/2. - self.simConfig.simSize/2.))
            self.dmShape = dmShape[coord:-coord, coord:-coord].astype("float32")

        else:
            pad = int(round((self.simConfig.simSize - self.dmSize)/2))
            self.dmShape = numpy.pad(
                    dmShape, ((pad,pad), (pad,pad)), mode="constant"
                    ).astype("float32")

        return self.dmShape

class Phase(numpy.ndarray):
    def __new__(cls, input_array, altitude=0):
        obj = numpy.asarray(input_array).view(cls)
        obj.altitude = altitude
        return obj
        
    def __array_finalize__(self, obj):
        if obj is None: return
        self.info = getattr(obj, 'info', None)
 