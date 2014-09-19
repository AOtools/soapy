"""
Config File for python tomographic AO simulation.

This configuration file contains a python dictionary in which all simulation parameters are saved. The main dictionary ``simConfig``, is split into several sections corresponding to different parts of the simulation, each of which is itself another python dictionary

This file will be parsed and missing parameters will trigger warnings, either defaulting to a reasonable value or exiting the simulation. Python syntax errors in this file can stop the simulation from running, ensure that every pararmeter finishes with a comma, including the sub-dictionaries.
"""
import numpy

simConfiguration = {

"Sim":{
    "filePrefix"    :  None,
    "pupilSize"     :   128, #this is the phase SAMPLING size
                            #(not number of detector pxls)
    "nGS"           :   1,
    "nDM"           :   1,
    "nSci"          :   1,

    "nIters"        :   1000,
    "loopTime"      :   1/250.0,
    "gain"          :   0.6,
    "aoloopMode"      :   "closed", #Can be "closedSCAO" or "openSCAO" so far...
    "reconstructor" :   "MVM", #Can be "MVM", "WooferTweeter, LearnAndApply1"
 
    "learnIters"    :   1000, #Only applicable if reconstructor is 

    "saveCMat"      :   False,
    "saveSlopes"    :   True,
    "saveDmCommands":   False,
    "saveLgsPsf"    :   False,
    "saveSciPsf"    :   True,

    "tipTilt"       :   False,
    "ttGain"        :   0.9,
    "wfsMP"            :   False,

    "verbosity"     :   3,
    },

"Atmosphere":{
    "scrnNo"        :  1,
    "scrnHeights"   :   numpy.array([0,5000,10000,25000]),
    "scrnStrengths" :   numpy.array([0.5,0.3,0.1,0.1]),
    "windDirs"      :   numpy.array([0,45,90,135]),
    "windSpeeds"    :   numpy.array([10,10,15,20]),
    "newScreens"    :   True, #Bool
    "wholeScrnSize" :   512,
    "r0"            :   0.15,
    },

"Telescope":{
   "telDiam"        :   4.2,  #Metres
   "obs"            :   1.2, #Central Obscuration
   "mask"           :   "circle",
    },

"WFS":{
    "GSPosition"   :   numpy.array([    [0,0]   ,
                                        [0,-20],
                                        [-20,0],
                                        [20,0],
                                        [0,20]
                                                ]),
    "GSHeight"      :   numpy.array([0 ]*5),
    "wfsPropMode"   :   ["geo"]*5,
    "subaps"        :   numpy.array([8]*5),

    "pxlsPerSubap"  :   numpy.array([14]*5),

    "subapFOV"      :   numpy.array([4.0]*5), #arcsecs

    "subapOversamp" :  numpy.array([4]*5),
    "wavelength"    :   numpy.array( [ 600e-9] *5),
    "subapThreshold":   [0.7,]*5,
    "bitDepth"      :   numpy.array([8]*5),
    "SNR"           :   numpy.array([0]*5),
    "removeTT"      :   numpy.array([0,0,0,0,0]),
    "angleEquivNoise":  numpy.array([0]*5), #arcsecs
    },

"LGS":{
    "lgsPupilSize"  :   0.5, #Metres
    "lgsUplink"     :   numpy.array([   1 ]*5),
    "wavelength"         :   numpy.array([   600e-9  ]*5),
    "propogationMode" :   ["physical"]*5,
    "height"     :   numpy.array([90000]*5),
    "elongationDepth" :   [0,]*5,
    "elongationLayers"   :   [10,]*5,
    },

"DM":{

    "dmType"        :   [ "peizo","peizo"], #peizo,TT or zernike so far...
    "dmActs"        :   numpy.array( [9**2,33**2]),
    "dmCond"        :   numpy.array( [0.05, 0.05] ),
    },

"Science":{
    "position"        :   numpy.array( [ [0,0], [0,0],[0,0],[0,0] ] ),
    "FOV"        :   numpy.array( [3.0]*4), #Arcsecs
    "wavelength"       :   numpy.array( [ 1500e-9, 800e-9, 1000e-9, 1500e-9]),
    "pxls"       :   numpy.array( [128]*4),
    "oversamp"   :   numpy.array( [4]*4 ),
    }


}



