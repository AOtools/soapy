"""
Config File for python tomographic AO simulation.

This configuration file contains a python dictionary in which all simulation parameters are saved. The main dictionary ``simConfig``, is split into several sections corresponding to different parts of the simulation, each of which is itself another python dictionary

This file will be parsed and missing parameters will trigger warnings, either defaulting to a reasonable value or exiting the simulation. Python syntax errors in this file can stop the simulation from running, ensure that every pararmeter finishes with a comma, including the sub-dictionaries.
"""
import numpy

simConfiguration = {

"Sim":{
    "filePrefix"    :  "sh_8x8",
    "pupilSize"     :   128, 
    "nGS"           :   1,
    "nDM"           :   1,
    "nSci"          :   1,
    "nIters"        :   1000,
    "loopTime"      :   1/250.0,
    "gain"          :   0.6,
    "aoloopMode"    :   "closed", 
    "reconstructor" :   "MVM", 

    "verbosity"     :   2,

    "saveCMat"      :   False,
    "saveSlopes"    :   True,
    "saveDmCommands":   False,
    "saveLgsPsf"    :   False,
    "saveSciPsf"    :   True,
    },

"Atmosphere":{
    "scrnNo"        :  4,
    "scrnHeights"   :   numpy.array([0,5000,10000,25000]),
    "scrnStrengths" :   numpy.array([0.5,0.3,0.1,0.1]),
    "windDirs"      :   numpy.array([0,45,90,135]),
    "windSpeeds"    :   numpy.array([10,10,15,20]),
    "newScreens"    :   True, #Bool
    "wholeScrnSize" :   1024,
    "r0"            :   0.16,
    },

"Telescope":{
   "telDiam"        :   4.2,  #Metres
   "obs"            :   1.2, #Central Obscuration
   "mask"           :   "circle",
    },

"WFS":{
    "GSPosition"    :   [(0,0)],
    "GSHeight"      :   [0],
    "subaps"        :   [8],
    "pxlsPerSubap"  :   [10],
    "subapFOV"      :   [4.0],
    "subapOversamp" :   [2],
    "wavelength"    :   [600e-9],
    "bitDepth"      :   [8],
    "lgs"           :   [False],
    },

"LGS":{

    },

"DM":{

    "dmType"        :   [ "peizo"],
    "dmActs"        :   [9**2],
    "dmCond"        :   [0.05],
    },

"Science":{
    "position"      :   [(0,0)],
    "FOV"           :   [3.0],
    "wavelength"    :   [1.65e-6],
    "pxls"          :   [128],
    "oversamp"      :   [4],
    }


}



