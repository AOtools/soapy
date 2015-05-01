"""
Config File for python tomographic AO simulation.

This configuration file contains a python dictionary in which all simulation parameters are saved. The main dictionary ``simConfig``, is split into several sections corresponding to different parts of the simulation, each of which is itself another python dictionary

This file will be parsed and missing parameters will trigger warnings, either defaulting to a reasonable value or exiting the simulation. Python syntax errors in this file can stop the simulation from running, ensure that every pararmeter finishes with a comma, including the sub-dictionaries.
"""
import numpy

simConfiguration = {

"Sim":{
    "simName"       :  "sh_8x8_lgs",
    "logfile"       :   "sh_8x8_lgs.log",
    "pupilSize"     :   128, 
    "nGS"           :   1,
    "nDM"           :   2,
    "nSci"          :   1,
    "nIters"        :   1000,
    "loopTime"      :   1/250.0,
    "gain"          :   0.6,
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
    "scrnHeights"   :   numpy.array([0,5000,10000,15000]),
    "scrnStrengths" :   numpy.array([0.5,0.3,0.1,0.1]),
    "windDirs"      :   numpy.array([0,45,90,135]),
    "windSpeeds"    :   numpy.array([10,10,15,20]),
    "newScreens"    :   True, #Bool
    "wholeScrnSize" :   1024,
    "r0"            :   0.16,
    },

"Telescope":{
   "telDiam"        :   8.,  #Metres
   "obsDiam"        :   1.2, #Central Obscuration
   "mask"           :   "circle",
    },

"WFS":{
    "GSPosition"    :   [(0,0)]*2,
    "GSHeight"      :   [90e3],
    "GSMag"         :   [8],
    "nxSubaps"      :   [8],
    "pxlsPerSubap"  :   [20],
    "subapFOV"      :   [6.0],
    "subapOversamp" :   [3],
    "wavelength"    :   [589e-9],
    "lgs"           :   [True],
    "centMethod"    :   ["brightestPxl"],
    "centThreshold" :   [0.2],
    },

"LGS":{
    "uplink"            :   [False],
    "height"            :   [90e3],
    "elongationDepth"   :  [8e3],
    "elongationLayers"  : [8],
    "launchPosition"    :   [(0,0)],
    },

"DM":{

    "type"              :   ["TT",     "Piezo"],
    "nxActuators"       :   [2,         9],
    "svdConditioning"   :   [1e-15,      0.05],
    "closed"            :   [False,      False],
    "gain"              :   [0.6,       0.6],
    "iMatValue"         :   [50,        10  ],
    },

"Science":{
    "position"      :   [(0,0)],
    "FOV"           :   [3.0],
    "wavelength"    :   [1.65e-6],
    "pxls"          :   [128],
    "fftOversamp"      :   [2],
    }


}



