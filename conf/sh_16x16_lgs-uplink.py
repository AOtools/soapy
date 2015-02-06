"""
Config File for python tomographic AO simulation.

This configuration file contains a python dictionary in which all simulation parameters are saved. The main dictionary ``simConfig``, is split into several sections corresponding to different parts of the simulation, each of which is itself another python dictionary

This file will be parsed and missing parameters will trigger warnings, either defaulting to a reasonable value or exiting the simulation. Python syntax errors in this file can stop the simulation from running, ensure that every pararmeter finishes with a comma, including the sub-dictionaries.
"""
import numpy

simConfiguration = {

"Sim":{
    "filePrefix"    :  "sh_8x8_lgs",
    "logfile"       :   "sh_8x8_lgs.log",
    "pupilSize"     :   128, 
    "nGS"           :   2,
    "nDM"           :   2,
    "nSci"          :   1,
    "nIters"        :   5000,
    "loopTime"      :   1/400.0,
    "gain"          :   0.6,
    "reconstructor" :   "MVM_SeperateDMs", 
    "wfsMP"         :   False,
    
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
    "wholeScrnSize" :   2048,
    "r0"            :   0.16,
    },

"Telescope":{
   "telDiam"        :   8.,  #Metres
   "obs"            :   1.1, #Central Obscuration
   "mask"           :   "circle",
    },

"WFS":{
    "GSPosition"    :   [(0,0),    (0,0)],
    "GSHeight"      :   [0,         90e3],
    "subaps"        :   [1,         16],
    "pxlsPerSubap"  :   [10,         14],
    "subapFOV"      :   [1.5,       3.0],
    "subapOversamp" :   [4,         4],
    "wavelength"    :   [600e-9]*2,
    "bitDepth"      :   [32]*2,
    "lgs"           :   [False,     True],
    "centMethod"    :   ["brightestPxl"]*2,
    "centThreshold" :   [0.2]*2,
    "exposureTime"  :   [None,      None],
    "removeTT"      :   [False,     True],
    },

"LGS":{
    "lgsUplink"     :   [True]*2,
    "lgsPupilDiam"  :   [0.3]*2,
    "wavelength"    :   [600e-9]*2,
    "propagationMode":  ["physical"]*2,
    "height"        :   [90e3]*2,
    "elongationDepth":  [0]*2,
    "elongationLayers": [5]*2,
    },

"DM":{
    "dmType"        :   ["TT",     "Piezo"],
    "dmActs"        :   [2,         17**2],
    "dmCond"        :   [1e-15,      0.09],
    "closed"        :   [True,      True],
    "gain"          :   [0.6,       0.6],
    "iMatValue"     :   [0.2,        0.2  ],
    "wfs"           :   [0,         1],
    },

"Science":{
    "position"      :   [(0,0)],
    "FOV"           :   [1.5],
    "wavelength"    :   [1.65e-6],
    "pxls"          :   [128],
    "oversamp"      :   [2],
    }
}



