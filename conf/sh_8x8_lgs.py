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
    "nGS"           :   2,
    "nDM"           :   2,
    "nSci"          :   1,
    "nIters"        :   5000,
    "loopTime"      :   1/400.0,
    "reconstructor" :   "MVM_SeparateDMs", 

    "verbosity"     :   2,

    "saveCMat"      :   False,
    "saveSlopes"    :   True,
    "saveDmCommands":   False,
    "saveLgsPsf"    :   False,
    "saveSciPsf"    :   True,
    },

"Atmosphere":{
    "scrnNo"        :   4,
    "scrnHeights"   :   numpy.array([0,5000,10000,15000]),
    "scrnStrengths" :   numpy.array([0.5,0.3,0.1,0.1]),
    "windDirs"      :   numpy.array([0,45,90,135]),
    "windSpeeds"    :   numpy.array([10,10,15,20]),
    "wholeScrnSize" :   2048,
    "r0"            :   0.16,
    },

"Telescope":{
   "telDiam"        :   8.,  #Metres
   "obsDiam"        :   1.1, #Central Obscuration
   "mask"           :   "circle",
    },

"WFS":{
    "GSPosition"    :   [(10,0),(0,0)],
    "GSHeight"      :   [0, 90000],
    "GSMag"         :   [8,     8],
    "nxSubaps"      :   [1, 8],
    "pxlsPerSubap"  :   [20, 10],
    "subapFOV"      :   [3., 2.5],
    "fftOversamp"   :   [3, 3],
    "wavelength"    :   [600e-9, 589e-9],
    "lgs"           :   [False, False],
    "centMethod"    :   ["brightestPxl", "brightestPxl"],
    "removeTT"      :   [False, True],

    },

"LGS":{

    },

"DM":{
    "type"              :   ["TT",     "Piezo"],
    "nxActuators"       :   [2,         9],
    "svdConditioning"   :   [1e-15,     0.05],
    "closed"            :   [True,      True],
    "gain"              :   [0.6,       0.7],
    "iMatValue"         :   [0.2,       0.2 ],
    "wfs"               :   [ 0,    1],
    },

"Science":{
    "position"      :   [(0,0)],
    "FOV"           :   [2.0],
    "wavelength"    :   [1.65e-6],
    "pxls"          :   [128],
    "fftOversamp"   :   [2],
    }
}



