'''
Config File for python tomographic AO simulation.

numpy is imported in the sim, so numpy funcs can be used here.
'''
import numpy

parameters={
    "filePrefix"    :  None,
    "pupilSize"     :   128, #this is the phase SAMPLING size
                            #(not number of detector pxls)

    #Atmosphere Parameters
    "scrnNo"        :  1,
    "scrnNames"     :   {   0:"data/phaseScreen_size-4096_r0-10.fits"},
#                            1:"../screen2.fits",
#                            2:"../screen3.fits",
#                            3:"../screen4.fits"
#                            },
    "scrnHeights"   :   numpy.array([0,5000,10000,25000]),
    #"scrnStrengths" :   numpy.array([0.20,0.30,0.50,0.50]),
    "scrnStrengths":   numpy.array([0.5,0.3,0.1,0.1]),
    "windDirs"      :   numpy.array([0,45,90,135]),
    "windSpeeds"    :   numpy.array([10,10,15,20]),
    "newScreens"    :   True, #Bool
    "scrnSize"      :   512,
    "r0"            :   0.15,

    
    #Telescope Parameters
   "telDiam"        :   4.2,  #Metres
   "obs"            :   1.2, #Central Obscuration
   "mask"           :   "circle",

    #WFS Parameters
    "GSNo"          :   1,
    "GSPositions"   :   numpy.array([   [0,0]   ,
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

    "subapOversamp":  numpy.array([4]*5),
    "waveLengths"   :   numpy.array( [ 600e-9] *5),
    "subapThreshold":   0.7,
    "mp_wfs"        :   False,
    "wfsBitDepth"   :   numpy.array([8]*5),
    "wfsSNR"        :   numpy.array([0]*5),
    "removeTT"      :   numpy.array([0,0,0,0,0]),
    "angleEquivNoise":  numpy.array([0]*5), #arcsecs

    #LGS Parameter
    "LGSPupilSize"  :   0.5, #Metres
    "LGSUplink"     :   numpy.array([   0 ]*5),
    "LGSFFTPadding" :   numpy.array([   128     ]*5),
    "LGSWL"         :   numpy.array([   600e-9  ]*5),
    "lgsPropMethod" :   ["physical"]*5,
    "lgsHeight"     :   numpy.array([0]*5),
    "lgsElongation" :   0,
    "elongLayers"   :   10,

    #DM Parameter9
    "tipTilt"       :   False,
    "ttGain"        :   0.9,
    "dmNo"          :   1,
    "dmTypes"        :   [ "peizo","peizo"], #peizo,TT or zernike so far...
    "dmActs"        :   numpy.array( [9**2,33**2]),
    "dmCond"        :   numpy.array( [0.05, 0.05] ),


    #Science Parameters
    "sciNo"         : 1,
    "sciPos"        :   numpy.array( [ [0,0], [0,0],[0,0],[0,0] ] ),
    "sciFOV"        :   numpy.array( [3.0]*4), #Arcsecs
    "sciWvls"       :   numpy.array( [ 1500e-9, 800e-9, 1000e-9, 1500e-9]),
    "sciPxls"       :   numpy.array( [128]*4),
    "sciOversamp"   :   numpy.array( [4]*4 ),

    #Loop Parameters
    "nIters"        :   1000,
    "loopTime"      :   1/250.0,
    "gain"          :   0.6,
    "loopMode"      :   "closed", #Can be "closedSCAO" or "openSCAO" so far...
    "reconstructor" :   "MVM", #Can be "MVM", "WooferTweeter, LearnAndApply1"
    
    #L&A Params  
    "learnIters"    :   1000, #Only applicable if reconstructor is "LearnAndApply"
    "LAscrnNo"        :  4,
    "LAscrnNames"     :   {   0:"data/phaseScreen_size-4096_r0-10.fits"},
#                            1:"../screen2.fits",
#                            2:"../screen3.fits",
#                            3:"../screen4.fits"
#                            },
    "LAscrnHeights"   :   numpy.array([0,5000,10000,25000]),
    #"scrnStrengths" :   numpy.array([0.20,0.30,0.50,0.50]),
    "LAscrnStrengths":   numpy.array([0.5,0.3,0.1,0.1]),
    "LAwindDirs"      :   numpy.array([0,45,90,135]),
    "LAwindSpeeds"    :   numpy.array([10,10,15,20]),
    "LAnewScreens"    :   True, #Bool
    "LAscrnSize"      :   1024,
    "LAr0"            :   0.15,
    


    #Debugging
    "logging"       :   1,

    #Data Saving
    "saveCMat"      :   False,
    "saveSlopes"    :   True,
    "saveDmCommands":   False,
    "saveLgsPsf"    :   False,
    "saveSciPsf"    :   True,

    }





