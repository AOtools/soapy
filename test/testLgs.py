from soapy import confParse, LGS, aoSimLib
import unittest
import numpy
import os
CONFIG_PATH = os.path.join(os.path.dirname(os.path.abspath(__file__)), "../conf/")

class TestLgs(unittest.TestCase):

    def testa_initLgs(self):

        config = confParse.Configurator(os.path.join(CONFIG_PATH, "sh_8x8_lgs-uplink.py"))
        config.loadSimParams()

        mask = aoSimLib.circle(config.sim.pupilSize/2., config.sim.simSize)

        lgs = LGS.LGS(config.sim, config.wfss[1], config.lgss[1], config.atmos)


    def testb_initGeoLgs(self):
        config = confParse.Configurator(os.path.join(CONFIG_PATH, "sh_8x8_lgs-uplink.py"))
        config.loadSimParams()
        config.lgss[1].propagationMode = "Geometric"

        mask = aoSimLib.circle(config.sim.pupilSize/2., config.sim.simSize)

        lgs = LGS.LGS_Geometric(
                config.sim, config.wfss[1], config.lgss[1], config.atmos)

    def testc_geoLgsPsf(self):
        config = confParse.Configurator(os.path.join(CONFIG_PATH, "sh_8x8_lgs-uplink.py"))
        config.loadSimParams()

        mask = aoSimLib.circle(config.sim.pupilSize/2., config.sim.simSize)
        config.lgss[1].propagationMode = "Geometric"
        lgs = LGS.LGS_Geometric(
                config.sim, config.wfss[1], config.lgss[1], config.atmos)
        psf = lgs.getLgsPsf(
                [numpy.zeros((config.sim.simSize, config.sim.simSize))])

    def testd_initPhysLgs(self):
        config = confParse.Configurator(os.path.join(CONFIG_PATH, "sh_8x8_lgs-uplink.py"))
        config.loadSimParams()
        config.lgss[1].propagationMode = "Physical"

        mask = aoSimLib.circle(config.sim.pupilSize/2., config.sim.simSize)

        lgs = LGS.LGS_Physical(config.sim, config.wfss[1], config.lgss[1], config.atmos)

    def teste_physLgsPsf(self):
        config = confParse.Configurator(os.path.join(CONFIG_PATH, "sh_8x8_lgs-uplink.py"))
        config.loadSimParams()

        config.lgss[1].propagationMode = "Physical"
        mask = aoSimLib.circle(config.sim.pupilSize/2., config.sim.simSize)

        lgs = LGS.LGS_Physical(
                config.sim, config.wfss[1], config.lgss[1], config.atmos,
                nOutPxls=10)
        psf = lgs.getLgsPsf(
                [numpy.zeros((config.sim.simSize, config.sim.simSize))])

if __name__=="__main__":
    unittest.main()
