from soapy import confParse, WFS, aoSimLib
import unittest
import numpy
import os
CONFIG_PATH = os.path.join(os.path.dirname(os.path.abspath(__file__)), "../conf/")

class TestWfs(unittest.TestCase):

    def testa_initWfs(self):

        config = confParse.Configurator(os.path.join(CONFIG_PATH, "sh_8x8.py"))
        config.loadSimParams()

        mask = aoSimLib.circle(config.sim.pupilSize/2., config.sim.simSize)

        wfs = WFS.WFS(config.sim, config.wfss[0], config.atmos, config.lgss[0], mask)


    def testb_wfsFrame(self):
        config = confParse.Configurator(os.path.join(CONFIG_PATH, "sh_8x8.py"))
        config.loadSimParams()
        mask = aoSimLib.circle(config.sim.pupilSize/2., config.sim.simSize)

        wfs = WFS.WFS(config.sim, config.wfss[0], config.atmos, config.lgss[0], mask)

        wfs.frame(numpy.zeros((config.sim.simSize, config.sim.simSize)))


    def testc_initSHWfs(self):

        config = confParse.Configurator(os.path.join(CONFIG_PATH, "sh_8x8.py"))
        config.loadSimParams()

        mask = aoSimLib.circle(config.sim.pupilSize/2., config.sim.simSize)

        wfs = WFS.ShackHartmann(config.sim, config.wfss[0], config.atmos, config.lgss[0], mask)

    def testd_SHWfsFrame(self):
        config = confParse.Configurator(os.path.join(CONFIG_PATH, "sh_8x8.py"))
        config.loadSimParams()
        mask = aoSimLib.circle(config.sim.pupilSize/2., config.sim.simSize)

        wfs = WFS.ShackHartmann(config.sim, config.wfss[0], config.atmos, config.lgss[0], mask)

        wfs.frame(numpy.zeros((config.sim.simSize, config.sim.simSize)))

    # def teste_initPyrWfs(self):
    #     config = confParse.Configurator(os.path.join(CONFIG_PATH, "sh_8x8.py"))
    #     config.loadSimParams()
    #
    #     mask = aoSimLib.circle(config.sim.pupilSize/2., config.sim.simSize)
    #
    #     wfs = WFS.Pyramid(config.sim, config.wfss[0], config.atmos, config.lgss[0], mask)
    #
    # def testf_PyrWfsFrame(self):
    #     config = confParse.Configurator(os.path.join(CONFIG_PATH, "sh_8x8.py"))
    #     config.loadSimParams()
    #     mask = aoSimLib.circle(config.sim.pupilSize/2., config.sim.simSize)
    #
    #     wfs = WFS.Pyramid(config.sim, config.wfss[0], config.atmos, config.lgss[0], mask)
    #
    #     wfs.frame(numpy.zeros((config.sim.simSize, config.sim.simSize)))


if __name__=="__main__":
    unittest.main()
