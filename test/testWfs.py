from soapy import confParse, WFS
from soapy.aotools import circle
import unittest
import numpy
import os
CONFIG_PATH = os.path.join(os.path.dirname(os.path.abspath(__file__)), "../conf/")

class TestWfs(unittest.TestCase):

    def testa_initWfs(self):

        config = confParse.Configurator(os.path.join(CONFIG_PATH, "sh_8x8.py"))
        config.loadSimParams()

        mask = circle.circle(config.sim.pupilSize/2., config.sim.simSize)

        wfs = WFS.WFS(config, mask=mask)


    def testb_wfsFrame(self):
        config = confParse.Configurator(os.path.join(CONFIG_PATH, "sh_8x8.py"))
        config.loadSimParams()
        mask = circle.circle(config.sim.pupilSize/2., config.sim.simSize)

        wfs = WFS.WFS(config, mask=mask)

        wfs.frame(numpy.zeros((config.sim.simSize, config.sim.simSize)))


    def testc_initSHWfs(self):

        config = confParse.Configurator(os.path.join(CONFIG_PATH, "sh_8x8.py"))
        config.loadSimParams()

        mask = circle.circle(config.sim.pupilSize/2., config.sim.simSize)

        wfs = WFS.ShackHartmann(config, mask=mask)

    def testd_SHWfsFrame(self):
        config = confParse.Configurator(os.path.join(CONFIG_PATH, "sh_8x8.py"))
        config.loadSimParams()
        mask = circle.circle(config.sim.pupilSize/2., config.sim.simSize)

        wfs = WFS.ShackHartmann(config, mask=mask)

        wfs.frame(numpy.zeros((config.sim.simSize, config.sim.simSize)))

    def test_PhysWfs(self):

        config = confParse.Configurator(os.path.join(CONFIG_PATH, "sh_8x8.py"))
        config.loadSimParams()
        config.wfss[0].propagationMode = "Physical"

        mask = circle.circle(config.sim.pupilSize/2., config.sim.simSize)

        wfs = WFS.WFS(config, mask=mask)

        wfs.frame([numpy.zeros((config.sim.scrnSize,)*2)]*config.atmos.scrnNo)

    def testc_initGradWfs(self):

        config = confParse.Configurator(os.path.join(CONFIG_PATH, "sh_8x8.py"))
        config.loadSimParams()

        mask = circle.circle(config.sim.pupilSize/2., config.sim.simSize)

        wfs = WFS.Gradient(config, mask=mask)

    def testd_GradWfsFrame(self):
        config = confParse.Configurator(os.path.join(CONFIG_PATH, "sh_8x8.py"))
        config.loadSimParams()
        mask = circle.circle(config.sim.pupilSize/2., config.sim.simSize)

        wfs = WFS.Gradient(config, mask=mask)

        wfs.frame(numpy.zeros((config.sim.simSize, config.sim.simSize)))

    # def teste_initPyrWfs(self):
    #     config = confParse.Configurator(os.path.join(CONFIG_PATH, "sh_8x8.py"))
    #     config.loadSimParams()
    #
    #     mask = circle.circle(config.sim.pupilSize/2., config.sim.simSize)
    #
    #     wfs = WFS.Pyramid(config, config.lgss[0], mask)
    #
    # def testf_PyrWfsFrame(self):
    #     config = confParse.Configurator(os.path.join(CONFIG_PATH, "sh_8x8.py"))
    #     config.loadSimParams()
    #     mask = circle.circle(config.sim.pupilSize/2., config.sim.simSize)
    #
    #     wfs = WFS.Pyramid(config, config.lgss[0], mask)
    #
    #     wfs.frame(numpy.zeros((config.sim.simSize, config.sim.simSize)))


if __name__=="__main__":
    unittest.main()
