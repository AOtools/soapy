from soapy import confParse, WFS
import aotools
import unittest
import numpy
import os
CONFIG_PATH = os.path.join(os.path.dirname(os.path.abspath(__file__)), "../conf/")

class TestWfs(unittest.TestCase):

    def testa_initWfs(self):

        config = confParse.loadSoapyConfig(os.path.join(CONFIG_PATH, "sh_8x8.yaml"))

        mask = aotools.circle(config.sim.pupilSize/2., config.sim.simSize)

        wfs = WFS.WFS(config, mask=mask)


    def testb_wfsFrame(self):
        config = confParse.loadSoapyConfig(os.path.join(CONFIG_PATH, "sh_8x8.yaml"))

        mask = aotools.circle(config.sim.pupilSize/2., config.sim.simSize)

        wfs = WFS.WFS(config, mask=mask)

        wfs.frame(numpy.zeros((config.sim.simSize, config.sim.simSize)))


    def testc_initSHWfs(self):

        config = confParse.loadSoapyConfig(os.path.join(CONFIG_PATH, "sh_8x8.yaml"))

        mask = aotools.circle(config.sim.pupilSize/2., config.sim.simSize)

        wfs = WFS.ShackHartmann(config, mask=mask)

    def testd_SHWfsFrame(self):
        config = confParse.loadSoapyConfig(os.path.join(CONFIG_PATH, "sh_8x8.yaml"))
        mask = aotools.circle(config.sim.pupilSize/2., config.sim.simSize)

        wfs = WFS.ShackHartmann(config, mask=mask)

        wfs.frame(numpy.zeros((config.sim.simSize, config.sim.simSize)))

    def testc_initLegacySHWfs(self):
        config = confParse.loadSoapyConfig(os.path.join(CONFIG_PATH, "sh_8x8.yaml"))

        mask = aotools.circle(config.sim.pupilSize / 2., config.sim.simSize)

        wfs = WFS.ShackHartmannLegacy(config, mask=mask)

    def testd_LegacySHWfsFrame(self):
        config = confParse.loadSoapyConfig(os.path.join(CONFIG_PATH, "sh_8x8.yaml"))
        mask = aotools.circle(config.sim.pupilSize / 2., config.sim.simSize)

        wfs = WFS.ShackHartmannLegacy(config, mask=mask)

        wfs.frame(numpy.zeros((config.atmos.scrnNo, config.sim.simSize, config.sim.simSize)))

    def test_PhysWfs(self):

        config = confParse.loadSoapyConfig(os.path.join(CONFIG_PATH, "sh_8x8.yaml"))
        config.wfss[0].propagationMode = "Physical"

        mask = aotools.circle(config.sim.pupilSize/2., config.sim.simSize)

        wfs = WFS.WFS(config, mask=mask)

        wfs.frame(numpy.zeros((config.atmos.scrnNo, config.sim.scrnSize, config.sim.scrnSize)))

    def testc_initGradWfs(self):

        config = confParse.loadSoapyConfig(os.path.join(CONFIG_PATH, "sh_8x8.yaml"))
        config.sim.pupilSize = 10*config.wfss[0].nxSubaps
        mask = aotools.circle(config.sim.pupilSize/2., config.sim.simSize)

        wfs = WFS.Gradient(config, mask=mask)

    def testd_GradWfsFrame(self):
        config = confParse.loadSoapyConfig(os.path.join(CONFIG_PATH, "sh_8x8.yaml"))
        config.sim.pupilSize = 10*config.wfss[0].nxSubaps

        mask = aotools.circle(config.sim.pupilSize/2., config.sim.simSize)

        wfs = WFS.Gradient(config, mask=mask)

        wfs.frame(numpy.zeros((config.sim.simSize, config.sim.simSize)))

    def teste_initPyrWfs(self):
        config = confParse.loadSoapyConfig(os.path.join(CONFIG_PATH, "pwfs.yaml"))
        mask = aotools.circle(config.sim.pupilSize/2., config.sim.simSize)
    
        wfs = WFS.Pyramid(config, mask=mask)
    
    def testf_PyrWfsFrame(self):
        config = confParse.loadSoapyConfig(os.path.join(CONFIG_PATH, "pwfs.yaml"))
        mask = aotools.circle(config.sim.pupilSize/2., config.sim.simSize)
    
        wfs = WFS.Pyramid(config, mask=mask)
    
        wfs.frame(numpy.zeros((config.sim.simSize, config.sim.simSize)))


if __name__=="__main__":
    unittest.main()
