from soapy import confParse, scienceinstrument
import aotools
import unittest
import numpy
import os
CONFIG_PATH = os.path.join(os.path.dirname(os.path.abspath(__file__)), "../conf/")

class TestSci(unittest.TestCase):

    def test_sciInit(self):

        config = confParse.loadSoapyConfig(os.path.join(CONFIG_PATH, "sh_8x8.yaml"))

        mask = aotools.circle(config.sim.pupilSize/2., config.sim.simSize)

        sci = scienceinstrument.PSFCamera(config, 0, mask)

    def test_sciFrame(self):
        config = confParse.loadSoapyConfig(os.path.join(CONFIG_PATH, "sh_8x8.yaml"))

        mask = aotools.circle(config.sim.pupilSize/2., config.sim.simSize)

        sci = scienceinstrument.PSFCamera(config, 0, mask)

        sci.frame(numpy.ones((config.atmos.scrnNo, config.sim.scrnSize, config.sim.scrnSize)))

    def test_sciStrehl(self):
        config = confParse.loadSoapyConfig(os.path.join(CONFIG_PATH, "sh_8x8.yaml"))

        mask = aotools.circle(config.sim.pupilSize/2., config.sim.simSize)

        sci = scienceinstrument.PSFCamera(
                config, 0, mask)

        sci.frame(numpy.ones((config.atmos.scrnNo, config.sim.scrnSize, config.sim.scrnSize)))

        self.assertTrue(numpy.allclose(sci.instStrehl, 1.))

    def test_fibreInit(self):

        config = confParse.loadSoapyConfig(os.path.join(CONFIG_PATH, "sh_8x8.yaml"))

        mask = aotools.circle(config.sim.pupilSize/2., config.sim.simSize)

        sci = scienceinstrument.singleModeFibre(config, 0, mask)

    def test_fibreFrame(self):
        config = confParse.loadSoapyConfig(os.path.join(CONFIG_PATH, "sh_8x8.yaml"))

        mask = aotools.circle(config.sim.pupilSize/2., config.sim.simSize)

        sci = scienceinstrument.singleModeFibre(config, 0, mask)
        sci.frame(numpy.ones((config.atmos.scrnNo, config.sim.scrnSize, config.sim.scrnSize)))


if __name__=="__main__":
    unittest.main()
