from soapy import confParse, SCI
from soapy.aotools import circle
import unittest
import numpy
import os
CONFIG_PATH = os.path.join(os.path.dirname(os.path.abspath(__file__)), "../conf/")

class TestSci(unittest.TestCase):

    def test_sciInit(self):

        config = confParse.loadSoapyConfig(os.path.join(CONFIG_PATH, "sh_8x8.yaml"))

        mask = circle.circle(config.sim.pupilSize/2., config.sim.simSize)

        sci = SCI.PSF(
                config, 0, mask)

    def test_sciFrame(self):
        config = confParse.loadSoapyConfig(os.path.join(CONFIG_PATH, "sh_8x8.yaml"))

        mask = circle.circle(config.sim.pupilSize/2., config.sim.simSize)

        sci = SCI.PSF(
                config, 0, mask)

        sci.frame(numpy.ones((config.sim.simSize, config.sim.simSize)))

    def test_sciStrehl(self):
        config = confParse.loadSoapyConfig(os.path.join(CONFIG_PATH, "sh_8x8.yaml"))

        mask = circle.circle(config.sim.pupilSize/2., config.sim.simSize)

        sci = SCI.PSF(
                config, 0, mask)

        sci.frame(numpy.ones((config.sim.simSize, config.sim.simSize)))

        self.assertTrue(numpy.allclose(sci.instStrehl, 1.))

    def test_fibreInit(self):

        config = confParse.loadSoapyConfig(os.path.join(CONFIG_PATH, "sh_8x8.yaml"))

        mask = circle.circle(config.sim.pupilSize/2., config.sim.simSize)

        sci = SCI.singleModeFibre(
                config, 0, mask)

    def test_fibreFrame(self):
        config = confParse.loadSoapyConfig(os.path.join(CONFIG_PATH, "sh_8x8.yaml"))

        mask = circle.circle(config.sim.pupilSize/2., config.sim.simSize)

        sci = SCI.singleModeFibre(
                config, 0, mask)


if __name__=="__main__":
    unittest.main()
