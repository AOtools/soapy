from soapy import confParse, SCI, aoSimLib
import unittest
import numpy
import os
CONFIG_PATH = os.path.join(os.path.dirname(os.path.abspath(__file__)), "../conf/")

class TestSci(unittest.TestCase):

    def test_sciInit(self):

        config = confParse.Configurator(os.path.join(CONFIG_PATH, "sh_8x8.py"))
        config.loadSimParams()

        mask = aoSimLib.circle(config.sim.pupilSize/2., config.sim.simSize)

        sci = SCI.ScienceCam(
                config.sim, config.tel, config.atmos, config.scis[0], mask)

    def test_sciFrame(self):
        config = confParse.Configurator(os.path.join(CONFIG_PATH, "sh_8x8.py"))
        config.loadSimParams()

        mask = aoSimLib.circle(config.sim.pupilSize/2., config.sim.simSize)

        sci = SCI.ScienceCam(
                config.sim, config.tel, config.atmos, config.scis[0], mask)

        sci.frame(numpy.ones((config.sim.simSize, config.sim.simSize)))

    def test_sciStrehl(self):
        config = confParse.Configurator(os.path.join(CONFIG_PATH, "sh_8x8.py"))
        config.loadSimParams()

        mask = aoSimLib.circle(config.sim.pupilSize/2., config.sim.simSize)

        sci = SCI.ScienceCam(
                config.sim, config.tel, config.atmos, config.scis[0], mask)

        sci.frame(numpy.ones((config.sim.simSize, config.sim.simSize)))

        self.assertTrue(numpy.allclose(sci.instStrehl, 1.))


if __name__=="__main__":
    unittest.main()
