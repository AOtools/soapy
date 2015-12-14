from soapy import confParse, SCI, aoSimLib
import unittest
import numpy


class TestWfs(unittest.TestCase):

    def test_sciWfs(self):

        config = confParse.Configurator("../conf/sh_8x8.py")
        config.loadSimParams()

        mask = aoSimLib.circle(config.sim.pupilSize/2., config.sim.simSize)

        sci = SCI.scienceCam(
                config.sim, config.tel, config.atmos, config.scis[0], mask)

    def test_sciFrame(self):
        config = confParse.Configurator("../conf/sh_8x8.py")
        config.loadSimParams()

        mask = aoSimLib.circle(config.sim.pupilSize/2., config.sim.simSize)

        sci = SCI.scienceCam(
                config.sim, config.tel, config.atmos, config.scis[0], mask)

        sci.frame(numpy.ones((config.sim.simSize, config.sim.simSize)))

    def test_sciStrehl(self):
        config = confParse.Configurator("../conf/sh_8x8.py")
        config.loadSimParams()

        mask = aoSimLib.circle(config.sim.pupilSize/2., config.sim.simSize)

        sci = SCI.scienceCam(
                config.sim, config.tel, config.atmos, config.scis[0], mask)

        sci.frame(numpy.ones((config.sim.simSize, config.sim.simSize)))

        self.assertTrue(numpy.allclose(sci.instStrehl, 1.))


if __name__=="__main__":
    unittest.main()
