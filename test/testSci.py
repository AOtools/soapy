from soapy import confParse, SCI, aoSimLib
import unittest
import numpy


class TestWfs(unittest.TestCase):

    def testa_initWfs(self):

        config = confParse.Configurator("../conf/sh_8x8.py")
        config.loadSimParams()

        mask = aoSimLib.circle(config.sim.pupilSize/2., config.sim.simSize)

        sci = SCI.scienceCam(
                config.sim, config.tel, config.atmos, config.scis[0], mask)

    def testb_wfsFrame(self):
        config = confParse.Configurator("../conf/sh_8x8.py")
        config.loadSimParams()

        mask = aoSimLib.circle(config.sim.pupilSize/2., config.sim.simSize)

        sci = SCI.scienceCam(
                config.sim, config.tel, config.atmos, config.scis[0], mask)

        sci.frame(numpy.ones((config.sim.simSize, config.sim.simSize)))



if __name__=="__main__":
    unittest.main()
