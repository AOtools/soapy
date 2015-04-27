from soapy import confParse, WFS, aoSimLib
import unittest
import numpy


class TestWfs(unittest.TestCase):
    
    def testa_initWfs(self):
        
        config = confParse.Configurator("../conf/sh_8x8.py")
        config.loadSimParams()
        
        mask = aoSimLib.circle(config.sim.pupilSize/2., config.sim.simSize)

        wfs = WFS.WFS(config.sim, config.wfs[0], config.atmos, config.lgs[0], mask)
            

    def testb_wfsFrame(self):
        config = confParse.Configurator("../conf/sh_8x8.py")
        config.loadSimParams()
        mask = aoSimLib.circle(config.sim.pupilSize/2., config.sim.simSize)

        wfs = WFS.WFS(config.sim, config.wfs[0], config.atmos, config.lgs[0], mask)

        wfs.frame(numpy.zeros((config.sim.simSize, config.sim.simSize)))

    
    def testc_initSHWfs(self):
        
        config = confParse.Configurator("../conf/sh_8x8.py")
        config.loadSimParams()
        
        mask = aoSimLib.circle(config.sim.pupilSize/2., config.sim.simSize)

        wfs = WFS.ShackHartmann(config.sim, config.wfs[0], config.atmos, config.lgs[0], mask)

    def testd_SHWfsFrame(self):
        config = confParse.Configurator("../conf/sh_8x8.py")
        config.loadSimParams()
        mask = aoSimLib.circle(config.sim.pupilSize/2., config.sim.simSize)

        wfs = WFS.ShackHartmann(config.sim, config.wfs[0], config.atmos, config.lgs[0], mask)

        wfs.frame(numpy.zeros((config.sim.simSize, config.sim.simSize)))

    def teste_initPyrWfs(self):
        config = confParse.Configurator("../conf/sh_8x8.py")
        config.loadSimParams()
        
        mask = aoSimLib.circle(config.sim.pupilSize/2., config.sim.simSize)

        wfs = WFS.Pyramid(config.sim, config.wfs[0], config.atmos, config.lgs[0], mask)

    def testf_PyrWfsFrame(self):
        config = confParse.Configurator("../conf/sh_8x8.py")
        config.loadSimParams()
        mask = aoSimLib.circle(config.sim.pupilSize/2., config.sim.simSize)

        wfs = WFS.Pyramid(config.sim, config.wfs[0], config.atmos, config.lgs[0], mask)

        wfs.frame(numpy.zeros((config.sim.simSize, config.sim.simSize)))


if __name__=="__main__":
    unittest.main()

