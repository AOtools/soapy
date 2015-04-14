import unittest

import pyAOS

import numpy

pyAOS.logger.setLoggingLevel(3)

RESULTS = {
        "8x8": 0.76,
        "8x8_offAxis": 0.30,
        "8x8_zernike": 0.65,
        "8x8_lgs"    : 0.65
        }


class TestSimpleSCAO(unittest.TestCase):

    def testOnAxis(self):
        sim = pyAOS.Sim("../conf/sh_8x8.py")
        sim.config.sim.filePrefix = None
        sim.config.sim.logfile = None
        sim.config.sim.nIters = 100
        sim.config.wfs[0].GSPosition=(0,0)
        
        sim.aoinit()

        sim.makeIMat(forceNew=True)

        sim.aoloop()

        #Check results are ok
        assert numpy.allclose(sim.longStrehl[0,-1], RESULTS["8x8"], atol=0.2)


    def testOffAxis(self):
        sim = pyAOS.Sim("../conf/sh_8x8.py")
        sim.config.sim.filePrefix = None
        sim.config.sim.logfile = None
        sim.config.sim.nIters = 100
        sim.config.wfs[0].GSPosition = (20,0)
        
        sim.aoinit()

        sim.makeIMat(forceNew=True)

        sim.aoloop()

        #Check results are ok
        assert numpy.allclose(
                sim.longStrehl[0,-1], RESULTS["8x8_offAxis"], atol=0.2)

    
    def testZernikeDM(self):
        sim = pyAOS.Sim("../conf/sh_8x8.py")
        sim.config.sim.filePrefix = None
        sim.config.sim.logfile = None
        sim.config.sim.nIters = 100
        sim.config.wfs[0].GSPosition = (0,0)
        
        sim.config.sim.nDM = 1
        sim.config.dm[0].dmType = "Zernike"
        sim.config.dm[0].dmActs = 45
        sim.config.dm[0].dmCond = 0.01
        
        sim.aoinit()

        sim.makeIMat(forceNew=True)

        sim.aoloop()

        #Check results are ok
        assert numpy.allclose(
                sim.longStrehl[0,-1], RESULTS["8x8_zernike"], atol=0.2)



    def testCone(self):

        sim = pyAOS.Sim("../conf/sh_8x8_lgs.py")
        sim.config.sim.filePrefix = None
        sim.config.sim.logfile = None
        sim.config.sim.nIters = 100

        sim.aoinit()

        sim.makeIMat(forceNew=True)

        sim.aoloop()

        #Check results are ok
        assert numpy.allclose(
                sim.longStrehl[0,-1], RESULTS["8x8_lgs"], atol=0.2)


if __name__ == '__main__':
    unittest.main()
