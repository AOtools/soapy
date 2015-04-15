import unittest

import pyAOS

import numpy

pyAOS.logger.setLoggingLevel(3)

RESULTS = {
        "8x8": 0.76,
        "8x8_offAxis": 0.30,
        "8x8_zernike": 0.65,
        "8x8_lgs"    : 0.65,
        "8x8_phys": 0.76,
        }


class TestSimpleSCAO(unittest.TestCase):

    def testOnAxis(self):
        
        print("\n\nTest On Axis NGS\n")

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

    def testPhysProp(self):

        print("\n\nTest Physica propagation...\n")

        sim = pyAOS.Sim("../conf/sh_8x8.py")
        sim.config.sim.filePrefix = None
        sim.config.sim.logfile = None
        sim.config.sim.nIters = 100
        sim.config.wfs[0].GSPosition=(0,0)
        sim.config.wfs[0].propagationMode="physical"
        
        sim.aoinit()

        sim.makeIMat(forceNew=True)

        sim.aoloop()

        #Check results are ok
        assert numpy.allclose(sim.longStrehl[0,-1], RESULTS["8x8_phys"], atol=0.2)

    def testOffAxis(self):

        print("\n\nTest Off Axis NGS...\n")

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

        print("\n\nTest Zernike DM...\n")
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

        print("\n\nTest Cone Effect...\n")
        sim = pyAOS.Sim("../conf/sh_8x8.py")
        sim.config.sim.filePrefix = None
        sim.config.sim.logfile = None
        sim.config.sim.nIters = 100
        sim.config.sim.GSHeight=25000

        sim.aoinit()

        sim.makeIMat(forceNew=True)

        sim.aoloop()

        #Check results are ok
        assert numpy.allclose(
                sim.longStrehl[0,-1], RESULTS["8x8_lgs"], atol=0.2)


if __name__ == '__main__':
    unittest.main()
