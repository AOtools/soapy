import unittest

import soapy

import numpy
import os
CONFIG_PATH = os.path.join(os.path.dirname(os.path.abspath(__file__)), "../conf/")

soapy.logger.setLoggingLevel(3)

RESULTS = {
        "8x8": 0.47,
        "8x8_open": 0.3,
        "8x8_offAxis": 0.22,
        "8x8_zernike": 0.36,
        "8x8_lgs"    : 0.27,
        "8x8_phys": 0.53,
        }


class TestSimpleSCAO(unittest.TestCase):

    def testOnAxis(self):

        sim = soapy.Sim(os.path.join(CONFIG_PATH, "sh_8x8.py"))

        sim.config.sim.simName = None
        sim.config.sim.logfile = None
        sim.config.sim.nIters = 100
        sim.config.wfss[0].GSPosition=(0,0)

        sim.aoinit()

        sim.makeIMat(forceNew=True)

        sim.aoloop()

        #Check results are ok
        assert numpy.allclose(sim.longStrehl[0,-1], RESULTS["8x8"], atol=0.2)

    def testPhysProp(self):
        sim = soapy.Sim(os.path.join(CONFIG_PATH, "sh_8x8.py"))

        sim.config.sim.simName = None
        sim.config.sim.logfile = None
        sim.config.sim.nIters = 100
        sim.config.wfss[0].GSPosition=(0,0)
        sim.config.wfss[0].propagationMode="physical"

        sim.aoinit()

        sim.makeIMat(forceNew=True)

        sim.aoloop()

        #Check results are ok
        assert numpy.allclose(sim.longStrehl[0,-1], RESULTS["8x8_phys"], atol=0.2)

    def testOffAxis(self):
        sim = soapy.Sim(os.path.join(CONFIG_PATH, "sh_8x8.py"))

        sim.config.sim.simName = None
        sim.config.sim.logfile = None
        sim.config.sim.nIters = 100
        sim.config.wfss[0].GSPosition = (20,0)

        sim.aoinit()

        sim.makeIMat(forceNew=True)

        sim.aoloop()

        #Check results are ok
        assert numpy.allclose(
                sim.longStrehl[0,-1], RESULTS["8x8_offAxis"], atol=0.2)


    def testZernikeDM(self):
        sim = soapy.Sim(os.path.join(CONFIG_PATH, "sh_8x8.py"))

        sim.config.sim.simName = None
        sim.config.sim.logfile = None
        sim.config.sim.nIters = 100
        sim.config.wfss[0].GSPosition = (0,0)

        sim.config.sim.nDM = 1
        sim.config.dms[0].type = "Zernike"
        sim.config.dms[0].nxActuators = 45
        sim.config.dms[0].svdConditioning = 0.01
        sim.config.dms[0].iMatValue=100

        sim.aoinit()

        sim.makeIMat(forceNew=True)

        sim.aoloop()

        #Check results are ok
        assert numpy.allclose(
                sim.longStrehl[0,-1], RESULTS["8x8_zernike"], atol=0.2)



    def testCone(self):

        sim = soapy.Sim(os.path.join(CONFIG_PATH, "sh_8x8_lgs.py"))
        sim.config.sim.simName= None
        sim.config.sim.logfile = None
        sim.config.sim.nIters = 100
        sim.config.sim.GSHeight=25000

        sim.aoinit()

        sim.makeIMat(forceNew=True)

        sim.aoloop()

        #Check results are ok
        assert numpy.allclose(
                sim.longStrehl[0,-1], RESULTS["8x8_lgs"], atol=0.2)

    def testOpenLoop(self):
        sim = soapy.Sim(os.path.join(CONFIG_PATH, "sh_8x8.py"))
        sim.config.sim.simName = None
        sim.config.sim.logfile = None
        sim.config.sim.nIters = 100
        sim.config.wfss[0].GSPosition=(0,0)

        for i in range(sim.config.sim.nDM-1):
            sim.config.dms[i+1].closed = False

        sim.aoinit()

        sim.makeIMat(forceNew=True)

        sim.aoloop()

        #Check results are ok
        assert numpy.allclose(sim.longStrehl[0,-1], RESULTS["8x8_open"], atol=0.2)


if __name__ == '__main__':
    unittest.main()
