import unittest

import pyAOS

class TestSimpleSCAO(unittest.TestCase):

    def testAOInit(self):
        sim = pyAOS.Sim("../conf/sh_8x8.py")
        sim.config.sim.filePrefix = None
        sim.config.sim.logfile = None
        sim.config.sim.nIters = 100
        sim.aoinit()

    #def testIMat(self):
        sim.makeIMat(forceNew=True)

    #def testLoop(self):
        sim.aoloop()

if __name__ == '__main__':
    unittest.main()
