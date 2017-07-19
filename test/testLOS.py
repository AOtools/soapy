from soapy import confParse, lineofsight
import unittest
import numpy
import os
CONFIG_PATH = os.path.join(os.path.dirname(os.path.abspath(__file__)), "../conf/")

class TestLOS(unittest.TestCase):

    def test_initLOS(self):
        config = confParse.loadSoapyConfig(os.path.join(CONFIG_PATH, "sh_8x8.yaml"))

        los = lineofsight.LineOfSight(config.wfss[0], config)

    def test_runLOS(self):
        config = confParse.loadSoapyConfig(os.path.join(CONFIG_PATH, "sh_8x8.yaml"))

        los = lineofsight.LineOfSight(config.wfss[0], config)

        testPhase = numpy.arange(config.sim.simSize**2).reshape(
                (config.sim.simSize,)*2)

        phs = los.frame(testPhase)
