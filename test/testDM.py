from soapy import confParse, DM, WFS
from soapy.aotools import circle
import unittest
import numpy
import os
CONFIG_PATH = os.path.join(os.path.dirname(os.path.abspath(__file__)), "../conf/")

class TestDM(unittest.TestCase):

    def testa_initDM(self):

        config = confParse.Configurator(os.path.join(CONFIG_PATH, "sh_8x8.py"))
        config.loadSimParams()

        mask = circle.circle(config.sim.pupilSize/2., config.sim.simSize)

        wfs = WFS.ShackHartmann(config, mask=mask)
        dm = DM.DM(config, wfss=[wfs], mask=mask)


    def testb_initPiezo(self):

        config = confParse.Configurator(os.path.join(CONFIG_PATH, "sh_8x8.py"))
        config.loadSimParams()

        mask = circle.circle(config.sim.pupilSize/2., config.sim.simSize)

        wfs = WFS.ShackHartmann(config, mask=mask)
        self.dm = DM.Piezo(config, wfss=[wfs], mask=mask)

    def testd_initGauss(self):

        config = confParse.Configurator(os.path.join(CONFIG_PATH, "sh_8x8.py"))
        config.loadSimParams()

        mask = circle.circle(config.sim.pupilSize/2., config.sim.simSize)

        wfs = WFS.ShackHartmann(config, mask=mask)
        dm = DM.GaussStack(config, wfss=[wfs], mask=mask)


    def testf_initTT(self):

        config = confParse.Configurator(os.path.join(CONFIG_PATH, "sh_8x8.py"))
        config.loadSimParams()

        mask = circle.circle(config.sim.pupilSize/2., config.sim.simSize)

        wfs = WFS.ShackHartmann(config, mask=mask)
        dm = DM.TT(config, wfss=[wfs], mask=mask)




    def testf_initFastPiezo(self):

        config = confParse.Configurator(os.path.join(CONFIG_PATH, "sh_8x8.py"))
        config.loadSimParams()

        mask = circle.circle(config.sim.pupilSize/2., config.sim.simSize)

        wfs = WFS.ShackHartmann(config, mask=mask)
        dm = DM.FastPiezo(config, wfss=[wfs], mask=mask)

