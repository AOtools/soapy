from soapy import confParse, aoSimLib, atmosphere
import unittest
import numpy
import os
CONFIG_PATH = os.path.join(os.path.dirname(os.path.abspath(__file__)), "../conf/")

class TestAtmos(unittest.TestCase):

    def test_initAtmos(self):

        config = confParse.Configurator(os.path.join(CONFIG_PATH,"sh_8x8.py"))
        config.loadSimParams()

        atmos = atmosphere.atmos(config)

    def test_moveAtmos(self):
        config = confParse.Configurator(os.path.join(CONFIG_PATH,"sh_8x8.py"))
        config.loadSimParams()

        atmos = atmosphere.atmos(config)
        atmos.moveScrns()

    def test_randomAtmos(self):
        config = confParse.Configurator(os.path.join(CONFIG_PATH,"sh_8x8.py"))
        config.loadSimParams()

        atmos = atmosphere.atmos(config)
        atmos.randomScrns()

    def test_ftScrn(self):

        scrn = atmosphere.ft_phase_screen(0.2, 512, 4.2/128, 30., 0.01)

    def test_ftShScrn(self):
        scrn = atmosphere.ft_sh_phase_screen(0.2, 512, 4.2/128, 30., 0.01)
