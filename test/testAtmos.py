from soapy import confParse, aoSimLib, atmosphere
import unittest
import numpy


class TestAtmos(unittest.TestCase):
    
    def test_initAtmos(self):
        
        config = confParse.Configurator("../conf/sh_8x8.py")
        config.loadSimParams()
        
        atmos = atmosphere.atmos(config.sim, config.atmos)

    def test_moveAtmos(self):
        config = confParse.Configurator("../conf/sh_8x8.py")
        config.loadSimParams()
        
        atmos = atmosphere.atmos(config.sim, config.atmos)
        atmos.moveScrns()

    def test_randomAtmos(self):
        config = confParse.Configurator("../conf/sh_8x8.py")
        config.loadSimParams()
        
        atmos = atmosphere.atmos(config.sim, config.atmos)
        atmos.randomScrns()

    def test_ftScrn(self):
        
        scrn = atmosphere.ft_phase_screen(0.2, 512, 4.2/128, 30., 0.01)

    def test_ftShScrn(self):
        scrn = atmosphere.ft_sh_phase_screen(0.2, 512, 4.2/128, 30., 0.01)
        
