from soapy import confParse, atmosphere
import unittest
import numpy
import os
CONFIG_PATH = os.path.join(os.path.dirname(os.path.abspath(__file__)), "../conf/")

import shutil

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

    def test_saveloadScrn(self):
        # test the saving and loading phase screens
        
        config = confParse.Configurator(os.path.join(CONFIG_PATH,"sh_8x8.py"))
        config.loadSimParams()
        config.atmos.wholeScrnSize = 512

        atmos = atmosphere.atmos(config)
        
        # Make a directory to save the screens in 
        if os.path.exists('testscrns'):
            shutil.rmtree('testscrns')

        os.makedirs('testscrns')
        atmos.saveScrns('testscrns')
        try:
            # change config to load scrns
            config.atmos.scrnNames = []
            for i in range(config.atmos.scrnNo):
                config.atmos.scrnNames.append('testscrns/scrn{}.fits'.format(i))
                
            atmos2 = atmosphere.atmos(config)
            
            # Check that all scrns are identical
            
            for i in range(config.atmos.scrnNo):
                assert numpy.allclose(atmos.wholeScrns[i], atmos2.wholeScrns[i])
            
            # delete that dir
            shutil.rmtree('testscrns')
        
        except:
            # always delete that dir - even in case of failure
            shutil.rmtree('testscrns')
            raise