from soapy import confParse
import unittest
import os
CONFIG_PATH = os.path.join(os.path.dirname(os.path.abspath(__file__)), "../conf/")

class TestConf(unittest.TestCase):

    def test_loadSh8x8_py(self):

        config = confParse.loadSoapyConfig(
                os.path.join(CONFIG_PATH, "sh_8x8.py"))

    def test_loadSh8x8_yaml(self):

        config = confParse.loadSoapyConfig(
                os.path.join(CONFIG_PATH, "sh_8x8.yaml"))

    def testa_loadSh8x8_lgsElong_py(self):

        config = confParse.loadSoapyConfig(
                os.path.join(CONFIG_PATH, "sh_8x8_lgs-elongation.py"))


    def testa_loadSh8x8_lgsElong_yaml(self):

        config = confParse.loadSoapyConfig(
                os.path.join(CONFIG_PATH, "sh_8x8_lgs-elongation.yaml"))

    def testa_loadSh8x8_lgsUplink_py(self):

        config = confParse.loadSoapyConfig(
                os.path.join(CONFIG_PATH, "sh_8x8_lgs-uplink.py"))


    def testa_loadSh8x8_lgsUplink_yaml(self):

        config = confParse.loadSoapyConfig(
                os.path.join(CONFIG_PATH, "sh_8x8_lgs-uplink.yaml"))
