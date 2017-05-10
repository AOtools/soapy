from soapy import confParse, DM, WFS
import unittest
import numpy
import aotools
import os
CONFIG_PATH = os.path.join(os.path.dirname(os.path.abspath(__file__)), "../conf/")

def testa_initDM():

    config = confParse.loadSoapyConfig(os.path.join(CONFIG_PATH, "sh_8x8.yaml"))

    mask = aotools.circle(config.sim.pupilSize/2., config.sim.simSize)

    wfs = WFS.ShackHartmann(config, mask=mask)
    dm = DM.DM(config, wfss=[wfs], mask=mask)


def testb_initPiezo():

    config = confParse.loadSoapyConfig(os.path.join(CONFIG_PATH, "sh_8x8.yaml"))

    mask = aotools.circle(config.sim.pupilSize/2., config.sim.simSize)

    wfs = WFS.ShackHartmann(config, mask=mask)
    dm = DM.Piezo(config, n_dm=1, wfss=[wfs], mask=mask)

def testd_initGauss():

    config = confParse.loadSoapyConfig(os.path.join(CONFIG_PATH, "sh_8x8.yaml"))

    mask = aotools.circle(config.sim.pupilSize/2., config.sim.simSize)

    wfs = WFS.ShackHartmann(config, mask=mask)
    dm = DM.GaussStack(config, n_dm=1, wfss=[wfs], mask=mask)


def testf_initTT():

    config = confParse.loadSoapyConfig(os.path.join(CONFIG_PATH, "sh_8x8.yaml"))

    mask = aotools.circle(config.sim.pupilSize/2., config.sim.simSize)

    wfs = WFS.ShackHartmann(config, mask=mask)
    dm = DM.TT(config, wfss=[wfs], mask=mask)




def testf_initFastPiezo():

    config = confParse.loadSoapyConfig(os.path.join(CONFIG_PATH, "sh_8x8.yaml"))

    mask = aotools.circle(config.sim.pupilSize/2., config.sim.simSize)

    wfs = WFS.ShackHartmann(config, mask=mask)
    dm = DM.FastPiezo(config, n_dm=1, wfss=[wfs], mask=mask)

def test_set_valid_actuators():
    """
    Tests that when you set the "valid actuators", the DM computes how many valid actuators there are correctly
    """

    config = confParse.loadSoapyConfig(os.path.join(CONFIG_PATH, "sh_8x8.yaml"))

    mask = aotools.circle(config.sim.pupilSize/2., config.sim.simSize)

    wfs = WFS.ShackHartmann(config, mask=mask)
    dm = DM.DM(config, n_dm=1, wfss=[wfs], mask=mask)

    valid_actuators = numpy.ones(dm.n_acts, dtype=int)
    valid_actuators[0] = valid_actuators[-1] = 0

    dm.valid_actuators = valid_actuators

    assert dm.n_valid_actuators == (dm.n_acts - 2)

def test_Piezo_valid_actuators():
    """
    Tests that when you set the "valid actuators", the DM doesn't use actuators marked 'invalid'
    """

    config = confParse.loadSoapyConfig(os.path.join(CONFIG_PATH, "sh_8x8.yaml"))

    mask = aotools.circle(config.sim.pupilSize / 2., config.sim.simSize)

    wfs = WFS.ShackHartmann(config, mask=mask)
    dm = DM.FastPiezo(config, n_dm=1, wfss=[wfs], mask=mask)

    act_coord1 = dm.valid_act_coords[0]
    act_coord_last = dm.valid_act_coords[-1]
    act_coord2 = dm.valid_act_coords[1]

    valid_actuators = numpy.ones(dm.n_acts, dtype=int)
    valid_actuators[0] = valid_actuators[-1] = 0

    dm.valid_actuators = valid_actuators

    assert dm.n_valid_actuators == (dm.n_acts - 2)
    assert not numpy.array_equal(dm.valid_act_coords[0], act_coord1)
    assert not numpy.array_equal(dm.valid_act_coords[-1], act_coord_last)
    assert numpy.array_equal(dm.valid_act_coords[0], act_coord2)