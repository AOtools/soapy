import unittest

import soapy

import numpy
from astropy.io import fits

import os
import shutil
CONFIG_PATH = os.path.join(os.path.dirname(os.path.abspath(__file__)), "../conf/")

soapy.logger.setLoggingLevel(3)

RESULTS = {
        "8x8": 0.5,
        "8x8_open": 0.5,
        "8x8_offAxis": 0.22,
        "8x8_zernike": 0.36,
        "8x8_lgs"    : 0.45,
        "8x8_phys": 0.50,
        "8x8_lgsuplink":0.35,
        }


class TestSimpleSCAO(unittest.TestCase):

    def testOnAxis(self):
        sim = soapy.Sim(os.path.join(CONFIG_PATH, "sh_8x8.yaml"))
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
        sim = soapy.Sim(os.path.join(CONFIG_PATH, "sh_8x8.yaml"))
        sim.config.sim.simName = None
        sim.config.sim.logfile = None
        sim.config.sim.nIters = 100
        sim.config.wfss[0].GSPosition=(0,0)
        sim.config.wfss[0].propagationMode="Physical"

        sim.aoinit()

        sim.makeIMat(forceNew=True)

        sim.aoloop()

        #Check results are ok
        assert numpy.allclose(sim.longStrehl[0,-1], RESULTS["8x8_phys"], atol=0.2)

    def testOffAxis(self):
        sim = soapy.Sim(os.path.join(CONFIG_PATH, "sh_8x8.yaml"))
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
        sim = soapy.Sim(os.path.join(CONFIG_PATH, "sh_8x8.yaml"))
        sim.config.sim.simName = None
        sim.config.sim.logfile = None
        sim.config.sim.nIters = 100
        sim.config.wfss[0].GSPosition = (0,0)

        sim.config.sim.nDM = 1
        sim.config.dms[0].type = "Zernike"
        sim.config.dms[0].nxActuators = 45
        sim.config.dms[0].svdConditioning = 0.01
        sim.config.dms[0].iMatValue = 100

        sim.aoinit()

        sim.makeIMat(forceNew=True)

        sim.aoloop()

        #Check results are ok
        assert numpy.allclose(
                sim.longStrehl[0,-1], RESULTS["8x8_zernike"], atol=0.2)


    def testCone(self):

        sim = soapy.Sim(os.path.join(CONFIG_PATH, "sh_8x8_lgs.yaml"))
        sim.config.sim.simName= None
        sim.config.sim.logfile = None
        sim.config.sim.nIters = 100

        sim.aoinit()

        sim.makeIMat(forceNew=True)

        sim.aoloop()

        #Check results are ok
        assert numpy.allclose(
                sim.longStrehl[0,-1], RESULTS["8x8_lgs"], atol=0.2)

    def testOpenLoop(self):
        sim = soapy.Sim(os.path.join(CONFIG_PATH, "sh_8x8_openloop.yaml"))
        sim.config.sim.simName = None
        sim.config.sim.logfile = None
        sim.config.sim.nIters = 100
        sim.config.wfss[0].GSPosition=(0,0)

        for i in range(sim.config.sim.nDM):
            sim.config.dms[i].closed = False

        sim.aoinit()

        sim.makeIMat(forceNew=True)

        sim.aoloop()

        #Check results are ok
        assert numpy.allclose(sim.longStrehl[0,-1], RESULTS["8x8_open"], atol=0.2)

    def testLgsUplink_phys(self):
        sim = soapy.Sim(os.path.join(CONFIG_PATH, "sh_8x8_lgs-uplink.yaml"))
        sim.config.sim.simName = None
        sim.config.sim.logfile = None
        sim.config.sim.nIters = 100
        sim.config.wfss[0].GSPosition = (0, 0)
        sim.config.wfss[1].GSPosition = (0, 0)
        sim.config.wfss[1].lgs.propagationMode = "Physical"
        sim.aoinit()

        sim.makeIMat(forceNew=True)

        sim.aoloop()

        #Check results are ok
        assert numpy.allclose(sim.longStrehl[0,-1], RESULTS["8x8_lgsuplink"], atol=0.2)

    def testLgsUplink_geo(self):
        sim = soapy.Sim(os.path.join(CONFIG_PATH, "sh_8x8_lgs-uplink.yaml"))
        sim.config.sim.simName = None
        sim.config.sim.logfile = None
        sim.config.sim.nIters = 100
        sim.config.wfss[0].GSPosition = (0, 0)
        sim.config.wfss[1].GSPosition = (0, 0)
        sim.config.wfss[1].lgs.propagationMode = "Geometric"
        sim.aoinit()

        sim.makeIMat(forceNew=True)

        sim.aoloop()

        #Check results are ok
        assert numpy.allclose(sim.longStrehl[0,-1], RESULTS["8x8_lgsuplink"], atol=0.2)


    def testSaveData(self):
        sim = soapy.Sim(os.path.join(CONFIG_PATH, "sh_8x8.yaml"))
        sim.config.sim.simName = 'test_sh8x8'
        sim.config.sim.logfile = False

        sim.config.sim.saveSlopes = True
        sim.config.sim.saveDmCommands = True
        sim.config.sim.saveLgsPsf = True
        sim.config.sim.saveWfe = True
        sim.config.sim.saveStrehl = True
        sim.config.sim.saveSciPsf = True
        sim.config.sim.saveInstPsf = True
        sim.config.sim.saveCalib = True
        sim.config.sim.saveWfsFrames = True

        sim.config.sim.saveSciRes = False
        sim.config.sim.saveInstScieField = False

        sim.config.sim.nIters = 2
        wdir = os.path.dirname(os.path.abspath(__file__)) + '/'

        sim.aoinit()
        sim.makeIMat()
        sim.aoloop()

        try:
            assert os.path.isfile(wdir + sim.path + '/slopes.fits') &\
                os.path.isfile(wdir + sim.path + '/dmCommands.fits') &\
                os.path.isfile(wdir + sim.path + '/lgsPsf.fits') &\
                os.path.isfile(wdir + sim.path + '/WFE.fits') &\
                os.path.isfile(wdir + sim.path + '/instStrehl.fits') &\
                os.path.isfile(wdir + sim.path + '/longStrehl.fits') &\
                os.path.isfile(wdir + sim.path + '/sciPsf_00.fits') &\
                os.path.isfile(wdir + sim.path + '/sciPsfInst_00.fits') &\
                os.path.isfile(wdir + sim.path + '/iMat.fits') &\
                os.path.isfile(wdir + sim.path + '/cMat.fits') &\
                os.path.isfile(wdir + sim.path + '/wfsFPFrames/wfs-0_frame-0.fits')
                # os.path.isfile(wdir + sim.path + '/sciResidual_00.fits') &\
                # os.path.isfile(wdir + sim.path + '/scieFieldInst_00_real.fits') &\
                # os.path.isfile(wdir + sim.path + '/scieFieldInst_00_imag.fits') &\
        except:
            raise
        finally:
            dd = os.path.dirname(sim.path)
            shutil.rmtree(wdir + dd)


def testMaskLoad():

    sim = soapy.Sim(os.path.join(CONFIG_PATH, "sh_8x8.yaml"))
    sim.config.sim.simName = None
    sim.config.sim.logfile = None

    sim.aoinit()

    mask = numpy.ones((sim.config.sim.pupilSize, sim.config.sim.pupilSize))

    # save mask
    if os.path.isfile('testmask.fits'):
        os.remove('testmask.fits')
    
    hdu = fits.PrimaryHDU(mask)
    hdulist = fits.HDUList([hdu])
    hdulist.writeto('testmask.fits')
    hdulist.close()
    
    try:
        # attempt to load it
        sim.config.tel.mask = 'testmask.fits'
        sim.aoinit()

        # check its good
        p = sim.config.sim.simPad
        pad_mask = numpy.pad(mask, mode="constant", pad_width=((p,p),(p,p)))
        assert numpy.array_equal(sim.mask, pad_mask)
    except:
        raise
    finally:
        os.remove('testmask.fits') 

if __name__ == '__main__':
    unittest.main()
