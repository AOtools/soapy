import soapy
import numpy
import os
from astropy.io import fits
CONFIG_PATH = os.path.join(os.path.dirname(os.path.abspath(__file__)), "../conf/")

import shutil

def test_save_imat():
    sim = soapy.Sim(os.path.join(CONFIG_PATH, "sh_8x8.yaml"))
    sim.config.sim.simName = "test_sim"
    sim.config.sim.logfile = None

    # 1 scrn for fast init
    sim.config.atmos.scrnNo = 1
    try:
        sim.aoinit()

        recon = sim.recon

        recon.makeIMat()
        recon.save_interaction_matrix()

        imat_filename = "test_sim/iMat.fits"

        imat = fits.getdata(imat_filename)

        assert (numpy.array_equal(imat, sim.recon.interaction_matrix))

    finally:
        shutil.rmtree("test_sim")




def test_load_imat():
    sim = soapy.Sim(os.path.join(CONFIG_PATH, "sh_8x8.yaml"))
    sim.config.sim.simName = "test_sim"
    sim.config.sim.logfile = None

    # 1 scrn for fast init
    sim.config.atmos.scrnNo = 1

    try:

        sim.aoinit()

        recon = sim.recon

        # Make an imat
        recon.makeIMat()

        # Save it for later
        recon.save_interaction_matrix()
        imat = recon.interaction_matrix.copy()

        # Set the internat soapy imat to 0
        recon.interaction_matrix[:] = 0

        # And attempt to load saved one
        recon.load_interaction_matrix()

        # Ensure its been loaded as expected
        assert (numpy.array_equal(imat, recon.interaction_matrix))

    finally:
        shutil.rmtree("test_sim")