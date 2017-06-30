"""
Turbulence gradient temporal power spectra calculation and plotting

:author: Andrew Reeves
:date: September 2016
"""

import numpy
from matplotlib import pyplot
from scipy import optimize


def calc_slope_temporalps(slope_data):
    """
    Calculates the temporal power spectra of the loaded centroid data.

    Calculates the Fourier transform over the number frames of the centroid
    data, then returns the square of the  mean of all sub-apertures, for x
    and y. This is a temporal power spectra of the slopes, and should adhere
    to a -11/3 power law for the slopes in the wind direction, and -14/3 in
    the direction tranverse to the wind direction. See Conan, 1995 for more.

    The phase slope data should be split into X and Y components, with all X data first, then Y (or visa-versa).

    Parameters:
        slope_data (ndarray): 2-d array of shape (..., nFrames, nCentroids)

    Returns:
        ndarray: The temporal power spectra of the data for X and Y components.
    """

    n_frames = slope_data.shape[-2]

    # Only take half result, as FFT mirrors
    tps = abs(numpy.fft.fft(slope_data, axis=-2)[..., :n_frames/2, :])**2

    # Find mean across all sub-aps
    tps = (abs(tps)**2)
    mean_tps = tps.mean(-1)
    tps_err = tps.std(-1)/numpy.sqrt(tps.shape[-1])

    return mean_tps, tps_err


def get_tps_time_axis(frame_rate, n_frames):
    """
    Parameters:
        frame_rate (float): Frame rate of detector observing slope gradients (Hz)
        n_frames: (int): Number of frames used for temporal power spectrum

    Returns:
        ndarray: Time values for temporal power spectra plots
    """

    t_vals = numpy.fft.fftfreq(n_frames, 1./frame_rate)[:n_frames/2.]

    return t_vals



def plot_tps(slope_data, frame_rate):
    """
    Generates a plot of the temporal power spectrum/a for a data set of phase gradients

    Parameters:
        slope_data (ndarray):  2-d array of shape (..., nFrames, nCentroids)
        frame_rate (float): Frame rate of detector observing slope gradients (Hz)

    Returns:
        tuple: The computed temporal power spectrum/a, and the time axis data

    """
    n_frames = slope_data.shape[-2]

    tps, tps_err = calc_slope_temporalps(slope_data)

    t_axis_data = get_tps_time_axis(frame_rate, n_frames)

    fig = pyplot.figure()
    ax = fig.add_subplot(111)

    # plot each power spectrum
    for i_ps, ps in enumerate(tps):
        ax.loglog(t_axis_data, ps, label="Spectrum {}".format(i_ps))

    ax.legend()

    ax.set_xlabel("Frequency (Hz)")
    ax.set_ylabel("Power (arbitrary units)")

    pyplot.show()

    return tps, tps_err, t_axis_data

def fit_tps(tps, t_axis, D, V_guess=10, f_noise_guess=20, A_guess=9, tps_err=None, plot=False):
    """
    Runs minimization routines to get t0.

    Parameters:
        tps (ndarray): The temporal power spectrum to fit
        axis (str): fit parallel ('par') or perpendicular ('per') to wind direction
        D (float): (Sub-)Aperture diameter

    Returns:
    """
    if plot:
        fig = pyplot.figure()

    opt_result = optimize.minimize(
            test_tps_fit_minimize_func,
            x0=(V_guess, f_noise_guess, A_guess),
            args=(tps, t_axis, D, tps_err, plot),
            method="COBYLA")

    print(opt_result)


def test_tps_fit(tps, t_axis_data, D, V, f_noise, A=1, tps_err=None, plot=False):
    """
    Tests the validaty of a fit to the temporal power spectrum.

    Uses the temporal power spectrum and time-axis data to test the validity of a coherence time. A frequency above which fitting is not performaed should also be given, as noise will be the dominant contributor above this.

    Parameters:
        tps (ndarray): Temporal power spectrum to fit
        t_axis_data (ndarray): Time axis data
        D (float): (sub-) Aperture diameter
        V (float): Integrated wind speed
        f_noise (float): Frequency above which noise dominates.
        A (float): Initial Guess of
    """
    f0 = 0.3 * V/D

    if f0<t_axis_data[0] or f0>f_noise or f_noise>t_axis_data.max():
        return 10**99

    tps_tt_indices = numpy.where((t_axis_data<f0) & (t_axis_data>0))[0]
    tt_t_data = t_axis_data[tps_tt_indices]
    tt_fit = 10**A * tt_t_data**(-2./3)

    # get scaling for next part of distribution so it matches up at cutof freq.
    tps_ho_indices = numpy.where((t_axis_data>f0) & (t_axis_data<f_noise))
    ho_t_data = t_axis_data[tps_ho_indices]
    B = tt_fit[-1]/(ho_t_data[0] ** (-11./3))
    ho_fit = B * ho_t_data ** (-11./3)

    ps_fit = numpy.append(tt_fit, ho_fit)
    fit_t_data = numpy.append(tt_t_data, ho_t_data)
    fit_t_coords = numpy.append(tps_tt_indices, tps_ho_indices)


    if tps_err is None:
        err = numpy.mean((numpy.sqrt(numpy.square(ps_fit - tps[fit_t_coords]))))
    else:
        chi2 = numpy.square((ps_fit - tps[fit_t_coords])/tps_err[fit_t_coords]).sum()
        err = chi2/ps_fit.shape[0]



    if plot:
        print("V: {}, f_noise: {}, A: {}".format(V, f_noise, A))
        pyplot.cla()
        ax = pyplot.gca()
        ax.set_xscale('log')
        ax.set_yscale('log')
        if tps_err is None:
            ax.loglog(t_axis_data, tps)
        else:
            ax.errorbar(t_axis_data, tps, tps_err)
        ax.plot(fit_t_data, ps_fit, linewidth=2)

        ax.plot([f0]*2, [0.1, tps.max()], color='k')
        pyplot.draw()
        pyplot.pause(0.01)
        print("Error Func: {}".format(err))

    return err

def test_tps_fit_minimize_func(args, tps, t_axis, D, tps_err=None, plot=False):

    V, f_noise, A = args

    return test_tps_fit(tps, t_axis, D=D, V=V, f_noise=f_noise, A=A, tps_err=None, plot=plot)


if __name__ == "__main__":

    from astropy.io import fits

    data = fits.getdata("t0_5ms_canary_slopes.fits")

    tps, tps_err, t_data = plot_tps(data, 150)
    fit_tps(tps[0], t_data, D=4.2/7, V_guess=20, f_noise_guess=50, A_guess=9, tps_err=tps_err[0], plot=True)