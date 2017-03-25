# Standard library
from collections import OrderedDict

# Third-party
import numpy as np
from scipy.optimize import minimize, leastsq
from scipy.stats import scoreatpercentile
from scipy.signal import argrelmin, argrelmax

# Project
from .models import voigt_polynomial

__all__ = ['fit_emission_line']

def errfunc(p, pix, flux, flux_ivar):
    amp, x_0, std_G, fwhm_L, *bg_coef = p
    return (flux - voigt_polynomial(pix, amp, x_0, std_G, fwhm_L, bg_coef)) * np.sqrt(flux_ivar)

def fit_emission_line(pix, flux, flux_ivar=None,
                      amp0=None, x0=None, std_G0=None, fwhm_L0=None, n_bg_coef=1):
    """
    TODO: maybe deprecated? see fit_spec_line() below

    Parameters
    ----------
    pix : array_like
        Must be the same shape as ``flux``.
    flux : array_like
        Must be the same shape as ``pix_grid``.
    amp0 : numeric (optional)
        Initial guess for line amplitude.
    x0 : numeric (optional)
        Initial guess for line centroid.
    n_bg_coef : int
        Number of terms in the background polynomial fit.
    """

    if x0 is None: # then estimate the initial guess for the centroid
        x0 = pix[np.argmax(flux)]

    bg0 = np.array([0.] * n_bg_coef)
    bg0[0] = scoreatpercentile(flux[flux>0], 5.)

    if std_G0 is None:
        std_G0 = 2. # MAGIC NUMBER

    if fwhm_L0 is None:
        fwhm_L0 = 0.5 # MAGIC NUMBER

    int_ctrd0 = int(round(x0-pix.min()))
    if amp0 is None: # then estimate the initial guess for amplitude
        amp0 = flux[int_ctrd0] * np.sqrt(2*np.pi*std_G0) # flux at initial guess

    if flux_ivar is None:
        flux_ivar = 1.

    p0 = [amp0, x0, std_G0, fwhm_L0] + bg0.tolist()
    p_opt,p_cov,*_,mesg,ier = leastsq(errfunc, p0, args=(pix, flux, flux_ivar),
                                      full_output=True, maxfev=10000, ftol=1E-11, xtol=1E-11)

    fit_amp, fit_x0, fit_std_G, fit_fwhm_L, *fit_bg = p_opt

    fail_msg = "Fitting spectral line in comp lamp spectrum failed. {msg}"

    if ier < 1 or ier > 4:
        raise RuntimeError(fail_msg.format(msg=mesg))

    if fit_x0 < min(pix) or fit_x0 > max(pix):
        raise ValueError(fail_msg.format(msg="Unphysical peak centroid: {:.3f}".format(fit_x0)))

    return dict(amp=fit_amp, x_0=fit_x0,
                std_G=fit_std_G, fwhm_L=fit_fwhm_L,
                bg_coef=fit_bg)

def fit_spec_line(x, flux, flux_ivar=None,
                  amp0=None, x0=None, std_G0=None, fwhm_L0=None,
                  bg0=None, n_bg_coef=2, target_x=None,
                  absorp_emiss=1.):
    """
    Assumes that the input arrays are trimmed to be a sensible region around
    the line of interest.

    Parameters
    ----------
    x : array_like
        Pixel or wavelength. Must be the same shape as ``flux``.
    flux : array_like
        Must be the same shape as ``pix_grid``.
    amp0 : numeric (optional)
        Initial guess for line amplitude.
    x0 : numeric (optional)
        Initial guess for line centroid.
    n_bg_coef : int
        Number of terms in the background polynomial fit.
    absorp_emiss : float
        -1 for absorption line
        +1 for emission line
    """

    sort_idx = np.argsort(x)
    x = np.array(x)[sort_idx]
    flux = np.array(flux)[sort_idx]
    if flux_ivar is not None:
        flux_ivar = np.array(flux_ivar)[sort_idx]
    else:
        flux_ivar = np.ones_like(flux)

    if x0 is None: # estimate the initial guess for the centroid
        relmins = argrelmin(-absorp_emiss*flux)[0]
        if len(relmins) > 1 and target_x is None:
            raise ValueError("If auto-finding x0, must supply a target value for x.")
        elif len(relmins) == 1:
            x0 = x[relmins[0]]
        else:
            x0_idx = relmins[np.abs(x[relmins] - target_x).argmin()]
            x0 = x[x0_idx]

    # shift x array so that line is approximately at 0
    _x = np.array(x, copy=True)
    x = np.array(_x) - x0

    # background polynomial parameters
    if bg0 is None:
        if n_bg_coef < 2:
            bg0 = np.array([0.])
            bg0[0] = np.median(flux)

        else:
            # estimate linear background model
            bg0 = np.array([0.] * n_bg_coef)
            bg0[1] = (flux[-1]-flux[0])/(x[-1]-x[0]) # slope
            bg0[0] = flux[-1] - bg0[1]*x[-1] # estimate constant term

    else:
        if len(bg0) != n_bg_coef:
            n_bg_coef = len(bg0)

    if std_G0 is None:
        std_G0 = 2. # MAGIC NUMBER

    if fwhm_L0 is None:
        fwhm_L0 = 0.5 # MAGIC NUMBER

    if amp0 is None: # then estimate the initial guess for amplitude
        _i = np.argmin(np.abs(x))
        amp0 = np.sqrt(2*np.pi) * (flux[_i] - (bg0[0] + bg0[1]*x[_i]))

    print([amp0, 0, std_G0, fwhm_L0, bg0.tolist()])

    p0 = [amp0, 0., std_G0, fwhm_L0] + bg0.tolist()
    p_opt,p_cov,*_,mesg,ier = leastsq(errfunc, p0, args=(x, flux, flux_ivar),
                                      full_output=True, maxfev=1000000,
                                      ftol=1E-10, xtol=1E-10)

    fit_amp, fit_x0, fit_std_G, fit_fwhm_L, *fit_bg = p_opt
    fit_x0 = fit_x0 + x0

    fail_msg = "Fitting spectral line in comp lamp spectrum failed. {msg}"

    if ier < 1 or ier > 4:
        raise RuntimeError(fail_msg.format(msg=mesg))

    if fit_x0 < min(_x) or fit_x0 > max(_x):
        raise ValueError(fail_msg.format(msg="Unphysical peak centroid: {:.3f}".format(fit_x0)))

    return OrderedDict(amp=fit_amp, x_0=fit_x0,
                       std_G=fit_std_G, fwhm_L=fit_fwhm_L,
                       bg_coef=fit_bg)
