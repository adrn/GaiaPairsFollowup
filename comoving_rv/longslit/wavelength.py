# Third-party
import numpy as np
from scipy.optimize import minimize, leastsq
from scipy.stats import scoreatpercentile

# Project
from .models import voigt_polynomial

__all__ = ['fit_emission_line']

def errfunc(p, pix, flux, flux_ivar):
    amp, x_0, std_G, fwhm_L, *bg_coef = p
    return (voigt_polynomial(pix, amp, x_0, std_G, fwhm_L, bg_coef) - flux) * np.sqrt(flux_ivar)

def fit_emission_line(pix, flux, flux_ivar=None,
                      amp0=None, x0=None, std_G0=None, fwhm_L0=None, n_bg_coef=1):
    """
    TODO:

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

    int_ctrd0 = int(round(x0-pix.min()))
    if amp0 is None: # then estimate the initial guess for amplitude
        amp0 = flux[int_ctrd0] # flux at initial guess

    bg0 = np.array([0.] * n_bg_coef)
    bg0[0] = scoreatpercentile(flux[flux>0], 5.)

    if std_G0 is None:
        std_G0 = 2. # MAGIC NUMBER

    if fwhm_L0 is None:
        fwhm_L0 = 0.5 # MAGIC NUMBER

    if flux_ivar is None:
        flux_ivar = 1.

    p0 = [amp0, x0, std_G0, fwhm_L0] + bg0.tolist()
    print(p0)
    p_opt,p_cov,*_,mesg,ier = leastsq(errfunc, p0, args=(pix, flux, flux_ivar),
                                      full_output=True)
    print(p_opt)

    # res = minimize(_errfunc, x0=p0, args=(pix_grid, flux, flux_ivar))
    # p = res.x

    fit_amp, fit_x0, fit_std_G, fit_fwhm_L, *fit_bg = p_opt

    fail_msg = "Fitting spectral line in comp lamp spectrum failed. {msg}"

    if ier < 1 or ier > 4:
        raise RuntimeError(fail_msg.format(msg=mesg))

    if fit_x0 < min(pix) or fit_x0 > max(pix):
        raise ValueError(fail_msg.format(msg="Unphysical peak centroid: {:.3f}".format(fit_x0)))

    return dict(amp=fit_amp, x_0=fit_x0,
                std_G=fit_std_G, fwhm_L=fit_fwhm_L,
                bg_coef=fit_bg)
