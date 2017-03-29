# Third-party
import numpy as np
from numpy.polynomial.polynomial import polyval
from scipy.special import wofz

__all__ = ['gaussian_1d', 'voigt', 'voigt_constant', 'voigt_polynomial']

def gaussian_1d(x, amp=1., x0=0., stddev=1.):
    return amp/np.sqrt(2*np.pi)/stddev * np.exp(-0.5 * (x-x0)**2/stddev**2)

def voigt(x, amp, x0, G_std, L_fwhm):
    """
    Voigt profile - convolution of a Gaussian and Lorentzian.

    When G_std -> 0, the profile approaches a Lorentzian. When L_fwhm=0,
    the profile is a Gaussian.

    Parameters
    ----------
    x : numeric, array_like
    amp : numeric
        Amplitude of the profile (integral).
    x0 : numeric
        Centroid.
    G_std : numeric
        Standard of deviation of the Gaussian component.
    L_fwhm : numeric
        FWHM of the Lorentzian component.
    """

    _x = x-x0
    z = (_x + 1j*L_fwhm/2.) / (np.sqrt(2.)*G_std)

    return amp * wofz(z).real / (np.sqrt(2.*np.pi)*G_std)

def voigt_constant(x, amp, x_0, std_G, fwhm_L, C):
    """
    Voigt profile plus a constant background.

    Parameters
    ----------
    x : numeric, array_like
    amp : numeric
        Amplitude of the profile (integral).
    x0 : numeric
        Centroid.
    G_std : numeric
        Standard of deviation of the Gaussian component.
    L_fwhm : numeric
        FWHM of the Lorentzian component.
    C : numeric
        Background
    """
    return voigt(x, amp, x_0, std_G, fwhm_L) + C

def voigt_polynomial(x, amp, x_0, std_G, fwhm_L, bg_coef):
    """
    Voigt profile plus a constant background.

    Parameters
    ----------
    x : numeric, array_like
    amp : numeric
        Amplitude of the profile (integral).
    x0 : numeric
        Centroid.
    G_std : numeric
        Standard of deviation of the Gaussian component.
    L_fwhm : numeric
        FWHM of the Lorentzian component.
    bg_coef : iterable
        List of polynomial coefficients.
    """
    return voigt(x, amp, x_0, std_G, fwhm_L) + polyval(x-x_0, np.atleast_1d(bg_coef))
