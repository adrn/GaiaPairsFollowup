# Third-party
import numpy as np
from numpy.polynomial.polynomial import polyval
from scipy.special import wofz

__all__ = ['gaussian', 'voigt', 'voigt_constant', 'voigt_polynomial']

def gaussian(x, amp=1., x0=0., std=1.):
    """
    1D Gaussian profile.

    Parameters
    ----------
    x : numeric, array-like
    amp : numeric
        Amplitude of the normalized Gaussian.
    x0 : numeric
        Centroid.
    std : numeric
        Standard of deviation.
    """
    return amp/np.sqrt(2*np.pi)/std * np.exp(-0.5 * (x-x0)**2/std**2)

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
    std_G : numeric
        Standard of deviation of the Gaussian component.
    fwhm_L : numeric
        FWHM of the Lorentzian component.
    """

    _x = x-x0
    z = (_x + 1j*L_fwhm/2.) / (np.sqrt(2.)*G_std)

    return amp * wofz(z).real / (np.sqrt(2.*np.pi)*G_std)

def voigt_constant(x, amp, x0, std_G, fwhm_L, C):
    """
    Voigt profile plus a constant background.

    Parameters
    ----------
    x : numeric, array_like
    amp : numeric
        Amplitude of the profile (integral).
    x0 : numeric
        Centroid.
    std_G : numeric
        Standard of deviation of the Gaussian component.
    fwhm_L : numeric
        FWHM of the Lorentzian component.
    C : numeric
        Background
    """
    return voigt(x, amp, x0, std_G, fwhm_L) + C

def voigt_polynomial(x, amp, x0, std_G, fwhm_L, bg_coef):
    """
    Voigt profile plus a constant background.

    Parameters
    ----------
    x : numeric, array_like
    amp : numeric
        Amplitude of the profile (integral).
    x0 : numeric
        Centroid.
    std_G : numeric
        Standard of deviation of the Gaussian component.
    fwhm_L : numeric
        FWHM of the Lorentzian component.
    bg_coef : iterable
        List of polynomial coefficients for the background
        component. The polynomial is evaluated at positions
        relative to the input ``x0``.
    """
    return voigt(x, amp, x0, std_G, fwhm_L) + polyval(x-x0, np.atleast_1d(bg_coef))
