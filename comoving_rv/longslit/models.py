# Third-party
import numpy as np
from numpy.polynomial.polynomial import polyval
from scipy.special import wofz, erf
from mpmath import hyp2f2

__all__ = ['gaussian', 'gaussian_polynomial', 'integrated_gaussian_polynomial',
           'voigt', 'voigt_constant', 'voigt_polynomial',
           'integrated_voigt_polynomial']

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

def gaussian_polynomial(x, amp, x0, std, bg_coef):
    """
    Gaussian profile plus a polynomial background.

    Parameters
    ----------
    x : numeric, array_like
    amp : numeric
        Amplitude of the profile (integral).
    x0 : numeric
        Centroid.
    std : numeric
        Standard of deviation.
    bg_coef : iterable
        List of polynomial coefficients for the background
        component. The polynomial is evaluated at positions
        relative to the input ``x0``.
    """
    return gaussian(x, amp, x0, std) + polyval(x-x0, np.atleast_1d(bg_coef))

def integrated_gaussian_polynomial(x, amp, x0, std, bg_coef):
    """
    Gaussian profile integrated over a pixel grid plus a polynomial background.

    Parameters
    ----------
    x : numeric, array_like
    amp : numeric
        Amplitude of the profile (integral).
    x0 : numeric
        Centroid.
    std : numeric
        Standard of deviation.
    bg_coef : iterable
        List of polynomial coefficients for the background
        component. The polynomial is evaluated at positions
        relative to the input ``x0``.
    """
    x = np.array(x)

    x_ = np.round(x)
    a = x_ - 0.5
    b = x_ + 0.5

    sqrt2s = np.sqrt(2)*std
    G_val = amp * 0.5 * (erf((x0 - a) / sqrt2s) - erf((x0 - b) / sqrt2s))
    return G_val + polyval(x-x0, np.atleast_1d(bg_coef))

# ------------------------------------------------------------------------------

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

def _voigt_helper(x, amp, x0, std_G, fwhm_L):
    z = (x - x0 + 1j*fwhm_L/2.) / np.sqrt(2*np.pi*std_G**2)
    hyperg = np.array([complex(hyp2f2(1, 1, 3/2., 2, arg)) for arg in -z**2])
    val = erf(z)/2. + 1j*z**2/np.pi * hyperg
    return amp * val

def integrated_voigt_polynomial(x, amp, x0, std_G, fwhm_L, bg_coef):
    """
    Voigt profile integrated over a pixel grid plus a polynomial background.

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
    x = np.array(x)
    if x.shape == ():
        is_scalar = True
    else:
        is_scalar = False
    x = np.atleast_1d(x)

    x_ = np.round(x)
    a = x_ - 0.5
    b = x_ + 0.5

    V_val = (_voigt_helper(b, amp, x0, std_G, fwhm_L) -
             _voigt_helper(a, amp, x0, std_G, fwhm_L))
    return_val = V_val.real + polyval(x-x0, np.atleast_1d(bg_coef))

    if is_scalar:
        return return_val[0]
    else:
        return return_val
