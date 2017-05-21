# Third-party
import numpy as np
from numpy.polynomial.polynomial import polyval
from scipy.special import wofz, erf
from mpmath import hyp2f2

__all__ = ['gaussian', 'binned_gaussian',
           'gaussian_polynomial', 'binned_gaussian_polynomial',
           'lorentzian', 'binned_lorentzian',
           'voigt', 'binned_voigt',
           'voigt_polynomial', 'binned_voigt_polynomial',
           'exact_voigt', 'binned_exact_voigt']

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

def binned_gaussian(x, amp, x0, std):
    """
    Gaussian profile integrated over a pixel grid.

    Parameters
    ----------
    x : numeric, array_like
    amp : numeric
        Amplitude of the profile (integral).
    x0 : numeric
        Centroid.
    std : numeric
        Standard of deviation.
    """
    x = np.array(x)

    x_ = np.round(x)
    a = x_ - 0.5
    b = x_ + 0.5

    sqrt2s = np.sqrt(2)*std
    return amp * 0.5 * (erf((x0 - a) / sqrt2s) - erf((x0 - b) / sqrt2s))

# ------------------------------------------------------------------------------

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

def binned_gaussian_polynomial(x, amp, x0, std, bg_coef):
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
    G_val = binned_gaussian(x, amp, x0, std)
    return G_val + polyval(x-x0, np.atleast_1d(bg_coef))

# ------------------------------------------------------------------------------

def lorentzian(x, amp, x0, hwhm):
    """
    1D Lorentzian/Cauchy profile.

    Parameters
    ----------
    x : numeric, array-like
    amp : numeric
        Amplitude of the normalized Gaussian.
    x0 : numeric
        Centroid.
    hwhm : numeric
        Half-width at half-maximum parameter. Sometimes called gamma.
    """
    return amp * hwhm / (np.pi * ((x-x0)**2 + hwhm**2))

def binned_lorentzian(x, amp, x0, hwhm):
    """
    1D Lorentzian/Cauchy profile integrated over a pixel grid.

    Parameters
    ----------
    x : numeric, array-like
    amp : numeric
        Amplitude of the normalized Gaussian.
    x0 : numeric
        Centroid.
    hwhm : numeric
        Half-width at half-maximum parameter. Sometimes called gamma.
    """
    x = np.array(x)
    x_ = np.round(x)
    a = x_ - 0.5
    b = x_ + 0.5
    return (np.arctan((b-x0)/hwhm) - np.arctan((a-x0)/hwhm)) / np.pi

# ------------------------------------------------------------------------------

def _eta_f(amp, x0, std_G, hwhm_L):
    fG = 2 * np.sqrt(2*np.log(2)) * std_G
    fL = 2*hwhm_L
    f = (fG**5 +
         2.69269*fG**4*fL +
         2.42843*fG**3*fL**2 +
         4.47163*fG**2*fL**3 +
         0.07842*fG*fL**4 +
         fL**5) ** (1/5.)

    n = 1.36603 * (fL/f) - 0.47719 * (fL/f)**2 + 0.11116*(fL/f)**3
    return n, f

def voigt(x, amp, x0, std_G, hwhm_L):
    """
    Approximate Voigt profile computed as a mixture of a Gaussian and a
    Lorentzian instead of a convolution. See Wikipedia page on
    `Voigt Profile <https://en.wikipedia.org/wiki/Voigt_profile>`_.

    Parameters
    ----------
    x : numeric, array-like
    amp : numeric
        Amplitude of the normalized Gaussian.
    x0 : numeric
        Centroid.
    std_G : numeric
        Standard of deviation of the Gaussian.
    hwhm_L : numeric
        Half-width at half-maximum of the Lorentzian.
    """
    n, f = _eta_f(amp, x0, std_G, hwhm_L)
    Vx = (n * lorentzian(x, 1., x0, f/2.) +
          (1-n) * gaussian(x, 1., x0, f/(2*np.sqrt(2*np.log(2)))))
    return amp * Vx

def binned_voigt(x, amp, x0, std_G, hwhm_L):
    """
    Approximate Voigt profile computed as a mixture of a Gaussian and a
    Lorentzian instead of a convolution, integrated over a pixel grid. See
    Wikipedia page on `Voigt Profile
    <https://en.wikipedia.org/wiki/Voigt_profile>`_.

    Parameters
    ----------
    x : numeric, array-like
    amp : numeric
        Amplitude of the normalized Gaussian.
    x0 : numeric
        Centroid.
    std_G : numeric
        Standard of deviation of the Gaussian.
    hwhm_L : numeric
        Half-width at half-maximum of the Lorentzian.
    """
    n, f = _eta_f(amp, x0, std_G, hwhm_L)
    Vx = (n * binned_lorentzian(x, 1., x0, f/2.) +
          (1-n) * binned_gaussian(x, 1., x0, f/(2*np.sqrt(2*np.log(2)))))
    return amp * Vx

# ------------------------------------------------------------------------------

def voigt_polynomial(x, amp, x0, std_G, hwhm_L, bg_coef):
    """
    Approximate voigt profile plus a constant background.

    Parameters
    ----------
    x : numeric, array_like
    amp : numeric
        Amplitude of the profile (integral).
    x0 : numeric
        Centroid.
    std_G : numeric
        Standard of deviation of the Gaussian component.
    hwhm_L : numeric
        Half-width at half-maximum of the Lorentzian.
    bg_coef : iterable
        List of polynomial coefficients for the background
        component. The polynomial is evaluated at positions
        relative to the input ``x0``.
    """
    return voigt(x, amp, x0, std_G, hwhm_L) + polyval(x-x0, np.atleast_1d(bg_coef))

def binned_voigt_polynomial(x, amp, x0, std_G, hwhm_L, bg_coef):
    """
    Approximate voigt profile plus a constant background.

    Parameters
    ----------
    x : numeric, array_like
    amp : numeric
        Amplitude of the profile (integral).
    x0 : numeric
        Centroid.
    std_G : numeric
        Standard of deviation of the Gaussian component.
    hwhm_L : numeric
        Half-width at half-maximum of the Lorentzian.
    bg_coef : iterable
        List of polynomial coefficients for the background
        component. The polynomial is evaluated at positions
        relative to the input ``x0``.
    """
    return binned_voigt(x, amp, x0, std_G, hwhm_L) + polyval(x-x0, np.atleast_1d(bg_coef))

# ------------------------------------------------------------------------------

def exact_voigt(x, amp, x0, std_G, hwhm_L):
    """
    Voigt profile - convolution of a Gaussian and Lorentzian.

    When std_G -> 0, the profile approaches a Lorentzian. When fwhm_L=0,
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
    hwhm_L : numeric
        Half-width at half-maximum of the Lorentzian.
    """

    _x = x-x0
    z = (_x + 1j*hwhm_L) / (np.sqrt(2.)*std_G)

    return amp * wofz(z).real / (np.sqrt(2.*np.pi)*std_G)

def _voigt_helper(x, amp, x0, std_G, hwhm_L):
    z = (x - x0 + 1j*hwhm_L) / np.sqrt(2*np.pi*std_G**2)
    hyperg = np.array([complex(hyp2f2(1, 1, 3/2., 2, arg)) for arg in -z**2])
    val = erf(z)/2. + 1j*z**2/np.pi * hyperg
    return amp * val

def binned_exact_voigt(x, amp, x0, std_G, hwhm_L, bg_coef):
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
    hwhm_L : numeric
        HWHM of the Lorentzian component.
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

    V_val = (_voigt_helper(b, amp, x0, std_G, hwhm_L) -
             _voigt_helper(a, amp, x0, std_G, hwhm_L))
    return_val = V_val.real + polyval(x-x0, np.atleast_1d(bg_coef))

    if is_scalar:
        return return_val[0]
    else:
        return return_val
