# Third-party
import numpy as np
from scipy.special import wofz

__all__ = ['voigt']

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
