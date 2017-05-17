# Third-party
import numpy as np
from numpy.polynomial.polynomial import polyval
from celerite.modeling import Model
from celerite import terms, GP

# Project
from comoving_rv.log import logger

__all__ = ['fit_all_lines', 'MeanModel', 'GPModel']

def fit_all_lines(pixels, flux, flux_ivar, line_waves, line_pixels, half_width=5):
    """
    Given a wavelength guess (a list of rough pixel-wavelength correspondences),
    measure more precise line centroids to then fit for a wavelength
    calibratrion function.

    Parameters
    ----------
    pixels : array_like
        Pixel array for a 1D comp. lamp spectrum.
    flux : array_like
        Flux array for a 1D comp. lamp spectrum.
    flux_ivar : array_like
        Inverse-variance array for a 1D comp. lamp spectrum.
    line_waves : array_like
        List of wavelength values for emission lines in a comp. lamp spectrum.
    line_pixels : array_like
        List of pixel centroids for emission lines in a comp. lamp spectrum.
    half_width : int
        Number of pixels on either side of line
    """

    _idx = np.argsort(line_waves)
    wvln = np.array(line_waves)[_idx]
    pixl = np.array(line_pixels)[_idx]

    fit_centroids = []

    for pix_ctr,wave in zip(pixl, wvln):

        logger.debug("Fitting line at predicted pix={:.2f}, λ={:.2f}"
                     .format(pix_ctr, wave))

        # indices for region around line
        i1 = int(np.floor(pix_ctr-half_width))
        i2 = int(np.ceil(pix_ctr+half_width)) + 1

        # recenter window
        i0 = i1 + flux[i1:i2].argmax()
        i1 = int(np.floor(i0-half_width))
        i2 = int(np.ceil(i0+half_width)) + 1

        _pixl = pixels[i1:i2]
        _flux = flux[i1:i2]
        _ivar = flux_ivar[i1:i2]

        # instead of doing anything fancy (e.g., fitting a profile), just
        # estimate the mean...
        x0 = np.sum(_pixl * _flux**2 * _ivar) / np.sum(_flux**2 * _ivar)
        fit_centroids.append(x0)

    return np.array(fit_centroids)

class MeanModel(Model):

    def __init__(self, n_bg_coef, **p0):
        self._n_bg_coef = n_bg_coef
        self.parameter_names = (["a{}".format(i) for i in range(n_bg_coef)])
        super(MeanModel, self).__init__(**p0)

    def get_value(self, x):
        return polyval(x, np.atleast_1d([getattr(self, "a{}".format(i))
                                         for i in range(self._n_bg_coef)]))

class GPModel(object):

    def __init__(self, x, y, n_bg_coef, wave_err=0.05, # MAGIC NUMBER: wavelength error hack
                 log_sigma0=0., log_rho0=np.log(10.), # initial params for GP
                 x_shift=None):

        self.x = np.array(x)
        self.y = np.array(y)

        if n_bg_coef >= 2:
            a_kw = dict([('a{}'.format(i),0.) for i in range(2,n_bg_coef)])

            # estimate background
            a_kw['a1'] = (y[-1]-y[0])/(x[-1]-x[0]) # slope
            a_kw['a0'] = y[-1] - a_kw['a1']*x[-1] # estimate constant term

        else:
            a_kw = dict(a0=np.mean([y[0], y[-1]]))

        # initialize model
        self.mean_model = MeanModel(n_bg_coef=n_bg_coef, **a_kw)
        self.kernel = terms.Matern32Term(log_sigma=log_sigma0, log_rho=log_rho0)

        # set up the gp
        self.gp = GP(self.kernel, mean=self.mean_model, fit_mean=True)
        self.gp.compute(x, yerr=wave_err)
        logger.debug("Initial log-likelihood: {0}"
                     .format(self.gp.log_likelihood(y)))

        if x_shift is None:
            self.x_shift = 0.
        else:
            self.x_shift = x_shift

    def neg_ln_like(self, params):
        # minimize -log(likelihood)
        self.gp.set_parameter_vector(params)
        ll = self.gp.log_likelihood(self.y)
        if np.isnan(ll):
            return np.inf
        return -ll

    def __call__(self, params):
        return self.neg_ln_like(params)
