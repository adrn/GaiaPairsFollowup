# Third-party
import matplotlib.pyplot as plt
import numpy as np
from numpy.polynomial.polynomial import polyval
from celerite.modeling import Model
from celerite import terms, GP
from celerite.solver import LinAlgError

# Project
from .fitting.gaussian import GaussianLineFitter
from .utils import extract_region
from ..log import logger

__all__ = ['fit_all_lines', 'MeanModel', 'GPModel']

def fit_all_lines(pixels, flux, flux_ivar, line_waves, line_pixels,
                  half_width=4):
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

        x_, (flux_, ivar_) = extract_region(pixels, center=pix_ctr,
                                            width=2*half_width,
                                            arrs=[flux, flux_ivar])

        lf = GaussianLineFitter(x_, flux_, ivar_, absorp_emiss=1.)
        try:
            lf.fit()
        except LinAlgError:
            # lf.init_gp()

            # fig = lf.plot_fit()
            # ax = fig.axes[1]
            # ax.plot(x_, flux_, drawstyle='steps-mid', marker='', linestyle='-')
            # ax.errorbar(x_, flux_, 1/np.sqrt(ivar_),
            #             marker='', linestyle='none', zorder=-1, alpha=0.5)

            # fig.tight_layout()
            # plt.show()
            logger.warning("Failed to fit line! Skipping...but you should be "
                           "careful if many fits fail, you might have a bad "
                           "comparison lamp spectrum.")
            lf.success = False

        # TODO: need to get plot path into here
        # fig = lf.plot_fit()
        # fig.suptitle(r'$\lambda={:.2f}\,\AA$'.format(wave), y=0.95)
        # fig.subplots_adjust(top=0.9)
        # fig.savefig()
        # plt.close(fig)

        fit_pars = lf.get_gp_mean_pars()
        if (not lf.success or
                abs(fit_pars['x0']-pix_ctr) > 8 or
                abs(fit_pars['amp']) < 10):
            x0 = np.nan # failed

        else:
            x0 = fit_pars['x0']

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
        logger.log(0, "Initial log-likelihood: {0}"
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
