"""
TODO:
-
"""

# Third-party
import numpy as np
from numpy.polynomial.polynomial import polyval
from celerite.modeling import Model
from celerite import terms, GP

# Project
from comoving_rv.log import logger

__all__ = ['MeanModel', 'GPModel']

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
