# Standard library
from collections import OrderedDict
import inspect

# Third-party
from celerite import terms, GP
from celerite.solver import LinAlgError
import matplotlib.pyplot as plt
import numpy as np
from scipy.optimize import minimize
from scipy.signal import argrelmin

# Project
from ..utils import argmedian
from ...log import logger

__all__ = ['LineFitter']

class LineFitter(object):

    def __init__(self, x, flux, ivar=None,
                 absorp_emiss=None, n_bg_coef=2, target_x=None):
        """

        Parameters
        ----------
        x : array_like
            Dispersion parameter. Usually either pixel or wavelength.
        flux : array_like
            Flux array.
        ivar : array_like
            Inverse-variance array.
        absorp_emiss : numeric [-1, 1] (optional)
            If -1, absorption line. If +1, emission line. If not specified,
            will try to guess.
        n_bg_coef : int (optional)
            The order of the polynomial used to fit the background.
            1 = constant, 2 = linear, 3 = quadratic, etc.
        target_x : numeric (optional)
            Where we think the line of interest is.
        """
        self.x = np.array(x)
        self.flux = np.array(flux)

        if ivar is not None:
            self.ivar = np.array(ivar)
            self._err = 1/np.sqrt(self.ivar)

        else:
            self.ivar = np.ones_like(flux)
            self._err = None

        if absorp_emiss is None:
            raise NotImplementedError("You must supply an absorp_emiss for now")
        self.absorp_emiss = float(absorp_emiss)
        self.target_x = target_x

        self.n_bg_coef = int(n_bg_coef)

        # result of fitting
        self.gp = None
        self.success = None

    @classmethod
    def transform_pars(cls, p):
        new_p = OrderedDict()
        for k,v in p.items():
            if k in cls._log_pars:
                k = 'ln_{0}'.format(k)
                v = np.log(v)
            new_p[k] = v
        return new_p

    def get_init_generic(self, amp=None, x0=None, bg=None, bg_buffer=None):
        """
        Get initial guesses for the generic line parameters.

        Parameters
        ----------
        amp : numeric
            Line amplitude. This should be strictly positive; the
            ``absorp_emiss`` parameter controls whether it is absorption or
            emission.
        x0 : numeric
            Line centroid.
        bg : iterable
            Background polynomial coefficients.
        bg_buffer : int
            The number of values to use on either end of the flux array to
            estimate the background parameters.
        """
        target_x = self.target_x
        if x0 is None: # estimate the initial guess for the centroid
            relmins = argrelmin(-self.absorp_emiss * self.flux)[0]

            if len(relmins) > 1 and target_x is None:
                logger.log(0, "no target_x specified - taking largest line "
                           "in spec region")
                target_x = self.x[np.argmin(-self.absorp_emiss*self.flux)]

            if len(relmins) == 1:
                x0 = self.x[relmins[0]]

            else:
                x0_idx = relmins[np.abs(self.x[relmins] - target_x).argmin()]
                x0 = self.x[x0_idx]

        # shift x array so that line is approximately at 0
        _x = np.array(self.x, copy=True)
        x = np.array(_x) - x0

        # background polynomial parameters
        if bg_buffer is None:
            bg_buffer = len(x) // 4
            bg_buffer = max(1, bg_buffer)

        if bg is None:
            if self.n_bg_coef < 2:
                bg = np.array([0.])
                bg[0] = np.median(self.flux)

            else:
                # estimate linear background model
                bg = np.array([0.] * self.n_bg_coef)

                i1 = argmedian(self.flux[:bg_buffer])
                i2 = argmedian(self.flux[-bg_buffer:])
                f1 = self.flux[:bg_buffer][i1]
                f2 = self.flux[-bg_buffer:][i2]
                x1 = x[:bg_buffer][i1]
                x2 = x[-bg_buffer:][i2]
                bg[1] = (f2-f1) / (x2-x1) # slope
                bg[0] = f2 - bg[1]*x2 # estimate constant term

        else:
            if len(bg) != self.n_bg_coef:
                raise ValueError('Number of bg polynomial coefficients does '
                                 'not match n_bg_coef specified when fitter '
                                 'was created.')

        if amp is None: # then estimate the initial guess for amplitude
            _i = np.argmin(np.abs(x))
            if len(bg) > 1:
                _bg = bg[0] + bg[1]*x[_i]
            else:
                _bg = bg[0]
            # amp = np.sqrt(2*np.pi) * (flux[_i] - bg)
            amp = self.flux[_i] - _bg

        p0 = OrderedDict()
        p0['amp'] = np.abs(amp)
        p0['x0'] = x0
        p0['bg_coef'] = bg
        return p0

    def get_init_gp(self, log_sigma=None, log_rho=None, x0=None):
        """
        Set up the GP kernel parameters.
        """

        if log_sigma is None:
            if x0 is not None:
                # better initial guess for GP parameters
                mask = np.abs(self.x - x0) > 2.
                y_MAD = np.median(np.abs(self.flux[mask] - np.median(self.flux[mask])))
            else:
                y_MAD = np.median(np.abs(self.flux - np.median(self.flux)))
            log_sigma = np.log(3*y_MAD)

        if log_rho is None:
            log_rho = np.log(5.)

        p0 = OrderedDict()
        p0['log_sigma'] = log_sigma
        p0['log_rho'] = log_rho
        return p0

    def init_gp(self, log_sigma=None, log_rho=None, amp=None, x0=None, bg=None,
                bg_buffer=None, **kwargs):
        """
        **kwargs:
        """

        # Call the different get_init methods with the correct kwargs passed
        sig = inspect.signature(self.get_init)
        kw = OrderedDict()
        for k in list(sig.parameters.keys()):
            kw[k] = kwargs.pop(k, None)
        p0 = self.get_init(**kw)

        # Call the generic init method - all line models must have these params
        p0_generic = self.get_init_generic(amp=amp, x0=x0, bg=bg,
                                           bg_buffer=bg_buffer)
        for k,v in p0_generic.items():
            p0[k] = v

        # expand bg parameters
        bgs = p0.pop('bg_coef')
        for i in range(self.n_bg_coef):
            p0['bg{}'.format(i)] = bgs[i]

        # transform
        p0 = self.transform_pars(p0)

        # initialize model
        mean_model = self.MeanModel(n_bg_coef=self.n_bg_coef,
                                    absorp_emiss=self.absorp_emiss, **p0)

        p0_gp = self.get_init_gp(log_sigma=log_sigma, log_rho=log_rho)
        kernel = terms.Matern32Term(log_sigma=p0_gp['log_sigma'],
                                    log_rho=p0_gp['log_rho'])

        # set up the gp model
        self.gp = GP(kernel, mean=mean_model, fit_mean=True)

        if self._err is not None:
            self.gp.compute(self.x, self._err)
        else:
            self.gp.compute(self.x)

        init_params = self.gp.get_parameter_vector()
        init_ll = self.gp.log_likelihood(self.flux)
        logger.log(0, "Initial log-likelihood: {0}".format(init_ll))

        return init_params

    def get_gp_mean_pars(self):
        """
        Return a parameter dictionary for the mean model parameters only.
        """
        fit_pars = OrderedDict()
        for k,v in self.gp.get_parameter_dict().items():
            if 'mean' not in k:
                continue

            k = k[5:] # remove 'mean:'
            if k.startswith('ln'):
                if 'amp' in k:
                    fit_pars[k[3:]] = self.absorp_emiss * np.exp(v)
                else:
                    fit_pars[k[3:]] = np.exp(v)

            elif k.startswith('bg'):
                if 'bg_coef' not in fit_pars:
                    fit_pars['bg_coef'] = []
                fit_pars['bg_coef'].append(v)

            else:
                fit_pars[k] = v

        return fit_pars

    def _neg_log_like(self, params):
        self.gp.set_parameter_vector(params)

        try:
            ll = self.gp.log_likelihood(self.flux)
        except (RuntimeError, LinAlgError):
            return np.inf

        if np.isnan(ll):
            return np.inf
        return -ll

    def fit(self, bounds=None, **kwargs):
        """
        """
        init_params = self.init_gp()

        if bounds is None:
            bounds = OrderedDict()

        # default bounds for mean model parameters
        i = self.gp.models['mean'].get_parameter_names().index('x0')
        mean_bounds = self.gp.models['mean'].get_parameter_bounds()
        if mean_bounds[i][0] is None and mean_bounds[i][1] is None:
            mean_bounds[i] = (min(self.x), max(self.x))

        for i,k in enumerate(self.gp.models['mean'].parameter_names):
            mean_bounds[i] = bounds.get(k, mean_bounds[i])
        self.gp.models['mean'].parameter_bounds = mean_bounds

        # HACK: default bounds for kernel parameters
        self.gp.models['kernel'].parameter_bounds = [(None, None),
                                                     (np.log(0.5), None)]

        soln = minimize(self._neg_log_like, init_params,
                        method="L-BFGS-B",
                        bounds=self.gp.get_parameter_bounds())
        self.success = soln.success

        self.gp.set_parameter_vector(soln.x)

        if self.success:
            logger.debug("Success: {0}, Final log-likelihood: {1}"
                         .format(soln.success, -soln.fun))
        else:
            logger.warning("Fit failed! Final log-likelihood: {0}, "
                           "Final parameters: {1}".format(-soln.fun, soln.x))

        return self

    def plot_fit(self, axes=None, fit_alpha=0.5):
        unbinned_color = '#3182bd'
        binned_color = '#2ca25f'
        gp_color = '#ff7f0e'
        noise_color = '#de2d26'

        if self.gp is None:
            raise ValueError("You must run .fit() first!")

        # ----------------------------------------------------------------------
        # Plot the maximum likelihood model
        if axes is None:
            fig,axes = plt.subplots(1, 3, figsize=(15, 5), sharex=True)

        # data
        for ax in axes[:2]:
            ax.plot(self.x, self.flux, drawstyle='steps-mid',
                    marker='', color='#777777', zorder=2)

            if self._err is not None:
                ax.errorbar(self.x, self.flux, self._err,
                            marker='', ls='none', ecolor='#666666', zorder=1)

        # mean model
        wave_grid = np.linspace(self.x.min(), self.x.max(), 256)
        mu, var = self.gp.predict(self.flux, self.x, return_var=True)
        std = np.sqrt(var)

        axes[0].plot(wave_grid, self.gp.mean.get_unbinned_value(wave_grid),
                     marker='', alpha=fit_alpha, zorder=10, color=unbinned_color)
        axes[0].plot(self.x, self.gp.mean.get_value(self.x),
                     marker='', alpha=fit_alpha, zorder=10, drawstyle='steps-mid',
                     color=binned_color)

        # full GP model
        axes[1].plot(self.x, mu, color=gp_color, drawstyle='steps-mid',
                     marker='', alpha=fit_alpha, zorder=10)
        axes[1].fill_between(self.x, mu+std, mu-std, color=gp_color,
                             alpha=fit_alpha/10., edgecolor="none", step='mid')

        # just GP noise component
        mean_model = self.gp.mean.get_value(self.x)
        axes[2].plot(self.x, mu - mean_model,
                     marker='', alpha=fit_alpha, zorder=10,
                     drawstyle='steps-mid', color=noise_color)

        axes[2].plot(self.x, self.flux - mean_model, drawstyle='steps-mid',
                     marker='', color='#777777', zorder=2)

        if self._err is not None:
            axes[2].errorbar(self.x, self.flux - mean_model, self._err,
                             marker='', ls='none', ecolor='#666666', zorder=1)

        axes[0].set_ylabel('flux')
        axes[0].set_xlim(self.x.min(), self.x.max())

        axes[0].set_title('Line model')
        axes[1].set_title('Line + noise model')
        axes[2].set_title('Noise model')

        fig = axes[0].figure
        fig.tight_layout()
        return fig
