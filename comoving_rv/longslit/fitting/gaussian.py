# Standard library
from collections import OrderedDict

# Third-party
import numpy as np
from celerite.modeling import Model

# Project
from .core import LineFitter
from ..models import gaussian_polynomial, binned_gaussian_polynomial

__all__ = ['GaussianLineFitter']

class GaussianMeanModel(Model):

    def __init__(self, n_bg_coef, absorp_emiss, *args, **kwargs):
        self._n_bg_coef = n_bg_coef
        self._absorp_emiss = absorp_emiss
        self.parameter_names = (["ln_amp", "x0", "ln_std"] +
                                ["bg{}".format(i) for i in range(n_bg_coef)])

        # parameter bounds
        bounds = kwargs.pop('bounds', dict())

        default_bounds = OrderedDict()
        default_bounds['ln_amp'] = np.log([1., 1E6])
        default_bounds['ln_std'] = np.log([0.25, 32.])
        default_bounds.update(bounds)
        kwargs['bounds'] = default_bounds

        super(GaussianMeanModel, self).__init__(*args, **kwargs)

    def get_value(self, x):
        f = binned_gaussian_polynomial
        return f(x, self._absorp_emiss*np.exp(self.ln_amp),
                 self.x0, np.exp(self.ln_std),
                 [getattr(self, "bg{}".format(i))
                  for i in range(self._n_bg_coef)])

    def get_unbinned_value(self, x):
        f = gaussian_polynomial
        return f(x, self._absorp_emiss*np.exp(self.ln_amp),
                 self.x0, np.exp(self.ln_std),
                 [getattr(self, "bg{}".format(i))
                  for i in range(self._n_bg_coef)])

class GaussianLineFitter(LineFitter):

    _log_pars = ['amp', 'std']
    MeanModel = GaussianMeanModel

    @classmethod
    def pack_pars(cls, p):
        return [p['amp'], p['x0'], p['std']] + p['bg_coef'].tolist()

    def get_init(self, std=None):
        """ """
        if std is None:
            std = 1.

        p0 = OrderedDict()
        p0['std'] = std
        return p0

# ------------------------------------------------------------------------------
# Old stuff:
# def errfunc(p, pix, flux, flux_ivar):
#     amp, x0, std, *bg_coef = p
#     f = binned_gaussian_polynomial
#     return (flux - f(pix, amp, x0, std, bg_coef)) * np.sqrt(flux_ivar)

# def fit_spec_line(x, flux, flux_ivar=None,
#                   amp0=None, x0=None, std0=None,
#                   bg0=None, n_bg_coef=2, target_x=None,
#                   absorp_emiss=1., return_cov=False,
#                   leastsq_kw=None):
#     """
#     Assumes that the input arrays are trimmed to be a sensible region around
#     the line of interest.

#     Parameters
#     ----------
#     x : array_like
#         Pixel or wavelength array.
#     flux : array_like
#         Array of fluxes. Must be the same shape as ``x``.
#     flux_var : array_like
#         Invere-variance uncertainty array. Must be the same shape
#         as ``x``.
#     amp0 : numeric (optional)
#         Initial guess for line amplitude.
#     x0 : numeric (optional)
#         Initial guess for line centroid.
#     n_bg_coef : int
#         Number of terms in the background polynomial fit.
#     absorp_emiss : float
#         -1 for absorption line
#         +1 for emission line
#     """

#     sort_idx = np.argsort(x)
#     x = np.array(x)[sort_idx]
#     flux = np.array(flux)[sort_idx]
#     if flux_ivar is not None:
#         flux_ivar = np.array(flux_ivar)[sort_idx]
#     else:
#         flux_ivar = np.ones_like(flux)

#     # initial guess
#     p0 = get_init_guess(x=x, flux=flux, ivar=flux_ivar,
#                         amp0=amp0, x0=x0, std0=std0,
#                         bg0=bg0, n_bg_coef=n_bg_coef, target_x=target_x,
#                         absorp_emiss=absorp_emiss)

#     # shift x array so that line is approximately at 0
#     _x = np.array(x, copy=True)
#     _x0 = float(p0['x0'])
#     x = np.array(_x) - _x0
#     p0['x0'] = 0. # recenter to initial guess

#     # kwargs for leastsq:
#     if leastsq_kw is None:
#         leastsq_kw = dict()
#     leastsq_kw.setdefault('ftol', 1e-10)
#     leastsq_kw.setdefault('xtol', 1e-10)
#     leastsq_kw.setdefault('maxfev', 100000)

#     fail_msg = "Fitting spectral line in comp lamp spectrum failed. {msg}"

#     args = (x, flux, flux_ivar)
#     p_opt,p_cov,*_,mesg,ier = leastsq(errfunc, par_dict_to_list(p0),
#                                       args=args, full_output=True, **leastsq_kw)

#     if p_cov is None:
#         raise RuntimeError(fail_msg.format(msg=mesg))

#     s_sq = (errfunc(p_opt, *args)**2).sum() / (len(flux)-len(p0))
#     p_cov = p_cov * s_sq

#     fit_amp, fit_x0, fit_std, *fit_bg = p_opt
#     fit_x0 = fit_x0 + _x0

#     if ier < 1 or ier > 4:
#         raise RuntimeError(fail_msg.format(msg=mesg))

#     if fit_x0 < min(_x) or fit_x0 > max(_x):
#         raise ValueError(fail_msg.format(msg="Unphysical peak centroid: {:.3f}".format(fit_x0)))

#     par_dict = OrderedDict(amp=fit_amp, x0=fit_x0,
#                            std=fit_std, bg_coef=fit_bg)

#     if return_cov:
#         return par_dict, p_cov

#     else:
#         return par_dict
