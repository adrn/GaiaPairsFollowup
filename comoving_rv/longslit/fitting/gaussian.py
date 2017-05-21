# Standard library
from collections import OrderedDict

# Third-party
import numpy as np
from scipy.optimize import minimize, leastsq
from scipy.signal import argrelmin
from celerite.modeling import Model
from celerite import terms, GP

# Project
from ...log import logger
from ..models import binned_gaussian_polynomial

__all__ = ['fit_spec_line', 'fit_spec_line_GP', 'gp_to_fit_pars']

def par_dict_to_list(p):
    return [p['amp'], p['x0'], p['std']] + p['bg_coef'].tolist()

def get_init_guess(x, flux, ivar,
                   amp0=None, x0=None, std0=None,
                   bg0=None, n_bg_coef=2, target_x=None,
                   absorp_emiss=1.):

    if x0 is None: # estimate the initial guess for the centroid
        relmins = argrelmin(-absorp_emiss*flux)[0]

        if len(relmins) > 1 and target_x is None:
            logger.log(0, "no target_x specified - taking largest line "
                       "in spec region")
            target_x = x[np.argmin(-absorp_emiss*flux)]

        if len(relmins) == 1:
            x0 = x[relmins[0]]

        else:
            x0_idx = relmins[np.abs(x[relmins] - target_x).argmin()]
            x0 = x[x0_idx]

    # shift x array so that line is approximately at 0
    _x = np.array(x, copy=True)
    x = np.array(_x) - x0

    # background polynomial parameters
    if bg0 is None:
        if n_bg_coef < 2:
            bg0 = np.array([0.])
            bg0[0] = np.median(flux)

        else:
            # estimate linear background model
            bg0 = np.array([0.] * n_bg_coef)
            bg0[1] = (flux[-1]-flux[0])/(x[-1]-x[0]) # slope
            bg0[0] = flux[-1] - bg0[1]*x[-1] # estimate constant term

    else:
        if len(bg0) != n_bg_coef:
            n_bg_coef = len(bg0)

    if std0 is None:
        std0 = 1. # MAGIC NUMBER

    if amp0 is None: # then estimate the initial guess for amplitude
        _i = np.argmin(np.abs(x))
        if len(bg0) > 1:
            bg = bg0[0] + bg0[1]*x[_i]
        else:
            bg = bg0[0]
        amp0 = np.sqrt(2*np.pi) * (flux[_i] - bg)

    p0 = OrderedDict()
    p0['amp'] = np.abs(amp0)
    p0['x0'] = x0
    p0['std'] = std0
    p0['bg_coef'] = bg0
    return p0

# ---

def errfunc(p, pix, flux, flux_ivar):
    amp, x0, std, *bg_coef = p
    f = binned_gaussian_polynomial
    return (flux - f(pix, amp, x0, std, bg_coef)) * np.sqrt(flux_ivar)

def fit_spec_line(x, flux, flux_ivar=None,
                  amp0=None, x0=None, std0=None,
                  bg0=None, n_bg_coef=2, target_x=None,
                  absorp_emiss=1., return_cov=False,
                  leastsq_kw=None):
    """
    Assumes that the input arrays are trimmed to be a sensible region around
    the line of interest.

    Parameters
    ----------
    x : array_like
        Pixel or wavelength array.
    flux : array_like
        Array of fluxes. Must be the same shape as ``x``.
    flux_var : array_like
        Invere-variance uncertainty array. Must be the same shape
        as ``x``.
    amp0 : numeric (optional)
        Initial guess for line amplitude.
    x0 : numeric (optional)
        Initial guess for line centroid.
    n_bg_coef : int
        Number of terms in the background polynomial fit.
    absorp_emiss : float
        -1 for absorption line
        +1 for emission line
    """

    sort_idx = np.argsort(x)
    x = np.array(x)[sort_idx]
    flux = np.array(flux)[sort_idx]
    if flux_ivar is not None:
        flux_ivar = np.array(flux_ivar)[sort_idx]
    else:
        flux_ivar = np.ones_like(flux)

    # initial guess
    p0 = get_init_guess(x=x, flux=flux, ivar=flux_ivar,
                        amp0=amp0, x0=x0, std0=std0,
                        bg0=bg0, n_bg_coef=n_bg_coef, target_x=target_x,
                        absorp_emiss=absorp_emiss)

    # shift x array so that line is approximately at 0
    _x = np.array(x, copy=True)
    _x0 = float(p0['x0'])
    x = np.array(_x) - _x0
    p0['x0'] = 0. # recenter to initial guess

    # kwargs for leastsq:
    if leastsq_kw is None:
        leastsq_kw = dict()
    leastsq_kw.setdefault('ftol', 1e-10)
    leastsq_kw.setdefault('xtol', 1e-10)
    leastsq_kw.setdefault('maxfev', 100000)

    fail_msg = "Fitting spectral line in comp lamp spectrum failed. {msg}"

    args = (x, flux, flux_ivar)
    p_opt,p_cov,*_,mesg,ier = leastsq(errfunc, par_dict_to_list(p0),
                                      args=args, full_output=True, **leastsq_kw)

    if p_cov is None:
        raise RuntimeError(fail_msg.format(msg=mesg))

    s_sq = (errfunc(p_opt, *args)**2).sum() / (len(flux)-len(p0))
    p_cov = p_cov * s_sq

    fit_amp, fit_x0, fit_std, *fit_bg = p_opt
    fit_x0 = fit_x0 + _x0

    if ier < 1 or ier > 4:
        raise RuntimeError(fail_msg.format(msg=mesg))

    if fit_x0 < min(_x) or fit_x0 > max(_x):
        raise ValueError(fail_msg.format(msg="Unphysical peak centroid: {:.3f}".format(fit_x0)))

    par_dict = OrderedDict(amp=fit_amp, x0=fit_x0,
                           std=fit_std, bg_coef=fit_bg)

    if return_cov:
        return par_dict, p_cov

    else:
        return par_dict

class MeanModel(Model):

    def __init__(self, n_bg_coef, absorp_emiss, *args, **kwargs):
        self._n_bg_coef = n_bg_coef
        self._absorp_emiss = absorp_emiss
        self.parameter_names = (["ln_amp", "x0", "ln_std"] +
                                ["bg{}".format(i) for i in range(n_bg_coef)])
        super(MeanModel, self).__init__(*args, **kwargs)

    def get_value(self, x):
        f = binned_gaussian_polynomial
        return f(x, self._absorp_emiss*np.exp(self.ln_amp),
                 self.x0, np.exp(self.ln_std),
                 [getattr(self, "bg{}".format(i))
                  for i in range(self._n_bg_coef)])

def gp_to_fit_pars(gp, absorp_emiss):
    """
    Given a GP instance, return a parameter dictionary for the
    mean model parameters only.
    """
    fit_pars = OrderedDict()
    for k,v in gp.get_parameter_dict().items():
        if 'mean' not in k:
            continue

        k = k[5:] # remove 'mean:'
        if k.startswith('ln'):
            if 'amp' in k:
                fit_pars[k[3:]] = absorp_emiss * np.exp(v)
            else:
                fit_pars[k[3:]] = np.exp(v)

        elif k.startswith('bg'):
            if 'bg_coef' not in fit_pars:
                fit_pars['bg_coef'] = []
            fit_pars['bg_coef'].append(v)

        else:
            fit_pars[k] = v

    return fit_pars

def fit_spec_line_GP(x, flux, flux_ivar=None,
                     amp0=None, x0=None, std0=None,
                     bg0=None, n_bg_coef=2, target_x=None,
                     log_sigma0=None, log_rho0=None, # GP parameters
                     absorp_emiss=1., minimize_kw=None):

    sort_idx = np.argsort(x)
    x = np.array(x)[sort_idx]
    flux = np.array(flux)[sort_idx]
    if flux_ivar is not None:
        flux_ivar = np.array(flux_ivar)[sort_idx]
    else:
        flux_ivar = np.ones_like(flux)

    flux_err = 1 / np.sqrt(flux_ivar)

    # initial guess for mean model
    p0 = get_init_guess(x=x, flux=flux, ivar=flux_ivar,
                        amp0=amp0, x0=x0, std0=std0,
                        bg0=bg0, n_bg_coef=n_bg_coef, target_x=target_x,
                        absorp_emiss=absorp_emiss)

    # replace log parameters
    p0['ln_amp'] = np.log(p0.pop('amp'))
    p0['ln_std'] = np.log(p0.pop('std'))

    # expand bg parameters
    bgs = p0.pop('bg_coef')
    for i in range(n_bg_coef):
        p0['bg{}'.format(i)] = bgs[i]

    # initialize model
    mean_model = MeanModel(n_bg_coef=n_bg_coef, absorp_emiss=absorp_emiss, **p0)

    # Set up the GP model
    if log_sigma0 is None:
        y_MAD = np.median(np.abs(flux - np.median(flux)))
        log_sigma0 = np.log(y_MAD)

    if log_rho0 is None:
        log_rho0 = np.log(5.)

    # kernel = terms.RealTerm(log_a=np.log(y_MAD), log_c=-np.log(0.1)) # MAGIC NUMBERs
    kernel = terms.Matern32Term(log_sigma=log_sigma0, log_rho=log_rho0) # MAGIC NUMBERs

    # set up the gp model
    gp = GP(kernel, mean=mean_model, fit_mean=True)
    gp.compute(x, flux_err) # need to do this
    init_params = gp.get_parameter_vector()
    logger.debug("Initial log-likelihood: {0}".format(gp.log_likelihood(flux)))

    # Define a cost function
    def neg_log_like(params, y, gp):
        gp.set_parameter_vector(params)
        ll = gp.log_likelihood(y)
        if np.isnan(ll):
            return np.inf
        return -ll

    # Fit for the maximum likelihood parameters
    bounds = gp.get_parameter_bounds()
    bounds = [(None,None), (np.log(0.5), None), (None,None), (min(x), max(x)),
              (np.log(0.25), np.log(32.))] + [(None,None)]*n_bg_coef
    soln = minimize(neg_log_like, init_params, method="L-BFGS-B",
                    bounds=bounds, args=(flux, gp))
    gp.set_parameter_vector(soln.x)
    logger.debug("Success: {}, Final log-likelihood: {}".format(soln.success, -soln.fun))

    return gp
