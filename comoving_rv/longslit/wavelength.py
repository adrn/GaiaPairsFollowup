# Standard library
from collections import OrderedDict

# Third-party
import numpy as np
from scipy.optimize import minimize, leastsq
from scipy.stats import scoreatpercentile
from scipy.signal import argrelmin
from celerite.modeling import Model
from celerite import terms, GP

# Project
from .models import voigt_polynomial

__all__ = ['fit_spec_line', 'fit_spec_line_GP']

def par_dict_to_list(p):
    return [p['amp'], p['x0'], p['std_G'], p['fwhm_L']] + p['bg'].tolist()

def get_init_guess(x, flux, ivar,
                   amp0=None, x0=None, std_G0=None, fwhm_L0=None,
                   bg0=None, n_bg_coef=2, target_x=None,
                   absorp_emiss=1.):

    if x0 is None: # estimate the initial guess for the centroid
        relmins = argrelmin(-absorp_emiss*flux)[0]

        if len(relmins) > 1 and target_x is None:
            raise ValueError("If auto-finding x0, must supply a target value for x.")
        elif len(relmins) == 1:
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

    if std_G0 is None:
        std_G0 = 2. # MAGIC NUMBER

    if fwhm_L0 is None:
        fwhm_L0 = 0.5 # MAGIC NUMBER

    if amp0 is None: # then estimate the initial guess for amplitude
        _i = np.argmin(np.abs(x))
        if len(bg0) > 1:
            bg = bg0[0] + bg0[1]*x[_i]
        else:
            bg = bg0[0]
        amp0 = np.sqrt(2*np.pi) * (flux[_i] - bg)

    p0 = OrderedDict()
    p0['amp'] = amp0
    p0['x0'] = x0
    p0['std_G'] = std_G0
    p0['fwhm_L'] = fwhm_L0
    p0['bg'] = bg0
    return p0

# ---

def errfunc(p, pix, flux, flux_ivar):
    amp, x0, std_G, fwhm_L, *bg_coef = p
    return (flux - voigt_polynomial(pix, amp, x0, std_G, fwhm_L, bg_coef)) * np.sqrt(flux_ivar)

def fit_spec_line(x, flux, flux_ivar=None,
                  amp0=None, x0=None, std_G0=None, fwhm_L0=None,
                  bg0=None, n_bg_coef=2, target_x=None,
                  absorp_emiss=1., return_cov=False,
                  leastsq_kw=None):
    """
    Assumes that the input arrays are trimmed to be a sensible region around
    the line of interest.

    Parameters
    ----------
    x : array_like
        Pixel or wavelength. Must be the same shape as ``flux``.
    flux : array_like
        Must be the same shape as ``pix_grid``.
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
                        amp0=amp0, x0=x0, std_G0=std_G0, fwhm_L0=fwhm_L0,
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

    p_opt,p_cov,*_,mesg,ier = leastsq(errfunc, par_dict_to_list(p0),
                                      args=(x, flux, flux_ivar),
                                      full_output=True, **leastsq_kw)

    fit_amp, fit_x0, fit_std_G, fit_fwhm_L, *fit_bg = p_opt
    fit_x0 = fit_x0 + _x0

    fail_msg = "Fitting spectral line in comp lamp spectrum failed. {msg}"

    if ier < 1 or ier > 4:
        raise RuntimeError(fail_msg.format(msg=mesg))

    if fit_x0 < min(_x) or fit_x0 > max(_x):
        raise ValueError(fail_msg.format(msg="Unphysical peak centroid: {:.3f}".format(fit_x0)))

    par_dict = OrderedDict(amp=fit_amp, x0=fit_x0,
                           std_G=fit_std_G, fwhm_L=fit_fwhm_L,
                           bg_coef=fit_bg)

    if return_cov:
        return par_dict, p_cov

    else:
        return par_dict

class MeanModel(Model):

    def __init__(self, n_bg_coef, absorp_emiss, *args, **kwargs):
        self._n_bg_coef = n_bg_coef
        self._absorp_emiss = absorp_emiss
        self.parameter_names = (["ln_amp", "x0", "ln_std_G", "ln_fwhm_L"] +
                                ["bg{}".format(i) for i in range(n_bg_coef)])
        super(MeanModel, self).__init__(*args, **kwargs)

    def get_value(self, x):
        return voigt_polynomial(x, self._absorp_emiss*np.exp(self.ln_amp), self.x0,
                                np.exp(self.ln_std_G), np.exp(self.ln_fwhm_L),
                                [getattr(self, "bg{}".format(i))
                                 for i in range(self._n_bg_coef)])

def fit_spec_line_GP(x, flux, flux_ivar=None,
                     amp0=None, x0=None, std_G0=None, fwhm_L0=None,
                     bg0=None, n_bg_coef=2, target_x=None,
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
                        amp0=amp0, x0=x0, std_G0=std_G0, fwhm_L0=fwhm_L0,
                        bg0=bg0, n_bg_coef=n_bg_coef, target_x=target_x,
                        absorp_emiss=absorp_emiss)

    # shift x array so that line is approximately at 0
    # _x = np.array(x, copy=True)
    # _x0 = float(p0['x0'])
    # x = np.array(_x) - _x0
    # p0['x0'] = 0. # recenter to initial guess

    # replace log parameters
    p0['ln_amp'] = np.log(absorp_emiss * p0.pop('amp'))
    p0['ln_std_G'] = np.log(p0.pop('std_G'))
    p0['ln_fwhm_L'] = np.log(p0.pop('fwhm_L'))

    # expand bg parameters
    bgs = p0.pop('bg')
    for i in range(n_bg_coef):
        p0['bg{}'.format(i)] = bgs[i]

    # initialize model
    mean_model = MeanModel(n_bg_coef=n_bg_coef, absorp_emiss=absorp_emiss, **p0)

    # Set up the GP model
    y_MAD = np.median(np.abs(flux - np.median(flux)))
    # kernel = terms.RealTerm(log_a=np.log(y_MAD), log_c=-np.log(0.1)) # MAGIC NUMBERs
    kernel = terms.Matern32Term(log_sigma=np.log(y_MAD), log_rho=np.log(10.)) # MAGIC NUMBERs

    # set up the gp model
    gp = GP(kernel, mean=mean_model, fit_mean=True)
    gp.compute(x, flux_err) # need to do this
    init_params = gp.get_parameter_vector()
    print("Initial log-likelihood: {0}".format(gp.log_likelihood(flux)))

    # Define a cost function
    def neg_log_like(params, y, gp):
        gp.set_parameter_vector(params)
        return -gp.log_likelihood(y)

    # Fit for the maximum likelihood parameters
    bounds = gp.get_parameter_bounds()
    soln = minimize(neg_log_like, init_params, method="L-BFGS-B",
                    bounds=bounds, args=(flux, gp))
    gp.set_parameter_vector(soln.x)
    print("Success: {}".format(soln.success))
    print("Final log-likelihood: {0}".format(-soln.fun))

    # pars = gp.get_parameter_dict()

    return gp
