# Third-party
import matplotlib.pyplot as plt
import numpy as np
from scipy.signal import argrelmax

# Project
# from .gaussian import fit_spec_line_GP, gp_to_fit_pars
# from ..models import gaussian_polynomial, binned_gaussian_polynomial

__all__ = ['fit_sky_region']

def fit_sky_region(x, flux, ivar, center, width, plot=False):
    _idx = np.abs(x - center) < width # x pixels on either side of center

    x = x[_idx]
    flux = flux[_idx]
    ivar = ivar[_idx]

    sort_ = x.argsort()
    wave_data = x[sort_]
    flux_data = flux[sort_]
    err_data = 1/np.sqrt(ivar[sort_])

    # better initial guess for GP parameters
    mask = np.abs(wave_data - center) > 2.
    y_MAD = np.median(np.abs(flux[mask] - np.median(flux[mask])))
    log_sigma0 = np.log(5*y_MAD)

    # find largest line closest to center
    max_idx, = argrelmax(flux_data)
    max_waves = wave_data[max_idx]
    max_idx = max_idx[np.argmin(np.abs(max_waves - center))]
    gp = fit_spec_line_GP(wave_data, flux_data, ivar, absorp_emiss=1.,
                          std0=1., x0=wave_data[max_idx],
                          log_sigma0=log_sigma0, log_rho0=np.log(10.))

    wave_grid = np.linspace(wave_data.min(), wave_data.max(), 256)
    mu, var = gp.predict(flux_data, wave_grid, return_var=True)
    std = np.sqrt(var)

    fit_pars = gp_to_fit_pars(gp, 1.)

    if plot:
        # ------------------------------------------------------------------------
        # Plot the maximum likelihood model
        fig,axes = plt.subplots(1, 3, figsize=(15, 5), sharex=True, sharey=True)

        # data
        for ax in axes:
            ax.plot(wave_data, flux_data, drawstyle='steps-mid',
                    marker='', color='#777777', zorder=-11)
            ax.errorbar(wave_data, flux_data, err_data,
                        marker='', ls='none', ecolor='#666666', zorder=-10)

        # mean model
        axes[0].plot(wave_grid, gaussian_polynomial(wave_grid, **fit_pars),
                     marker='', alpha=0.5, zorder=100, color='r')
        axes[1].plot(wave_data, binned_gaussian_polynomial(wave_data, **fit_pars),
                     marker='', alpha=0.5, zorder=101, drawstyle='steps-mid', color='g')

        # full GP model
        gp_color = "#ff7f0e"
        axes[2].plot(wave_data, mu, color=gp_color, marker='')
        axes[2].fill_between(wave_data, mu+std, mu-std, color=gp_color,
                             alpha=0.3, edgecolor="none")

        for ax in axes: ax.set_xlabel(r'wavelength [$\AA$]')
        axes[0].set_ylabel('flux')

        fig.tight_layout()

    success = True

    # some validation
    max_ = fit_pars['amp'] / np.sqrt(2*np.pi*fit_pars['std']**2)
    SNR = max_ / np.median(err_data)

    if ((abs(fit_pars['x0']-center) > 4) or (fit_pars['amp'] < 10) or
            (fit_pars['std'] > 4) or (SNR < 2)):
        # FAILED
        success = False

    if plot:
        return fit_pars, success, fig
    else:
        return fit_pars, success
