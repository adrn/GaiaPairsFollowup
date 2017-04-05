# coding: utf-8

"""
TODO:
- n1.0073 Halpha is emission
- Save results to a database file of some kind...sqlite? yaml file?

"""

# Standard library
import os
from os import path
from collections import OrderedDict
import glob

# Third-party
from astropy.constants import c
import astropy.coordinates as coord
import astropy.units as u
from astropy.io import fits
from astropy.table import Table, Column
import numpy as np
import matplotlib.pyplot as plt
plt.style.use('apw-notebook')
import emcee
import corner
import schwimmbad

# Project
from comoving_rv.log import logger
from comoving_rv.longslit.wavelength import fit_spec_line_GP
from comoving_rv.longslit.models import voigt_polynomial

def get_fit_pars(gp):
    fit_pars = OrderedDict()
    for k,v in gp.get_parameter_dict().items():
        if 'mean' not in k:
            continue

        k = k[5:] # remove 'mean:'
        if k.startswith('ln'):
            if 'amp' in k:
                fit_pars[k[3:]] = -np.exp(v)
            else:
                fit_pars[k[3:]] = np.exp(v)

        elif k.startswith('bg'):
            if 'bg_coef' not in fit_pars:
                fit_pars['bg_coef'] = []
            fit_pars['bg_coef'].append(v)

        else:
            fit_pars[k] = v

    if 'std_G' not in fit_pars:
        fit_pars['std_G'] = 1E-10

    return fit_pars

def log_probability(params, gp, flux_data):
    gp.set_parameter_vector(params)
    lp = gp.log_prior()
    if not np.isfinite(lp):
        return -np.inf

    # HACK: Gaussian prior on log(rho)
    var = 1.
    lp += -0.5*(params[1]-1)**2/var - 0.5*np.log(2*np.pi*var)

    if params[4] < -10. or params[5] < -10.:
        return -np.inf

    ll = gp.log_likelihood(flux_data)
    if not np.isfinite(ll):
        return -np.inf

    return ll + lp

def main(overwrite=False):
    night = 'n5'

    plot_path = path.abspath('../plots/')
    root_path = path.abspath('../data/mdm-spring-2017/processed/')
    table_path = path.join(root_path, 'rough_velocity.ecsv')
    Halpha = 6562.8 * u.angstrom

    if not path.exists(plot_path):
        os.makedirs(plot_path, exist_ok=True)

    if not path.exists(table_path):
        logger.debug('Creaing table at {}'.format(table_path))
        tbl_init = [Column(name='object_name', dtype='|S30', data=[], length=0),
                    Column(name='group_id', dtype=int, length=0),
                    Column(name='filename', dtype='|S128', data=[], length=0),
                    Column(name='wave0', dtype=float, unit=u.angstrom, length=0),
                    Column(name='rv', dtype=float, unit=u.km/u.s, length=0),
                    Column(name='rv_precision', dtype=float, unit=u.km/u.s, length=0)]
        velocity_tbl = Table(tbl_init)
        velocity_tbl.write(table_path, format='ascii.ecsv', delimiter=',')
        logger.debug('Table: {}'.format(velocity_tbl.colnames))

    else:
        logger.debug('Table exists, reading ({})'.format(table_path))
        velocity_tbl = Table.read(table_path, format='ascii.ecsv', delimiter=',')

    # read master_wavelength file
    pix_wav = np.genfromtxt(path.join(root_path, night, 'master_wavelength.csv'),
                            delimiter=',', names=True)

    idx = (pix_wav['wavelength'] < 6950) & (pix_wav['wavelength'] > 6000)
    pix_wav = pix_wav[idx] # HACK
    pix_range = [min(pix_wav['pixel']), max(pix_wav['pixel'])]

    # '1d_{}.{:04d}.fit'.format(night, frame)
    for file_path in glob.glob(path.join(root_path, night, '1d_*.fit*')):
        filename = path.basename(file_path)
        filebase,ext = path.splitext(filename)

        # read FITS header
        hdr = fits.getheader(file_path, 0)
        object_name = hdr['OBJECT']

        if object_name in velocity_tbl['object_name']:
            if overwrite:
                logger.debug('Object {} already done - overwriting!'.format(object_name))
                idx, = np.where(velocity_tbl['object_name'] == object_name)
                velocity_tbl.remove_row(idx[0])

            else:
                logger.debug('Object {} already done.'.format(object_name))
                continue

        # read the spectrum data and get wavelength solution
        spec = Table.read(file_path)
        coef = np.polynomial.polynomial.polyfit(pix_wav['pixel'], pix_wav['wavelength'], deg=9)

        # compute wavelength array for the pixels
        wave = np.polynomial.polynomial.polyval(spec['pix'], coef)
        wave[(spec['pix'] > max(pix_range)) | (spec['pix'] < min(pix_range))] = np.nan

        # Define data arrays to be used in fitting below
        near_Ha = (wave > 6510) & (wave < 6615)
        flux_data = spec['source_flux'][near_Ha]
        ivar_data = spec['source_ivar'][near_Ha]
        absorp_emiss = -1.
        target_x = Halpha.value

        wave_data = wave[near_Ha]

        _idx = wave_data.argsort()
        wave_data = wave_data[_idx]
        flux_data = flux_data[_idx]
        ivar_data = ivar_data[_idx]
        err_data = 1/np.sqrt(ivar_data)

        # grid of wavelengths for plotting
        wave_grid = np.linspace(wave_data.min(), wave_data.max(), 256)

        # start by doing a maximum likelihood GP
        gp = fit_spec_line_GP(wave_data, flux_data, ivar_data,
                              absorp_emiss=absorp_emiss, target_x=target_x,
                              fwhm_L0=4., std_G0=1., n_bg_coef=2)

        fit_pars = get_fit_pars(gp)

        # Make the maximum likelihood prediction
        mu, var = gp.predict(flux_data, wave_grid, return_var=True)
        std = np.sqrt(var)

        # ------------------------------------------------------------------------
        # Plot the maximum likelihood model
        fig,ax = plt.subplots()

        # data
        ax.plot(wave_data, flux_data, drawstyle='steps-mid', marker='')
        ax.errorbar(wave_data, flux_data, err_data,
                    marker='', ls='none', ecolor='#666666', zorder=-10)

        # mean model
        ax.plot(wave_grid, voigt_polynomial(wave_grid, **fit_pars),
                marker='', alpha=0.5)

        # full GP model
        gp_color = "#ff7f0e"
        ax.plot(wave_grid, mu, color=gp_color, marker='')
        ax.fill_between(wave_grid, mu+std, mu-std, color=gp_color,
                        alpha=0.3, edgecolor="none")

        ax.set_xlabel(r'wavelength [$\AA$]')
        ax.set_ylabel('flux')

        fig.tight_layout()
        fig.savefig(path.join(plot_path, '{}_maxlike.png'.format(filebase)), dpi=256)
        plt.close(fig)
        # ------------------------------------------------------------------------

        # Run `emcee` instead to sample over GP model parameters:
        if fit_pars['std_G'] < 1E-2:
            gp.freeze_parameter('mean:ln_std_G')

        initial = np.array(gp.get_parameter_vector())
        if initial[4] < -10:
            initial[4] = -8.
        if initial[5] < -10:
            initial[5] = -8.
        ndim, nwalkers = len(initial), 64

        with schwimmbad.MultiPool() as pool:
            # with schwimmbad.SerialPool() as pool:
            sampler = emcee.EnsembleSampler(nwalkers, ndim, log_probability, pool=pool,
                                            args=(gp, flux_data))

            logger.debug("Running burn-in...")
            p0 = initial + 1e-6 * np.random.randn(nwalkers, ndim)
            p0, lp, _ = sampler.run_mcmc(p0, 256)

            logger.debug("Running 2nd burn-in...")
            sampler.reset()
            p0 = p0[lp.argmax()] + 1e-3 * np.random.randn(nwalkers, ndim)
            p0, lp, _ = sampler.run_mcmc(p0, 512)

            logger.debug("Running production...")
            sampler.reset()
            pos, lp, _ = sampler.run_mcmc(p0, 512)

        # --------------------------------------------------------------------
        # plot MCMC traces
        fig,axes = plt.subplots(2,4,figsize=(18,6))
        for i in range(sampler.dim):
            for walker in sampler.chain[...,i]:
                axes.flat[i].plot(walker, marker='', drawstyle='steps-mid', alpha=0.2)
            axes.flat[i].set_title(gp.get_parameter_names()[i], fontsize=12)
        fig.tight_layout()
        fig.savefig(path.join(plot_path, '{}_mcmc_trace.png'.format(filebase)), dpi=256)
        plt.close(fig)
        # --------------------------------------------------------------------

        # --------------------------------------------------------------------
        # plot samples
        fig,axes = plt.subplots(3, 1, figsize=(6,9), sharex=True)

        samples = sampler.flatchain
        for s in samples[np.random.randint(len(samples), size=32)]:
            gp.set_parameter_vector(s)

            fit_pars = get_fit_pars(gp)
            _mean_model = voigt_polynomial(wave_grid, **fit_pars)
            axes[0].plot(wave_grid, _mean_model,
                         marker='', alpha=0.25, color='#3182bd', zorder=-10)

            mu = gp.predict(flux_data, wave_grid, return_cov=False)
            axes[1].plot(wave_grid, mu-_mean_model, color=gp_color, alpha=0.25, marker='')
            axes[2].plot(wave_grid, mu, color='#756bb1', alpha=0.25, marker='')

        axes[2].plot(wave_data, flux_data, drawstyle='steps-mid', marker='', zorder=-6)
        axes[2].errorbar(wave_data, flux_data, err_data,
                         marker='', ls='none', ecolor='#666666', zorder=-10)

        axes[2].set_ylabel('flux')
        axes[2].set_xlabel(r'wavelength [$\AA$]')
        axes[0].set_title('mean model (voigt + poly.)')
        axes[1].set_title('noise model (GP)')
        axes[2].set_title('full model')

        fig.tight_layout()
        fig.savefig(path.join(plot_path, '{}_mcmc_fits.png'.format(filebase)), dpi=256)
        plt.close(fig)
        # --------------------------------------------------------------------

        # --------------------------------------------------------------------
        # corner plot
        fig = corner.corner(sampler.flatchain[::10, :],
                            labels=[x.split(':')[1] for x in gp.get_parameter_names()])
        fig.savefig(path.join(plot_path, '{}_corner.png'.format(filebase)), dpi=256)
        plt.close(fig)

        x0 = sampler.flatchain[:, 3] * u.angstrom
        MAD = np.median(np.abs(x0 - np.median(x0)))
        v_precision = 1.48 * MAD / Halpha * c.to(u.km/u.s)
        centroid = np.median(x0)
        rv = (centroid - Halpha) / Halpha * c.to(u.km/u.s)

        if '-' in object_name:
            group_id,*_ = object_name.split('-')

        else:
            group_id = 0

        velocity_tbl.add_row(dict(object_name=object_name, group_id=group_id,
                                  filename=file_path, wave0=centroid,
                                  rv=rv, rv_precision=v_precision))

        logger.info('{} [{}]: x0={x0:.3f} σ={err:.3f} rv={rv:.3f}'
                    .format(object_name, filebase, x0=centroid,
                            err=v_precision, rv=rv))

        velocity_tbl.write(table_path, format='ascii.ecsv',
                           overwrite=True, delimiter=',')

if __name__ == "__main__":
    from argparse import ArgumentParser
    import logging

    # Define parser object
    parser = ArgumentParser(description="")

    vq_group = parser.add_mutually_exclusive_group()
    vq_group.add_argument('-v', '--verbose', action='count', default=0, dest='verbosity')
    vq_group.add_argument('-q', '--quiet', action='count', default=0, dest='quietness')

    parser.add_argument('-s', '--seed', dest='seed', default=None,
                        type=int, help='Random number generator seed.')
    parser.add_argument('-o', '--overwrite', action='store_true', dest='overwrite',
                        default=False, help='Destroy everything.')

    args = parser.parse_args()

    # Set logger level based on verbose flags
    if args.verbosity != 0:
        if args.verbosity == 1:
            logger.setLevel(logging.DEBUG)
        else: # anything >= 2
            logger.setLevel(1)

    elif args.quietness != 0:
        if args.quietness == 1:
            logger.setLevel(logging.WARNING)
        else: # anything >= 2
            logger.setLevel(logging.ERROR)

    else: # default
        logger.setLevel(logging.INFO)

    if args.seed is not None:
        np.random.seed(args.seed)

    main(overwrite=args.overwrite)

