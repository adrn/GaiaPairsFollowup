# coding: utf-8

"""
TODO:
- n1.0073 Halpha is emission
"""

# Standard library
import os
from os import path

# Third-party
import astropy.coordinates as coord
from astropy.time import Time
from astropy.constants import c
import astropy.units as u
from astropy.io import fits
from astropy.table import Table, Column
import numpy as np
import matplotlib.pyplot as plt
plt.style.use('apw-notebook')
import emcee
import corner
import schwimmbad
from schwimmbad import choose_pool

# Project
from comoving_rv.log import logger
from comoving_rv.longslit import GlobImageFileCollection
from comoving_rv.longslit.fitting import fit_spec_line_GP, gp_to_fit_pars
from comoving_rv.longslit.models import voigt_polynomial
from comoving_rv.velocity import bary_vel_corr, kitt_peak

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

def main(night_path, overwrite=False, pool=None):

    if pool is None:
        pool = schwimmbad.SerialPool()

    night_path = path.realpath(path.expanduser(night_path))
    if not path.exists(night_path):
        raise IOError("Path '{}' doesn't exist".format(night_path))

    if path.isdir(night_path):
        data_file = None
        logger.info("Reading data from path: {}".format(night_path))

    elif path.isfile(night_path):
        data_file = night_path
        base_path, name = path.split(night_path)
        night_path = base_path
        logger.info("Reading file: {}".format(data_file))

    else:
        raise RuntimeError("how?!")

    plot_path = path.join(night_path, 'plots')
    root_path = path.abspath(path.join(night_path, '..'))
    table_path = path.join(root_path, 'velocity.fits')

    # air wavelength of Halpha -- wavelength calibration from comp lamp is done
    #   at air wavelengths, so this is where Halpha should be, right?
    Halpha = 6562.8 * u.angstrom

    # [OI] emission lines -- wavelengths from:
    #   http://physics.nist.gov/PhysRefData/ASD/lines_form.html
    sky_lines = [5577.3387, 6300.304, 6363.776]

    if not path.exists(plot_path):
        os.makedirs(plot_path, exist_ok=True)

    if not path.exists(table_path):
        logger.debug('Creating table at {}'.format(table_path))
        tbl_init = [Column(name='object_name', dtype='|S30', data=[], length=0),
                    Column(name='group_id', dtype=int, length=0),
                    Column(name='smoh_index', dtype=int, length=0),
                    Column(name='ra', dtype=float, unit=u.degree, length=0),
                    Column(name='dec', dtype=float, unit=u.degree, length=0),
                    Column(name='secz', dtype=float, length=0),
                    Column(name='filename', dtype='|S128', data=[], length=0),
                    Column(name='Ha_centroid', dtype=float, unit=u.angstrom, length=0),
                    Column(name='Ha_centroid_err', dtype=float, unit=u.angstrom, length=0),
                    Column(name='bary_rv_shift', dtype=float, unit=u.km/u.s, length=0),
                    Column(name='sky_shift_flag', dtype=int, length=0),
                    Column(name='sky_wave_shift', dtype=float, unit=u.angstrom, length=0, shape=(len(sky_lines,))),
                    Column(name='rv', dtype=float, unit=u.km/u.s, length=0),
                    Column(name='rv_err', dtype=float, unit=u.km/u.s, length=0)]

        velocity_tbl = Table(tbl_init)
        velocity_tbl.write(table_path, format='fits')
        logger.debug('Table: {}'.format(velocity_tbl.colnames))

    else:
        logger.debug('Table exists, reading ({})'.format(table_path))
        velocity_tbl = Table.read(table_path, format='fits')

    if data_file is None:
        ic = GlobImageFileCollection(night_path, glob_include='1d_*')
        files = ic.files_filtered(imagetyp='OBJECT')
    else:
        files = [data_file]

    for filename in files:
        file_path = path.join(night_path, filename)
        filebase,ext = path.splitext(filename)

        # read FITS header
        hdr = fits.getheader(file_path, 0)
        object_name = hdr['OBJECT']

        # HACK: for testing
        # if 'HIP' not in object_name:
        #     continue

        if object_name in velocity_tbl['object_name']:
            if overwrite:
                logger.debug('Object {} already done - overwriting!'
                             .format(object_name))
                idx, = np.where(velocity_tbl['object_name'] == object_name)
                for i in idx:
                    velocity_tbl.remove_row(i)

            else:
                logger.debug('Object {} already done.'.format(object_name))
                continue

        # read the spectrum data and get wavelength solution
        spec = Table.read(file_path)

        # Define data arrays to be used in fitting below
        near_Ha = (np.isfinite(spec['wavelength']) &
                   (spec['wavelength'] > 6510) & (spec['wavelength'] < 6615))
        flux_data = np.array(spec['source_flux'][near_Ha])
        ivar_data = np.array(spec['source_ivar'][near_Ha])
        wave_data = np.array(spec['wavelength'][near_Ha])

        _idx = wave_data.argsort()
        wave_data = wave_data[_idx]
        flux_data = flux_data[_idx]
        ivar_data = ivar_data[_idx]
        err_data = 1/np.sqrt(ivar_data)

        # grid of wavelengths for plotting
        wave_grid = np.linspace(wave_data.min(), wave_data.max(), 256)

        # start by doing a maximum likelihood GP fit

        # TODO: figure out if it's emission or absorption...for now just assume
        #   absorption
        absorp_emiss = -1.
        gp = fit_spec_line_GP(wave_data, flux_data, ivar_data,
                              absorp_emiss=absorp_emiss,
                              fwhm_L0=4., std_G0=1., n_bg_coef=2)

        if gp.get_parameter_dict()['mean:ln_amp'] < 0.5: # MAGIC NUMBER
            # try again with emission line
            logger.error('absorption line has tiny amplitude! did '
                         'auto-determination of absorption/emission fail?')
            # TODO: what now?
            continue

        fit_pars = gp_to_fit_pars(gp, absorp_emiss)

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
        fig.savefig(path.join(plot_path, '{}_maxlike.png'.format(filebase)),
                    dpi=256)
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

        sampler = emcee.EnsembleSampler(nwalkers, ndim, log_probability,
                                        pool=pool, args=(gp, flux_data))

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

        pool.close()

        # --------------------------------------------------------------------
        # plot MCMC traces
        fig,axes = plt.subplots(2,4,figsize=(18,6))
        for i in range(sampler.dim):
            for walker in sampler.chain[...,i]:
                axes.flat[i].plot(walker, marker='',
                                  drawstyle='steps-mid', alpha=0.2)
            axes.flat[i].set_title(gp.get_parameter_names()[i], fontsize=12)
        fig.tight_layout()
        fig.savefig(path.join(plot_path, '{}_mcmc_trace.png'.format(filebase)),
                    dpi=256)
        plt.close(fig)
        # --------------------------------------------------------------------

        # --------------------------------------------------------------------
        # plot samples
        fig,axes = plt.subplots(3, 1, figsize=(6,9), sharex=True)

        samples = sampler.flatchain
        for s in samples[np.random.randint(len(samples), size=32)]:
            gp.set_parameter_vector(s)

            fit_pars = gp_to_fit_pars(gp, absorp_emiss)
            _mean_model = voigt_polynomial(wave_grid, **fit_pars)
            axes[0].plot(wave_grid, _mean_model,
                         marker='', alpha=0.25, color='#3182bd', zorder=-10)

            mu = gp.predict(flux_data, wave_grid, return_cov=False)
            axes[1].plot(wave_grid, mu-_mean_model, color=gp_color,
                         alpha=0.25, marker='')
            axes[2].plot(wave_grid, mu, color='#756bb1',
                         alpha=0.25, marker='')

        axes[2].plot(wave_data, flux_data, drawstyle='steps-mid',
                     marker='', zorder=-6)
        axes[2].errorbar(wave_data, flux_data, err_data,
                         marker='', ls='none', ecolor='#666666', zorder=-10)

        axes[2].set_ylabel('flux')
        axes[2].set_xlabel(r'wavelength [$\AA$]')
        axes[0].set_title('mean model (voigt + poly.)')
        axes[1].set_title('noise model (GP)')
        axes[2].set_title('full model')

        fig.tight_layout()
        fig.savefig(path.join(plot_path, '{}_mcmc_fits.png'.format(filebase)),
                    dpi=256)
        plt.close(fig)
        # --------------------------------------------------------------------

        # --------------------------------------------------------------------
        # corner plot
        fig = corner.corner(sampler.flatchain[::10, :],
                            labels=[x.split(':')[1]
                                    for x in gp.get_parameter_names()])
        fig.savefig(path.join(plot_path, '{}_corner.png'.format(filebase)),
                    dpi=256)
        plt.close(fig)
        # --------------------------------------------------------------------

        # object naming stuff
        if '-' in object_name:
            group_id,smoh_index,*_ = object_name.split('-')
            smoh_index = int(smoh_index)

        else:
            group_id = 0
            smoh_index = 0

        # Now estimate raw radial velocity and precision:
        x0 = sampler.flatchain[:, 3] * u.angstrom
        MAD = np.median(np.abs(x0 - np.median(x0)))
        centroid = np.median(x0)
        centroid_err = 1.48 * MAD # convert to stddev

        # compute shifts for sky lines, uncertainty, quality flag
        width = 100. # window size in angstroms, centered on line
        absorp_emiss = 1. # all sky lines are emission lines

        wave_shifts = np.full(len(sky_lines), np.nan) * u.angstrom
        for j,sky_line in enumerate(sky_lines):
            mask = ((spec['wavelength'] > (sky_line-width/2)) &
                    (spec['wavelength'] < (sky_line+width/2)))
            flux_data = spec['background_flux'][mask]
            ivar_data = spec['background_ivar'][mask]
            wave_data = spec['wavelength'][mask]

            _idx = wave_data.argsort()
            wave_data = wave_data[_idx]
            flux_data = flux_data[_idx]
            ivar_data = ivar_data[_idx]
            err_data = 1/np.sqrt(ivar_data)

            gp = fit_spec_line_GP(wave_data, flux_data, ivar_data,
                                  absorp_emiss=absorp_emiss,
                                  fwhm_L0=2., std_G0=1., n_bg_coef=2)

            pars = gp.get_parameter_dict()
            dlam = sky_line - pars['mean:x0']

            if ((pars['mean:ln_fwhm_L'] < -0.5 and pars['mean:ln_std_G'] < (-0.5)) or
                    pars['mean:ln_amp'] > 10. or pars['mean:ln_amp'] < 3.5):
                title = 'fucked'

            else:
                title = '{:.2f}'.format(pars['mean:ln_amp'])
                wave_shifts[j] = dlam * u.angstrom

            # Make the maximum likelihood prediction
            wave_grid = np.linspace(wave_data.min(), wave_data.max(), 256)
            mu, var = gp.predict(flux_data, wave_grid, return_var=True)
            std = np.sqrt(var)

            # ----------------------------------------------------------------
            # Plot the fit and data
            fig,ax = plt.subplots(1,1)

            # data
            ax.plot(wave_data, flux_data, drawstyle='steps-mid', marker='')
            ax.errorbar(wave_data, flux_data, err_data,
                        marker='', ls='none', ecolor='#666666', zorder=-10)

            # full GP model
            ax.plot(wave_grid, mu, color=gp_color, marker='')
            ax.fill_between(wave_grid, mu+std, mu-std,
                            color=gp_color, alpha=0.3, edgecolor="none")
            ax.set_title(title)
            fig.tight_layout()
            fig.savefig(path.join(plot_path, '{}_maxlike_sky_{:.0f}.png'
                                  .format(filebase, sky_line)), dpi=256)
            plt.close(fig)

        # some quality checks on sky velocity shifts
        filter_ = np.isnan(wave_shifts) | (np.abs(wave_shifts) > (5*u.angstrom))
        wave_shifts[filter_] = np.nan * u.angstrom

        # compute barycenter velocity given coordinates of where the
        #   telescope was pointing
        t = Time(hdr['JD'], format='jd', scale='utc')
        sc = coord.SkyCoord(ra=hdr['RA'], dec=hdr['DEC'],
                            unit=(u.hourangle, u.degree))
        vbary = bary_vel_corr(t, sc, location=kitt_peak)

        # sky shift flag is:
        #   - 0 if both lines were fit
        #   - 1 if only 6300
        #   - 2 if neither lines are there
        sky_flag = np.isnan(wave_shifts).sum()

        shift = np.nanmedian(wave_shifts, axis=-1)
        if np.isnan(shift):
            shift = 0. * u.angstrom

        rv = (centroid + shift - Halpha) / Halpha * c.to(u.km/u.s) + vbary
        rv_err = centroid_err / Halpha * c.to(u.km/u.s) # formal precision

        # convert ra,dec to quantities
        ra = sc.ra.degree * u.deg
        dec = sc.dec.degree * u.deg
        velocity_tbl.add_row(dict(object_name=object_name, group_id=group_id,
                                  smoh_index=smoh_index,
                                  ra=ra, dec=dec, secz=hdr['AIRMASS'],
                                  filename=file_path,
                                  Ha_centroid=centroid,
                                  Ha_centroid_err=centroid_err,
                                  bary_rv_shift=vbary,
                                  sky_shift_flag=sky_flag,
                                  sky_wave_shift=wave_shifts,
                                  rv=rv,
                                  rv_err=rv_err))

        logger.info('{} [{}]: x0={x0:.3f} σ={err:.3f} rv={rv:.3f}'
                    .format(object_name, filebase, x0=centroid,
                            err=centroid_err, rv=rv))

        velocity_tbl.write(table_path, format='fits', overwrite=True)

if __name__ == "__main__":
    from argparse import ArgumentParser
    import logging

    # Define parser object
    parser = ArgumentParser(description="")

    vq_group = parser.add_mutually_exclusive_group()
    vq_group.add_argument('-v', '--verbose', action='count',
                          default=0, dest='verbosity')
    vq_group.add_argument('-q', '--quiet', action='count',
                          default=0, dest='quietness')

    parser.add_argument('-s', '--seed', dest='seed', default=None,
                        type=int, help='Random number generator seed.')
    parser.add_argument('-o', '--overwrite', action='store_true',
                        dest='overwrite', default=False,
                        help='Destroy everything.')

    parser.add_argument('-p', '--path', dest='night_path', required=True,
                        help='Path to a PROCESSED night or chunk of data to '
                             'process. Or, path to a specific comp file.')

    # multiprocessing options
    group = parser.add_mutually_exclusive_group()
    group.add_argument('--ncores', dest='n_cores', default=1,
                       type=int, help='Number of CPU cores to use.')
    group.add_argument('--mpi', dest='mpi', default=False,
                       action='store_true', help='Run with MPI.')

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

    pool = choose_pool(mpi=args.mpi, processes=args.n_cores)
    logger.info("Using pool: {}".format(pool.__class__))

    main(night_path=args.night_path, overwrite=args.overwrite, pool=pool)
