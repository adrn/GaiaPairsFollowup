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
from comoving_rv.longslit import GlobImageFileCollection, extract_region
from comoving_rv.longslit.fitting import VoigtLineFitter, GaussianLineFitter
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
                    Column(name='sky_centroids', dtype=float, unit=u.angstrom, length=0, shape=(len(sky_lines,))),
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

        # Extract region around Halpha
        x, (flux, ivar) = extract_region(spec['wavelength'], center=6563.,
                                         width=100,
                                         arrs=[spec['source_flux'],
                                               spec['source_ivar']])

        # TODO: need to figure out if it's emission or absorption...for now just
        #   assume absorption
        absorp_emiss = -1.
        lf = VoigtLineFitter(x, flux, ivar, absorp_emiss=absorp_emiss)
        lf.fit()
        fit_pars = lf.get_gp_mean_pars()

        if (not lf.success or
                abs(fit_pars['x0']-6562.8) > 16. or # 16 Å = ~700 km/s
                abs(fit_pars['amp']) < 10): # MAGIC NUMBER
            # TODO: should try again with emission line
            logger.error('absorption line has tiny amplitude! did '
                         'auto-determination of absorption/emission fail?')
            # TODO: what now?
            continue

        fig = lf.plot_fit()
        fig.savefig(path.join(plot_path, '{}_maxlike.png'.format(filebase)),
                    dpi=256)
        plt.close(fig)
        # ----------------------------------------------------------------------

        # Run `emcee` instead to sample over GP model parameters:
        if fit_pars['std_G'] < 1E-2:
            lf.gp.freeze_parameter('mean:ln_std_G')

        initial = np.array(lf.gp.get_parameter_vector())
        if initial[4] < -10:
            initial[4] = -8.
        if initial[5] < -10:
            initial[5] = -8.
        ndim, nwalkers = len(initial), 64

        sampler = emcee.EnsembleSampler(nwalkers, ndim, log_probability,
                                        pool=pool, args=(lf.gp, flux))

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
                axes.flat[i].plot(walker, marker='',
                                  drawstyle='steps-mid', alpha=0.2)
            axes.flat[i].set_title(lf.gp.get_parameter_names()[i], fontsize=12)
        fig.tight_layout()
        fig.savefig(path.join(plot_path, '{}_mcmc_trace.png'.format(filebase)),
                    dpi=256)
        plt.close(fig)
        # --------------------------------------------------------------------

        # --------------------------------------------------------------------
        # plot samples
        fig,axes = plt.subplots(3, 1, figsize=(10, 10), sharex=True)

        samples = sampler.flatchain
        for s in samples[np.random.randint(len(samples), size=32)]:
            lf.gp.set_parameter_vector(s)
            lf.plot_fit(axes=axes, fit_alpha=0.2)

        fig.tight_layout()
        fig.savefig(path.join(plot_path, '{}_mcmc_fits.png'.format(filebase)),
                    dpi=256)
        plt.close(fig)
        # --------------------------------------------------------------------

        # --------------------------------------------------------------------
        # corner plot
        fig = corner.corner(sampler.flatchain[::10, :],
                            labels=[x.split(':')[1]
                                    for x in lf.gp.get_parameter_names()])
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

        # compute centroids for sky lines
        sky_centroids = []
        for j,sky_line in enumerate(sky_lines):
            # Extract region around Halpha
            x, (flux, ivar) = extract_region(spec['wavelength'],
                                             center=sky_line,
                                             width=32., # angstroms
                                             arrs=[spec['background_flux'],
                                                   spec['background_ivar']])

            lf = GaussianLineFitter(x, flux, ivar, absorp_emiss=1.) # all emission
            lf.fit()
            fit_pars = lf.get_gp_mean_pars()

            # HACK: hackish signal-to-noise
            max_ = fit_pars['amp'] / np.sqrt(2*np.pi*fit_pars['std']**2)
            SNR = max_ / np.median(1/np.sqrt(ivar))

            if (not lf.success or abs(fit_pars['x0']-sky_line) > 4 or
                    fit_pars['amp'] < 10 or fit_pars['std'] > 4 or SNR < 2.5):
                # failed
                x0 = np.nan * u.angstrom
                title = 'fucked'
            else:
                x0 = fit_pars['x0'] * u.angstrom
                title = '{:.2f}'.format(fit_pars['amp'])

            fig = lf.plot_fit()
            fig.suptitle(title, y=0.95)
            fig.subplots_adjust(top=0.8)
            fig.savefig(path.join(plot_path, '{}_maxlike_sky_{:.0f}.png'
                                  .format(filebase, sky_line)), dpi=256)
            plt.close(fig)

            sky_centroids.append(x0)
        sky_centroids = u.Quantity(sky_centroids)

        # compute barycenter velocity given coordinates of where the
        #   telescope was pointing
        t = Time(hdr['JD'], format='jd', scale='utc')
        sc = coord.SkyCoord(ra=hdr['RA'], dec=hdr['DEC'],
                            unit=(u.hourangle, u.degree))
        vbary = bary_vel_corr(t, sc, location=kitt_peak)

        raw_rv = (centroid - Halpha) / Halpha * c.to(u.km/u.s) + vbary
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
                                  sky_centroids=sky_centroids,
                                  rv=raw_rv,
                                  rv_err=rv_err))

        logger.info('{} [{}]: x0={x0:.3f} σ={err:.3f} rv={rv:.3f}'
                    .format(object_name, filebase, x0=centroid,
                            err=centroid_err, rv=raw_rv))

        velocity_tbl.write(table_path, format='fits', overwrite=True)

    pool.close()

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
