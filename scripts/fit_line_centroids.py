# coding: utf-8

"""
TODO:
- n1.0073 Halpha is emission
"""

# Standard library
import os
from os import path

# Third-party
import astropy.units as u
from astropy.table import Table
import numpy as np
import matplotlib.pyplot as plt
import emcee
import corner
import schwimmbad
from schwimmbad import choose_pool

# Project
from comoving_rv.log import logger
from comoving_rv.db import Session, db_connect
from comoving_rv.longslit import extract_region
from comoving_rv.longslit.fitting import VoigtLineFitter, GaussianLineFitter
from comoving_rv.db.model import (SpectralLineMeasurement, SpectralLineInfo,
                                  Run, Observation)

def log_probability(params, gp, flux_data):
    gp.set_parameter_vector(params)
    lp = gp.log_prior()
    if not np.isfinite(lp):
        return -np.inf

    # HACK: Gaussian prior on log(rho)
    var = 1.
    lp += -0.5*(params[1]-1)**2/var - 0.5*np.log(2*np.pi*var)

    if (params[4] < -8. or params[4] > 2 or params[5] < -8. or params[5] > 2):
        return -np.inf

    ll = gp.log_likelihood(flux_data)
    if not np.isfinite(ll):
        return -np.inf

    return ll + lp

def main(db_path, run_name, data_root_path=None,
         filename=None, overwrite=False, pool=None):

    if pool is None:
        pool = schwimmbad.SerialPool()

    # connect to the database
    engine = db_connect(db_path)
    # engine.echo = True
    logger.debug("Connected to database at '{}'".format(db_path))

    # create a new session for interacting with the database
    session = Session()

    root_path, _ = path.split(db_path)
    if data_root_path is None:
        data_root_path = root_path

    plot_path = path.join(root_path, 'plots', run_name)
    if not path.exists(plot_path):
        os.makedirs(plot_path, exist_ok=True)

    # TODO: there might be some bugs here...
    n_lines = session.query(SpectralLineInfo).count()
    Halpha = session.query(SpectralLineInfo)\
                    .filter(SpectralLineInfo.name == 'Halpha').one()
    OI_lines = session.query(SpectralLineInfo)\
                      .filter(SpectralLineInfo.name.contains('[OI]')).all()

    if filename is None: # grab all unfinished sources
        observations = session.query(Observation).join(Run)\
                              .filter(Run.name == run_name).all()

    else: # only process the observation corresponding to this filename
        observations = session.query(Observation).join(Run)\
                              .filter(Run.name == run_name)\
                              .filter(Observation.filename_raw == filename).all()

    for obs in observations:
        measurements = session.query(SpectralLineMeasurement)\
                              .join(Observation)\
                              .filter(Observation.id == obs.id).all()

        if len(measurements) == n_lines and not overwrite:
            logger.debug('All line measurements already complete for object '
                         '{0} in file {1}'.format(obs.object, obs.filename_raw))
            continue

        # Read the spectrum data and get wavelength solution
        filebase, _ = path.splitext(obs.filename_1d)
        filename_1d = obs.path_1d(data_root_path)
        spec = Table.read(filename_1d)
        logger.debug('Loaded 1D spectrum for object {0} from file {1}'
                     .format(obs.object, filename_1d))

        # Extract region around Halpha
        x, (flux, ivar) = extract_region(spec['wavelength'],
                                         center=Halpha.wavelength.value,
                                         width=100,
                                         arrs=[spec['source_flux'],
                                               spec['source_ivar']])

        # We start by doing maximum likelihood estimation to fit the line, then
        # use the best-fit parameters to initialize an MCMC run.
        # TODO: need to figure out if it's emission or absorption...for now just
        #   assume absorption
        absorp_emiss = -1.
        lf = VoigtLineFitter(x, flux, ivar, absorp_emiss=absorp_emiss)
        lf.fit()
        fit_pars = lf.get_gp_mean_pars()

        if (not lf.success or
                abs(fit_pars['x0'] - Halpha.wavelength.value) > 16. or # 16 Å = ~700 km/s
                abs(fit_pars['amp']) < 10): # minimum amplitude - MAGIC NUMBER
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
        if initial[4] < -10: # TODO: ???
            initial[4] = -8.
        if initial[5] < -10: # TODO: ???
            initial[5] = -8.
        ndim, nwalkers = len(initial), 64

        sampler = emcee.EnsembleSampler(nwalkers, ndim, log_probability,
                                        pool=pool, args=(lf.gp, flux))

        logger.debug("Running burn-in...")
        p0 = initial + 1e-6 * np.random.randn(nwalkers, ndim)
        p0, lp, _ = sampler.run_mcmc(p0, 128)

        logger.debug("Running 2nd burn-in...")
        sampler.reset()
        p0 = p0[lp.argmax()] + 1e-3 * np.random.randn(nwalkers, ndim)
        p0, lp, _ = sampler.run_mcmc(p0, 512)

        logger.debug("Running production...")
        sampler.reset()
        pos, lp, _ = sampler.run_mcmc(p0, 1024)

        fit_kw = dict()
        for i,par_name in enumerate(lf.gp.get_parameter_names()):
            if 'kernel' in par_name: continue

            # remove 'mean:'
            par_name = par_name[5:]

            # skip bg
            if par_name.startswith('bg'): continue

            samples = sampler.flatchain[:,i]

            if par_name.startswith('ln_'):
                par_name = par_name[3:]
                samples = np.exp(samples)

            MAD = np.median(np.abs(samples - np.median(samples)))
            fit_kw[par_name] = np.median(samples)
            fit_kw[par_name+'_error'] = 1.5 * MAD # convert to ~stddev

        # remove all previous line measurements
        q = session.query(SpectralLineMeasurement).join(Observation)\
                   .filter(Observation.id == obs.id)
        if q.count() > 0:
            for meas in q.all():
                session.delete(meas)
            session.commit()

        slm = SpectralLineMeasurement(**fit_kw)
        slm.info = Halpha
        slm.observation = obs
        session.add(slm)
        session.commit()

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

        # compute centroids for sky lines
        sky_centroids = []
        for j,sky_line in enumerate(OI_lines):
            wvln = sky_line.wavelength.value
            x, (flux, ivar) = extract_region(spec['wavelength'],
                                             center=wvln,
                                             width=32., # angstroms
                                             arrs=[spec['background_flux'],
                                                   spec['background_ivar']])

            lf = GaussianLineFitter(x, flux, ivar, absorp_emiss=1.) # all emission lines

            try:
                lf.fit()
                fit_pars = lf.get_gp_mean_pars()

            except Exception as e:
                logger.warn("Failed to fit sky line {0}:\n{1}".format(sky_line,
                                                                      e))
                lf.success = False
                fit_pars = lf.get_init()
                fit_pars['amp'] = 0.

            # HACK: hackish signal-to-noise
            max_ = fit_pars['amp'] / np.sqrt(2*np.pi*fit_pars['std']**2)
            SNR = max_ / np.median(1/np.sqrt(ivar))

            if (not lf.success or abs(fit_pars['x0']-wvln) > 4 or
                    fit_pars['amp'] < 10 or fit_pars['std'] > 4 or SNR < 2.5):
                # failed
                x0 = np.nan * u.angstrom
                title = 'fucked'
                fit_pars['amp'] = 0.

            else:
                x0 = fit_pars['x0'] * u.angstrom
                title = '{:.2f}'.format(fit_pars['amp'])

            if lf.success:
                fig = lf.plot_fit()
                fig.suptitle(title, y=0.95)
                fig.subplots_adjust(top=0.8)
                fig.savefig(path.join(plot_path, '{}_maxlike_sky_{:.0f}.png'
                                      .format(filebase, wvln)), dpi=256)
                plt.close(fig)

            # store the sky line measurements
            fit_pars['std_G'] = fit_pars.pop('std') # HACK
            fit_pars.pop('bg_coef') # HACK
            slm = SpectralLineMeasurement(**fit_pars)
            slm.info = sky_line
            slm.observation = obs
            session.add(slm)
            session.commit()

            sky_centroids.append(x0)
        sky_centroids = u.Quantity(sky_centroids)

        logger.info('{} [{}]: x0={x0:.3f} σ={err:.3f}\n--------'
                    .format(obs.object, filebase,
                            x0=fit_kw['x0'],
                            err=fit_kw['x0_error']))

        session.commit()

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

    parser.add_argument('-d', '--db', dest='db_path', required=True,
                        help='Path to sqlite database file')
    parser.add_argument('-r', '--run', dest='run_name', required=True,
                        help='Name of the observing run')
    parser.add_argument('-f', '--file', dest='filename', default=None,
                        help='Only run on one particular file.')
    parser.add_argument('--data-root', dest='data_root_path', default=None,
                        help='Root data path.')

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

    main(db_path=args.db_path, data_root_path=args.data_root_path,
         run_name=args.run_name, filename=args.filename,
         overwrite=args.overwrite, pool=pool)
