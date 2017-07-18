""" Fit isochrones to all stars in the confirmed pairs """

# Standard library
from collections import OrderedDict
from os import path
import sys
from pathlib import Path

# Third-party
import emcee
import numpy as np
from isochrones import StarModel
from isochrones.dartmouth import Dartmouth_Isochrone

# Package
from comoving_rv.log import logger
from comoving_rv.db import Session, Base, db_connect
from comoving_rv.db.model import (Run, Observation, TGASSource, SimbadInfo,
                                  GroupToObservations, SpectralLineInfo,
                                  SpectralLineMeasurement, RVMeasurement,
                                  Photometry)

from schwimmbad import choose_pool

# Map from `isochrones` filter name to database filter name
phot_name_map = dict()

phot_name_map['J'] = ('j_m', 'j_msigcom')
phot_name_map['H'] = ('h_m', 'h_msigcom')
phot_name_map['K'] = ('ks_m', 'ks_msigcom')

phot_name_map['W1'] = ('w1mpro', 'w1mpro_error')
phot_name_map['W2'] = ('w2mpro', 'w2mpro_error')
phot_name_map['W3'] = ('w3mpro', 'w3mpro_error')

phot_name_map['B'] = ('bt_mag', 'e_bt_mag')
phot_name_map['V'] = ('vt_mag', 'e_vt_mag')

def obs_to_mags(obs):
    """ Given an Observation, return a dict of magnitude and errors """
    p = obs.photometry

    mag_dict = dict()
    for k,v in phot_name_map.items():
        if getattr(p, v[0]) is None or getattr(p, v[1]) is None:
            continue

        mag_dict[k] = (getattr(p, v[0]), getattr(p, v[1]))

    return mag_dict

def obs_to_starmodel(obs, iso=None):
    """ Given an Observation, return a StarModel object """

    if iso is None:
        iso = Dartmouth_Isochrone()

    mags = obs_to_mags(obs)
    parallax = (obs.tgas_source.parallax, obs.tgas_source.parallax_error)

    model = StarModel(iso, use_emcee=True, parallax=parallax, **mags)
    model.set_bounds(mass=(0.01, 20),
                     feh=(-1, 1),
                     distance=(0, 300),
                     AV=(0, 1))

    return model

class Worker(object):

    def __init__(self, session, samples_path,
                 nwalkers=128, ninit=256, nburn=1024, niter=4096,
                 overwrite=False):
        self.session = session
        self.samples_path = samples_path
        self.overwrite = overwrite

        self.nwalkers = nwalkers
        self.ninit = ninit
        self.nburn = nburn
        self.niter = niter

    def work(self, id):
        obs = self.session.query(Observation).filter(Observation.id == id).one()
        model = obs_to_starmodel(obs)

        # initial conditions for emcee walkers
        p0 = []
        m0, age0, feh0 = model.ic.random_points(self.nwalkers,
                                                minmass=0.01, maxmass=10.,
                                                minfeh=-1, maxfeh=1)
        _, max_distance = model.bounds('distance')
        _, max_AV = model.bounds('AV')
        d0 = 10**(np.random.uniform(0, np.log10(max_distance),
                                    size=self.nwalkers))
        AV0 = np.random.uniform(0, max_AV, size=self.nwalkers)
        p0 += [m0]
        p0 += [age0, feh0, d0, AV0]

        p0 = np.array(p0).T
        npars = p0.shape[1]

        logger.debug('Running emcee - initial sampling...')
        sampler = emcee.EnsembleSampler(self.nwalkers, npars, model.lnpost)

        pos, prob, _ = sampler.run_mcmc(p0, self.ninit)

        # cull the weak walkers
        best_ix = sampler.flatlnprobability.argmax()
        best_p0 = (sampler.flatchain[best_ix][None] +
                   np.random.normal(0, 1E-5, size=(self.nwalkers, npars)))

        sampler.reset()
        logger.debug('burn-in...')
        pos, prob, _ = sampler.run_mcmc(best_p0, self.nburn)

        sampler.reset()
        logger.debug('sampling...')
        _ = sampler.run_mcmc(pos, self.niter)

        model._sampler = sampler
        model._make_samples(0.01)

        return id, model

    def __call__(self, args):
        return self.work(*args)

    def callback(self, result):
        if result is None:
            pass

        else:
            id, model = result # stuff

            samples_file = path.join(self.samples_path, '{0}.hdf5'.format(id))
            model.samples.to_hdf(samples_file, key='samples')

def main(db_file, pool, overwrite=False):

    # HACK:
    base_path = '../data/'
    db_path = path.join(base_path, 'db.sqlite')
    engine = db_connect(db_path)
    session = Session()

    # HACK:
    from astropy.table import Table
    tbl = Table.read('../paper/figures/group_llr_dv_tbl.ecsv',
                     format='ascii.ecsv')

    worker = Worker(session=session,
                    samples_path=path.abspath('../data/isochrone_samples'),
                    overwrite=overwrite)

    # A little bit of a hack
    comoving = tbl['R_RV'] > tbl['R_mu']
    tasks = session.query(Observation.id)\
                   .filter(Observation.group_id.in_(tbl['group_id'][comoving]))\
                   .group_by(Observation.group_id).all()
    tasks = tasks[:1]

    for r in pool.map(worker, tasks, callback=worker.callback):
        pass

    pool.close()
    sys.exit(0)

if __name__ == '__main__':
    from argparse import ArgumentParser
    import logging

    # Define parser object
    parser = ArgumentParser(description="")

    vq_group = parser.add_mutually_exclusive_group()
    vq_group.add_argument('-v', '--verbose', action='count', default=0, dest='verbosity')
    vq_group.add_argument('-q', '--quiet', action='count', default=0, dest='quietness')

    parser.add_argument('-o', '--overwrite', action='store_true', dest='overwrite',
                        default=False, help='Destroy everything.')

    group = parser.add_mutually_exclusive_group()
    group.add_argument('--ncores', dest='n_cores', default=1,
                       type=int, help='Number of CPU cores to use.')
    group.add_argument('--mpi', dest='mpi', default=False,
                       action='store_true', help='Run with MPI.')

    parser.add_argument('--db', dest='db_file', required=True,
                        help='Path to database file.')

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

    pool = choose_pool(mpi=args.mpi, processes=args.n_cores)
    logger.info("Using pool: {}".format(pool.__class__))

    main(db_file=args.db_file, pool=pool, overwrite=args.overwrite)
