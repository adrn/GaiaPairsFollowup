"""
Fit isochrones to the stars in the 3 highlighted pairs

"""

# Standard library
from collections import OrderedDict
from os import path
import os

# Third-party
from astropy.table import Table
import matplotlib.pyplot as plt
import numpy as np
import emcee
from tqdm import tqdm

# Package
from comoving_rv.log import logger
from comoving_rv.db import Session, Base, db_connect
from comoving_rv.db.model import (Run, Observation, TGASSource, SimbadInfo,
                                  GroupToObservations, SpectralLineInfo,
                                  SpectralLineMeasurement, RVMeasurement)

from isochrones import StarModel
# from isochrones.mist import MIST_Isochrone
# iso = MIST_Isochrone() # interpolation issues with MIST isochrones
from isochrones.dartmouth import Dartmouth_Isochrone
iso = Dartmouth_Isochrone()

def main():
    # TODO: bad, hard-coded...
    # base_path = '/Volumes/ProjectData/gaia-comoving-followup/'
    base_path = '../../data/'
    db_path = path.join(base_path, 'db.sqlite')
    engine = db_connect(db_path)
    session = Session()

    chain_path = path.abspath('./isochrone_chains')
    os.makedirs(chain_path, exist_ok=True)

    # Check out the bottom of "Color-magnitude diagram.ipynb":
    interesting_group_ids = [1500, 1229, 1515]

    all_photometry = OrderedDict([
        ('1500-8455', OrderedDict([('J', (6.8379998, 0.021)),
                                   ('H', (6.4640002, 0.017000001)),
                                   ('K', (6.3369999, 0.017999999)),
                                   ('W1', (6.2950001, 0.093000002)),
                                   ('W2', (6.2490001, 0.026000001)),
                                   ('W3', (6.3330002, 0.015)),
                                   ('B', (9.5950003, 0.022)),
                                   ('V', (8.5120001, 0.014))])),
        ('1500-1804', OrderedDict([('J', (6.9039998, 0.041000001)),
                                   ('H', (6.8559999, 0.027000001)),
                                   ('K', (6.7989998, 0.017000001)),
                                   ('W1', (6.803, 0.064999998)),
                                   ('W2', (6.7600002, 0.018999999)),
                                   ('W3', (6.8270001, 0.016000001)),
                                   ('B', (7.4980001, 0.015)),
                                   ('V', (7.289, 0.011))])),
        ('1229-1366', OrderedDict([('J', (6.7290001, 0.024)),
                                   ('H', (6.2449999, 0.02)),
                                   ('K', (6.1529999, 0.023)),
                                   ('W1', (6.1799998, 0.096000001)),
                                   ('W2', (6.04, 0.035)),
                                   ('W3', (6.132, 0.016000001)),
                                   ('B', (9.5539999, 0.021)),
                                   ('V', (8.4619999, 0.014))])),
        ('1229-7470', OrderedDict([('J', (9.1709995, 0.024)),
                                   ('H', (8.7959995, 0.026000001)),
                                   ('K', (8.7299995, 0.022)),
                                   ('W1', (8.6669998, 0.023)),
                                   ('W2', (8.7189999, 0.02)),
                                   ('W3', (8.6680002, 0.025)),
                                   ('B', (11.428, 0.054000001)),
                                   ('V', (10.614, 0.039999999))])),
        ('1515-3584', OrderedDict([('J', (5.363999843597412, 0.024000000208616257)),
                                   ('H', (4.965000152587891, 0.035999998450279236)),
                                   ('K', (4.815999984741211, 0.032999999821186066)),
                                   ('W1', (4.758, 0.215)),
                                   ('W2', (4.565, 0.115)),
                                   ('W3', (4.771, 0.015)),
                                   ('B', (8.347999572753906, 0.01600000075995922)),
                                   ('V', (7.182000160217285, 0.009999999776482582))])),
        ('1515-1834', OrderedDict([('J', (8.855999946594238, 0.024000000208616257)),
                                   ('H', (8.29699993133545, 0.020999999716877937)),
                                   ('K', (8.178999900817871, 0.017999999225139618)),
                                   ('W1', (8.117, 0.022)),
                                   ('W2', (8.15, 0.019)),
                                   ('W3', (8.065, 0.02)),
                                   ('B', (12.309000015258789, 0.11999999731779099)),
                                   ('V', (11.069999694824219, 0.054999999701976776))]))
    ])

    for k in all_photometry:
        samples_file = path.join(chain_path, '{0}.hdf5'.format(k))

        if path.exists(samples_file):
            logger.info("skipping {0} - samples exist at {1}"
                        .format(k, samples_file))
            continue

        phot = all_photometry[k]
        obs = session.query(Observation).filter(Observation.object == k).one()
        plx = (obs.tgas_source.parallax, obs.tgas_source.parallax_error)

        # fit an isochrone
        model = StarModel(iso, use_emcee=True, parallax=plx, **phot)
        model.set_bounds(mass=(0.01, 20),
                         feh=(-1, 1),
                         distance=(0, 300),
                         AV=(0, 1))

        # initial conditions for emcee walkers
        nwalkers = 128

        p0 = []
        m0, age0, feh0 = model.ic.random_points(nwalkers,
                                                minmass=0.01, maxmass=10.,
                                                minfeh=-1, maxfeh=1)
        _, max_distance = model.bounds('distance')
        _, max_AV = model.bounds('AV')
        d0 = 10**(np.random.uniform(0,np.log10(max_distance),size=nwalkers))
        AV0 = np.random.uniform(0, max_AV, size=nwalkers)
        p0 += [m0]
        p0 += [age0, feh0, d0, AV0]

        p0 = np.array(p0).T
        npars = p0.shape[1]

        # run emcee
        ninit = 256
        nburn = 1024
        niter = 4096

        logger.debug('Running emcee - initial sampling...')
        sampler = emcee.EnsembleSampler(nwalkers, npars, model.lnpost)
        # pos, prob, state = sampler.run_mcmc(p0, ninit)

        for pos, prob, state in tqdm(sampler.sample(p0, iterations=ninit),
                                     total=ninit):
            pass

        # cull the weak walkers
        best_ix = sampler.flatlnprobability.argmax()
        best_p0 = (sampler.flatchain[best_ix][None] +
                   np.random.normal(0, 1E-5, size=(nwalkers, npars)))

        sampler.reset()
        logger.debug('burn-in...')
        for pos, prob, state in tqdm(sampler.sample(best_p0, iterations=nburn),
                                     total=nburn):
            pass
        # pos,_,_ = sampler.run_mcmc(best_p0, nburn)

        sampler.reset()
        logger.debug('sampling...')
        # _ = sampler.run_mcmc(pos, niter)
        for pos, prob, state in tqdm(sampler.sample(pos, iterations=niter),
                                     total=niter):
            pass

        model._sampler = sampler
        model._make_samples(0.08)

        model.samples.to_hdf(samples_file, key='samples')
        # np.save('isochrone_chains/chain.npy', sampler.chain)
        logger.debug('...done and saved!')

if __name__ == '__main__':
    import logging
    logger.setLevel(logging.DEBUG)
    main()
