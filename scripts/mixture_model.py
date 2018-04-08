# Standard library
from math import log
import os
import sys

# Third-party
from astropy.io import fits
import astropy.units as u
import matplotlib.pyplot as plt
import numpy as np
from tqdm import tqdm
import emcee
from emcee.utils import MPIPool

# Project
from gwb.data import TGASData

def ln_normal(x, mu, var):
    return -0.5*((x-mu)**2 / var + np.log(2*np.pi*var))

class MixtureModel:

    def __init__(self, data1, data2, field_vdisp=27.*u.km/u.s, name=''):
        """
        Parameters
        ----------
        data1 : `~gwb.data.TGASData`
            Data for one star in each pair.
        data2 : `~gwb.data.TGASData`
            Data for the other star in each pair.
        """

        assert len(data1) == len(data2)

        self.n_data = len(data1)
        self.name = name

        self._field_vdisp = field_vdisp.to(u.km/u.s).value

        self.rv1 = data1.rv.to(u.km/u.s).value
        self.rv_err1 = data1.rv_err.to(u.km/u.s).value

        self.rv2 = data2.rv.to(u.km/u.s).value
        self.rv_err2 = data2.rv_err.to(u.km/u.s).value

    def ln_likelihood(self, p):
        f = p[0]

        comov_term = ln_normal(self.rv1-self.rv2, 0., self.rv_err1**2 + self.rv_err2**2 + 2.5**2)
        field_term = ln_normal(self.rv1-self.rv2, 0., self.rv_err1**2 + self.rv_err2**2 + self._field_vdisp**2)

        return np.logaddexp(comov_term + np.log(f), field_term + np.log(1-f)), (comov_term, field_term)

    def ln_prior(self, p):
        f = p[0]

        if f <= 0 or f >= 1:
            return -np.inf

        return 0.

    def ln_posterior(self, p):
        lp = self.ln_prior(p)
        if not np.isfinite(lp):
            return -np.inf, None

        ll, blobs = self.ln_likelihood(p)
        if np.any(np.logical_not(np.isfinite(ll))):
            return -np.inf, None

        return lp + ll.sum(), blobs

    def __call__(self, p):
        # print(p[0])
        return self.ln_posterior(p)

def plot_posterior(mm):
    # Test: plot the posterior curve
    lls = []
    fs = np.linspace(0.15, 0.7, 32)
    for f in tqdm(fs):
        ll = mm.ln_likelihood([f])[0].sum()
        print(f, ll)
        lls.append(ll)

    lls = np.array(lls)
    plt.plot(fs, np.exp(lls - lls.max()))
    plt.show()

def run_emcee(model, pool, chain_file, blobs_file):

    n_walkers = 28
    n_steps = 1024
    n_batches = 8

    p0 = np.random.normal(0.5, 1E-3, size=(n_walkers, 1))
    sampler = emcee.EnsembleSampler(n_walkers, 1, model)

    for batch in range(n_batches):
        print("Batch: {0}".format(batch))

        pos, *_ = sampler.run_mcmc(p0, n_steps // n_batches)
        p0 = pos

        np.save('../data/sampler_chain{0}_{1}.npy'.format(batch, model.name),
                sampler.chain)
        np.save('../data/sampler_blobs{0}_{1}.npy'.format(batch, model.name),
                sampler.blobs)

        sampler.reset()

    # Now collect all the individual files into one...
    chains = []
    blobs = []
    for batch in range(n_batches):
        chains.append(np.load('../data/sampler_chain{0}_{1}.npy'
                              .format(batch, model.name)))
        blobs.append(np.load('../data/sampler_blobs{0}_{1}.npy'
                             .format(batch, model.name)))

    chain = np.hstack(chains)
    blobs = np.vstack(blobs)
    np.save(chain_file, chain)
    np.save(blobs_file, blobs)

    # Now clean up / delete the files
    for batch in range(n_batches):
        os.remove('../data/sampler_chain{0}_{1}.npy'.format(batch, model.name))
        os.remove('../data/sampler_blobs{0}_{1}.npy'.format(batch, model.name))

def analyze_chain(chain, blobs, probs_file):
    # MAGIC NUMBER: index after which walkers are converged
    ix = 256
    trim_chain = chain[:,ix:]
    trim_blobs = blobs[ix:]

    # Downsample chains because correlation
    flat_f = np.vstack(trim_chain[:,::8])[:,0]
    med_f = np.median(flat_f)
    std_f = 1.5 * np.median(np.abs(flat_f - med_f))
    print('f = {0:.2f} +/- {1:.2f}'.format(med_f, std_f))

    # Now we compute the per-pair probability
    norm = 0.0
    post_prob = np.zeros(blobs.shape[-1])
    for i in range(trim_chain.shape[1]): # steps
        for j in range(trim_chain.shape[0]): # walkers
            ll_fg, ll_bg = trim_blobs[i][j]
            post_prob += np.exp(ll_fg - np.logaddexp(ll_fg, ll_bg))
            norm += 1
    post_prob /= norm

    np.save(probs_file, post_prob)

if __name__ == "__main__":
    import schwimmbad
    from argparse import ArgumentParser

    # Define parser object
    parser = ArgumentParser(description="")
    parser.add_argument('--mpi', action='store_true', default=False,
                        dest='mpi')
    parser.add_argument('--sim', action='store_true', default=False,
                        dest='simulated_data')
    parser.add_argument('--name', required=True, dest='name',
                        help='Name of the data - can be "apw" or "rave"')

    args = parser.parse_args()

    if args.mpi:
        pool = MPIPool()

        if not pool.is_master():
            pool.wait()
            sys.exit(0)

    else:
        pool = schwimmbad.SerialPool()

    if args.simulated_data:
        print("Loading simulated data")

        # Load simulated data
        _tbl1 = fits.getdata('../notebooks/data1.fits')
        data1 = TGASData(_tbl1, rv=_tbl1['RV']*u.km/u.s,
                         rv_err=_tbl1['RV_err']*u.km/u.s)

        _tbl2 = fits.getdata('../notebooks/data2.fits')
        data2 = TGASData(_tbl2, rv=_tbl2['RV']*u.km/u.s,
                         rv_err=_tbl2['RV_err']*u.km/u.s)

    else:
        print("Loading real data")

        if args.name not in ['apw', 'rave']:
            raise ValueError("Invalid name '{0}'".format(args.name))

        # Load real data
        _tbl1 = fits.getdata('../data/tgas_{0}1.fits'.format(args.name))
        data1 = TGASData(_tbl1, rv=_tbl1['RV']*u.km/u.s,
                         rv_err=_tbl1['RV_err']*u.km/u.s)

        _tbl2 = fits.getdata('../data/tgas_{0}2.fits'.format(args.name))
        data2 = TGASData(_tbl2, rv=_tbl2['RV']*u.km/u.s,
                         rv_err=_tbl2['RV_err']*u.km/u.s)

    print("Data loaded, creating model...")

    mm = MixtureModel(data1, data2, name=args.name, field_vdisp=25.*u.km/u.s)
    print("Model created")
    # plot_posterior(data1, data2)

    chain_file = '../data/sampler_chain_{0}.npy'.format(args.name)
    blobs_file = '../data/sampler_blobs_{0}.npy'.format(args.name)

    if not os.path.exists(chain_file):
        print("Couldn't find cached chain file - starting sampling")
        run_emcee(mm, pool, chain_file=chain_file, blobs_file=blobs_file)
        pool.close()

    analyze_chain(np.load(chain_file),
                  np.load(blobs_file),
                  '../data/pair_probs_{0}.npy'.format(args.name))

    sys.exit(0)
