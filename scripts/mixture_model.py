# Standard library
from math import log
import os
import sys

# Third-party
from astropy.io import fits
import astropy.units as u
import matplotlib.pyplot as plt
import numpy as np
from scipy.misc import logsumexp
from scipy.integrate import simps
from scipy.stats import norm
from tqdm import tqdm
import emcee
from emcee.utils import MPIPool

# Project
from gwb.coords import get_tangent_basis
from gwb.data import TGASData

pc_mas_yr_per_km_s = (1*u.km/u.s).to(u.pc*u.mas/u.yr, u.dimensionless_angles()).value
km_s_per_pc_mas_yr = 1/pc_mas_yr_per_km_s

def get_icrs_samples(data, size=1, seed=None):
    if seed is not None:
        np.random.seed(seed)

    all_samples = []
    for i in range(len(data)):
        star = data[i]
        y = np.array([star.parallax.value, star.pmra.value, star.pmdec.value,
                      star.rv.to(u.km/u.s).value])
        Cov = star.get_cov()[2:,2:]
        Cov[-1,-1] = star.rv_err.to(u.km/u.s).value**2

        all_samples.append(np.random.multivariate_normal(y, Cov, size=size))
    all_samples = np.array(all_samples)

    ra = np.repeat(data.ra.to(u.radian).value[:,None], size, axis=1)
    dec = np.repeat(data.dec.to(u.radian).value[:,None], size, axis=1)

    # ra : radian
    # dec : radian
    # parallax : mas
    # pm_ra_cosdec : mas/yr
    # pm_dec : mas/yr
    # rv : km/s
    return np.dstack((ra[...,None], dec[...,None], all_samples))

def ln_dv_pdf(x, sigma):
    return 2*np.log(x) - x**2/(4*sigma**2) - 1.2655121234846456 - 3*np.log(sigma)

class MixtureModel:

    def __init__(self, data1, data2, n_dv_samples=512, n_dist_grid=5,
                 field_vdisp=25.*u.km/u.s):
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
        self.n_dv_samples = int(n_dv_samples)
        self.n_dist_grid = int(n_dist_grid)

        self._field_vdisp = field_vdisp.to(u.km/u.s).value

        self._icrs1 = get_icrs_samples(data1, n_dv_samples)
        self._icrs2 = get_icrs_samples(data2, n_dv_samples)

        # We can pre-compute the tangent basis matrices given the sky positions
        # of each star. We transpose it using swapaxes()
        self.M1 = np.swapaxes(get_tangent_basis(self._icrs1[:,0,0],
                                                self._icrs1[:,0,1]), 1, 2)
        self.M2 = np.swapaxes(get_tangent_basis(self._icrs2[:,0,0],
                                                self._icrs2[:,0,1]), 1, 2)

        # We can also pre-compute sample vectors in proper motion components
        # and in radial velocity and make units consistent. To get velocity
        # samples we just need to multiply in the distances
        self._v1_samples = np.array([self._icrs1[...,3].T * km_s_per_pc_mas_yr,
                                     self._icrs1[...,4].T * km_s_per_pc_mas_yr,
                                     self._icrs1[...,5].T])

        self._v2_samples = np.array([self._icrs2[...,3].T * km_s_per_pc_mas_yr,
                                     self._icrs2[...,4].T * km_s_per_pc_mas_yr,
                                     self._icrs2[...,5].T])

        self._plx1 = data1.parallax.to(u.mas).value
        self._plx1_err = data1.parallax_error.to(u.mas).value
        self._plx2 = data2.parallax.to(u.mas).value
        self._plx2_err = data2.parallax_error.to(u.mas).value

        # The last thing we can cache are the distance grids
        self.d1_grids = self._get_d_grids(self._plx1, self._plx1_err,
                                          n_dist_grid)
        self.d2_grids = self._get_d_grids(self._plx2, self._plx2_err,
                                          n_dist_grid)

    def _get_d_grids(self, plx, plx_err, size=1):
        # distance grid ends in pc
        d_min = 1000 / (plx + 3*plx_err)
        d_max = 1000 / (plx - 3*plx_err)
        return np.array([np.linspace(d_min[i], d_max[i], size)
                         for i in range(self.n_data)])

    def get_dv_samples(self, d1, d2):
        v1_tmp = self._v1_samples* np.vstack((d1, d1, np.ones_like(d1)))[:,None]
        v2_tmp = self._v2_samples* np.vstack((d2, d2, np.ones_like(d2)))[:,None]

        v1_samples = np.array([self.M1[n].dot(v1_tmp[...,n])
                               for n in range(self.n_data)])
        v2_samples = np.array([self.M2[n].dot(v2_tmp[...,n])
                               for n in range(self.n_data)])

        return np.linalg.norm(v1_samples - v2_samples, axis=1).T

    def ln_likelihood_at_d1d2(self, p, d1, d2):
        f = p[0]

        dv_samples = self.get_dv_samples(d1, d2)

        term1 = ln_dv_pdf(dv_samples, 1.) + log(f)
        term2 = ln_dv_pdf(dv_samples, self._field_vdisp) + log(1-f)

        return (logsumexp(term1, axis=0) - log(self.n_dv_samples),
                logsumexp(term2, axis=0) - log(self.n_dv_samples))

    def ln_likelihood(self, p):
        ll_grid1 = np.zeros((self.n_data, self.n_dist_grid, self.n_dist_grid))
        ll_grid2 = np.zeros((self.n_data, self.n_dist_grid, self.n_dist_grid))

        terms = self.ln_likelihood_at_d1d2(p, self.d1_grids[:,2],
                                           self.d2_grids[:,2])

        for i in range(self.n_dist_grid):
            for j in range(self.n_dist_grid):
                terms = self.ln_likelihood_at_d1d2(p,
                                                   self.d1_grids[:,i],
                                                   self.d2_grids[:,j])
                log_d_pdf = (norm.logpdf(1000 / self.d1_grids[:,i],
                                         self._plx1, self._plx1_err) +
                             norm.logpdf(1000 / self.d2_grids[:,j],
                                         self._plx2, self._plx2_err))
                ll_grid1[:,i,j] = terms[0] + log_d_pdf
                ll_grid2[:,i,j] = terms[1] + log_d_pdf

        l_grid1 = np.exp(ll_grid1)
        lls1 = np.log([simps(simps(l_grid1[n], self.d2_grids[n]),
                             self.d1_grids[n])
                       for n in range(self.n_data)])

        l_grid2 = np.exp(ll_grid2)
        lls2 = np.log([simps(simps(l_grid2[n], self.d2_grids[n]),
                             self.d2_grids[n])
                       for n in range(self.n_data)])

        return np.logaddexp(lls1, lls2), (lls1, lls2)

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

        np.save('../data/sampler_chain{0}.npy'.format(batch),
                sampler.chain)
        np.save('../data/sampler_blobs{0}.npy'.format(batch),
                sampler.blobs)

        sampler.reset()

    # Now collect all the individual files into one...
    chains = []
    blobs = []
    for batch in range(n_batches):
        chains.append(np.load('../data/sampler_chain{0}.npy'.format(batch)))
        blobs.append(np.load('../data/sampler_blobs{0}.npy'.format(batch)))

    chain = np.hstack(chains)
    blobs = np.vstack(blobs)
    np.save(chain_file, chain)
    np.save(blobs_file, blobs)

    # Now clean up / delete the files
    for batch in range(n_batches):
        os.remove('../data/sampler_chain{0}.npy'.format(batch))
        os.remove('../data/sampler_blobs{0}.npy'.format(batch))

def analyze_chain(chain, blobs):
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

    np.save('../data/pair_probs.npy', post_prob)

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

    mm = MixtureModel(data1, data2, field_vdisp=25.*u.km/u.s)
    print("Model created")
    # plot_posterior(data1, data2)

    chain_file = '../data/sampler_chain_{0}.npy'.format(args.name)
    blobs_file = '../data/sampler_blobs_{0}.npy'.format(args.name)

    if not os.path.exists(chain_file):
        print("Couldn't find cached chain file - starting sampling")
        run_emcee(mm, pool, chain_file=chain_file, blobs_file=blobs_file)
        pool.close()

    analyze_chain(np.load(chain_file),
                  np.load(blobs_file))

    sys.exit(0)
