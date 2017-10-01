# Standard library
from math import log
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
        print(p[0])
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

def run_emcee(model, pool):

    n_walkers = 28
    p0 = np.random.normal(0.5, 1E-3, size=(n_walkers, 1))
    sampler = emcee.EnsembleSampler(n_walkers, 1, model)
    sampler.run_mcmc(p0, 1024)

    np.save('../data/sampler.chain', sampler.chain)
    np.save('../data/sampler.blobs', sampler.blobs)

if __name__ == "__main__":
    import schwimmbad
    from argparse import ArgumentParser

    # Define parser object
    parser = ArgumentParser(description="")
    parser.add_argument('--mpi', action='store_true', default=False,
                        dest='mpi')

    args = parser.parse_args()

    if args.mpi:
        pool = schwimmbad.MPIPool()

        if not pool.is_master():
            pool.wait()
            sys.exit(0)

    else:
        pool = schwimmbad.SerialPool()

    # # Load simulated data
    # _tbl1 = fits.getdata('../notebooks/data1.fits')
    # data1 = TGASData(_tbl1, rv=_tbl1['RV']*u.km/u.s,
    #                  rv_err=_tbl1['RV_err']*u.km/u.s)

    # _tbl2 = fits.getdata('../notebooks/data2.fits')
    # data2 = TGASData(_tbl2, rv=_tbl2['RV']*u.km/u.s,
    #                  rv_err=_tbl2['RV_err']*u.km/u.s)

    # Load real data
    _tbl1 = fits.getdata('../data/tgas_apw1.fits')
    data1 = TGASData(_tbl1, rv=_tbl1['RV']*u.km/u.s,
                     rv_err=_tbl1['RV_err']*u.km/u.s)

    _tbl2 = fits.getdata('../data/tgas_apw2.fits')
    data2 = TGASData(_tbl2, rv=_tbl2['RV']*u.km/u.s,
                     rv_err=_tbl2['RV_err']*u.km/u.s)

    mm = MixtureModel(data1, data2, field_vdisp=25.*u.km/u.s)
    # plot_posterior(data1, data2)

    run_emcee(mm, pool)
