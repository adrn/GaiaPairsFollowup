# Third-party
from astropy.io import fits
import astropy.units as u
import matplotlib.pyplot as plt
import numpy as np
from scipy.misc import logsumexp
from scipy.integrate import simps
from scipy.stats import norm
from tqdm import tqdm

# Project
from gwb.coords import get_tangent_basis
from gwb.data import TGASData

pc_mas_yr_per_km_s = (1*u.km/u.s).to(u.pc*u.mas/u.yr, u.dimensionless_angles()).value
km_s_per_pc_mas_yr = 1/pc_mas_yr_per_km_s

def get_icrs_samples(data, size=1):
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

        icrs1 = get_icrs_samples(data1, n_dv_samples)
        icrs2 = get_icrs_samples(data2, n_dv_samples)

        # We can pre-compute the tangent basis matrices given the sky positions
        # of each star
        self.M1 = get_tangent_basis(icrs1[:,0,0], icrs1[:,0,1])
        self.M2 = get_tangent_basis(icrs2[:,0,0], icrs2[:,0,1])

        # We can also pre-compute sample vectors in proper motion components
        # and in radial velocity and make units consistent. To get velocity
        # samples we just need to multiply in the distances
        self._v1_samples = np.array([icrs1[...,3] * km_s_per_pc_mas_yr,
                                     icrs1[...,4] * km_s_per_pc_mas_yr,
                                     icrs1[...,5]])

        self._v2_samples = np.array([icrs2[...,3] * km_s_per_pc_mas_yr,
                                     icrs2[...,4] * km_s_per_pc_mas_yr,
                                     icrs2[...,5]])

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
        d1_tmp = np.vstack((d1, d1, np.ones_like(d1)))
        d2_tmp = np.vstack((d2, d2, np.ones_like(d2)))

        v1_samples = np.einsum('nij,ink->jkn', self.M1,
                               self._v1_samples * d1_tmp[...,None])
        v2_samples = np.einsum('nij,ink->jkn', self.M2,
                               self._v2_samples * d2_tmp[...,None])
        return np.linalg.norm(v1_samples - v2_samples, axis=0)

    def ln_likelihood_at_d1d2(self, p, d1, d2):
        f = p[0]
        b = np.vstack((np.full(self.n_dv_samples, f),
                       np.full(self.n_dv_samples, 1-f)))

        dv_samples = self.get_dv_samples(d1, d2)
        term1 = ln_dv_pdf(dv_samples, 1.)
        term2 = ln_dv_pdf(dv_samples, self._field_vdisp)

        ll = np.zeros(term1.shape[1])
        for n in range(self.n_data):
            ll[n] += logsumexp([term1[:,n], term2[:,n]], b=b) - np.log(self.n_dv_samples)

        return ll

    def ln_likelihood(self, p):
        ll_grid = np.zeros((self.n_data, self.n_dist_grid, self.n_dist_grid))

        for i in range(self.n_dist_grid):
            for j in range(self.n_dist_grid):
                ll_grid[:,i,j] = (self.ln_likelihood_at_d1d2(p,
                                                             self.d1_grids[:,i],
                                                             self.d2_grids[:,j]) +
                                  norm.logpdf(1000/self.d1_grids[:,i],
                                              self._plx1, self._plx1_err) +
                                  norm.logpdf(1000/self.d2_grids[:,j],
                                              self._plx2, self._plx2_err))

        l_grid = np.exp(ll_grid)
        likes = np.array([simps(simps(l_grid[n], self.d2_grids[n]), self.d1_grids[n])
                          for n in range(self.n_data)])
        return np.log(likes)


def main(data1, data2):
    mm = MixtureModel(data1, data2, field_vdisp=15.*u.km/u.s)

    lls = []
    fs = np.linspace(0., 0.75, 32)
    for f in tqdm(fs):
        lls.append(mm.ln_likelihood([f]).sum())

    lls = np.array(lls)
    plt.plot(fs, np.exp(lls - lls.max()))
    plt.show()

if __name__ == "__main__":
    # Load simulated data
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

    _tbl2 = fits.getdata('../data/tgas_apw1.fits')
    data2 = TGASData(_tbl2, rv=_tbl2['RV']*u.km/u.s,
                     rv_err=_tbl2['RV_err']*u.km/u.s)

    main(data1, data2)
