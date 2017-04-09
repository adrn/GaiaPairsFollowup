# Standard library
from os import path

# Third-party
from astropy.nddata.nduncertainty import StdDevUncertainty
from astropy.visualization import ZScaleInterval
from astropy.table import Table
import ccdproc
from ccdproc import CCDData
import matplotlib.pyplot as plt
import numpy as np
from scipy.optimize import leastsq
from scipy.stats import scoreatpercentile

# Project
from ..log import logger

__all__ = ['CCDExtractor']

class CCDExtractor(object):

    def __init__(self, filename,
                 ccd_gain, ccd_readnoise, # ccd properties
                 oscan_idx, oscan_size, # overscan region
                 plot_path=None, zscaler=None, cmap=None, # for plotting
                 **read_kwargs):

        # read CCD frame
        self.ccd = CCDData.read(filename, **read_kwargs)
        self.filename = filename
        self._filename_base = path.splitext(path.basename(self.filename))[0]
        self._obj_name = self.ccd.header['OBJECT']

        # CCD properties
        self.ccd_gain = ccd_gain
        self.ccd_readnoise = ccd_readnoise
        self.oscan_idx = oscan_idx
        self.oscan_size = oscan_size

        # Plot settings
        self.plot_path = plot_path

        if zscaler is None:
            self.zscaler = ZScaleInterval(32768, krej=5., max_iterations=16)

        if cmap is None:
            self.cmap = 'Greys_r'

    def make_nearby_source_mask(self, mask_spec):
        """
        Construct and return a boolean array mask to remove pixels
        from the CCD image where there is a nearby source. The mask
        is True where the pixels should be removed, or where the
        uncertainty should be set to inf.

        Parameters
        ----------
        mask_spec : list
            Read from a mask specification yaml file. Should be a list
            of dicts that contain 2 keys:

                'top_bottom' : iterable
                    A length-2 iterable specifying the mask trace column
                    center at the top and bottom of the CCD region.
                'width' : numeric
                    Width of the source mask.

        """

        n_row,n_col = self.ccd.shape
        mask = np.zeros((n_row, n_col)).astype(bool)

        for spec in mask_spec:
            x1,x2 = spec['top_bottom'][::-1] # trace position to mask at top and bottom of CCD
            y1,y2 = 250,n_row-250 # remove top and bottom of CCD

            def mask_cen_func(row):
                return (x2-x1)/(y2-y1) * (row - y1) + x1

            for i in range(n_row):
                j1 = int(np.floor(mask_cen_func(i) - spec['width']/2.))
                j2 = int(np.ceil(mask_cen_func(i) + spec['width']/2.)) + 1
                j1 = max(0, j1)
                j2 = min(n_col, j2)
                mask[i,j1:j2] = 1

        return mask

    def process_raw_frame(self, master_bias, master_flat, pixel_mask_spec=None):
        """
        Bias and flat-correct a raw CCD frame. Trim off the overscan
        region. Identify cosmic rays using "lacosmic" and inflat
        uncertainties where CR's are found. If specified, mask out
        nearby sources by setting pixel uncertainty to infinity (or
        inverse-variance to 0).

        Returns
        -------
        nccd : `ccdproc.CCDData`
            A copy of the original ``CCDData`` object but after the
            above procedures have been run.
        """

        oscan_fits_section = "[{}:{},:]".format(self.oscan_idx,
                                                self.oscan_idx+self.oscan_size)

        # make a copy of the object
        nccd = self.ccd.copy()

        # apply the overscan correction
        poly_model = Polynomial1D(2)
        nccd = ccdproc.subtract_overscan(nccd, fits_section=oscan_fits_section,
                                         model=poly_model)

        # trim the image (remove overscan region)
        nccd = ccdproc.trim_image(nccd, fits_section='[1:{},:]'.format(self.oscan_idx))

        # create the error frame
        nccd = ccdproc.create_deviation(nccd, gain=self.ccd_gain,
                                        readnoise=self.ccd_readnoise)

        # now correct for the ccd gain
        nccd = ccdproc.gain_correct(nccd, gain=self.ccd_gain)

        # correct for master bias frame
        # - this does some crazy shit at the blue end, but we can live with it
        nccd = ccdproc.subtract_bias(nccd, master_bias)

        # correct for master flat frame
        nccd = ccdproc.flat_correct(nccd, master_flat)

        # comsic ray cleaning - this updates the uncertainty array as well
        nccd = ccdproc.cosmicray_lacosmic(nccd, sigclip=8.)

        # replace ccd with processed ccd
        self.ccd = nccd

        # check for a pixel mask
        if pixel_mask_spec is not None:
            mask = self.make_nearby_source_mask(pixel_mask_spec)
            logger.debug("\t\tSource mask loaded.")

            stddev = nccd.uncertainty.array
            stddev[mask] = np.inf
            nccd.uncertainty = StdDevUncertainty(stddev)

        if self.plot_path is not None:
            # TODO: this assumes vertical CCD
            aspect_ratio = nccd.shape[1]/nccd.shape[0]

            fig,axes = plt.subplots(2, 1, figsize=(10,2 * 12*aspect_ratio),
                                    sharex=True, sharey=True)

            vmin,vmax = self.zscaler.get_limits(nccd.data)
            axes[0].imshow(nccd.data.T, origin='bottom',
                           cmap=self.cmap, vmin=max(0,vmin), vmax=vmax)

            stddev = nccd.uncertainty.array
            vmin,vmax = self.zscaler.get_limits(stddev[np.isfinite(stddev)])
            axes[1].imshow(stddev.T, origin='bottom',
                           cmap=self.cmap, vmin=max(0,vmin), vmax=vmax)

            axes[0].set_title('Object: {0}, flux'.format(self._obj_name))
            axes[1].set_title('root-variance'.format(self._obj_name))

            fig.tight_layout()
            fig.savefig(path.join(self.plot_path, '{}_frame.png'.format(self._filename_base)))
            plt.close(fig)

        return nccd

    def get_lsf_pars(self, row_idx=800): # MAGIC NUMBER
        """
        Fit a Voigt profile + background to the specified row to
        get the LSF parameters.
        """

        def lsf_model(p, x):
            amp, x0, std_G, fwhm_L, C = p
            return voigt(x, amp, x0, std_G, fwhm_L) + C

        def lsf_chi(p, pix, flux, flux_ivar):
            return (lsf_model(p, pix) - flux) * np.sqrt(flux_ivar)

        flux = self.ccd.data[row_idx]
        flux_err = self.ccd.uncertainty.array[row_idx]

        flux_ivar = 1/flux_err**2.
        flux_ivar[~np.isfinite(flux_ivar)] = 0.

        pix = np.arange(len(flux))

        # initial guess for optimization
        p0 = [flux.max(), pix[np.argmax(flux)], 1., 1., scoreatpercentile(flux[flux>0], 16.)]
        p_opt,ier = leastsq(lsf_chi, x0=p0, args=(pix, flux, flux_ivar))

        if self.plot_path is not None:
            fig,ax = plt.subplots(1,1,figsize=(8,5))
            ax.plot(pix, flux, marker='', drawstyle='steps-mid')
            ax.errorbar(pix, flux, 1/np.sqrt(flux_ivar), linewidth=1.,
                        marker='', linestyle='none', ecolor='#777777', zorder=-10)

            _grid = np.linspace(pix.min(), pix.max(), 1024)
            ax.plot(_grid, lsf_model(p_opt, _grid),
                    marker='', drawstyle='steps-mid', zorder=10, alpha=0.7)

            ax.set_xlabel('column pixel')
            ax.set_ylabel('flux')
            ax.set_yscale('log')
            ax.set_title('Object: {0}, fit ier: {1}'.format(self._obj_name, ier))

            fig.tight_layout()
            fig.savefig(path.join(self.plot_path, '{0}_lsf.png'.format(self._filename_base)))
            plt.close(fig)

        if ier < 1 or ier > 4:
            raise RuntimeError("Failed to fit for LSF at row {}".format(row_idx))

        lsf_p = dict()
        lsf_p['std_G'] = p_opt[2]
        lsf_p['fwhm_L'] = p_opt[3]

        return lsf_p

    def extract_1d(self, lsf_p):
        """
        Use the fit LSF, but fit for amplitude and background at each row
        on the detector to get source and background flux.
        """

        def row_model(p, lsf_p, x):
            amp, x0, C = p
            return voigt(x, amp, x0, G_std=lsf_p['std_G'], L_fwhm=lsf_p['fwhm_L']) + C

        def row_chi(p, pix, flux, flux_ivar, lsf_p):
            return (row_model(p, lsf_p, pix) - flux) * np.sqrt(flux_ivar)

        n_rows,n_cols = self.ccd.data.shape
        pix = np.arange(n_cols)

        # LSF extraction
        trace_1d = np.zeros(n_rows).astype(float)
        flux_1d = np.zeros(n_rows).astype(float)
        flux_1d_err = np.zeros(n_rows).astype(float)
        sky_flux_1d = np.zeros(n_rows).astype(float)
        sky_flux_1d_err = np.zeros(n_rows).astype(float)

        for i in range(self.ccd.data.shape[0]):
            flux = self.ccd.data[i]
            flux_err = self.ccd.uncertainty.array[i]
            flux_ivar = 1/flux_err**2.
            flux_ivar[~np.isfinite(flux_ivar)] = 0.

            p0 = [flux.max(), pix[np.argmax(flux)], scoreatpercentile(flux[flux>0], 16.)]
            p_opt,p_cov,*_,mesg,ier = leastsq(row_chi, x0=p0, full_output=True,
                                              args=(pix, flux, flux_ivar, lsf_p))

            if ier < 1 or ier > 4 or p_cov is None:
                flux_1d[i] = np.nan
                sky_flux_1d[i] = np.nan
                logger.log(0, "Fit failed for {}".format(i)) # TODO: ignored for now
                continue

            flux_1d[i] = p_opt[0]
            trace_1d[i] = p_opt[1]
            sky_flux_1d[i] = p_opt[2]

            # TODO: ignores centroiding covariances...
            flux_1d_err[i] = np.sqrt(p_cov[0,0])
            sky_flux_1d_err[i] = np.sqrt(p_cov[2,2])

        # clean up the 1d spectra
        flux_1d_ivar = 1/flux_1d_err**2
        sky_flux_1d_ivar = 1/sky_flux_1d_err**2

        pix_1d = np.arange(n_rows)
        mask_1d = (pix_1d < 50) | (pix_1d > 1600) # MAGIC NUMBERS: remove near top and bottom of CCD

        flux_1d[mask_1d] = 0.
        flux_1d_ivar[mask_1d] = 0.

        sky_flux_1d[mask_1d] = 0.
        sky_flux_1d_ivar[mask_1d] = 0.

        tbl = Table()
        tbl['pix'] = pix_1d
        tbl['trace'] = trace_1d
        tbl['source_flux'] = flux_1d
        tbl['source_ivar'] = flux_1d_ivar
        tbl['background_flux'] = sky_flux_1d
        tbl['background_ivar'] = sky_flux_1d_ivar

        if self.plot_path is not None:
            fig,axes = plt.subplots(2, 1, figsize=(12,8), sharex=True)

            axes[0].plot(tbl['pix'], tbl['source_flux'],
                         marker='', drawstyle='steps-mid', linewidth=1.)
            axes[0].errorbar(tbl['pix'], tbl['source_flux'], 1/np.sqrt(tbl['source_ivar']),
                             linestyle='none', marker='', ecolor='#666666', alpha=1., zorder=-10)
            axes[0].set_ylim(tbl['source_flux'][200]/4, np.nanmax(tbl['source_flux']))
            axes[0].set_yscale('log')

            axes[1].plot(tbl['pix'], tbl['background_flux'],
                         marker='', drawstyle='steps-mid', linewidth=1.)
            axes[1].errorbar(tbl['pix'], tbl['background_flux'], 1/np.sqrt(tbl['background_ivar']),
                             linestyle='none', marker='', ecolor='#666666', alpha=1., zorder=-10)
            axes[1].set_ylim(1e-1, np.nanmax(tbl['background_flux']))
            axes[1].set_yscale('log')

            fig.tight_layout()
            fig.savefig(path.join(self.plot_path, '{0}_1d.png'.format(self._filename_base)))
            plt.close(fig)

        return tbl
