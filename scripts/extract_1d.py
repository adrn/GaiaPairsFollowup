"""
This script is part of a spectroscopic reduction pipeline used to
extract 1D optical spectra from data taken at MDM observatory with
the ModSpec spectrograph mounted on the 2.4m Hiltner telescope.

This script processes and extracts 1D spectra from the 2D raw images
from a single night or block of observing. The input 2D images are
expected to be the raw frames that are output directly from the
instrument software (OWL). This script does the following:

    - Bias-subtract
    - Flat-correct
    - Trim 2D CCD images
    - 2D-to-1D extraction

The resulting 1D (flux vs. pixel) spectra are then saved to new files
that contain the source flux, source flux uncertainty, sky (background)
flux, sky flux uncertainty, and the position of the trace centroid
along the spatial dimension of the CCD as a function of pixel in the
disperion direction.

Wavelength calibration and radial velocity corrections are handled in
subsequent scripts.

"""

# Standard library
import os
from os import path
import logging

# Third-party
from astropy.table import Table
from astropy.io import fits
from astropy.modeling.models import Polynomial1D
from astropy.nddata.nduncertainty import StdDevUncertainty
from astropy.visualization import ZScaleInterval
import astropy.units as u
import ccdproc
from ccdproc import CCDData
import matplotlib.pyplot as plt
import numpy as np
from scipy.optimize import leastsq
from scipy.stats import scoreatpercentile
import yaml

# Project
from comoving_rv.log import logger
from comoving_rv.longslit import GlobImageFileCollection
from comoving_rv.longslit.models import voigt

# -------------------------------
# CCD properties
#
ccd_gain = 2.7 * u.electron/u.adu
ccd_readnoise = 7.9 * u.electron
oscan_idx = 300
oscan_size = 64
#
# -------------------------------

# Plot settings
zscaler = ZScaleInterval(32768, krej=5., max_iterations=16)
cmap = 'Greys_r'

class CCDExtractor(object):

    def __init__(self, filename, plot_path=None, **read_kwargs):
        # TODO: put overscan stuff, gain, etc. into CCDData.meta instead of globals

        # read CCD frame
        self.ccd = CCDData.read(filename, **read_kwargs)
        self.filename = filename
        self._filename_base = path.splitext(path.basename(self.filename))[0]
        self._obj_name = self.ccd.header['OBJECT']

        self.plot_path = plot_path

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

        oscan_fits_section = "[{}:{},:]".format(oscan_idx, oscan_idx+oscan_size)

        # make a copy of the object
        nccd = self.ccd.copy()

        # apply the overscan correction
        poly_model = Polynomial1D(2)
        nccd = ccdproc.subtract_overscan(nccd, fits_section=oscan_fits_section,
                                         model=poly_model)

        # trim the image (remove overscan region)
        nccd = ccdproc.trim_image(nccd, fits_section='[1:{},:]'.format(oscan_idx))

        # create the error frame
        nccd = ccdproc.create_deviation(nccd, gain=ccd_gain,
                                        readnoise=ccd_readnoise)

        # now correct for the ccd gain
        nccd = ccdproc.gain_correct(nccd, gain=ccd_gain)

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

            vmin,vmax = zscaler.get_limits(nccd.data)
            axes[0].imshow(nccd.data.T, origin='bottom',
                           cmap=cmap, vmin=max(0,vmin), vmax=vmax)

            stddev = nccd.uncertainty.array
            vmin,vmax = zscaler.get_limits(stddev[np.isfinite(stddev)])
            axes[1].imshow(stddev.T, origin='bottom',
                           cmap=cmap, vmin=max(0,vmin), vmax=vmax)

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

def main(night_path, skip_list_file, mask_file, overwrite=False, plot=False):
    """
    See argparse block at bottom of script for description of parameters.
    """

    night_path = path.realpath(path.expanduser(night_path))
    if not path.exists(night_path):
        raise IOError("Path '{}' doesn't exist".format(night_path))
    logger.info("Reading data from path: {}".format(night_path))

    base_path, name = path.split(night_path)
    output_path = path.realpath(path.join(base_path, 'processed', name))
    os.makedirs(output_path, exist_ok=True)
    logger.info("Saving processed files to path: {}".format(output_path))

    if plot: # if we're making plots
        plot_path = path.realpath(path.join(base_path, 'processed', name, 'plots'))
        logger.debug("Will make and save plots to: {}".format(plot_path))
        os.makedirs(plot_path, exist_ok=True)
    else:
        plot_path = None

    # check for files to skip (e.g., saturated or errored exposures)
    if skip_list_file is not None: # a file containing a list of filenames to skip
        with open(skip_list_file, 'r') as f:
            skip_list = [x.strip() for x in f if x.strip()]
    else:
        skip_list = []

    # look for pixel mask file
    if mask_file is not None:
        with open(mask_file, 'r') as f: # load YAML file specifying pixel masks for nearby sources
            pixel_mask_spec = yaml.load(f.read())
    else:
        pixel_mask_spec = None

    # generate the raw image file collection to process
    ic = GlobImageFileCollection(night_path, skip_filenames=skip_list)
    logger.info("Frames to process:")
    logger.info("- Bias frames: {}".format(len(ic.files_filtered(imagetyp='BIAS'))))
    logger.info("- Flat frames: {}".format(len(ic.files_filtered(imagetyp='FLAT'))))
    logger.info("- Comparison lamp frames: {}".format(len(ic.files_filtered(imagetyp='COMP'))))
    logger.info("- Object frames: {}".format(len(ic.files_filtered(imagetyp='OBJECT'))))

    # ============================
    # Create the master bias frame
    # ============================

    # overscan region of the CCD, using FITS index notation
    oscan_fits_section = "[{}:{},:]".format(oscan_idx, oscan_idx+oscan_size)

    master_bias_file = path.join(output_path, 'master_bias.fits')

    if not os.path.exists(master_bias_file) or overwrite:
        # get list of overscan-subtracted bias frames as 2D image arrays
        bias_list = []
        for hdu, fname in ic.hdus(return_fname=True, imagetyp='BIAS'):
            ccd = CCDData.read(path.join(ic.location, fname), unit='adu')
            ccd = ccdproc.gain_correct(ccd, gain=ccd_gain)
            ccd = ccdproc.subtract_overscan(ccd, overscan=ccd[:,oscan_idx:])
            ccd = ccdproc.trim_image(ccd, fits_section="[1:{},:]".format(oscan_idx))
            bias_list.append(ccd)

        # combine all bias frames into a master bias frame
        logger.info("Creating master bias frame")
        master_bias = ccdproc.combine(bias_list, method='average', clip_extrema=True,
                                      nlow=1, nhigh=1, error=True)
        master_bias.write(master_bias_file, overwrite=True)

    else:
        logger.info("Master bias frame file already exists: {}".format(master_bias_file))
        master_bias = CCDData.read(master_bias_file)

    if plot:
        # TODO: this assumes vertical CCD
        assert master_bias.shape[0] > master_bias.shape[1]
        aspect_ratio = master_bias.shape[1]/master_bias.shape[0]

        fig,ax = plt.subplots(1, 1, figsize=(10,12*aspect_ratio))
        vmin,vmax = zscaler.get_limits(master_bias.data)
        cs = ax.imshow(master_bias.data.T, origin='bottom',
                       cmap=cmap, vmin=max(0,vmin), vmax=vmax)
        ax.set_title('master bias frame [zscale]')

        fig.colorbar(cs)
        fig.tight_layout()
        fig.savefig(path.join(plot_path, 'master_bias.png'))
        plt.close(fig)

    # ============================
    # Create the master flat field
    # ============================

    master_flat_file = path.join(output_path, 'master_flat.fits')

    if not os.path.exists(master_flat_file) or overwrite:
        # create a list of flat frames
        flat_list = []
        for hdu, fname in ic.hdus(return_fname=True, imagetyp='FLAT'):
            ccd = CCDData.read(path.join(ic.location, fname), unit='adu')
            ccd = ccdproc.gain_correct(ccd, gain=ccd_gain)
            ccd = ccdproc.ccd_process(ccd, oscan=oscan_fits_section,
                                      trim="[1:{},:]".format(oscan_idx),
                                      master_bias=master_bias)
            flat_list.append(ccd)

        # combine into a single master flat - use 3*sigma sigma-clipping
        logger.info("Creating master flat frame")
        master_flat = ccdproc.combine(flat_list, method='average', sigma_clip=True,
                                      low_thresh=3, high_thresh=3)
        master_flat.write(master_flat_file, overwrite=True)

        # TODO: make plot if requested?

    else:
        logger.info("Master flat frame file already exists: {}".format(master_flat_file))
        master_flat = CCDData.read(master_flat_file)

    if plot:
        # TODO: this assumes vertical CCD
        assert master_flat.shape[0] > master_flat.shape[1]
        aspect_ratio = master_flat.shape[1]/master_flat.shape[0]

        fig,ax = plt.subplots(1, 1, figsize=(10,12*aspect_ratio))
        vmin,vmax = zscaler.get_limits(master_flat.data)
        cs = ax.imshow(master_flat.data.T, origin='bottom',
                       cmap=cmap, vmin=max(0,vmin), vmax=vmax)
        ax.set_title('master flat frame [zscale]')

        fig.colorbar(cs)
        fig.tight_layout()
        fig.savefig(path.join(plot_path, 'master_flat.png'))
        plt.close(fig)

    # =====================
    # Process object frames
    # =====================

    logger.info("Beginning object frame processing...")
    for hdu, fname in ic.hdus(return_fname=True, imagetyp='OBJECT'):
        new_fname = path.join(output_path, 'p_{}'.format(fname))

        # -------------------------------------------
        # First do the simple processing of the frame
        # -------------------------------------------

        logger.debug("Processing '{}' [{}]".format(hdu.header['OBJECT'], fname))
        if path.exists(new_fname) and not overwrite:
            logger.log(1, "\tAlready processed! {}".format(new_fname))
            ext = CCDExtractor(filename=path.join(ic.location, new_fname),
                               plot_path=plot_path)
            nccd = ext.ccd

            # HACK: FUCK this is a bad hack
            ext._filename_base = ext._filename_base[2:]

        else:
            # process the frame!
            ext = CCDExtractor(filename=path.join(ic.location, fname), plot_path=plot_path,
                               unit='adu')
            nccd = ext.process_raw_frame(pixel_mask_spec=pixel_mask_spec.get(fname, None),
                                         master_bias=master_bias,
                                         master_flat=master_flat)
            nccd.write(new_fname, overwrite=overwrite)

        # -------------------------------------------
        # Now do the 1D extraction
        # -------------------------------------------

        fname_1d = path.join(output_path, '1d_{0}'.format(fname))
        if path.exists(fname_1d) and not overwrite:
            logger.log(1, "\tAlready extracted! {}".format(fname_1d))
            continue

        else:
            logger.debug("\tExtracting to 1D")

            # first step is to fit a voigt profile to a middle-ish row to determine LSF
            lsf_p = ext.get_lsf_pars(row_idx=800) # MAGIC NUMBER

            try:
                tbl = ext.extract_1d(lsf_p)
            except Exception as e:
                logger.error('--- Failed! --- {}'.format(e))
                continue

            hdu0 = fits.PrimaryHDU(header=nccd.header)
            hdu1 = fits.table_to_hdu(tbl)
            hdulist = fits.HDUList([hdu0, hdu1])

            hdulist.writeto(fname_1d, overwrite=overwrite)

        del ext

    # ==============================
    # Process comparison lamp frames
    # ==============================

    logger.info("Beginning comp. lamp frame processing...")
    for hdu, fname in ic.hdus(return_fname=True, imagetyp='COMP'):
        new_fname = path.join(output_path, 'p_{}'.format(fname))

        logger.debug("\tProcessing '{}'".format(hdu.header['OBJECT']))
        if path.exists(new_fname) and not overwrite:
            logger.log(1, "\t\tAlready done! {}".format(new_fname))
            continue

        # process the frame!
        ext = CCDExtractor(filename=path.join(ic.location, fname), plot_path=plot_path,
                           unit='adu')
        nccd = ext.process_raw_frame(pixel_mask_spec=pixel_mask_spec.get(fname, None),
                                     master_bias=master_bias,
                                     master_flat=master_flat,)
        nccd.write(new_fname, overwrite=overwrite)

if __name__ == "__main__":
    from argparse import ArgumentParser

    # Define parser object
    parser = ArgumentParser(description="")

    vq_group = parser.add_mutually_exclusive_group()
    vq_group.add_argument('-v', '--verbose', action='count', default=0, dest='verbosity')
    vq_group.add_argument('-q', '--quiet', action='count', default=0, dest='quietness')

    parser.add_argument('-s', '--seed', dest='seed', default=None,
                        type=int, help='Random number generator seed.')
    parser.add_argument('-o', '--overwrite', action='store_true', dest='overwrite',
                        default=False, help='Destroy everything.')
    parser.add_argument('--plot', action='store_true', dest='plot',
                        default=False, help='Make plots along the way.')

    parser.add_argument('-p', '--path', dest='night_path', required=True,
                        help='Path to a single night or chunk of data to process.')
    parser.add_argument('--skiplist', dest='skip_list_file', default=None,
                        help='Path to a file containing a list of filenames (not '
                             'paths) to skip.')
    parser.add_argument('--mask', dest='mask_file', default=None,
                        help='Path to a YAML file containing pixel regions to ignore '
                             'in each frame. Useful for masking nearby sources.')

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

    main(night_path=args.night_path,
         skip_list_file=args.skip_list_file,
         overwrite=args.overwrite,
         mask_file=args.mask_file,
         plot=args.plot)
