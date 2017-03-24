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
import fnmatch

# Third-party
from astropy.table import Table
from astropy.io import fits
from astropy.modeling.models import Polynomial1D
import astropy.units as u
import ccdproc
from ccdproc import CCDData, ImageFileCollection
import matplotlib.pyplot as plt
import numpy as np
import six
from scipy.optimize import leastsq
from scipy.special import wofz
from scipy.stats import scoreatpercentile

# -------------------------------
# CCD properties
#
ccd_gain = 2.7 * u.electron/u.adu
ccd_readnoise = 7.9 * u.electron
oscan_idx = 300
oscan_size = 64
#
# -------------------------------

logger = logging.getLogger(__name__)
formatter = logging.Formatter('%(levelname)s:%(name)s:  %(message)s')
handler = logging.StreamHandler()
handler.setFormatter(formatter)
logger.addHandler(handler)
logger.propagate = False

class SkippableImageFileCollection(ImageFileCollection):

    def __init__(self, location=None, keywords=None, info_file=None,
                 filenames=None, glob_pattr=None, skip_filenames=None):

        if skip_filenames is None:
            self.skip_filenames = list()

        else:
            self.skip_filenames = list(skip_filenames)

        self.glob_pattr = glob_pattr

        super(SkippableImageFileCollection, self).__init__(location=location,
                                                           keywords=keywords,
                                                           info_file=info_file,
                                                           filenames=filenames)

    def _get_files(self):
        """ Helper method which checks whether ``files`` should be set
        to a subset of file names or to all file names in a directory.
        Returns
        -------
        files : list or str
            List of file names which will be added to the collection.
        """
        files = []
        if self._filenames:
            if isinstance(self._filenames, six.string_types):
                files.append(self._filenames)
            else:
                files = self._filenames
        else:
            _files = self._fits_files_in_directory()

            files = []
            for fn in _files:
                if fn in self.skip_filenames:
                    continue

                if self.glob_pattr is not None and not fnmatch.fnmatch(fn, self.glob_pattr):
                    continue

                files.append(fn)

        return files

def voigt(x, amp, x0, G_std, L_fwhm):
    """
    Voigt profile - convolution of a Gaussian and Lorentzian.

    When G_std -> 0, the profile approaches a Lorentzian. When L_fwhm=0,
    the profile is a Gaussian.

    Parameters
    ----------
    x : numeric, array_like
    amp : numeric
        Amplitude of the profile (integral).
    x0 : numeric
        Centroid.
    G_std : numeric
        Standard of deviation of the Gaussian component.
    L_fwhm : numeric
        FWHM of the Lorentzian component.
    """
    _x = x-x0
    z = (_x + 1j*L_fwhm/2.) / (np.sqrt(2.)*G_std)
    return amp * wofz(z).real / (np.sqrt(2.*np.pi)*G_std)

def get_psf_pars(ccd, row_idx=800): # MAGIC NUMBER
    """
    Fit a Voigt profile + background to the specified row to
    get the PSF parameters.
    """

    def psf_model(p, x):
        amp, x_0, std_G, fwhm_L, C = p
        return voigt(x, amp, x_0, std_G, fwhm_L) + C

    def psf_chi(p, pix, flux, flux_ivar):
        return (psf_model(p, pix) - flux) * np.sqrt(flux_ivar)

    flux = ccd.data[row_idx]
    flux_err = ccd.uncertainty.array[row_idx]

    flux_ivar = 1/flux_err**2.
    flux_ivar[~np.isfinite(flux_ivar)] = 0.

    pix = np.arange(len(flux))

    # initial guess for optimization
    p0 = [flux.max(), pix[np.argmax(flux)], 1., 1., scoreatpercentile(flux[flux>0], 16.)]
    p_opt,ier = leastsq(psf_chi, x0=p0, args=(pix, flux, flux_ivar))

    if ier < 1 or ier > 4:
        raise RuntimeError("Failed to fit for PSF at row {}".format(row_idx))

    psf_p = dict()
    psf_p['std_G'] = p_opt[2]
    psf_p['fwhm_L'] = p_opt[3]

    return psf_p

def extract_1d(ccd, psf_p):
    """
    Use the fit PSF, but fit for amplitude and background at each row
    on the detector to get source and background flux.
    """

    def row_model(p, psf_p, x):
        amp, x_0, C = p
        return voigt(x, amp, x_0, G_std=psf_p['std_G'], L_fwhm=psf_p['fwhm_L']) + C

    def row_chi(p, pix, flux, flux_ivar, psf_p):
        return (row_model(p, psf_p, pix) - flux) * np.sqrt(flux_ivar)

    n_rows,n_cols = ccd.data.shape
    pix = np.arange(n_cols)

    # PSF extraction
    trace_1d = np.zeros(n_rows).astype(float)
    flux_1d = np.zeros(n_rows).astype(float)
    flux_1d_err = np.zeros(n_rows).astype(float)
    sky_flux_1d = np.zeros(n_rows).astype(float)
    sky_flux_1d_err = np.zeros(n_rows).astype(float)

    for i in range(ccd.data.shape[0]):
        flux = ccd.data[i]
        flux_err = ccd.uncertainty.array[i]
        flux_ivar = 1/flux_err**2.
        flux_ivar[~np.isfinite(flux_ivar)] = 0.

        p0 = [flux.max(), pix[np.argmax(flux)], scoreatpercentile(flux[flux>0], 16.)]
        p_opt,p_cov,*_,mesg,ier = leastsq(row_chi, x0=p0, full_output=True,
                                          args=(pix, flux, flux_ivar, psf_p))

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

    return tbl

def main(night_path, skip_list_file, overwrite=False):
    """
    See argparse block at bottom of script for description of parameters.
    """

    night_path = path.realpath(path.expanduser(night_path))
    if not path.exists(night_path):
        raise IOError("Path '{}' doesn't exist".format(night_path))
    logger.debug("Reading data from path: {}".format(night_path))

    base_path, name = path.split(night_path)
    output_path = path.realpath(path.join(base_path, '{}_proc'.format(name)))
    os.makedirs(output_path, exist_ok=True)
    logger.debug("Saving processed files to path: {}".format(output_path))

    # check for files to skip (e.g., saturated or errored exposures)
    if skip_list_file is not None: # a file containing a list of filenames to skip
        with open(skip_list_file, 'r') as f:
            skip_list = [x.strip() for x in f if x.strip()]

    else:
        skip_list = []

    # generate the raw image file collection to process
    ic = SkippableImageFileCollection(night_path, skip_filenames=skip_list)
    logger.debug("Frames to process:")
    logger.debug("- Bias frames: {}".format(len(ic.files_filtered(imagetyp='BIAS'))))
    logger.debug("- Flat frames: {}".format(len(ic.files_filtered(imagetyp='FLAT'))))
    logger.debug("- Comparison lamp frames: {}".format(len(ic.files_filtered(imagetyp='COMP'))))
    logger.debug("- Object frames: {}".format(len(ic.files_filtered(imagetyp='OBJECT'))))

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
        logger.debug("Creating master bias frame")
        master_bias = ccdproc.combine(bias_list, method='average', clip_extrema=True,
                                      nlow=1, nhigh=1, error=True)
        master_bias.write(master_bias_file, overwrite=True)

        # TODO: make plot if requested?

    else:
        logger.debug("Master bias frame file already exists: {}".format(master_bias_file))
        master_bias = CCDData.read(master_bias_file)

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
        logger.debug("Creating master flat frame")
        master_flat = ccdproc.combine(flat_list, method='average', sigma_clip=True,
                                      low_thresh=3, high_thresh=3)
        master_flat.write(master_flat_file, overwrite=True)

        # TODO: make plot if requested?

    else:
        logger.debug("Master flat frame file already exists: {}".format(master_flat_file))
        master_flat = CCDData.read(master_flat_file)

    # =====================
    # Process object frames
    # =====================

    logger.debug("Beginning object frame processing...")
    for hdu, fname in ic.hdus(return_fname=True, imagetyp='OBJECT'):
        new_fname = path.join(output_path, 'proc_{}'.format(fname))

        logger.log(1, "\tProcessing '{}'".format(hdu.header['OBJECT']))
        if path.exists(new_fname) and not overwrite:
            logger.log(1, "\t\tAlready done! {}".format(new_fname))
            continue

        # read CCD frame
        ccd = CCDData.read(path.join(ic.location, fname), unit='adu')

        # make a copy of the object
        nccd = ccd.copy()

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

        nccd.write(new_fname, overwrite=overwrite)

    # ==================
    # Extract 1D spectra
    # ==================

    proc_ic = SkippableImageFileCollection(output_path, glob_pattr='proc_*')
    logger.debug("{} raw frames already processed".format(len(proc_ic.files)))

    logger.debug("Beginning 1D extraction...")
    for ccd, fname in proc_ic.ccds(return_fname=True):
        logger.log(1, "\tExtracting '{}'".format(ccd.header['OBJECT']))

        fname_1d = path.join(output_path, '1d_{}'.format(fname))
        if path.exists(fname_1d) and not overwrite:
            logger.log(1, "\t\tAlready extracted! {}".format(fname_1d))
            continue

        # first step is to fit a voigt profile to a middle-ish row to determine PSF
        psf_p = get_psf_pars(ccd, row_idx=800) # MAGIC NUMBER

        try:
            tbl = extract_1d(ccd, psf_p)
        except Exception as e:
            logger.error('--- Failed! --- {}'.format(e))
            continue

        # # PLOT!!!
        # fig,axes = plt.subplots(1, 2, figsize=(12,8), sharex='row')

        # axes[0].plot(tbl['pix'], tbl['source_flux'], marker='', drawstyle='steps-mid')
        # axes[0].errorbar(tbl['pix'], tbl['source_flux'], 1/np.sqrt(tbl['source_ivar']),
        #                  linestyle='none', marker='', ecolor='#666666', alpha=1., zorder=-10)
        # axes[0].set_ylim(1e2, np.nanmax(tbl['source_flux']))
        # axes[0].set_yscale('log')

        # axes[1].plot(tbl['pix'], tbl['background_flux'], marker='', drawstyle='steps-mid')
        # axes[1].errorbar(tbl['pix'], tbl['background_flux'], 1/np.sqrt(tbl['background_ivar']),
        #                  linestyle='none', marker='', ecolor='#666666', alpha=1., zorder=-10)
        # axes[1].set_ylim(1e-1, np.nanmax(tbl['background_flux']))
        # axes[1].set_yscale('log')

        # fig.tight_layout()

        # plt.show()
        # return

        hdu0 = fits.PrimaryHDU(header=ccd.header)
        hdu1 = fits.table_to_hdu(tbl)
        hdulist = fits.HDUList([hdu0, hdu1])

        hdulist.writeto(fname_1d)

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

    parser.add_argument('-p', '--path', dest='night_path', required=True,
                        help='Path to a single night or chunk of data to process.')
    parser.add_argument('--skip-list', dest='skip_list_file', default=None,
                        help='Path to a file containing a list of filenames (not '
                             'paths) to skip.')

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
         overwrite=args.overwrite)
