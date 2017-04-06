"""

TODO:
- Better propagate wavelength calibration fit (line centroid) uncertainties
- Do some cross-validation to figure out best power-law to use to fit? pix vs. wavelength?
- Need to do sky-spectrum adjustments here when we can

"""

# Standard library
from os import path

# Third-party
from astropy.table import Table
from astropy.io import fits
import numpy as np

# Project
from comoving_rv.log import logger
from comoving_rv.longslit import SkippableImageFileCollection
from comoving_rv.longslit.wavelength import fit_spec_line
from comoving_rv.longslit.models import voigt_polynomial

def sky_line_shift(wavelength, bg_flux, bg_ivar, plot=False):
    """
    Arc lamp spectrum determines non-linear relation between pixel and
    wavelength. This function tries to find a bright sky line to use to
    determine small absolute shifts to the wavelength solution. This
    works by fitting a Voigt profile to the [OI] line at 6300Å and
    5577Å and uses the line with a larger amplitude. If neither line has
    an amplitude > XX TODO, it raises an error.

    Parameters
    ----------
    TODO
    """

    wavelength = np.array(wavelength)
    bg_flux = np.array(bg_flux)
    bg_ivar = np.array(bg_ivar)

    # for target_wave in [5577.3387]:
    #   [OI] from: http://www.star.ucl.ac.uk/~msw/lines.html
    target_wave = 5577.3387
    # target_wave = 6300.30

    # extract region of SKY spectrum around line
    _i1 = np.argmin(np.abs(wavelength - (target_wave-25)))
    _i2 = np.argmin(np.abs(wavelength - (target_wave+25)))
    i1 = min(_i1, _i2)
    i2 = max(_i1, _i2)

    wave = wavelength[i1:i2+1]
    flux = bg_flux[i1:i2+1]
    ivar = bg_ivar[i1:i2+1]

    if plot:
        _grid = np.linspace(wave.min(), wave.max(), 512)

        import matplotlib.pyplot as plt
        plt.figure(figsize=(14,8))
        plt.plot(wave, flux, marker='', drawstyle='steps-mid', alpha=0.5)
        plt.errorbar(wave, flux, 1/np.sqrt(ivar), linestyle='none',
                     marker='', ecolor='#666666', alpha=0.75, zorder=-10)
        plt.show()

    try:
        OI_fit_p = fit_spec_line(wave, flux, ivar, std_G0=1.,
                                 n_bg_coef=2, target_x=target_wave,
                                 absorp_emiss=1.)
    except (RuntimeError, ValueError) as e:
        logger.warning("Failed to fit [OI] sky line - won't shift spectrum.")
        return 0.

    dlambda = OI_fit_p['x0']-target_wave
    logger.debug("[OI] {:.2f}, ∆λ: {:.3f}, amp: {:.3e}".format(target_wave,
                                                               dlambda,
                                                               OI_fit_p['amp']))

    if plot:
        _grid = np.linspace(wave.min(), wave.max(), 512)
        fit_flux = voigt_polynomial(_grid, **OI_fit_p)

        import matplotlib.pyplot as plt
        plt.figure(figsize=(14,8))
        plt.plot(wave, flux, marker='', drawstyle='steps-mid', alpha=0.5)
        plt.errorbar(wave, flux, 1/np.sqrt(ivar), linestyle='none',
                     marker='', ecolor='#666666', alpha=0.75, zorder=-10)
        plt.plot(_grid, fit_flux, marker='', alpha=0.75)
        plt.show()

    if OI_fit_p['amp'] < 10.:
        logger.warning("Failed to fit [OI] sky line - won't shift spectrum.")
        return 0.

    return dlambda

def add_wavelength(filename, wavelength_coef, overwrite=False, pix_range=None):
    hdulist = fits.open(filename)

    # read both hdu's
    logger.debug("\tObject: {}".format(hdulist[0].header['OBJECT']))

    # extract just the middle part of the CCD (we only really care about Halpha)
    tbl = Table(hdulist[1].data)

    if 'wavelength' in tbl.colnames and not overwrite:
        logger.debug("\tTable already contains wavelength values!")
        return

    if pix_range is not None:
        good_idx = (tbl['pix'] > min(pix_range)) & (tbl['pix'] < max(pix_range))
    else:
        good_idx = np.ones(len(tbl)).astype(bool)

    print(pix_range, good_idx)

    # compute wavelength array for the pixels
    tbl['wavelength'] = np.polynomial.polynomial.polyval(tbl['pix'],
                                                         wavelength_coef)
    tbl['wavelength'][~good_idx] = np.nan

    # here, need to do sky line adjustment to wavelength values
    dlambda = sky_line_shift(tbl['wavelength'][good_idx],
                             tbl['background_flux'][good_idx],
                             tbl['background_ivar'][good_idx],
                             plot=True)
    tbl['wavelength'] = tbl['wavelength'] - dlambda

    new_hdu1 = fits.table_to_hdu(tbl)
    new_hdulist = fits.HDUList([hdulist[0], new_hdu1])

    logger.debug("\tWriting out file with wavelength array.")
    new_hdulist.writeto(filename, overwrite=True)

def main(proc_path, polynomial_order, overwrite=False):

    proc_path = path.realpath(path.expanduser(proc_path))
    if not path.exists(proc_path):
        raise IOError("Path '{}' doesn't exist".format(proc_path))

    if path.isdir(proc_path):
        data_file = None
        logger.info("Reading data from path: {}".format(proc_path))

    elif path.isfile(proc_path):
        data_file = proc_path
        base_path, name = path.split(proc_path)
        proc_path = base_path
        logger.info("Reading file: {}".format(data_file))

    else:
        raise RuntimeError("how?!")

    # read master_wavelength file
    pix_wav = np.genfromtxt(path.join(proc_path, 'master_wavelength.csv'),
                            delimiter=',', names=True)

    idx = pix_wav['wavelength'] < 6929 # HACK: see notebook 'Wavelength fit residuals'
    pix_wav = pix_wav[idx] # HACK
    pix_range = [min(pix_wav['pixel']), max(pix_wav['pixel'])]

    # fit a polynomial to pixel vs. wavelength
    coef = np.polynomial.polynomial.polyfit(pix_wav['pixel'], pix_wav['wavelength'],
                                            deg=polynomial_order)
    pred = np.polynomial.polynomial.polyval(pix_wav['pixel'], coef) # TODO: plot residuals?
    if np.any((pred - pix_wav['wavelength']) > 0.01):
        logger.warning("Wavelength residuals are large! Check your wavelength "
                       "calibration init file." + str(pred - pix_wav['wavelength']))

    # ========================
    # Compute wavelength grids
    # ========================

    if data_file is not None: # filename passed - only operate on that
        add_wavelength(data_file, coef, overwrite=overwrite, pix_range=pix_range)

    else: # a path was passed - operate on all 1D extracted files
        proc_ic = SkippableImageFileCollection(proc_path, glob_pattr='1d_*')
        logger.info("{} 1D extracted spectra found".format(len(proc_ic.files)))

        logger.info("Beginning wavelength calibration...")
        for base_fname in proc_ic.files_filtered(imagetyp='OBJECT'):
            fname = path.join(proc_ic.location, base_fname)
            add_wavelength(fname, coef, overwrite=overwrite, pix_range=pix_range)

if __name__ == "__main__":
    from argparse import ArgumentParser
    import logging

    # Define parser object
    parser = ArgumentParser(description="")

    vq_group = parser.add_mutually_exclusive_group()
    vq_group.add_argument('-v', '--verbose', action='count', default=0, dest='verbosity')
    vq_group.add_argument('-q', '--quiet', action='count', default=0, dest='quietness')

    parser.add_argument('-o', '--overwrite', action='store_true', dest='overwrite',
                        default=False, help='Destroy everything.')

    parser.add_argument('-p', '--path', dest='proc_path', required=True,
                        help='Path to a PROCESSED night or chunk of data to process. Or, '
                             'path to a specific comp file.')

    parser.add_argument('--polyorder', dest='polynomial_order', default=9, type=int,
                        help='TODO')

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

    kwargs = vars(args)
    kwargs.pop('verbosity')
    kwargs.pop('quietness')
    main(**kwargs)
