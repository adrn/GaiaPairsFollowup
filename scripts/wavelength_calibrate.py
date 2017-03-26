"""

TODO:
- Some sky spectra have [OI] 5577Å, others have [OI] 6300Å, others have
    neither...wat do?!

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

def add_wavelength(filename, wavelength_coef, overwrite=False):
    hdulist = fits.open(filename)

    # read both hdu's
    logger.debug("\tObject: {}".format(hdulist[0].header['OBJECT']))

    # extract just the middle part of the CCD (we only really care about Halpha)
    tbl = Table(hdulist[1].data)

    if 'wavelength' in tbl.colnames and not overwrite:
        logger.debug("\tTable already contains wavelength values!")

    # compute wavelength array for the pixels
    tbl['wavelength'] = np.polynomial.polynomial.polyval(tbl['pix'],
                                                         wavelength_coef)
    new_hdu1 = fits.table_to_hdu(tbl)
    new_hdulist = fits.HDUList([hdulist[0], new_hdu1])

    logger.debug("\tWriting out file with wavelength array.")
    new_hdulist.writeto(filename, overwrite=overwrite)

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

    # fit a polynomial to pixel vs. wavelength
    coef = np.polynomial.polynomial.polyfit(pix_wav['pixel'], pix_wav['wavelength'],
                                            deg=polynomial_order)
    pred = np.polynomial.polynomial.polyval(pix_wav['pixel'], coef) # TODO: plot residuals?
    if np.any((pred - pix_wav['wavelength']) > 0.1):
        logger.warning("Wavelength residuals are large! Consider using a higher-order "
                       "polynomial, or check your wavelength calibration file." +
                       str(pred - pix_wav['wavelength']))

    # ========================
    # Compute wavelength grids
    # ========================

    if data_file is not None: # filename passed - only operate on that
        add_wavelength(data_file, coef, overwrite=overwrite)

    else: # a path was passed - operate on all 1D extracted files
        proc_ic = SkippableImageFileCollection(proc_path, glob_pattr='1d_proc_*')
        logger.info("{} 1D extracted spectra found".format(len(proc_ic.files)))

        logger.info("Beginning wavelength calibration...")
        for base_fname in proc_ic.files_filtered(imagetyp='OBJECT'):
            fname = path.join(proc_ic.location, base_fname)
            add_wavelength(fname, coef, overwrite=overwrite)

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

    # TODO: polynomial order

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
