"""

TODO:
-

"""

# Standard library
from os import path
import logging

# Third-party
from astropy.io import fits
import ccdproc
from ccdproc import CCDData
import matplotlib.pyplot as plt
import numpy as np

# Project
from comoving_rv.longslit import SkippableImageFileCollection
from comoving_rv.longslit import voigt_polynomial
from comoving_rv.longslit.wavelength import fit_emission_line

logger = logging.getLogger('wavelength_calibrate')
formatter = logging.Formatter('%(levelname)s:%(name)s:  %(message)s')
handler = logging.StreamHandler()
handler.setFormatter(formatter)
logger.addHandler(handler)
logger.propagate = False

def main(proc_path, polynomial_order, overwrite=False):
    """ """

    proc_path = path.realpath(path.expanduser(proc_path))
    if not path.exists(proc_path):
        raise IOError("Path '{}' doesn't exist".format(proc_path))
    logger.info("Reading data from path: {}".format(proc_path))

    # read master_wavelength file
    pix_wav = np.genfromtxt(path.join(proc_path, 'master_wavelength.csv'),
                            delimiter=',', names=True)

    # fit a polynomial to pixel vs. wavelength
    coef = np.polynomial.polynomial.polyfit(pix_wav['pixel'], pix_wav['wavelength'],
                                            deg=polynomial_order)
    # pred = np.polynomial.polynomial.polyval(pix, coef) # TODO: plot residuals

    # ========================
    # Compute wavelength grids
    # ========================

    proc_ic = SkippableImageFileCollection(proc_path, glob_pattr='1d_proc_*')
    logger.info("{} raw frames already processed".format(len(proc_ic.files)))

    logger.info("Beginning 1D extraction...")
    for fname in proc_ic.files_filtered(imagetyp='OBJECT'):
        fname = path.join(proc_ic.location, fname)
        hdulist = fits.open(fname)

        # read both hdu's
        hdu0 = hdulist[0]
        hdu1 = hdulist[1]
        logger.debug("\tWavelength calibrating '{}'".format(hdu0.header['OBJECT']))

        # TODO: check if file already wavelength calibrated
        # fname_1d = path.join(output_path, '1d_{}'.format(fname))
        # if path.exists(fname_1d) and not overwrite:
        #     logger.log(1, "\t\tAlready extracted! {}".format(fname_1d))
        #     continue

        tbl = hdu1.data
        wvln = np.polynomial.polynomial.polyval(tbl['pix'], coef)

        # PLOT!!!
        fig,axes = plt.subplots(1, 2, figsize=(12,8), sharex='row')

        axes[0].plot(wvln, tbl['source_flux'], marker='', drawstyle='steps-mid')
        axes[0].errorbar(wvln, tbl['source_flux'], 1/np.sqrt(tbl['source_ivar']),
                         linestyle='none', marker='', ecolor='#666666', alpha=1., zorder=-10)
        axes[0].set_ylim(1e2, np.nanmax(tbl['source_flux']))
        axes[0].set_yscale('log')

        axes[1].plot(wvln, tbl['background_flux'], marker='', drawstyle='steps-mid')
        axes[1].errorbar(wvln, tbl['background_flux'], 1/np.sqrt(tbl['background_ivar']),
                         linestyle='none', marker='', ecolor='#666666', alpha=1., zorder=-10)
        axes[1].set_ylim(1e-1, np.nanmax(tbl['background_flux']))
        axes[1].set_yscale('log')

        fig.tight_layout()

        plt.show()
        return

    # TODO: figure out shift to apply for sky line

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

    parser.add_argument('--polyorder', dest='polynomial_order', default=11, type=int,
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
