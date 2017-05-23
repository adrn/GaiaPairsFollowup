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
from astropy.io import fits
import astropy.units as u
import ccdproc
from ccdproc import CCDData
import matplotlib.pyplot as plt
import numpy as np
import yaml

# Project
from comoving_rv.log import logger
from comoving_rv.longslit import GlobImageFileCollection
from comoving_rv.longslit.extract import SourceCCDExtractor, CompCCDExtractor

# -------------------------------
# CCD properties
#
ccd_gain = 2.7 * u.electron/u.adu
ccd_readnoise = 7.9 * u.electron
oscan_idx = 300
oscan_size = 64
#
# -------------------------------

ccd_props = dict()
ccd_props['ccd_gain'] = ccd_gain
ccd_props['ccd_readnoise'] = ccd_readnoise
ccd_props['oscan_idx'] = oscan_idx
ccd_props['oscan_size'] = oscan_size

# plotting configuration
from astropy.visualization import ZScaleInterval
zscaler = ZScaleInterval(32768, krej=5., max_iterations=16)
cmap = 'Greys_r'

def main(night_path, skip_list_file, mask_file, overwrite=False, plot=False):
    """
    See argparse block at bottom of script for description of parameters.
    """

    night_path = path.realpath(path.expanduser(night_path))
    if not path.exists(night_path):
        raise IOError("Path '{}' doesn't exist".format(night_path))
    logger.info("Reading data from path: {}".format(night_path))

    base_path, night_name = path.split(night_path)
    data_path, run_name = path.split(base_path)
    output_path = path.realpath(path.join(data_path, 'processed',
                                          run_name, night_name))
    os.makedirs(output_path, exist_ok=True)
    logger.info("Saving processed files to path: {}".format(output_path))

    if plot: # if we're making plots
        plot_path = path.realpath(path.join(output_path, 'plots'))
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
            ext = SourceCCDExtractor(filename=path.join(ic.location, new_fname),
                                     plot_path=plot_path, zscaler=zscaler,
                                     cmap=cmap, **ccd_props)
            nccd = ext.ccd

            # HACK: FUCK this is a bad hack
            ext._filename_base = ext._filename_base[2:]

        else:
            # process the frame!
            ext = SourceCCDExtractor(filename=path.join(ic.location, fname),
                                     plot_path=plot_path, zscaler=zscaler,
                                     cmap=cmap, unit='adu', **ccd_props)
            nccd = ext.process_raw_frame(pixel_mask_spec=pixel_mask_spec.get(fname, None),
                                         master_bias=master_bias,
                                         master_flat=master_flat)
            nccd.write(new_fname, overwrite=overwrite)

        # -------------------------------------------
        # Now do the 1D extraction
        # -------------------------------------------

        fname_1d = path.join(output_path, '1d_{0}'.format(fname))
        # if path.exists(fname_1d) and not overwrite:
        if False:
            logger.log(1, "\tAlready extracted! {}".format(fname_1d))
            continue

        else:
            logger.debug("\tExtracting to 1D")

            # first step is to fit a voigt profile to a middle-ish row to determine LSF
            lsf_p = ext.get_lsf_pars() # MAGIC NUMBER

            try:
                tbl = ext.extract_1d(lsf_p)
            except Exception as e:
                logger.error('Failed! {}: {}'.format(e.__class__.__name__,
                                                     str(e)))
                continue

            hdu0 = fits.PrimaryHDU(header=nccd.header)
            hdu1 = fits.table_to_hdu(tbl)
            hdulist = fits.HDUList([hdu0, hdu1])

            # hdulist.writeto(fname_1d, overwrite=overwrite)

        del ext

    # ==============================
    # Process comparison lamp frames
    # ==============================

    logger.info("Beginning comp. lamp frame processing...")
    for hdu, fname in ic.hdus(return_fname=True, imagetyp='COMP'):
        new_fname = path.join(output_path, 'p_{}'.format(fname))

        logger.debug("\tProcessing '{}'".format(hdu.header['OBJECT']))

        if path.exists(new_fname) and not overwrite:
            logger.log(1, "\tAlready processed! {}".format(new_fname))
            ext = CompCCDExtractor(filename=path.join(ic.location, new_fname),
                                   plot_path=plot_path, zscaler=zscaler,
                                   cmap=cmap, **ccd_props)
            nccd = ext.ccd

            # HACK: FUCK this is a bad hack
            ext._filename_base = ext._filename_base[2:]

        else:
            # process the frame!
            ext = CompCCDExtractor(filename=path.join(ic.location, fname),
                                   plot_path=plot_path, unit='adu', **ccd_props)
            nccd = ext.process_raw_frame(pixel_mask_spec=pixel_mask_spec.get(fname, None),
                                         master_bias=master_bias,
                                         master_flat=master_flat,)
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

            try:
                tbl = ext.extract_1d()
            except Exception as e:
                logger.error('Failed! {}: {}'.format(e.__class__.__name__,
                                                     str(e)))
                continue

            hdu0 = fits.PrimaryHDU(header=nccd.header)
            hdu1 = fits.table_to_hdu(tbl)
            hdulist = fits.HDUList([hdu0, hdu1])

            hdulist.writeto(fname_1d, overwrite=overwrite)

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
