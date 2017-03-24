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
import sys
import logging

# Third-party
import astropy.units as u
import ccdproc
from ccdproc import CCDData, ImageFileCollection
import numpy as np
import six

# -------------------------------
# CCD properties
#
ccd_gain = 2.7 * u.electron/u.adu
readnoise = 7.9*u.electron
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
                 filenames=None, skip_filenames=None):

        if skip_filenames is None:
            self.skip_filenames = list()

        else:
            self.skip_filenames = list(skip_filenames)

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
            files = self._fits_files_in_directory()
            files = [fn for fn in files if fn not in self.skip_filenames]

        return files

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
        master_bias.write(master_bias_file, clobber=True)

        # TODO: make plot if requested?

    else:
        logger.debug("Master bias frame file already exists: {}".format(master_bias_file))

        # TODO: read from file

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
            ccd = ccdproc.ccd_process(ccd,
                                      oscan="[{}:{},:]".format(oscan_idx, oscan_idx+oscan_size),
                                      trim="[1:{},:]".format(oscan_idx),
                                      master_bias=master_bias)
            flat_list.append(ccd)

        # combine into a single master flat - use 3*sigma sigma-clipping
        master_flat = ccdproc.combine(flat_list, method='average', sigma_clip=True,
                                      low_thresh=3, high_thresh=3)
        master_flat.write(master_flat_file, clobber=True)

        # TODO: make plot if requested?

    else:
        logger.debug("Master flat frame file already exists: {}".format(master_flat_file))

        # TODO: read from file

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

