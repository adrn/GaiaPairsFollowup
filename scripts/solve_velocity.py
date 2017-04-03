"""

TODO: do both relative and absolute velocity stuff in here

Solve for the relative velocities between two spectra by fitting line profiles
to the Halpha absorption line (in pixel space) and computing the relative
velocity
"""

# Standard library
from os import path

# Third-party
from astropy.time import Time
import astropy.coordinates as coord
from astropy.constants import c
from astropy.io import fits
from astropy.table import Table
import astropy.units as u
import matplotlib.pyplot as plt
import numpy as np

# Project
from comoving_rv.log import logger
from comoving_rv.longslit import SkippableImageFileCollection, voigt_polynomial
from comoving_rv.longslit.wavelength import fit_spec_line
from comoving_rv.velocity import bary_vel_corr, kitt_peak

def solve_radial_velocity(wavelength, flux, flux_ivar, header, plot=False):
    good_idx = np.isfinite(wavelength) & np.isfinite(flux) & np.isfinite(flux_ivar)
    wavelength = wavelength[good_idx]
    flux = flux[good_idx]
    flux_ivar = flux_ivar[good_idx]

    _sort = wavelength.argsort()
    wavelength = wavelength[_sort]
    flux = flux[_sort]
    flux_ivar = flux_ivar[_sort]

    # from: http://www.star.ucl.ac.uk/~msw/lines.html
    Halpha = 6562.80 # air, STP

    # ==============================
    # Fit a Voigt profile to H-alpha
    # ==============================

    # TODO: could do this for H-beta too

    # extract region of source spectrum around Halpha
    target_wave = Halpha

    # select 30 angstroms to either side of Halpha
    i1 = np.argmin(np.abs(wavelength - (target_wave-30)))
    i2 = np.argmin(np.abs(wavelength - (target_wave+30))) + 1

    wave = wavelength[i1:i2]
    flux = flux[i1:i2]
    ivar = flux_ivar[i1:i2]

    halpha_fit_p = fit_spec_line(wave, flux, ivar,
                                 n_bg_coef=2, target_x=6563., absorp_emiss=-1.)

    if plot:
        _grid = np.linspace(wave.min(), wave.max(), 512)
        fit_flux = voigt_polynomial(_grid, **halpha_fit_p)

        plt.figure(figsize=(14,8))
        plt.title("OBJECT: {}, EXPTIME: {}".format(header['OBJECT'],
                                                   header['EXPTIME']))

        plt.plot(wave, flux, marker='', drawstyle='steps-mid', alpha=0.5)
        plt.errorbar(wave, flux, 1/np.sqrt(ivar), linestyle='none',
                     marker='', ecolor='#666666', alpha=0.75, zorder=-10)
        plt.plot(_grid, fit_flux, marker='', alpha=0.75)
        plt.show()# TODO

    dHalpha = halpha_fit_p['x0'] - target_wave
    RV = dHalpha / Halpha * c
    print("Object: {}".format(header['OBJECT']))
    print("Radial velocity: ", RV.to(u.km/u.s))

    sc = coord.SkyCoord(ra=header['RA'], dec=header['DEC'],
                        unit=(u.hourangle, u.degree))
    time = header['JD']*u.day + header['EXPTIME']/2.*u.second
    time = Time(time.to(u.day), format='jd', scale='utc')
    v_bary = bary_vel_corr(time, sc, location=kitt_peak)
    RV_corr = (RV + v_bary).to(u.km/u.s)
    print("Bary. correction: ", v_bary.to(u.km/u.s))
    print("Radial velocity (bary. corrected): ", RV_corr)
    print()

def main(proc_path, overwrite=False):
    """ """

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

    # # TODO: keep track of RV measurements?!
    # rv_file = path.join(proc_path, 'rvs.ecsv')
    # if path.exists(rv_file):
    #     rv_tbl = Table.read(rv_file, format='ecsv')
    # else:
    #     rv_tbl = None

    # ==================================
    # Solve for absolute radial velocity
    # ==================================

    if data_file is not None: # filename passed - only operate on that
        file_list = [data_file]

    else: # a path was passed - operate on all 1D extracted files
        proc_ic = SkippableImageFileCollection(proc_path, glob_pattr='1d_proc_*')
        logger.info("{} 1D extracted spectra found".format(len(proc_ic.files)))
        file_list = proc_ic.files_filtered(imagetyp='OBJECT')
        file_list = [path.join(proc_ic.location, base_fname) for base_fname in file_list]

    for fname in file_list:
        header = fits.getheader(fname, 0)
        tbl = fits.getdata(fname, 1)
        solve_radial_velocity(tbl['wavelength'], tbl['source_flux'], tbl['source_ivar'],
                              header, plot=False)

    # if rv_tbl is None:
    #     rv_tbl = Table()
    #     rv_tbl.add_column(Column(data=[hdu0.header['OBJECT']], name='name'))
    #     rv_tbl.add_column(Column(data=RV_corr.reshape(1), name='rv'))

    # else:
    #     rv_tbl.insert_row(len(rv_tbl), [hdu0.header['OBJECT'], RV_corr])

    # rv_tbl.write('/Users/adrian/Downloads/test.ecsv', format='ascii.ecsv', overwrite=True)

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
                             'path to a specific file.')

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
