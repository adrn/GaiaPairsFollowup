"""
Solve for the relative velocities between two spectra in all groups by fitting
line profiles to the Halpha absorption line (in pixel space) and computing the
relative velocity.

TODO:
- We currently skip groups when they contain more than two members
"""

# Standard library
from os import path
import logging

# Third-party
from astropy.io import fits
from astropy.table import Table
import astropy.units as u
import matplotlib.pyplot as plt
import numpy as np

# Project
from comoving_rv.log import logger
from comoving_rv.longslit import SkippableImageFileCollection, voigt_polynomial
from comoving_rv.longslit.wavelength import fit_spec_line

def solve_radial_velocity(filename, wavelength_coef, done_list=None, plot=False):
    hdulist = fits.open(filename)

    # read both hdu's
    hdu0 = hdulist[0]
    hdu1 = hdulist[1]
    name = hdu0.header['OBJECT']
    logger.debug("\tObject: {}".format(name))

    if done_list is not None and name in done_list:
        return

    # extract just the middle part of the CCD (we only really care about Halpha)
    tbl = hdu1.data[200:1600][::-1]

    # compute wavelength array for the pixels
    wvln = np.polynomial.polynomial.polyval(tbl['pix'], wavelength_coef)

    # ==============================
    # Fit a Voigt profile to H-alpha
    # ==============================

    # extract region of SOURCE spectrum around Halpha
    i1 = np.argmin(np.abs(wvln - 6460))
    i2 = np.argmin(np.abs(wvln - 6665))

    wave = wvln[i1:i2+1]
    flux = tbl['source_flux'][i1:i2+1]
    ivar = tbl['source_ivar'][i1:i2+1]
    halpha_fit_p = fit_spec_line(wave, flux, ivar,
                                 n_bg_coef=2, target_x=6563., absorp_emiss=-1.)

    if plot:
        _grid = np.linspace(wave.min(), wave.max(), 512)
        fit_flux = voigt_polynomial(_grid, **halpha_fit_p)

        plt.figure(figsize=(14,8))
        plt.title("OBJECT: {}, EXPTIME: {}".format(hdu0.header['OBJECT'],
                                                   hdu0.header['EXPTIME']))

        plt.plot(wave, flux, marker='', drawstyle='steps-mid', alpha=0.5)
        plt.errorbar(wave, flux, 1/np.sqrt(ivar), linestyle='none',
                     marker='', ecolor='#666666', alpha=0.75, zorder=-10)
        plt.plot(_grid, fit_flux, marker='', alpha=0.75)

    # =========================================
    # Fit a Voigt profile to [OI] 6300 and 5577
    # =========================================

    # needed for Barycenter correction
    # earth_loc = coord.EarthLocation.of_site('KPNO')

    for target_wave in [5577.3387, 6300.3]:
        # extract region of SKY spectrum around line
        i1 = np.argmin(np.abs(wvln - (target_wave-25)))
        i2 = np.argmin(np.abs(wvln - (target_wave+25)))

        wave = wvln[i1:i2+1]
        flux = tbl['background_flux'][i1:i2+1]
        ivar = tbl['background_ivar'][i1:i2+1]

        OI_fit_p = fit_spec_line(wave, flux, ivar, std_G0=1.,
                                 n_bg_coef=2, target_x=target_wave,
                                 absorp_emiss=1.)

        print('[OI] {:.2f}'.format(target_wave))
        print('∆x0: {:.3f}'.format(OI_fit_p['x0'] - target_wave))
        print('amp: {:.3e}'.format(OI_fit_p['amp']))

        chi2 = np.sum((voigt_polynomial(wave, **OI_fit_p) - flux)**2 * ivar)
        print('chi2: {}'.format(chi2))

        if plot:
            _grid = np.linspace(wave.min(), wave.max(), 512)
            fit_flux = voigt_polynomial(_grid, **OI_fit_p)

            plt.figure(figsize=(14,8))
            plt.title("OBJECT: {}, EXPTIME: {}".format(hdu0.header['OBJECT'],
                                                       hdu0.header['EXPTIME']))

            plt.plot(wave, flux, marker='', drawstyle='steps-mid', alpha=0.5)
            plt.errorbar(wave, flux, 1/np.sqrt(ivar), linestyle='none',
                         marker='', ecolor='#666666', alpha=0.75, zorder=-10)
            plt.plot(_grid, fit_flux, marker='', alpha=0.75)

    # from: http://www.star.ucl.ac.uk/~msw/lines.html
    # Halpha = 6562.80 # air, STP
    # OI_5577 = 5577.3387 # air, STP
    # OI_6300 = 6300.30 # air, STP

    # dOI = OI_fit_p['x0'] - OI_5577
    # dHalpha = halpha_fit_p['x0'] - Halpha
    # dlambda = dHalpha - dOI

    # RV = dlambda / Halpha * c
    # print("Radial velocity: ", RV.to(u.km/u.s))

    # sc = coord.SkyCoord(ra=hdu0.header['RA'], dec=hdu0.header['DEC'],
    #                     unit=(u.hourangle, u.degree))
    # time = hdu0.header['JD']*u.day + hdu0.header['EXPTIME']/2.*u.second
    # time = Time(time.to(u.day), format='jd', scale='utc')
    # v_bary = bary_vel_corr(time, sc, location=earth_loc)
    # RV_corr = (RV + v_bary).to(u.km/u.s)
    # print("Bary. correction: ", v_bary.to(u.km/u.s))
    # print("Radial velocity (bary. corrected): ", RV_corr)
    # print()

    if plot:
        plt.show()

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



    # file to keep track of RV measurements
    rv_file = path.join(proc_path, 'rvs.ecsv')
    if path.exists(rv_file):
        rv_tbl = Table.read(rv_file, format='ecsv')
    else:
        rv_tbl = None

    # ========================
    # Compute wavelength grids
    # ========================

    if data_file is not None: # filename passed - only operate on that
        solve_radial_velocity(data_file, coef, done_list=None, plot=True)

    else: # a path was passed - operate on all 1D extracted files
        proc_ic = SkippableImageFileCollection(proc_path, glob_pattr='1d_proc_*')
        logger.info("{} 1D extracted spectra found".format(len(proc_ic.files)))

        logger.info("Beginning wavelength calibration...")
        for base_fname in proc_ic.files_filtered(imagetyp='OBJECT'):
            fname = path.join(proc_ic.location, base_fname)
            solve_radial_velocity(fname)

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
                             'path to a specific comp file.')

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
