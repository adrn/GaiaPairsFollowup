"""

NOTES:
- It looks like some sky spectra have [OI] 5577Å, others have [OI] 6300Å - we
    may need to fit for both and check?

TODO:
-

"""

# Standard library
from os import path
import logging

# Third-party
import astropy.coordinates as coord
from astropy.constants import c
from astropy.io import fits
from astropy.time import Time
from astropy.table import Table, Column
import astropy.units as u
import matplotlib.pyplot as plt
import numpy as np

# Project
from comoving_rv.longslit import SkippableImageFileCollection
from comoving_rv.longslit import voigt_polynomial
from comoving_rv.longslit.wavelength import fit_spec_line

logger = logging.getLogger('wavelength_calibrate')
formatter = logging.Formatter('%(levelname)s:%(name)s:  %(message)s')
handler = logging.StreamHandler()
handler.setFormatter(formatter)
logger.addHandler(handler)
logger.propagate = False

def bary_vel_corr(time, skycoord, location=None):
    """
    Barycentric velocity correction.

    Uses the ephemeris set with  ``astropy.coordinates.solar_system_ephemeris.set``
    for corrections. For more information see `~astropy.coordinates.solar_system_ephemeris`.

    Parameters
    ----------
    time : `~astropy.time.Time`
        The time of observation.
    skycoord: `~astropy.coordinates.SkyCoord`
        The sky location to calculate the correction for.
    location: `~astropy.coordinates.EarthLocation`, optional
        The location of the observatory to calculate the correction for.
        If no location is given, the ``location`` attribute of the Time
        object is used

    Returns
    -------
    vel_corr : `~astropy.units.Quantity`
        The velocity correction to convert to Barycentric velocities. Should be added to the original
        velocity.
    """

    if location is None:
        if time.location is None:
            raise ValueError('An EarthLocation needs to be set or passed '
                             'in to calculate bary- or heliocentric '
                             'corrections')
        location = time.location

    # ensure sky location is ICRS compatible
    if not skycoord.is_transformable_to(coord.ICRS):
        raise ValueError("Given skycoord is not transformable to the ICRS")

    # ICRS position and velocity of Earth's geocenter
    ep, ev = coord.solar_system.get_body_barycentric_posvel('earth', time)

    # GCRS position and velocity of observatory
    op, ov = location.get_gcrs_posvel(time)

    # ICRS and GCRS are axes-aligned. Can add the velocities
    velocity = ev + ov # relies on PR5434 being merged

    # get unit ICRS vector in direction of SkyCoord
    sc_cartesian = skycoord.icrs.represent_as(coord.UnitSphericalRepresentation)\
                                .represent_as(coord.CartesianRepresentation)
    return sc_cartesian.dot(velocity).to(u.km/u.s) # similarly requires PR5434

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
        print('∆x_0: {:.3f}'.format(OI_fit_p['x_0'] - target_wave))
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

    # dOI = OI_fit_p['x_0'] - OI_5577
    # dHalpha = halpha_fit_p['x_0'] - Halpha
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

def main(proc_path, polynomial_order, overwrite=False):
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
