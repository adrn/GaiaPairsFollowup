"""

TODO:
- Make the functions below part of a class so we can store diagnostic
    plots along the way (easily keep track of paths)
- Need to do sky-spectrum adjustments here when we can
"""

# Standard library
from os import path
import pickle

# Third-party
from astropy.table import Table
import astropy.units as u
from astropy.io import fits
import matplotlib.pyplot as plt
import numpy as np
from numpy.polynomial.polynomial import polyval
from scipy.optimize import minimize
from celerite.modeling import Model
from celerite import terms, GP

# Project
from comoving_rv.log import logger
from comoving_rv.longslit import GlobImageFileCollection
# from comoving_rv.longslit.fitting import fit_spec_line_GP, gp_to_fit_pars
from comoving_rv.longslit.fitting import fit_spec_line
from comoving_rv.longslit.models import voigt_polynomial
from comoving_rv.longslit.wavelength import GPModel

# ----------------------------------------------------------------------------
# Settings, or arbitrary choices / magic numbers
#
n_bg_coef = 2 # linear
#
# ----------------------------------------------------------------------------

def fit_all_lines(pixels, flux, flux_ivar, line_waves, line_pixels):

    _idx = np.argsort(line_waves)
    wvln = np.array(line_waves)[_idx]
    pixl = np.array(line_pixels)[_idx]

    fit_centroids = []

    half_width = 5 # MAGIC NUMBER: number of pixels on either side of line to fit to
    for pix_ctr,wave in zip(pixl, wvln):

        logger.debug("Fitting line at predicted pix={:.2f}, λ={:.2f}"
                     .format(pix_ctr, wave))

        # indices for region around line
        i1 = int(np.floor(pix_ctr-half_width))
        i2 = int(np.ceil(pix_ctr+half_width))+1

        # recenter window
        i0 = i1 + flux[i1:i2].argmax()
        i1 = int(np.floor(i0-half_width))
        i2 = int(np.ceil(i0+half_width))+1

        _pixl = pixels[i1:i2]
        _flux = flux[i1:i2]

        # instead of doing anything fancy (e.g., fitting a profile), just
        # estimate the mean...
        x0 = np.sum(_pixl * _flux) / np.sum(_flux)
        fit_centroids.append(x0)

    return np.array(fit_centroids)

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

def generate_wavelength_model(comp_lamp_path, night_path, plot_path):

    # read 1D comp lamp spectrum
    spec = Table.read(comp_lamp_path)

    # read wavelength guess file
    guess_path = path.abspath(path.join(night_path,
                                        '..', 'wavelength_guess.csv'))
    pix_wav = np.genfromtxt(guess_path, delimiter=',', names=True)

    # fit line profiles to each emission line at the guessed positions of the lines
    pix_x0s = fit_all_lines(spec['pix'], spec['flux'], spec['ivar'],
                            pix_wav['wavelength'], pix_wav['pixel'])

    # --------------------------------------------------------------------------
    # fit a gaussian process to determine the pixel-to-wavelength transformation
    #
    idx = np.argsort(pix_x0s)
    med_x = np.median(pix_x0s[idx])
    x = pix_x0s[idx] - med_x
    y = pix_wav['wavelength'][idx]

    model = GPModel(x=x, y=y, n_bg_coef=n_bg_coef)

    # Fit for the maximum likelihood parameters
    bounds = model.gp.get_parameter_bounds()
    init_params = model.gp.get_parameter_vector()
    soln = minimize(model, init_params, method="L-BFGS-B",
                    bounds=bounds)
    model.gp.set_parameter_vector(soln.x)
    logger.debug("Success: {}, Final log-likelihood: {}".format(soln.success,
                                                                -soln.fun))

    # ---
    # residuals to the mean model
    x_grid = np.linspace(0, 1600, 1024) - med_x
    mu, var = model.gp.predict(y, x_grid, return_var=True)
    std = np.sqrt(var)

    _y_mean = model.mean_model.get_value(x)
    _mu_mean = model.mean_model.get_value(x_grid)

    # Plot the maximum likelihood model
    fig,ax = plt.subplots(1, 1, figsize=(8,8))

    # data
    ax.scatter(x + med_x, y - _y_mean, marker='o')

    # full GP model
    gp_color = "#ff7f0e"
    ax.plot(x_grid+med_x, mu - _mu_mean, color=gp_color, marker='')
    ax.fill_between(x_grid+med_x, mu+std-_mu_mean, mu-std-_mu_mean,
                    color=gp_color, alpha=0.3, edgecolor="none")

    ax.set_xlabel('pixel')
    ax.set_ylabel(r'wavelength [$\AA$]')
    ax.set_title(path.basename(comp_lamp_path))

    fig.tight_layout()
    fig.savefig(path.join(plot_path, 'wavelength_mean_subtracted.png'), dpi=200)
    # ---

    # ---
    # residuals to full GP model
    mu, var = model.gp.predict(y, x_grid, return_var=True)
    std = np.sqrt(var)

    y_mu, var = model.gp.predict(y, x, return_var=True)

    # Plot the maximum likelihood model
    fig,ax = plt.subplots(1, 1, figsize=(12,8))

    # data
    ax.scatter(x + med_x, y - y_mu, marker='o')

    gp_color = "#ff7f0e"
    ax.plot(x_grid+med_x, mu-mu, color=gp_color, marker='')
    ax.fill_between(x_grid+med_x, std, -std, color=gp_color,
                    alpha=0.3, edgecolor="none")

    ax.set_xlabel('pixel')
    ax.set_ylabel(r'wavelength residual [$\AA$]')
    ax.set_title(path.basename(comp_lamp_path))

    ax.set_ylim(-0.4, 0.4)
    ax.axvline(683., zorder=-10, color='#666666', alpha=0.5)

    ax2 = ax.twinx()
    ax2.set_ylim([x/6563*300000 for x in ax.get_ylim()])
    ax2.set_ylabel(r'velocity error at ${{\rm H}}_\alpha$ [{}]'
                   .format((u.km/u.s).to_string(format='latex_inline')))

    fig.tight_layout()
    fig.savefig(path.join(plot_path, 'wavelength_residuals.png'), dpi=200)
    # --------------------------------------------------------------------------

    return model

def main(night_path, wavelength_gp_path=None, comp_lamp_path=None,
         overwrite=False):

    night_path = path.realpath(path.expanduser(night_path))
    if not path.exists(night_path):
        raise IOError("Path '{}' doesn't exist".format(night_path))

    if path.isdir(night_path):
        data_file = None
        logger.info("Reading data from path: {}".format(night_path))

    elif path.isfile(night_path):
        data_file = night_path
        base_path, name = path.split(night_path)
        night_path = base_path
        logger.info("Reading file: {}".format(data_file))

    else:
        raise RuntimeError("how?!")

    plot_path = path.join(night_path, 'plots')

    # ==========================================================================
    # GP model does not exist yet
    #
    if wavelength_gp_path is None:

        # filename to save the GP model
        wavelength_gp_path = path.join(night_path, 'wavelength_GP_model.pickle')

        # see if a wavelength GP model file already exists
        if path.exists(wavelength_gp_path):
            logger.info('Loading wavelength GP model from {}'
                        .format(wavelength_gp_path))

            # GP model already exists -- just load it
            with open(wavelength_gp_path, 'rb') as f:
                model = pickle.load(f)

        else:
            logger.info('Generating wavelength GP model, saving to {}'
                        .format(wavelength_gp_path))

            if comp_lamp_path is None:
                ic = GlobImageFileCollection(night_path, glob_include='1d_*')

                hdu = None
                for hdu,wavelength_data_file in ic.hdus(return_fname=True, imagetyp='COMP'):
                    break
                else:
                    raise IOError("No COMP lamp file found in {}".format(night_path))

                comp_lamp_path = path.join(ic.location, wavelength_data_file)
                logger.info("No comp. lamp spectrum file specified - using: {}"
                            .format(comp_lamp_path))

            model = generate_wavelength_model(comp_lamp_path, night_path,
                                              plot_path)

        # pickle the model
        with open(wavelength_gp_path, 'wb') as f:
            pickle.dump(model, f)

    # ==========================================================================
    # GP model already exists -- just load it
    #
    else:
        logger.info('Loading wavelength GP model from {}'
                    .format(wavelength_gp_path))
        with open(wavelength_gp_path, 'rb') as f:
            model = pickle.load(f)

    return

    # ========================
    # Compute wavelength grids
    # ========================

    if data_file is not None: # filename passed - only operate on that
        add_wavelength(data_file, coef, overwrite=overwrite, pix_range=pix_range)

    else: # a path was passed - operate on all 1D extracted files
        proc_ic = GlobImageFileCollection(proc_path, glob_include='1d_*')
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

    parser.add_argument('-p', '--path', dest='night_path', required=True,
                        help='Path to a PROCESSED night or chunk of data to process. Or, '
                             'path to a specific comp file.')

    comp_gp_group = parser.add_mutually_exclusive_group()
    comp_gp_group.add_argument('--comp', dest='comp_lamp_path', default=None,
                               help='Path to a PROCESSED comparison lamp '
                                    'spectrum file.')
    comp_gp_group.add_argument('--gp', dest='wavelength_gp_path', default=None,
                               help='Path to a pickle file containing a GP '
                                    '(Gaussian Process) model for the '
                                    'wavelength solution.')

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
