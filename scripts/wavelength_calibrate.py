"""

TODO:
- Make the functions below part of a class so we can store diagnostic
    plots along the way (easily keep track of paths)
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
from scipy.optimize import minimize

# Project
from comoving_rv.log import logger
from comoving_rv.longslit import GlobImageFileCollection
from comoving_rv.longslit.wavelength import GPModel, fit_all_lines

# ----------------------------------------------------------------------------
# Settings, or arbitrary choices / magic numbers
#
n_bg_coef = 2 # linear
#
# ----------------------------------------------------------------------------

def generate_wavelength_model(comp_lamp_path, night_path, plot_path):
    """
    Fit a line + Gaussian Process model to the pixel vs. wavelength relation for
    identified and centroided comp. lamp spectrum emission lines.

    Parameters
    ----------
    comp_lamp_path : str
    night_path : str
    plot_path : str

    """

    # read 1D comp lamp spectrum
    spec = Table.read(comp_lamp_path)

    # read wavelength guess file
    guess_path = path.abspath(path.join(night_path,
                                        '..', 'wavelength_guess.csv'))
    pix_wav = np.genfromtxt(guess_path, delimiter=',', names=True)

    # get emission line centroids at the guessed positions of the lines
    pix_x0s = fit_all_lines(spec['pix'], spec['flux'], spec['ivar'],
                            pix_wav['wavelength'], pix_wav['pixel'])

    # --------------------------------------------------------------------------
    # fit a gaussian process to determine the pixel-to-wavelength transformation
    #
    idx = np.argsort(pix_x0s)
    med_x = np.median(pix_x0s[idx])
    x = pix_x0s[idx] - med_x
    y = pix_wav['wavelength'][idx]

    model = GPModel(x=x, y=y, n_bg_coef=n_bg_coef, x_shift=med_x)

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

    ax.set_ylim(-1, 1)
    ax.axvline(683., zorder=-10, color='#666666', alpha=0.5)

    ax2 = ax.twinx()
    ax2.set_ylim([x/6563*300000 for x in ax.get_ylim()])
    ax2.set_ylabel(r'velocity error at ${{\rm H}}_\alpha$ [{}]'
                   .format((u.km/u.s).to_string(format='latex_inline')))

    fig.tight_layout()
    fig.savefig(path.join(plot_path, 'wavelength_residuals.png'), dpi=200)
    # --------------------------------------------------------------------------

    return model

def add_wavelength(filename, model, std_tol, overwrite=False, plot_path=None):
    """
    Given an extracted, 1D spectrum FITS file, add wavelength and
    wavelength_prec columnes to the file.

    Parameters
    ----------
    filename : str
        Path to a 1D extracted spectrum file.
    model : `comoving_rv.longslit.GPModel`
    std_tol : quantity_like
        Set the wavelength grid to NaN when the root-variance of the prediction
        from the Gaussian process is larger than this tolerance.
    overwrite : bool (optional)
        Overwrite any existing wavelength information.
    plot_path : str (optional)
    """
    hdulist = fits.open(filename)

    # read both hdu's
    logger.debug("\tObject: {}".format(hdulist[0].header['OBJECT']))

    # extract just the middle part of the CCD (we only really care about Halpha)
    tbl = Table(hdulist[1].data)

    if 'wavelength' in tbl.colnames and not overwrite:
        logger.debug("\tTable already contains wavelength values!")
        return

    # compute wavelength array for the pixels
    wavelength, var = model.gp.predict(model.y, tbl['pix']-model.x_shift,
                                       return_var=True)
    bad_idx = np.sqrt(var) > std_tol.to(u.angstrom).value
    wavelength[bad_idx] = np.nan

    tbl['wavelength'] = wavelength
    tbl['wavelength_err'] = np.sqrt(var)

    new_hdu1 = fits.table_to_hdu(tbl)
    new_hdulist = fits.HDUList([hdulist[0], new_hdu1])

    logger.debug("\tWriting out file with wavelength array.")
    new_hdulist.writeto(filename, overwrite=True)

    if plot_path is not None:
        # plot the spectrum vs. wavelength
        fig,axes = plt.subplots(2, 1, figsize=(12,8), sharex=True)

        axes[0].plot(tbl['wavelength'], tbl['source_flux'],
                     marker='', drawstyle='steps-mid', linewidth=1.)
        axes[0].errorbar(tbl['wavelength'], tbl['source_flux'], 1/np.sqrt(tbl['source_ivar']),
                         linestyle='none', marker='', ecolor='#666666', alpha=1., zorder=-10)
        axes[0].set_ylim(tbl['source_flux'][200]/4, np.nanmax(tbl['source_flux']))
        axes[0].set_yscale('log')

        axes[1].plot(tbl['wavelength'], tbl['background_flux'],
                     marker='', drawstyle='steps-mid', linewidth=1.)
        axes[1].errorbar(tbl['wavelength'], tbl['background_flux'], 1/np.sqrt(tbl['background_ivar']),
                         linestyle='none', marker='', ecolor='#666666', alpha=1., zorder=-10)
        axes[1].set_ylim(1e-1, np.nanmax(tbl['background_flux']))
        axes[1].set_yscale('log')

        fig.tight_layout()
        _filename_base = path.splitext(path.basename(filename))[0]
        fig.savefig(path.join(plot_path, '{0}_1d_wvln.png'
                              .format(_filename_base)))

        plt.close(fig)

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

    # ===========================
    # GP model does not exist yet
    # ===========================
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

    # =======================================
    # GP model already exists -- just load it
    # =======================================
    else:
        logger.info('Loading wavelength GP model from {}'
                    .format(wavelength_gp_path))
        with open(wavelength_gp_path, 'rb') as f:
            model = pickle.load(f)

    # ========================
    # Compute wavelength grids
    # ========================

    # set the wavelength grid to NaN when the root-variance of the prediction is
    #   larger than this tolerance
    std_tol = 1. * u.angstrom

    if data_file is not None: # filename passed - only operate on that
        add_wavelength(data_file, model, overwrite=overwrite, std_tol=std_tol,
                       plot_path=plot_path)

    else: # a path was passed - operate on all 1D extracted files
        proc_ic = GlobImageFileCollection(night_path, glob_include='1d_*')
        logger.info("{} 1D extracted spectra found".format(len(proc_ic.files)))

        logger.info("Beginning wavelength calibration...")
        for base_fname in proc_ic.files_filtered(imagetyp='OBJECT'):
            fname = path.join(proc_ic.location, base_fname)
            add_wavelength(fname, model, overwrite=overwrite, std_tol=std_tol,
                           plot_path=plot_path)

if __name__ == "__main__":
    from argparse import ArgumentParser
    import logging

    # Define parser object
    parser = ArgumentParser(description="")

    vq_group = parser.add_mutually_exclusive_group()
    vq_group.add_argument('-v', '--verbose', action='count', default=0,
                          dest='verbosity')
    vq_group.add_argument('-q', '--quiet', action='count', default=0,
                          dest='quietness')

    parser.add_argument('-o', '--overwrite', action='store_true',
                        dest='overwrite', default=False,
                        help='Destroy everything.')

    parser.add_argument('-p', '--path', dest='night_path', required=True,
                        help='Path to a PROCESSED night or chunk of data to '
                             'process. Or, path to a specific comp file.')

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
