# coding: utf-8

"""
Major hack to add skyline shifts to the velocity table
"""

# Standard library
from os import path
from collections import OrderedDict

# Third-party
from astropy.constants import c
import astropy.units as u
from astropy.io import fits
from astropy.table import Table
import numpy as np
import matplotlib.pyplot as plt
plt.style.use('apw-notebook')

# Project
from comoving_rv.log import logger
from comoving_rv.longslit.wavelength import fit_spec_line_GP

def get_fit_pars(gp):
    fit_pars = OrderedDict()
    for k,v in gp.get_parameter_dict().items():
        if 'mean' not in k:
            continue

        k = k[5:] # remove 'mean:'
        if k.startswith('ln'):
            if 'amp' in k:
                fit_pars[k[3:]] = -np.exp(v)
            else:
                fit_pars[k[3:]] = np.exp(v)

        elif k.startswith('bg'):
            if 'bg_coef' not in fit_pars:
                fit_pars['bg_coef'] = []
            fit_pars['bg_coef'].append(v)

        else:
            fit_pars[k] = v

    if 'std_G' not in fit_pars:
        fit_pars['std_G'] = 1E-10

    return fit_pars

def main(overwrite=False):
    plot_path = path.abspath('../plots/')
    root_path = path.abspath('../data/mdm-spring-2017/processed/')
    table_path = path.join(root_path, 'rough_velocity.ecsv')

    # config settings
    sky_lines = [5577.3387, 6300.3] # [OI]
    width = 100. # angstroms centered on line
    absorp_emiss = 1. # all emission lines
    gp_color = "#ff7f0e"

    if not path.exists(plot_path) or not path.exists(table_path):
        raise IOError("You must run test_solve_velocity first!")

    logger.debug('Table exists, reading ({})'.format(table_path))
    velocity_tbl = Table.read(table_path, format='ascii.ecsv', delimiter=',')
    if 'skyline_shift' not in velocity_tbl.colnames:
        velocity_tbl['skyline_shift'] = np.full(len(velocity_tbl), np.nan)

    for row in velocity_tbl[velocity_tbl['object_name'] == '483-209']:
        night_path = path.split(row['filename'])[0]

        # read master_wavelength file
        pix_wav = np.genfromtxt(path.join(night_path, 'master_wavelength.csv'),
                                delimiter=',', names=True)

        idx = (pix_wav['wavelength'] < 6950) & (pix_wav['wavelength'] > 5400)
        pix_wav = pix_wav[idx] # HACK
        pix_range = [min(pix_wav['pixel']), max(pix_wav['pixel'])]
        coef = np.polynomial.polynomial.polyfit(pix_wav['pixel'], pix_wav['wavelength'], deg=5)

        file_path = row['filename']
        filename = path.basename(file_path)
        filebase,ext = path.splitext(filename)

        # read FITS header
        hdr = fits.getheader(file_path, 0)
        object_name = hdr['OBJECT']

        if object_name != row['object_name']:
            row['object_name'] = object_name

        # sky_lines
        logger.debug('{} {} [{}]'.format(object_name, row['object_name'],
                                         path.basename(row['filename'])))
        if not np.isnan(row['skyline_shift']) and not overwrite:
            logger.debug('has sky line shift already')
            continue

        else:
            logger.debug('computing sky line shift')

        # read the spectrum data and get wavelength solution
        spec = Table.read(file_path)

        # compute wavelength array for the pixels
        wave = np.polynomial.polynomial.polyval(spec['pix'], coef)
        wave[(spec['pix'] > max(pix_range)) | (spec['pix'] < min(pix_range))] = np.nan

        vshifts = np.full(len(sky_lines), np.nan)
        for j,sky_line in enumerate(sky_lines):
            mask = (wave > (sky_line-width/2)) & (wave < (sky_line+width/2))
            flux_data = spec['background_flux'][mask]
            ivar_data = spec['background_ivar'][mask]
            wave_data = wave[mask]

            _idx = wave_data.argsort()
            wave_data = wave_data[_idx]
            flux_data = flux_data[_idx]
            ivar_data = ivar_data[_idx]
            err_data = 1/np.sqrt(ivar_data)

            gp = fit_spec_line_GP(wave_data, flux_data, ivar_data,
                                  absorp_emiss=absorp_emiss,
                                  fwhm_L0=2., std_G0=1., n_bg_coef=2) # target_x=sky_line,

            pars = gp.get_parameter_dict()
            dlam = pars['mean:x0'] - sky_line
            vshift = (dlam/sky_line*c).to(u.km/u.s).value

            if ((pars['mean:ln_fwhm_L'] < 0 and pars['mean:ln_std_G'] < 0) or pars['mean:ln_amp'] > 10.):
                title = 'fucked'
            else:
                title = '{:.2f}'.format(pars['mean:ln_amp'])
                vshifts[j] = vshift

            # Make the maximum likelihood prediction
            wave_grid = np.linspace(wave_data.min(), wave_data.max(), 256)
            mu, var = gp.predict(flux_data, wave_grid, return_var=True)
            std = np.sqrt(var)

            # ----------------------------------------------------------------
            # Plot the fit and data
            fig,ax = plt.subplots(1,1)

            # data
            ax.plot(wave_data, flux_data, drawstyle='steps-mid', marker='')
            ax.errorbar(wave_data, flux_data, err_data,
                        marker='', ls='none', ecolor='#666666', zorder=-10)

            # full GP model
            ax.plot(wave_grid, mu, color=gp_color, marker='')
            ax.fill_between(wave_grid, mu+std, mu-std,
                            color=gp_color, alpha=0.3, edgecolor="none")
            ax.set_title(title)
            fig.tight_layout()
            fig.savefig(path.join(plot_path, '{}_maxlike_sky_{:.0f}.png'
                                  .format(filebase, sky_line)), dpi=256)
            plt.close(fig)

        row['skyline_shift'] = np.nanmean(vshifts)
        logger.debug('Shifts: {}, Mean: {:.2f}'.format(vshifts, row['skyline_shift']))
        velocity_tbl.write(table_path, format='ascii.ecsv',
                           overwrite=True, delimiter=',')

if __name__ == "__main__":
    from argparse import ArgumentParser
    import logging

    # Define parser object
    parser = ArgumentParser(description="")

    vq_group = parser.add_mutually_exclusive_group()
    vq_group.add_argument('-v', '--verbose', action='count', default=0, dest='verbosity')
    vq_group.add_argument('-q', '--quiet', action='count', default=0, dest='quietness')

    parser.add_argument('-s', '--seed', dest='seed', default=None,
                        type=int, help='Random number generator seed.')
    parser.add_argument('-o', '--overwrite', action='store_true', dest='overwrite',
                        default=False, help='Destroy everything.')

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

    main(overwrite=args.overwrite)

