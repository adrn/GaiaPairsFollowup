"""
Generate a rough, initial wavelength solution for a 1D spectrum.
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
from matplotlib.widgets import SpanSelector
import matplotlib.pyplot as plt
import numpy as np
from PyQt5 import QtCore, QtGui, QtWidgets
from PyQt5.QtCore import Qt
from scipy.interpolate import InterpolatedUnivariateSpline

# Package
from comoving_rv.longslit import voigt_polynomial
from comoving_rv.longslit.wavelength import fit_emission_line

logger = logging.getLogger('init_wavelength')
formatter = logging.Formatter('%(levelname)s:%(name)s:  %(message)s')
handler = logging.StreamHandler()
handler.setFormatter(formatter)
logger.addHandler(handler)
logger.propagate = False

def gui_solution(pixels, flux, flux_ivar, fig, ax, line_list=None):
    # map_dict = dict()
    # map_dict['wavelength'] = []
    # map_dict['pixel'] = []
    # line_widths = [] # for auto-solving lines

    # FOR TESTING:
    map_dict = {'wavelength': [5460.7399999999998, 7245.1665999999996, 6717.0429999999997, 6678.2762000000002, 6598.9529000000002, 6532.8822], 'pixel': [1226.9646056472734, 349.38080535972756, 610.93127457855871, 630.09556101866531, 668.9871368080278, 701.444940640303]}
    line_widths = [1.5] # for auto-solving lines

    # # A 1D span selector to highlight a given line
    # span = SpanSelector(ax, on_select, 'horizontal', useblit=True,
    #                     rectprops=dict(alpha=0.5, facecolor='red'))
    # span.set_active(False)

    # def enable_span():
    #     tb = fig.canvas.manager.toolbar

    #     if span.active:
    #         span.set_active(False)
    #     else:
    #         span.set_active(True)

    #     if tb._active == 'PAN':
    #         tb.pan()
    #         tb._actions['pan'].setChecked(False)

    #     if tb._active == 'ZOOM':
    #         tb.zoom()
    #         tb._actions['zoom'].setChecked(False)

    # span_control = QtWidgets.QPushButton('λ')
    # span_control.setCheckable(True)
    # span_control.clicked.connect(enable_span)
    # span_control.setStyleSheet("color: #de2d26; font-weight: bold;")
    # fig.canvas.manager.toolbar.addWidget(span_control)

    if line_list is not None:
        def auto_identify():
            _idx = np.argsort(map_dict['wavelength'])
            wvln = np.array(map_dict['wavelength'])[_idx]
            pixl = np.array(map_dict['pixel'])[_idx]

            # build an approximate wavelength solution to predict where lines are
            spl = InterpolatedUnivariateSpline(wvln, pixl, k=3)

            predicted_pixels = spl(line_list)

            new_waves = []
            new_pixels = []

            lw = np.median(line_widths)
            for pix_ctr,xmin,xmax,wave in zip(predicted_pixels,
                                              predicted_pixels-3*lw,
                                              predicted_pixels+3*lw,
                                              line_list):
                lp = get_line_props(pixels, flux, xmin, xmax,
                                    sigma0=lw)

                if lp is None or lp['amp'] < 1000.: # HACK
                    # logger.error("Failed to fit predicted line at {:.3f}A, {:.2f}pix"
                    #              .format(wave, pix_ctr))
                    continue

                draw_line_marker(lp, wave, ax)
                new_waves.append(wave)
                new_pixels.append(pix_ctr)

            fig.canvas.draw()

            fig2,axes2 = plt.subplots(2,1,figsize=(6,10))
            axes2[0].plot(new_pixels, new_waves, linestyle='none', marker='o')

            coef = np.polynomial.chebyshev.chebfit(new_pixels, new_waves, deg=5)
            pred = np.polynomial.chebyshev.chebval(new_pixels, coef)
            axes2[1].plot(new_pixels, new_waves-pred,
                          linestyle='none', marker='o')

            plt.show()

        autoid_control = QtWidgets.QPushButton('auto-identify')
        autoid_control.clicked.connect(auto_identify)
        fig.canvas.manager.toolbar.addWidget(autoid_control)

    plt.show()

    return map_dict

class GUIWavelengthSolver(object):

    def __init__(self, pix, flux, flux_ivar=None, line_list=None):

        # the spectrum data
        self.pix = pix
        self.flux = flux
        self.flux_ivar = flux_ivar

        if line_list is not None:
            self.line_list = np.array(line_list)
        else:
            self.line_list = None

        # make the plot
        fig,ax = plt.subplots(1, 1)
        ax.plot(pix, flux, marker='', drawstyle='steps-mid')
        ax.set_xlim(pix.min(), pix.max())

        # since we require QT anyways...
        figManager = plt.get_current_fig_manager()
        figManager.window.showMaximized()

        self.fig = fig
        self.ax = ax

        # for caching
        self._map_dict = dict()
        self._map_dict['wavel'] = []
        self._map_dict['pixel'] = []

        # for storing UI elements
        self._ui = dict()

        self._setup_ui()

    def _setup_ui(self):
        # Add a side menu for specifying the wavelength of the selected line
        panel = QtWidgets.QWidget()

        label = QtWidgets.QLabel("Enter wavelength [Å]:", panel)
        help_label = QtWidgets.QLabel("Enter a wavelength value, zoom in, then select the region "
                                      "containing the emission line at the specified wavelength.",
                                      panel)
        help_label.setStyleSheet("font-style: italic;")
        self._ui['textbox'] = QtWidgets.QLineEdit(parent=panel)

        panel.layout = QtWidgets.QGridLayout(panel)
        panel.layout.addWidget(label, 0, 0, 1, 1)
        panel.layout.addWidget(self._ui['textbox'], 0, 1, 1, 1)
        panel.layout.addWidget(help_label, 1, 0, 1, 2, Qt.AlignCenter)

        main_window = self.fig.canvas.manager.window
        dock = QtWidgets.QDockWidget("Enter wavelength:", main_window)
        main_window.addDockWidget(Qt.BottomDockWidgetArea, dock)
        dock.setWidget(panel)

        # A 1D span selector to highlight a given line
        span = SpanSelector(self.ax, self._on_select, 'horizontal', useblit=True,
                            rectprops=dict(alpha=0.5, facecolor='red'))
        span.set_active(False)

        def enable_span():
            tb = self.fig.canvas.manager.toolbar

            if span.active:
                span.set_active(False)
            else:
                span.set_active(True)

            if tb._active == 'PAN':
                tb.pan()
                tb._actions['pan'].setChecked(False)

            if tb._active == 'ZOOM':
                tb.zoom()
                tb._actions['zoom'].setChecked(False)

        span_control = QtWidgets.QPushButton('λ')
        span_control.setCheckable(True)
        span_control.clicked.connect(enable_span)
        span_control.setStyleSheet("color: #de2d26; font-weight: bold;")
        self.fig.canvas.manager.toolbar.addWidget(span_control)

        plt.show()

    def _on_select(self, xmin, xmax):
        wvln = self._ui['textbox'].text().strip()

        if wvln == '':
            self._ui['textbox'].setText('Error: please enter a wavelength value before selecting')
            return

        wave_val = float(wvln)

        # if line_list specified, find closest line from list:
        if self.line_list is not None:
            absdiff = np.abs(self.line_list - wave_val)
            idx = absdiff.argmin()
            if absdiff[idx] > 1.:
                logger.error("Couldn't find precise line corresponding to "
                             "input {:.3f}".format(wave_val))
                return

            logger.info("Snapping input wavelength {:.3f} to line list "
                        "value {:.3f}".format(wave_val, self.line_list[idx]))
            wave_val = self.line_list[idx]

        line_props = self.get_line_props(xmin, xmax)
        if line_props is None:
            return

        self.draw_line_marker(line_props, wave_val, xmin, xmax)

        self.fig.suptitle('')
        plt.draw()
        self.fig.canvas.draw()

        self._map_dict['wavel'].append(wave_val)
        self._map_dict['pixel'].append(line_props['x_0'])
        # line_widths.append(line_props['std_G'])

    def get_line_props(self, xmin, xmax):
        i1 = int(np.floor(xmin))
        i2 = int(np.ceil(xmax))+1

        pix = self.pix[i1:i2]
        flux = self.flux[i1:i2]

        if self.flux_ivar is not None:
            flux_ivar = self.flux_ivar[i1:i2]
        else:
            flux_ivar = None

        try:
            line_props = fit_emission_line(pix, flux, flux_ivar, n_bg_coef=2)
        except Exception as e:
            raise
            msg = "Failed to fit line!"
            logger.error(msg)
            logger.error(str(e))
            self._ui['textbox'].setText("ERROR: See terminal for more information.")
            return None

        return line_props

    def draw_line_marker(self, line_props, wavelength, xmin, xmax):
        pix_grid = np.linspace(xmin, xmax, 512)
        flux_grid = voigt_polynomial(pix_grid, **line_props)

        peak = flux_grid.max()
        x0 = line_props['x_0']

        space = 0.05*peak
        self.ax.plot([x0,x0], [peak+space,peak+3*space],
                     lw=1., linestyle='-', marker='', alpha=0.5, c='#2166AC')
        self.ax.plot(pix_grid, flux_grid,
                     lw=1., linestyle='-', marker='', alpha=0.25, c='#31a354')
        self.ax.text(x0, peak+4*space, "{:.3f} $\AA$".format(wavelength),
                     ha='center', va='bottom', rotation='vertical')

def main(proc_path, linelist_file, wavelength_data_file, overwrite=False):
    """ """

    proc_path = path.realpath(path.expanduser(proc_path))
    if not path.exists(proc_path):
        raise IOError("Path '{}' doesn't exist".format(proc_path))
    logger.info("Reading data from path: {}".format(proc_path))

    base_path, name = path.split(proc_path)
    output_path = path.realpath(path.join(base_path, '{}_proc'.format(name)))
    os.makedirs(output_path, exist_ok=True)
    logger.info("Saving processed files to path: {}".format(output_path))

    # read linelist if specified
    if linelist_file is not None:
        line_list = np.genfromtxt(linelist_file, usecols=[0], dtype=float)

    else:
        line_list = None

    if wavelength_data_file is None: # find a COMP lamp:
        ic = ccdproc.ImageFileCollection(proc_path)

        hdu = None
        for hdu,wavelength_data_file in ic.hdus(return_fname=True, imagetyp='COMP'):
            break
        else:
            raise IOError("No COMP lamp file found in {}".format(proc_path))

        wavelength_data_file = path.join(ic.location, wavelength_data_file)

    # read 2D CCD data
    ccd = CCDData.read(wavelength_data_file)

    # create 1D pixel and flux grids
    col_pix = np.arange(ccd.shape[0])

    # HACK: this is a total hack, but seems to be ok for the comp lamp spectra we have
    flux = np.mean(ccd.data[:,100:200], axis=1)
    flux_ivar = np.sum(1/ccd.uncertainty[:,100:200].array**2, axis=1)
    flux_ivar[np.isnan(flux_ivar)] = 0.

    gui = GUIWavelengthSolver(col_pix, flux, flux_ivar=flux_ivar,
                              line_list=line_list)

    # wave_pix_map = gui_solution(col_pix, flux, flux_ivar=flux_ivar,
    #                             fig=fig, ax=ax, line_list=linelist)
    # print(wave_pix_map)

    return

    # sort by pixel and write to file
    _ix = np.argsort(wave_to_pix['pixel'])
    pix_wvl = zip(np.array(wave_to_pix['pixel'])[_ix],
                  np.array(wave_to_pix['wavelength'])[_ix])
    with open(outputpath, 'w') as f:
        txt = ["# pixel wavelength"]
        for row in pix_wvl:
            txt.append("{:.5f} {:.5f}".format(*row))
        f.write("\n".join(txt))

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
                        help='Path to a PROCESSED night or chunk of data to process.')
    parser.add_argument('--linelist', dest='linelist_file', type=str, default=None,
                        help='Path to a text file where the 0th column is a list of '
                             'emission lines for the comparison lamp. Default is to '
                             'require the user to enter exact wavelengths.')
    parser.add_argument('--comp', dest='wavelength_data_file', type=str, default=None,
                        help='Path to a specific comp lamp file. Default is to find '
                             'the first one within the data directory structure.')

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
