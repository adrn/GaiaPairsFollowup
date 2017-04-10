# Standard library
from collections import OrderedDict

# Third-party
import numpy as np
from scipy.optimize import minimize, leastsq
from scipy.signal import argrelmin
from celerite.modeling import Model
from celerite import terms, GP

# Project
from ..log import logger
from .models import voigt_polynomial

__all__ = ['extract_1d_comp_spectrum']

def extract_1d_comp_spectrum(ccd_data, column_slice=slice(100,200)):
    # create 1D pixel and flux grids
    col_pix = np.arange(ccd_data.shape[0])

    # HACK: this is a hack, but seems to be ok for the comp lamp spectra we have
    flux_ivar = 1 / ccd_data.uncertainty[:,column_slice].array**2
    flux_ivar[np.isnan(flux_ivar)] = 0.

    flux = np.sum(flux_ivar * ccd_data.data[:,column_slice], axis=1) / np.sum(flux_ivar, axis=1)
    flux_ivar = np.sum(flux_ivar, axis=1)

    return col_pix, flux, flux_ivar

