"""
TODO:
- Cascading deletes. If we delete Observation, it should delete SimbadInfo,
  TGASSource, SpectralLineMeasurement.
"""

from __future__ import division, print_function

# Standard library
from os import path

# Third-party
import astropy.units as u
from sqlalchemy import Column, types
from sqlalchemy.schema import ForeignKey
from sqlalchemy.orm import relationship

# Project
from .connect import Base
from .custom_types import (QuantityTypeClassFactory, JDTimeType,
                           SexHourAngleType, SexDegAngleType, IntEnumType)
from .numpy_adapt import * # just need to execute code

__all__ = ['Run', 'Observation', 'SimbadInfo', 'TGASSource',
           'SpectralLineMeasurement', 'SpectralLineInfo']

VelocityType = QuantityTypeClassFactory(u.km/u.s)
WavelengthType = QuantityTypeClassFactory(u.angstrom)

class Run(Base):
    __tablename__ = 'run'

    id = Column(types.Integer, primary_key=True)
    name = Column('name', types.String, nullable=False)

    # Relationships
    observations = relationship('Observation')

class Observation(Base):
    __tablename__ = 'observation'

    id = Column(types.Integer, primary_key=True)

    night = Column('night', IntEnumType(1,2,3,4,5), nullable=False)

    filename_raw = Column('filename_raw', types.String, nullable=False)
    filename_p = Column('filename_p', types.String, nullable=False)
    filename_1d = Column('filename_1d', types.String, nullable=False)

    group_id = Column('group_id', types.Integer)
    v_bary = Column('v_bary', VelocityType, nullable=False)

    # taken straight from FITS header:
    object = Column('object', types.String, nullable=False)
    ccdpicno = Column('ccdpicno', types.SmallInteger, nullable=False)
    imagetyp = Column('imagetyp', types.String, nullable=False)
    observer = Column('observer', types.String, nullable=False)
    timesys = Column('timesys', types.String, nullable=False)
    equinox = Column('equinox', types.REAL, nullable=False)
    airmass = Column('airmass', types.REAL, nullable=False)
    exptime = Column('exptime', types.REAL, nullable=False)
    rotangle = Column('rotangle', types.REAL, nullable=False)
    telfocus = Column('telfocus', types.REAL, nullable=False)
    date_obs = Column('date_obs', types.String, nullable=False)
    time_obs = Column('time_obs', types.String, nullable=False)

    # special-case:
    st = Column('st', SexHourAngleType, nullable=False)
    ha = Column('ha', SexHourAngleType, nullable=False)
    ra = Column('ra', SexHourAngleType, nullable=False)
    dec = Column('dec', SexDegAngleType, nullable=False)
    jd = Column('jd', JDTimeType, nullable=False)

    # Relationships
    run_id = Column('run_id', types.Integer, ForeignKey('run.id'),
                    nullable=False)
    run = relationship('Run')

    simbad_info_id = Column('simbad_info_id', types.Integer,
                            ForeignKey('simbad_info.id'))
    simbad_info = relationship('SimbadInfo')

    tgas_source_id = Column('tgas_source_id', types.Integer,
                            ForeignKey('tgas_source.id'))
    tgas_source = relationship('TGASSource')

    measurements = relationship('SpectralLineMeasurement')

    def path_night(self, base_path):
        return path.join(base_path, self.run.name, 'n'.str(self.night))

    def path_raw(self, base_path):
        return path.join(self.path_night(base_path), self.filename_raw)

    def path_p(self, base_path):
        return path.join(self.path_night(base_path), self.filename_p)

    def path_1d(self, base_path):
        return path.join(self.path_night(base_path), self.filename_1d)


class SimbadInfo(Base):
    __tablename__ = 'simbad_info'

    id = Column(types.Integer, primary_key=True)
    hd_id = Column('hd_id', types.String)
    hip_id = Column('hip_id', types.String)
    tyc_id = Column('tyc_id', types.String)
    twomass_id = Column('twomass_id', types.String)

    rv = Column('rv', VelocityType)
    rv_qual = Column('rv_qual', types.String)
    rv_bibcode = Column('rv_bibcode', types.String)

class TGASSource(Base):
    __tablename__ = 'tgas_source'

    id = Column(types.Integer, primary_key=True)
    row_index = Column('row_index', types.Integer, nullable=False)

    # auto-generated from TGAS FITS files
    solution_id = Column('solution_id', types.Integer, nullable=False)
    source_id = Column('source_id', types.Integer, nullable=False)
    random_index = Column('random_index', types.Integer, nullable=False)
    ref_epoch = Column('ref_epoch', types.REAL, nullable=False)
    ra = Column('ra', SexDegAngleType, nullable=False)
    ra_error = Column('ra_error', types.REAL, nullable=False)
    dec = Column('dec', SexDegAngleType, nullable=False)
    dec_error = Column('dec_error', types.REAL, nullable=False)
    parallax = Column('parallax', types.REAL, nullable=False)
    parallax_error = Column('parallax_error', types.REAL, nullable=False)
    pmra = Column('pmra', types.REAL, nullable=False)
    pmra_error = Column('pmra_error', types.REAL, nullable=False)
    pmdec = Column('pmdec', types.REAL, nullable=False)
    pmdec_error = Column('pmdec_error', types.REAL, nullable=False)
    ra_dec_corr = Column('ra_dec_corr', types.REAL, nullable=False)
    ra_parallax_corr = Column('ra_parallax_corr', types.REAL, nullable=False)
    ra_pmra_corr = Column('ra_pmra_corr', types.REAL, nullable=False)
    ra_pmdec_corr = Column('ra_pmdec_corr', types.REAL, nullable=False)
    dec_parallax_corr = Column('dec_parallax_corr', types.REAL, nullable=False)
    dec_pmra_corr = Column('dec_pmra_corr', types.REAL, nullable=False)
    dec_pmdec_corr = Column('dec_pmdec_corr', types.REAL, nullable=False)
    parallax_pmra_corr = Column('parallax_pmra_corr', types.REAL, nullable=False)
    parallax_pmdec_corr = Column('parallax_pmdec_corr', types.REAL, nullable=False)
    pmra_pmdec_corr = Column('pmra_pmdec_corr', types.REAL, nullable=False)
    phot_g_n_obs = Column('phot_g_n_obs', types.SmallInteger, nullable=False)
    phot_g_mean_flux = Column('phot_g_mean_flux', types.REAL, nullable=False)
    phot_g_mean_flux_error = Column('phot_g_mean_flux_error', types.REAL, nullable=False)
    phot_g_mean_mag = Column('phot_g_mean_mag', types.REAL, nullable=False)

    l = Column('l', SexDegAngleType, nullable=False)
    b = Column('b', SexDegAngleType, nullable=False)
    ecl_lon = Column('ecl_lon', SexDegAngleType, nullable=False)
    ecl_lat = Column('ecl_lat', SexDegAngleType, nullable=False)

class SpectralLineMeasurement(Base):
    __tablename__ = 'spectral_line_measurement'

    id = Column(types.Integer, primary_key=True)

    x0 = Column('x0', types.REAL, nullable=False)
    x0_error = Column('x0_error', types.REAL)

    amp = Column('amp', types.REAL, nullable=False)
    amp_error = Column('amp_error', types.REAL)

    std_G = Column('std_G', types.REAL, nullable=False)
    std_G_error = Column('std_G_error', types.REAL, nullable=False)

    hwhm_L = Column('hwhm_L', types.REAL)
    hwhm_L_error = Column('hwhm_L_error', types.REAL)

    # Relationships
    observation_id = Column('observation_id', types.Integer,
                            ForeignKey('observation.id'))
    observation = relationship('Observation')

    line_id = Column('line_id', types.Integer,
                     ForeignKey('spectral_line_info.id'))
    info = relationship('SpectralLineInfo')

class SpectralLineInfo(Base):
    __tablename__ = 'spectral_line_info'

    id = Column(types.Integer, primary_key=True)
    name = Column('name', types.String, nullable=False)
    wavelength = Column('wavelength', WavelengthType, nullable=False)
