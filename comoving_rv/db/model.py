"""
TODO:
- Cascading deletes. If we delete Observation, it should delete SimbadInfo,
  TGASSource, SpectralLineMeasurement.
"""

from __future__ import division, print_function

# Standard library
from os import path

# Third-party
import astropy.coordinates as coord
import astropy.units as u
import numpy as np
from sqlalchemy import Column, types
from sqlalchemy.schema import ForeignKey
from sqlalchemy.orm import relationship, backref

# Project
from .connect import Base
from .custom_types import (QuantityTypeClassFactory, JDTimeType,
                           SexHourAngleType, SexDegAngleType, IntEnumType)
from .numpy_adapt import * # just need to execute code

__all__ = ['Run', 'Observation', 'SimbadInfo', 'TGASSource',
           'SpectralLineMeasurement', 'SpectralLineInfo', 'RVMeasurement']

VelocityType = QuantityTypeClassFactory(u.km/u.s)
WavelengthType = QuantityTypeClassFactory(u.angstrom)

class Run(Base):
    __tablename__ = 'run'

    id = Column(types.Integer, primary_key=True)
    name = Column('name', types.String, nullable=False)

    # Relationships
    observations = relationship('Observation', cascade='all, delete-orphan')

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
    simbad_info = relationship('SimbadInfo', cascade='all,delete-orphan',
                               backref=backref('observation', uselist=False),
                               single_parent=True)

    prior_rv_id = Column('prior_rv_id', types.Integer,
                         ForeignKey('prior_rv.id'))
    prior_rv = relationship('PriorRV', cascade='all,delete-orphan',
                            backref=backref('observation', uselist=False),
                            single_parent=True)

    tgas_source_id = Column('tgas_source_id', types.Integer,
                            ForeignKey('tgas_source.id'))
    tgas_source = relationship('TGASSource', cascade='all,delete-orphan',
                               backref=backref('observation', uselist=False),
                               single_parent=True)

    def __repr__(self):
        return ("<Observation {0.object} [{0.simbad_info.preferred_name}], "
                "{0.filename_raw}, {0.run.name}>".format(self))

    def path_raw(self, base_path):
        p = path.join(base_path, self.run.name, 'n'+str(self.night))
        return path.join(p, self.filename_raw)

    def path_p(self, base_path):
        p = path.join(base_path, 'processed', self.run.name,
                      'n'+str(self.night))
        return path.join(p, self.filename_p)

    def path_1d(self, base_path):
        p = path.join(base_path, 'processed', self.run.name,
                      'n'+str(self.night))
        return path.join(p, self.filename_1d)

    @property
    def utc_hour(self):
        return np.sum(np.array(list(map(float, self.time_obs.split(':')))) / np.array([1., 60., 3600.]))

    def tgas_star(self, with_rv=True):
        """gwb.data.TGASData instance"""
        from gwb.data import TGASStar

        names = list(self.tgas_source.row_dict.keys())
        dtype = dict(names=names, formats=['f8']*len(names))
        arr = np.array([tuple([self.tgas_source.row_dict[k] for k in names])],
                       dtype)

        kw = dict()
        if with_rv:
            kw['rv'] = self.rv_measurement.rv
            kw['rv_err'] = self.rv_measurement.err

        return TGASStar(arr[0], **kw)

    def icrs(self, with_rv=True):
        kw = dict()
        if with_rv:
            if hasattr(with_rv, 'unit'):
                rv = with_rv
            else:
                rv = self.rv_measurement.rv

            kw['radial_velocity'] = rv

        return coord.ICRS(ra=self.tgas_source.ra, dec=self.tgas_source.dec,
                          distance=1000./self.tgas_source.parallax*u.pc,
                          pm_ra_cosdec=self.tgas_source.pmra*u.mas/u.yr,
                          pm_dec=self.tgas_source.pmdec*u.mas/u.yr, **kw)

    def icrs_samples(self, size=1, custom_rv=None):
        y = np.zeros(6)
        Cov = np.zeros((6,6))

        y[:5] = [self.tgas_source.ra.value, self.tgas_source.dec.value,
                 self.tgas_source.parallax, self.tgas_source.pmra,
                 self.tgas_source.pmdec]
        Cov[:5,:5] = self.tgas_star().get_cov()[:5,:5]

        if custom_rv is None:
            rv = self.rv_measurement.rv
            rv_err = self.rv_measurement.err
        else:
            rv, rv_err = custom_rv

        Cov[5, 5] = rv_err.value**2
        y[5] = rv.value
        samples = np.random.multivariate_normal(y, Cov, size=size)

        return coord.ICRS(ra=samples[:,0]*u.deg, dec=samples[:,1]*u.deg,
                          distance=1000./samples[:,2]*u.pc,
                          pm_ra_cosdec=samples[:,3]*u.mas/u.yr,
                          pm_dec=samples[:,4]*u.mas/u.yr,
                          radial_velocity=samples[:,5]*u.km/u.s)

class SimbadInfo(Base):
    __tablename__ = 'simbad_info'

    id = Column(types.Integer, primary_key=True)
    hd_id = Column('hd_id', types.String)
    hip_id = Column('hip_id', types.String)
    tyc_id = Column('tyc_id', types.String)
    twomass_id = Column('twomass_id', types.String)

    def __repr__(self):
        names = []
        pres = ['HD', 'HIP', 'TYC', '2MASS']
        for pre,id_ in zip(pres, ['hd_id', 'hip_id', 'tyc_id', 'twomass_id']):
            if getattr(self, id_) is not None:
                names.append('{0} {1}'.format(pre, getattr(self, id_)))

        return ("<SimbadInfo {0} [{1}]>"
                .format(self.preferred_name, ', '.join(names)))

    @property
    def preferred_name(self):
        pres = ['HD', 'HIP', 'TYC', '2MASS']
        attrs = ['hd_id', 'hip_id', 'tyc_id', 'twomass_id']

        for pre,id_ in zip(pres, attrs):
            the_id = getattr(self, id_)
            if the_id is not None:
                return '{0} {1}'.format(pre, the_id)

        return '{0} {1}'.format('SMOH', self.observation.object)

class PriorRV(Base):
    __tablename__ = 'prior_rv'

    id = Column(types.Integer, primary_key=True)
    rv = Column('rv', VelocityType, nullable=False)
    err = Column('err', VelocityType, default=10.*u.km/u.s)
    qual = Column('qual', types.String)
    bibcode = Column('bibcode', types.String)
    source = Column('source', types.String, nullable=False)

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

    J = Column('J', types.REAL)
    J_err = Column('J_err', types.REAL)
    H = Column('H', types.REAL)
    H_err = Column('H_err', types.REAL)
    Ks = Column('Ks', types.REAL)
    Ks_err = Column('Ks_err', types.REAL)

    @property
    def skycoord(self):
        return coord.SkyCoord(ra=self.ra, dec=self.dec,
                              distance=1000./self.parallax*u.pc)

    @property
    def row_dict(self):
        row = dict()
        for k in self.__dict__:
            if not k.startswith('_'):
                v = getattr(self, k)

                if hasattr(v, 'value'):
                    row[k] = v.value

                elif not isinstance(v, float):
                    continue

                else:
                    row[k] = v

        return row

class SpectralLineMeasurement(Base):
    __tablename__ = 'spectral_line_measurement'

    id = Column(types.Integer, primary_key=True)

    x0 = Column('x0', types.REAL, nullable=False)
    x0_error = Column('x0_error', types.REAL)

    amp = Column('amp', types.REAL, nullable=False)
    amp_error = Column('amp_error', types.REAL)

    std_G = Column('std_G', types.REAL, nullable=False)
    std_G_error = Column('std_G_error', types.REAL)

    hwhm_L = Column('hwhm_L', types.REAL)
    hwhm_L_error = Column('hwhm_L_error', types.REAL)

    # Relationships
    observation_id = Column('observation_id', types.Integer,
                            ForeignKey('observation.id'))
    observation = relationship('Observation',
                               backref=backref('measurements',
                                               cascade="all,delete"))

    line_id = Column('line_id', types.Integer,
                     ForeignKey('spectral_line_info.id'))
    info = relationship('SpectralLineInfo')

    def __repr__(self):
        r = '<SpectralLineMeasurement {0} x0={1:.3f}>'.format(self.info.name,
                                                              self.x0)
        return r

class SpectralLineInfo(Base):
    __tablename__ = 'spectral_line_info'

    id = Column(types.Integer, primary_key=True)
    name = Column('name', types.String, nullable=False)
    wavelength = Column('wavelength', WavelengthType, nullable=False)

    def __repr__(self):
        return '<SpectralLineInfo {0} @ {1}>'.format(self.name, self.wavelength)

class RVMeasurement(Base):
    __tablename__ = 'rv_measurement'

    id = Column(types.Integer, primary_key=True)
    rv = Column('rv', VelocityType, nullable=False)
    err = Column('err', VelocityType, nullable=False)
    flag = Column('flag', types.Integer, nullable=False)

    # Relationships
    observation_id = Column('observation_id', types.Integer,
                            ForeignKey('observation.id'))
    observation = relationship('Observation', single_parent=True,
                               backref=backref('rv_measurement',
                                               cascade='all,delete-orphan',
                                               uselist=False))
