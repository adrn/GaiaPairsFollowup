# Standard library
from collections import OrderedDict
from os import path
import sys

# Third-party
from astroquery.gaia import Gaia
import numpy as np

# Package
from comoving_rv.log import logger
from comoving_rv.db import Session, Base, db_connect
from comoving_rv.db.model import (Run, Observation, TGASSource, SimbadInfo,
                                  GroupToObservations, SpectralLineInfo,
                                  SpectralLineMeasurement, RVMeasurement,
                                  Photometry)

from stellarparameters import get_photometry

result_columns = ['bt_mag',
                  'vt_mag',
                  'e_bt_mag',
                  'e_vt_mag',
                  'j_m',
                  'j_msigcom',
                  'h_m',
                  'h_msigcom',
                  'ks_m',
                  'ks_msigcom',
                  'w1mpro',
                  'w1mpro_error',
                  'w2mpro',
                  'w2mpro_error',
                  'w3mpro',
                  'w3mpro_error',
                  'w4mpro',
                  'w4mpro_error',
                  'u_mag',
                  'u_mag_error',
                  'g_mag',
                  'g_mag_error',
                  'r_mag',
                  'r_mag_error',
                  'i_mag',
                  'i_mag_error',
                  'z_mag',
                  'z_mag_error']

def main():
    # TODO: bad, hard-coded...
    # base_path = '/Volumes/ProjectData/gaia-comoving-followup/'
    base_path = '../data/'
    db_path = path.join(base_path, 'db.sqlite')
    engine = db_connect(db_path)
    session = Session()

    credentials = dict(user='apricewh', password='7Gy2^otQNj6FKz6')
    Gaia.login(**credentials)

    for obs in session.query(Observation).all():
        q = session.query(Photometry).join(Observation).filter(Observation.id == obs.id).count()
        if q > 0:
            logger.debug('Photometry already exists')
            continue

        if obs.tgas_source is None:
            continue

        tgas_source_id = obs.tgas_source.source_id
        res = get_photometry(tgas_source_id)

        phot_kw = dict()
        for col in result_columns:
            phot_kw[col] = res[col]

        phot = Photometry(**phot_kw)
        phot.observation = obs
        session.add(phot)
        session.commit()

if __name__ == '__main__':
    import logging
    logger.setLevel(logging.DEBUG)
    main()
