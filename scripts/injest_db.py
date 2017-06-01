"""
TODO:
- Get 2MASS magnitudes and load into tgas_source (see notebook
  "Fill in 2MASS magnitudes") - see below
"""

# Standard library
from os import path
import glob
from collections import OrderedDict

# Third-party
import numpy as np

import astropy.coordinates as coord
from astropy.time import Time
import astropy.units as u
from astropy.io import fits
from astropy.table import Table
from astropy.utils.console import ProgressBar

from astroquery.simbad import Simbad
from astroquery.gaia import Gaia
Simbad.add_votable_fields('rv_value', 'rvz_qual', 'rvz_bibcode')

# Project
from comoving_rv.db import Session, Base, db_connect
from comoving_rv.db.model import (Run, Observation, TGASSource, SimbadInfo,
                                  SpectralLineInfo)
from comoving_rv.velocity import bary_vel_corr, kitt_peak
from comoving_rv.log import logger

def fits_header_to_cols(hdr, colnames):
    kw = dict()
    for k,v in hdr.items():
        k = k.lower().replace('-', '_')

        if k in colnames:
            kw[k] = v
    return kw

def main(db_path, run_root_path, drop_all=False, overwrite=False, **kwargs):

    # Make sure the specified paths actually exist
    db_path = path.abspath(db_path)
    run_root_path = path.abspath(run_root_path)
    for path_ in [db_path, run_root_path]:
        if not path.exists(path_):
            raise ValueError("Path '{0}' doesn't exist!".format(path_))

    # --------------------------------------------------------------------------
    # These are relative paths, so the script needs to be run from the
    #   scripts path...

    # ID table for mapping group index to TGAS row
    ID_tbl = Table.read('../data/star_identifier.csv')

    # TGAS table
    logger.debug("Loading TGAS data...")
    tgas = Table.read('../../gaia-comoving-stars/data/stacked_tgas.fits')

    # Catalog of velocities for Bensby's HIP stars:
    bensby = Table.read('../data/bensbyrv_bestunique.csv')

    # --------------------------------------------------------------------------

    # connect to the database
    engine = db_connect(db_path)
    # engine.echo = True
    logger.debug("Connected to database at '{}'".format(db_path))

    if drop_all: # remove all tables and replace
        Base.metadata.drop_all()
        Base.metadata.create_all()

    # create a new session for interacting with the database
    session = Session()

    logger.debug("Loading SpectralLineInfo table")

    line_info = OrderedDict()
    # air wavelength of Halpha -- wavelength calibration from comp lamp is done
    #   at air wavelengths, so this is where Halpha should be, right?
    line_info['Halpha'] = 6562.8*u.angstrom

    # [OI] emission lines -- wavelengths from:
    #   http://physics.nist.gov/PhysRefData/ASD/lines_form.html
    line_info['[OI] 5577'] = 5577.3387*u.angstrom
    line_info['[OI] 6300'] = 6300.304*u.angstrom
    line_info['[OI] 6364'] = 6363.776*u.angstrom

    for name, wvln in line_info.items():
        n = session.query(SpectralLineInfo).filter(SpectralLineInfo.name == name).count()
        if n == 0:
            logger.debug('Loading line {0} at {1}'.format(name, wvln))
            line = SpectralLineInfo(name=name, wavelength=wvln)
            session.add(line)
            session.commit()

    # Create an entry for this observing run
    data_path, run_name = path.split(run_root_path)
    n = session.query(Run).filter(Run.name == run_name).count()
    if n == 0:
        run = Run(name=run_name)
        session.add(run)
        session.commit()

    # Now we need to go through each processed night of data and load all of the
    # relevant observations of sources.

    # First we get the column names for the Observation and TGASSource tables
    obs_columns = [str(c).split('.')[1] for c in Observation.__table__.columns]
    tgassource_columns = [str(c).split('.')[1]
                          for c in TGASSource.__table__.columns]

    # Here's where there's a bit of hard-coded bewitchery - the nights (within
    # each run) have to be labeled 'n1', 'n2', and etc. Sorry.
    glob_pattr_proc = path.join(data_path, 'processed', run_name, 'n?')
    for proc_night_path in glob.glob(glob_pattr_proc):
        night = path.basename(proc_night_path)
        night_id = int(night[1])
        logger.debug('Loading night {0}...'.format(night_id))

        observations = []
        tgas_sources = []

        glob_pattr_1d = path.join(proc_night_path, '1d_*.fit')
        for path_1d in ProgressBar(glob.glob(glob_pattr_1d)):
            hdr = fits.getheader(path_1d)

            # skip all except OBJECT observations
            if hdr['IMAGETYP'] != 'OBJECT':
                continue

            basename = path.basename(path_1d)[3:]
            logger.log(1, 'loading row for {0}'.format(basename))

            kw = dict()

            # construct filenames using hard-coded bullshit
            kw['filename_raw'] = basename
            kw['filename_p'] = 'p_' + basename
            kw['filename_1d'] = '1d_' + basename

            # check if this filename is already in the database, if so, drop it
            base_query = session.query(Observation)\
                                .filter(Observation.filename_raw == kw['filename_raw'])
            already_loaded = base_query.count() > 0

            if already_loaded and overwrite:
                base_query.delete()
                session.commit()

            elif already_loaded:
                logger.debug('Object {0} [{1}] already loaded'
                             .format(hdr['OBJECT'],
                                     path.basename(kw['filename_raw'])))
                continue

            # read in header of 1d file and store keywords that exist as columns
            kw.update(fits_header_to_cols(hdr, obs_columns))

            # get group id from object name
            if '-' in hdr['OBJECT']:
                split_name = hdr['OBJECT'].split('-')
                kw['group_id'] = int(split_name[0])

                # because: reasons
                if kw['group_id'] == 10:
                    tgas_row_idx = int(split_name[1])
                else:
                    smoh_idx = int(split_name[1])
                    tgas_row_idx = ID_tbl[smoh_idx]['tgas_row']
                tgas_row = tgas[tgas_row_idx]

                # query Simbad to get all possible names for this target
                if tgas_row['hip'] > 0:
                    object_name = 'HIP{0}'.format(tgas_row['hip'])
                else:
                    object_name = 'TYC {0}'.format(tgas_row['tycho2_id'])
                logger.log(1, 'common name: {0}'.format(object_name))

                try:
                    all_ids = Simbad.query_objectids(object_name)['ID'].astype(str)
                except Exception as e:
                    logger.warning('Simbad query_objectids failed for "{0}" '
                                   'with error: {1}'
                                   .format(object_name, str(e)))
                    all_ids = []

                logger.log(1, 'this is a group object')

                if len(all_ids) > 0:
                    logger.log(1, 'other names for this object: {0}'
                               .format(', '.join(all_ids)))
                else:
                    logger.log(1, 'simbad names for this object could not be '
                               'retrieved')

            else:
                object_name = hdr['OBJECT']
                logger.log(1, 'common name: {0}'.format(object_name))
                logger.log(1, 'this is not a group object')

                # query Simbad to get all possible names for this target
                try:
                    all_ids = Simbad.query_objectids(object_name)['ID'].astype(str)
                except Exception as e:
                    logger.warning('SKIPPING: Simbad query_objectids failed for '
                                   '"{0}" with error: {1}'
                                   .format(object_name, str(e)))
                    continue

                # get the Tycho 2 ID, if it has one
                tyc_id = [id_ for id_ in all_ids if 'TYC' in id_]
                if tyc_id:
                    tyc_id = tyc_id[0].replace('TYC', '').strip()
                    logger.log(1, 'source has tycho 2 id: {0}'.format(tyc_id))
                    tgas_row_idx = np.where(tgas['tycho2_id'] == tyc_id)[0]

                    if len(tgas_row_idx) == 0:
                        tgas_row_idx = None
                    else:
                        tgas_row = tgas[tgas_row_idx]

                else:
                    logger.log(1, 'source has no tycho 2 id.')
                    tgas_row_idx = None

            # store relevant names / IDs
            simbad_info_kw = dict()
            for id_ in all_ids:
                if id_.lower().startswith('hd'):
                    simbad_info_kw['hd_id'] = id_[2:]

                elif id_.lower().startswith('hip'):
                    simbad_info_kw['hip_id'] = id_[3:]

                elif id_.lower().startswith('tyc'):
                    simbad_info_kw['tyc_id'] = id_[3:]

                elif id_.lower().startswith('2mass'):
                    simbad_info_kw['twomass_id'] = id_[5:]

            for k,v in simbad_info_kw.items():
                simbad_info_kw[k] = v.strip()

            simbad_info = SimbadInfo(**simbad_info_kw)

            # Compute barycenter velocity given coordinates of where the
            # telescope was pointing and observation time
            t = Time(hdr['JD'], format='jd', scale='utc')
            sc = coord.SkyCoord(ra=hdr['RA'], dec=hdr['DEC'],
                                unit=(u.hourangle, u.degree))
            kw['v_bary'] = bary_vel_corr(t, sc, location=kitt_peak)

            obs = Observation(night=night_id, **kw)
            obs.run = run

            # Get the TGAS data if the source is in TGAS
            if tgas_row_idx is not None:
                logger.log(1, 'TGAS row: {0}'.format(tgas_row_idx))

                tgas_kw = dict()
                tgas_kw['row_index'] = tgas_row_idx
                for name in tgas.colnames:
                    if name in tgassource_columns:
                        tgas_kw[name] = tgas_row[name]

                # TODO:
                # query = """
                # SELECT TOP 10 j_m, j_msigcom, h_m, h_msigcom, ks_m, ks_msigcom
                # FROM gaiadr1.tmass_original_valid
                # JOIN gaiadr1.tmass_best_neighbour USING (tmass_oid)
                # JOIN gaiadr1.tgas_source USING (source_id)
                # WHERE source_id = {0.source_id}
                # """

                # job = Gaia.launch_job(query.format(src), dump_to_file=False)
                # res = job.get_results()

                # if len(res) == 0:
                #     print("No 2MASS data found for: {0}".format(src.source_id))

                # elif len(res) == 1:
                #     src.J = res['j_m'][0]
                #     src.J_err = res['j_msigcom'][0]
                #     src.H = res['h_m'][0]
                #     src.H_err = res['h_msigcom'][0]
                #     src.Ks = res['ks_m'][0]
                #     src.Ks_err = res['ks_msigcom'][0]

                tgas_source = TGASSource(**tgas_kw)
                tgas_sources.append(tgas_source)

                obs.tgas_source = tgas_source

            else:
                logger.log(1, 'TGAS row could not be found.')

            # object_name is never None?
            try:
                result = Simbad.query_object(object_name)
            except Exception as e:
                logger.warning('Simbad query_object failed for "{0}" '
                               'with error: {1}'
                               .format(object_name, str(e)))
                continue

            if result is not None and not np.any(result['RV_VALUE'].mask):
                k, = np.where(np.logical_not(result['RV_VALUE'].mask))
                simbad_info.rv = float(result['RV_VALUE'][k]) * u.km/u.s
                simbad_info.rv_qual = result['RVZ_QUAL'].astype(str)[k]
                simbad_info.rv_bibcode = result['RVZ_BIBCODE'].astype(str)[k]

            obs.simbad_info = simbad_info
            observations.append(obs)

            logger.log(1, '-'*68)

        session.add_all(observations)
        session.add_all(tgas_sources)
        session.commit()

    # Last thing to do is cross-match with the Bensby catalog to
    #   replace velocities when they are better
    for sim_info in session.query(SimbadInfo)\
                           .filter(SimbadInfo.hip_id != None).all():
        hip_id = 'HIP' + str(sim_info.hip_id)
        row = bensby[bensby['OBJECT'] == hip_id]
        if len(row) > 0:
            sim_info.rv = row['velValue']
            sim_info.rv_qual = row['quality']
            sim_info.rv_bibcode = row['bibcode']
            session.flush()

    session.close()

if __name__ == "__main__":
    from argparse import ArgumentParser
    import logging

    # Define parser object
    parser = ArgumentParser(description="Initialize the TwoFace project database.")

    vq_group = parser.add_mutually_exclusive_group()
    vq_group.add_argument('-v', '--verbose', action='count', default=0, dest='verbosity')
    vq_group.add_argument('-q', '--quiet', action='count', default=0, dest='quietness')
    parser.add_argument('-o', '--overwrite', action='store_true', dest='overwrite',
                        default=False, help='Destroy everything.')

    parser.add_argument('--db', dest='db_path', required=True,
                        type=str, help='Path to sqllite database file.')
    parser.add_argument('--run', dest='run_root_path', required=True,
                        type=str, help='Path to root observing run path.')
    parser.add_argument('--drop-all', action='store_true', dest='drop_all',
                        default=False, help='Destroy all tables.')

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

    main(**vars(args))
