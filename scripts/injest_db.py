# Standard library
from os import path
import glob

# Third-party
import astropy.coordinates as coord
from astropy.time import Time
import astropy.units as u
from astropy.io import fits
from astropy.table import Table
from astroquery.simbad import Simbad
import numpy as np

# Project
from comoving_rv.db import Session, Base, db_connect
from comoving_rv.db.model import (Run, Observation, TGASSource, SimbadInfo,
                                  SpectralLineInfo)
from comoving_rv.velocity import bary_vel_corr, kitt_peak
from comoving_rv.log import logger

def main(db_path, run_root_path, drop_all=False, **kwargs):
    drop_all = True # HACK

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
    lines = []
    lines.append(SpectralLineInfo(name='Halpha',
                                  wavelength=6562.8*u.angstrom))
    lines.append(SpectralLineInfo(name='[OI] 5577',
                                  wavelength=5577.3387*u.angstrom))
    lines.append(SpectralLineInfo(name='[OI] 6300',
                                  wavelength=6300.304*u.angstrom))
    lines.append(SpectralLineInfo(name='[OI] 6364',
                                  wavelength=6363.776*u.angstrom))
    session.add_all(lines)
    session.commit()

    # Create an entry for this observing run
    data_path, run_name = path.split(run_root_path)
    run = Run(name=run_name)
    session.add(run)
    session.commit()

    # Now we need to go through each processed night of data and load all of the
    # relevant observations of sources.
    obs_columns = [str(c).split('.')[1] for c in Observation.__table__.columns]
    tgassource_columns = [str(c).split('.')[1]
                          for c in TGASSource.__table__.columns]

    # Here's where there's a bit of hard-coded bewitchery - the nights (within
    # each run) have to be labeled 'n1', 'n2', and etc. Sorry.
    glob_pattr_proc = path.join(data_path, 'processed', run_name, 'n?')
    for proc_night_path in glob.glob(glob_pattr_proc):
        night = path.basename(proc_night_path)
        night_id = int(night[1])

        observations = []
        tgas_sources = []

        glob_pattr_1d = path.join(proc_night_path, '1d_*.fit')
        for path_1d in glob.glob(glob_pattr_1d):
            hdr = fits.getheader(path_1d)

            # skip all except OBJECT observations
            if hdr['IMAGETYP'] != 'OBJECT':
                continue

            basename = path.basename(path_1d)[3:]
            logger.log(1, 'loading row for {0}'.format(basename))

            kw = dict()

            # construct filenames using hard-coded bullshit
            kw['filename_raw'] = path.join(run_name, night, basename)
            kw['filename_p'] = path.join('processed', run_name,
                                         night, 'p_'+basename)
            kw['filename_1d'] = path.join('processed', run_name,
                                          night, '1d_'+basename)

            # read in header of 1d file and store keywords that exist as columns
            for k,v in hdr.items():
                k = k.lower().replace('-', '_')

                if k in obs_columns:
                    kw[k] = v

            # get group id from name
            if '-' in hdr['OBJECT']:
                split_name = hdr['OBJECT'].split('-')
                kw['group_id'] = int(split_name[0])
                smoh_idx = int(split_name[1])
                tgas_row_idx = ID_tbl[smoh_idx]['tgas_row']
                tgas_row = tgas[tgas_row_idx]

                # query Simbad to get all possible names for this target
                if tgas_row['hip'] > 0:
                    name = 'HIP{0}'.format(tgas_row['hip'])
                else:
                    name = 'TYC {0}'.format(tgas_row['tycho2_id'])
                logger.log(1, 'common name: {0}'.format(name))

                all_ids = Simbad.query_objectids(name)['ID'].astype(str)

                logger.log(1, 'this is a group object')

            else:
                name = hdr['OBJECT']
                logger.log(1, 'common name: {0}'.format(name))
                logger.log(1, 'this is not a group object')

                # query Simbad to get all possible names for this target
                all_ids = Simbad.query_objectids(name)['ID'].astype(str)

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

            # compute barycenter velocity given coordinates of where the
            #   telescope was pointing
            t = Time(hdr['JD'], format='jd', scale='utc')
            sc = coord.SkyCoord(ra=hdr['RA'], dec=hdr['DEC'],
                                unit=(u.hourangle, u.degree))
            kw['v_bary'] = bary_vel_corr(t, sc, location=kitt_peak)

            obs = Observation(night=night_id, **kw)
            obs.run = run
            obs.simbad_info = simbad_info

            # Get the TGAS data if the source is in TGAS
            if tgas_row_idx is not None:
                tgas_kw = dict()
                tgas_kw['row_index'] = tgas_row_idx
                for name in tgas.colnames:
                    if name in tgassource_columns:
                        tgas_kw[name] = tgas_row[name]

                tgas_source = TGASSource(**tgas_kw)
                tgas_sources.append(tgas_source)

                obs.tgas_source = tgas_source

            observations.append(obs)

            logger.log(1, '-'*68)

            # TODO HACK: remove this when running for real
            if len(observations) == 1:
                break

        session.add_all(observations)
        session.add_all(tgas_sources)
        session.commit()

    session.close()

if __name__ == "__main__":
    from argparse import ArgumentParser
    import logging

    # Define parser object
    parser = ArgumentParser(description="Initialize the TwoFace project database.")

    vq_group = parser.add_mutually_exclusive_group()
    vq_group.add_argument('-v', '--verbose', action='count', default=0, dest='verbosity')
    vq_group.add_argument('-q', '--quiet', action='count', default=0, dest='quietness')
    # parser.add_argument('-o', '--overwrite', action='store_true', dest='overwrite',
    #                     default=False, help='Destroy everything.')

    parser.add_argument("--db", dest="db_path", required=True,
                        type=str, help="Path to sqllite database file.")
    parser.add_argument("--run", dest="run_root_path", required=True,
                        type=str, help="Path to root observing run path.")

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
