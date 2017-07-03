# coding: utf-8

"""


"""

# Standard library
import os
from os import path

# Third-party
from astropy.constants import c
import astropy.units as u
import numpy as np
import matplotlib.pyplot as plt
plt.style.use('apw-notebook')
import schwimmbad
from schwimmbad import choose_pool

# Project
from comoving_rv.log import logger
from comoving_rv.db import Session, db_connect
from comoving_rv.db.model import (SpectralLineMeasurement, SpectralLineInfo,
                                  Run, Observation, PriorRV, RVMeasurement)

class RVCorrector(object):

    def __init__(self, session, run_name):
        self.session = session
        self.run_name = str(run_name)

        # get wavelength for Halpha
        self.Halpha, = session.query(SpectralLineInfo.wavelength)\
                              .filter(SpectralLineInfo.name == 'Halpha').one()

        self._compute_offset_corrections()

    def _compute_offset_corrections(self):
        session = self.session
        run_name = self.run_name

        q = session.query(Observation).join(Run, SpectralLineMeasurement, PriorRV)
        q = q.filter(Run.name == run_name)
        q = q.filter(SpectralLineMeasurement.x0 != None)
        q = q.filter(PriorRV.rv != None)
        logger.debug('{0} observations with prior RV measurements'
                     .format(q.distinct().count()))

        # retrieve all observations with measured centroids and previous RV's
        observations = q.all()

        # What we do below is look at the residual offsets between applying a naïve
        # sky-line correction and the true RV (with the barycentric velocity
        # applied)

        raw_offsets = np.zeros(len(observations)) * u.angstrom
        all_sky_offsets = np.full((len(observations), 3), np.nan) * u.angstrom
        true_rv = np.zeros(len(observations)) * u.km/u.s
        obs_time = np.zeros(len(observations))
        night_id = np.zeros(len(observations), dtype=int)
        corrected_rv = np.zeros(len(observations)) * u.km/u.s

        for i,obs in enumerate(observations):
            # convert obstime into decimal hour
            obs_time[i] = np.sum(np.array(list(map(float, obs.time_obs.split(':')))) / np.array([1., 60., 3600.]))

            # Compute the raw offset: difference between Halpha centroid and true
            # wavelength value
            x0 = obs.measurements[0].x0 * u.angstrom
            offset = (x0 - self.Halpha)
            raw_offsets[i] = offset

            night_id[i] = obs.night

            # For each sky line (that passes certain quality checks), compute the
            # offset between the predicted wavelength and measured centroid
            # TODO: generalize these quality cuts - see also below in
            # get_corrected_rv
            sky_offsets = []
            for j,meas in enumerate(obs.measurements[1:]):
                sky_offset = meas.x0*u.angstrom - meas.info.wavelength
                if (meas.amp > 16 and meas.std_G < 2 and meas.std_G > 0.3 and
                        np.abs(sky_offset) < 4*u.angstrom): # MAGIC NUMBER: quality cuts
                    sky_offsets.append(sky_offset)
                    all_sky_offsets[i,j] = sky_offset

            sky_offsets = u.Quantity(sky_offsets)

            if len(sky_offsets) > 0:
                sky_offset = np.mean(sky_offsets)
            else:
                sky_offset = np.nan * u.angstrom
                logger.debug("not correcting with sky line for {0}".format(obs))

            true_rv[i] = obs.prior_rv.rv - obs.v_bary

        raw_rv = raw_offsets / self.Halpha * c.to(u.km/u.s)

        # unique night ID's
        unq_night_id = np.unique(night_id)
        unq_night_id.sort()

        # Now we do a totally insane thing. From visualizing the residual
        # differences, there seems to be a trend with the observation time. We
        # fit a line to these residuals and use this to further correct the
        # wavelength solutions using just the (strongest) [OI] 5577 Å line.
        diff = all_sky_offsets[:,0] - ((raw_rv - true_rv)/c*5577*u.angstrom).decompose()
        diff[np.abs(diff) > 2*u.angstrom] = np.nan * u.angstrom # reject BIG offsets

        self._night_polys = dict()
        self._night_final_offsets = dict()
        for n in unq_night_id:
            mask = (night_id == n) & np.isfinite(diff)
            coef = np.polyfit(obs_time[mask], diff[mask], deg=1, w=np.full(mask.sum(), 1/0.1))
            poly = np.poly1d(coef)
            self._night_polys[n] = poly

            sky_offset = np.nanmean(all_sky_offsets[mask,:2], axis=1)
            sky_offset[np.isnan(sky_offset)] = 0.*u.angstrom
            sky_offset -= self._night_polys[n](obs_time[mask]) * u.angstrom

            corrected_rv[mask] = (raw_offsets[mask] - sky_offset) / self.Halpha * c.to(u.km/u.s)

            # Finally, we align the median of each night's ∆RV distribution with 0
            drv = corrected_rv[mask] - true_rv[mask]
            self._night_final_offsets[n] = np.nanmedian(drv)

        # now estimate the std. dev. uncertainty using the MAD
        all_drv = corrected_rv - true_rv
        self._abs_err = 1.5 * np.nanmedian(np.abs(all_drv - np.nanmedian(all_drv)))

    def get_corrected_rv(self, obs):
        """Compute a corrected radial velocity for the given observation"""

        # Compute the raw offset: difference between Halpha centroid and true
        # wavelength value
        x0 = obs.measurements[0].x0 * u.angstrom
        raw_offset = (x0 - self.Halpha)

        # precision estimate from line centroid error
        precision = (obs.measurements[0].x0_error * u.angstrom) / self.Halpha * c.to(u.km/u.s)

        # For each sky line (that passes certain quality checks), compute the
        # offset between the predicted wavelength and measured centroid
        # TODO: generalize these quality cuts - see also above in
        # _compute_offset_corrections
        sky_offsets = np.full(3, np.nan) * u.angstrom
        for j,meas in enumerate(obs.measurements[1:]):
            sky_offset = meas.x0*u.angstrom - meas.info.wavelength
            if (meas.amp > 16 and meas.std_G < 2 and meas.std_G > 0.3 and
                    np.abs(sky_offset) < 3.3*u.angstrom): # MAGIC NUMBER: quality cuts
                sky_offsets[j] = sky_offset

        # final sky offset to apply
        flag = 0
        sky_offset = np.nanmean(sky_offsets)
        if np.isnan(sky_offset.value):
            logger.debug("not correcting with sky line for {0}".format(obs))
            sky_offset = 0*u.angstrom
            flag = 1

        # apply global sky offset correction - see _compute_offset_corrections()
        sky_offset -= self._night_polys[obs.night](obs.utc_hour) * u.angstrom

        # compute radial velocity and correct for sky line
        rv = (raw_offset - sky_offset) / self.Halpha * c.to(u.km/u.s)

        # correct for offset of median of ∆RV distribution from targets with
        # prior/known RV's
        rv -= self._night_final_offsets[obs.night]

        # rv error
        err = np.sqrt(self._abs_err**2 + precision**2)

        return rv, err, flag

def main(db_path, run_name, overwrite=False, pool=None):

    if pool is None:
        pool = schwimmbad.SerialPool()

    # connect to the database
    engine = db_connect(db_path)
    # engine.echo = True
    logger.debug("Connected to database at '{}'".format(db_path))

    # create a new session for interacting with the database
    session = Session()

    root_path, _ = path.split(db_path)
    plot_path = path.join(root_path, 'plots', run_name)
    if not path.exists(plot_path):
        os.makedirs(plot_path, exist_ok=True)

    # get object to correct the observed RV's
    rv_corr = RVCorrector(session, run_name)

    observations = session.query(Observation).join(Run)\
                          .filter(Run.name == run_name).all()

    for obs in observations:
        q = session.query(RVMeasurement).join(Observation)\
                   .filter(Observation.id == obs.id)

        if q.count() > 0 and not overwrite:
            logger.debug('RV measurement already complete for object '
                         '{0} in file {1}'.format(obs.object, obs.filename_raw))
            continue

        elif q.count() > 1:
            raise RuntimeError('Multiple RV measurements found for object {0}'
                               .format(obs))

        corrected_rv, err, flag = rv_corr.get_corrected_rv(obs)

        # remove previous RV measurements
        if q.count() > 0:
            session.delete(q.one())
            session.commit()

        rv_meas = RVMeasurement(rv=corrected_rv, err=err, flag=flag)
        rv_meas.observation = obs
        session.add(rv_meas)
        session.commit()

    pool.close()

if __name__ == "__main__":
    from argparse import ArgumentParser
    import logging

    # Define parser object
    parser = ArgumentParser(description="")

    vq_group = parser.add_mutually_exclusive_group()
    vq_group.add_argument('-v', '--verbose', action='count',
                          default=0, dest='verbosity')
    vq_group.add_argument('-q', '--quiet', action='count',
                          default=0, dest='quietness')

    parser.add_argument('-s', '--seed', dest='seed', default=None,
                        type=int, help='Random number generator seed.')
    parser.add_argument('-o', '--overwrite', action='store_true',
                        dest='overwrite', default=False,
                        help='Destroy everything.')

    parser.add_argument('-d', '--db', dest='db_path', required=True,
                        help='Path to sqlite database file')
    parser.add_argument('-r', '--run', dest='run_name', required=True,
                        help='Name of the observing run')

    # multiprocessing options
    group = parser.add_mutually_exclusive_group()
    group.add_argument('--ncores', dest='n_cores', default=1,
                       type=int, help='Number of CPU cores to use.')
    group.add_argument('--mpi', dest='mpi', default=False,
                       action='store_true', help='Run with MPI.')

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

    pool = choose_pool(mpi=args.mpi, processes=args.n_cores)
    logger.info("Using pool: {}".format(pool.__class__))

    main(db_path=args.db_path, run_name=args.run_name,
         overwrite=args.overwrite, pool=pool)
