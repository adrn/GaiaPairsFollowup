"""
Functions to help retrieving velocities from the literature via
Vizier or Simbad.
"""

from astroquery.simbad import Simbad
Simbad.add_votable_fields('rv_value', 'rvz_qual', 'rvz_bibcode')
from astroquery.vizier import Vizier
import numpy as np

from ..log import logger

__all__ = ['get_GCRV_rv', 'get_GCS_rv', 'get_CRVAD_rv', 'get_simbad_rv',
           'get_best_rv']

def get_GCS_rv(obs):
    """
    Try to retrieve a velocity from the Geneva-Copenhagen Survey.

    Parameters
    ----------
    obs : `comoving_rv.db.model.Observation`
        An observation instance.

    Returns
    -------
    rv : float
        Radial velocity (in km/s).
    rv_err : float or None
        Radial velocity error (in km/s).
    rv_qual : str or None
        Quality flag for the measurement.
    bibcode : str
        Bibliographic code for the source of the measurement.

    or

    ``None``

    """
    bibcode = '2004A&A...418..989N'

    cat = Vizier(catalog="V/117/newcat", columns=['*', 'HIP', 'RVel', 'e_RVel'])

    sinfo = obs.simbad_info
    if sinfo.hd_id is not None:
        tbl = cat.query_constraints(Name='HD {}'.format(sinfo.hd_id))
        try:
            row = tbl[0][0]
        except IndexError:
            return None

    elif sinfo.hip_id is not None:
        tbl = cat.query_constraints(HIP=int(sinfo.hip_id))
        try:
            row = tbl[0][0]
        except IndexError:
            return None

    else:
        return None

    return float(row['RVel']), float(row['e_RVel']), None, bibcode

def get_CRVAD_rv(obs):
    """
    Try to retrieve a velocity from the CRVAD.

    Parameters
    ----------
    obs : `comoving_rv.db.model.Observation`
        An observation instance.

    Returns
    -------
    rv : float
        Radial velocity (in km/s).
    rv_err : float or None
        Radial velocity error (in km/s).
    rv_qual : str or None
        Quality flag for the measurement.
    bibcode : str
        Bibliographic code for the source of the measurement.

    or

    ``None``

    """
    bibcode = '2007AN....328..889K'
    cat = Vizier(catalog="III/254/crvad2", columns=['*', 'meRVel'])

    sinfo = obs.simbad_info
    if sinfo.hd_id is not None:
        tbl = cat.query_constraints(HD=sinfo.hd_id)
        try:
            row = tbl[0][0]
        except IndexError:
            return None

        err = float(row['e_RV'])
        qual = str(row['q_RV'].astype(str)).lower()

        if np.isnan(err):
            if qual in ['a', 'b']:
                err = 10.

        return float(row['RV']), err, qual, bibcode

    return None

def get_GCRV_rv(obs):
    """
    Try to retrieve a velocity from the General Catalogue of Mean
    Radial Velocities.

    Parameters
    ----------
    obs : `comoving_rv.db.model.Observation`
        An observation instance.

    Returns
    -------
    rv : float
        Radial velocity (in km/s).
    rv_err : float or None
        Radial velocity error (in km/s).
    rv_qual : str or None
        Quality flag for the measurement.
    bibcode : str
        Bibliographic code for the source of the measurement.

    or

    ``None``

    """
    bibcode = '1953GCRV..C......0W'

    sinfo = obs.simbad_info
    if sinfo.hd_id is not None:
        tbl = Vizier(catalog="III/21/gcrv").query_constraints(HD=str(sinfo.hd_id))
        try:
            row = tbl[0][0]
        except IndexError:
            return None

        return float(row['RV']), None, str(row['q_RV'].astype(str)), bibcode

    return None

def get_simbad_rv(obs):
    """
    Try to retrieve a velocity from the Simbad database.

    Parameters
    ----------
    obs : `comoving_rv.db.model.Observation`
        An observation instance.

    Returns
    -------
    rv : float
        Radial velocity (in km/s).
    rv_err : float or None
        Radial velocity error (in km/s).
    rv_qual : str or None
        Quality flag for the measurement.
    bibcode : str
        Bibliographic code for the source of the measurement.

    or

    ``None``

    """
    try:
        result = Simbad.query_object(obs.simbad_info.preferred_name)
    except Exception as e:
        logger.warning('Simbad query_object failed for "{0}" '
                       'with error: {1}'
                       .format(repr(obs), str(e)))
        return None

    if result is not None and not np.any(result['RV_VALUE'].mask):
        k, = np.where(np.logical_not(result['RV_VALUE'].mask))
        return (float(result['RV_VALUE'][k]), None,
                str(result['RVZ_QUAL'].astype(str)[k]),
                str(result['RVZ_BIBCODE'].astype(str)[k]))

    return None

def get_best_rv(obs):
    """
    Retrieve the "best" literature value of a radial velocity from a given
    source. In descending order of priority, the catalogs searched are:
    GCS, CRVAD, GCRV, Simbad. A quality of 'A' is required or an
    error < 10 km/s.

    Parameters
    ----------
    obs : `comoving_rv.db.model.Observation`
        An observation instance.

    Returns
    -------
    rv : float
        Radial velocity (in km/s).
    rv_err : float or None
        Radial velocity error (in km/s).
    rv_qual : str or None
        Quality flag for the measurement.
    bibcode : str
        Bibliographic code for the source of the measurement.

    or

    ``None``

    """

    sources = ['GCS', 'CRVAD', 'GCRV', 'SIMBAD']

    results = list()
    for i,func in enumerate([get_GCS_rv, get_CRVAD_rv, get_GCRV_rv, get_simbad_rv]):
        res = func(obs)
        if res is not None:
            err = res[1]
            if err is not None and not np.isnan(err) and err > 0 and err < 10: # it's a good measurement!
                return tuple(res) + (sources[i],)

            if res[1] is None and res[2] is None: # no error or quality flag
                continue

            results.append(tuple(res) + (sources[i],))

    if len(results) == 0:
        return None

    for res in results:
        if (res[1] is not None and res[1] < 10) or (res[2] is not None and res[2].lower() == 'a'):
            return res

    return None
