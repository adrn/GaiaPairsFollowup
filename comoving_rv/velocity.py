# Third-party
import astropy.coordinates as coord
import astropy.units as u

__all__ = ['bary_vel_corr']

kitt_peak = coord.EarthLocation.of_site('KPNO')

def bary_vel_corr(time, skycoord, location=None):
    """
    Barycentric velocity correction.

    Uses the ephemeris set with  ``astropy.coordinates.solar_system_ephemeris.set``
    for corrections. For more information see `~astropy.coordinates.solar_system_ephemeris`.

    Parameters
    ----------
    time : `~astropy.time.Time`
        The time of observation.
    skycoord: `~astropy.coordinates.SkyCoord`
        The sky location to calculate the correction for.
    location: `~astropy.coordinates.EarthLocation`, optional
        The location of the observatory to calculate the correction for.
        If no location is given, the ``location`` attribute of the Time
        object is used

    Returns
    -------
    vel_corr : `~astropy.units.Quantity`
        The velocity correction to convert to Barycentric velocities. Should be
        *added* to the original velocity.
    """

    if location is None:
        if time.location is None:
            raise ValueError('An EarthLocation needs to be set or passed '
                             'in to calculate bary- or heliocentric '
                             'corrections')
        location = time.location

    # ensure sky location is ICRS compatible
    if not skycoord.is_transformable_to(coord.ICRS):
        raise ValueError("Given skycoord is not transformable to the ICRS")

    # ICRS position and velocity of Earth's geocenter
    ep, ev = coord.solar_system.get_body_barycentric_posvel('earth', time)

    # GCRS position and velocity of observatory
    op, ov = location.get_gcrs_posvel(time)

    # ICRS and GCRS are axes-aligned. Can add the velocities
    velocity = ev + ov

    # get unit ICRS vector in direction of SkyCoord
    sc_cartesian = skycoord.icrs.represent_as(coord.UnitSphericalRepresentation)\
                                .represent_as(coord.CartesianRepresentation)
    return sc_cartesian.dot(velocity).to(u.km/u.s)
