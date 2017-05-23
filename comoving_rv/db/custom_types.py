# Third-party
from sqlalchemy.types import TypeDecorator, REAL, Integer
from astropy.coordinates import Angle
from astropy.time import Time
import astropy.units as u

__all__ = ['SexHourAngleType', 'SexDegAngleType', 'QuantityTypeClassFactory']

class SexHourAngleType(TypeDecorator):
    """
    Custom database type to handle `~astropy.coordinates.Angle` objects.
    """

    impl = REAL

    def process_bind_param(self, value, dialect):
        if isinstance(value, Angle):
            return value.to(u.degree).value
        elif isinstance(value, str):
            return Angle(value, u.hourangle).to(u.degree).value
        else:
            return value

    def process_result_value(self, value, dialect):
        if value is not None:
            value = Angle(value, u.degree)
        return value

class SexDegAngleType(TypeDecorator):
    """
    Custom database type to handle `~astropy.coordinates.Angle` objects.
    """

    impl = REAL

    def process_bind_param(self, value, dialect):
        if isinstance(value, Angle):
            return value.to(u.degree).value
        elif isinstance(value, str):
            return Angle(value, u.degree).to(u.degree).value
        else:
            return value

    def process_result_value(self, value, dialect):
        if value is not None:
            value = Angle(value, u.degree)
        return value

class JDTimeType(TypeDecorator):
    """
    Custom database type to handle `~astropy.time.Time` objects.
    """

    impl = REAL

    def process_bind_param(self, value, dialect):
        if isinstance(value, Time):
            return value.jd
        else:
            return value

    def process_result_value(self, value, dialect):
        if value is not None:
            value = Time(value, format='jd', scale='utc')
        return value

def QuantityTypeClassFactory(unit):
    class QuantityType(TypeDecorator):
        """ Custom type to handle `~astropy.units.Quantity` objects. """

        impl = REAL

        def process_bind_param(self, value, dialect):
            if isinstance(value, u.Quantity):
                return value.to(unit).value
            else:
                return value

        def process_result_value(self, value, dialect):
            if value is not None:
                value = value * unit
            return value

    QuantityType.__name__ = '{}QuantityType'.format(unit.physical_type.title().replace(' ', ''))
    return QuantityType

class IntEnumType(TypeDecorator):

    impl = Integer

    def __init__(self, *args, **kw):
        """ Enum type with integer values. """
        self.allowed = list(args)
        super(IntEnumType, self).__init__()

    def process_bind_param(self, value, dialect):
        if value not in self.allowed:
            raise ValueError("Invalid value '{0}'".format(value))
        return value
