__version__ = '0.1'

try:
    __COMOVINGRV_SETUP__
except NameError:
    __COMOVINGRV_SETUP__ = False

if not __COMOVINGRV_SETUP__:
    __all__ = ['longslit']

    from . import longslit
