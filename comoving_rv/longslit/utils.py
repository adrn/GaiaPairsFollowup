# Third-party
import numpy as np

__all__ = ['extract_region', 'argmedian']

def extract_region(x, center, width, arrs=(), sort=True, clean_nan=True):
    """
    Extract only a region of size ``width`` centered on ``center`` in ``x``
    units from the ``x`` array, and all other index-aligned arrays ``arrs``.

    Parameters
    ----------
    x : array_like
        The array used to define the region mask.
    center : numeric
        The center of the region mask in ``x`` units.
    width : numeric
        The width of the region mask in ``x`` units.
    arrs : iterable (optional)
        Any other arrays to mask in the same way as ``x``.
    sort : bool (optional)
        Sort the arrays on return.
    clean_nan : bool (optional)
        Mask out nan values.

    Returns
    -------
    x_new : `numpy.ndarray`
        Masked and sorted ``x`` array.
    arrs_new : tuple of `numpy.ndarray`
        Masked and sorted ``arrs`` arrays.
    """
    x = np.array(x)
    mask = np.abs(x - center) <= width/2.

    if clean_nan:
        mask &= np.isfinite(x)
        for arr in arrs:
            mask &= np.isfinite(arr)

    x = x[mask]
    arrs = [np.array(arr)[mask] for arr in arrs]

    if sort:
        sort_ = x.argsort()
        x = x[sort_]
        arrs = [arr[sort_] for arr in arrs]

    return x, arrs

def argmedian(arr):
    return np.argsort(arr)[len(arr)//2]
