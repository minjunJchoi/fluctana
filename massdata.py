import numpy as np
from scipy.interpolate import griddata
from scipy.ndimage import generic_filter

# functions to massage the raw data including data from bad channels.

def fill_bad_channel(pdata, rpos, zpos, good_channels, cutoff):
    # cutoff = 0.0345 [m]
    # ## fake data
    # dist = np.sqrt((rpos - 1.8)**2 + (zpos - 0)**2)
    # pdata = 0.1*(1 - (dist/0.5)**2)
    # pdata = pdata * good_channels

    # remove NaN
    pdata[np.isnan(pdata)] = 0
    
    # recovery
    for c in range(pdata.size):
        if good_channels[c] == 0:
            dist = np.sqrt((rpos - rpos[c])**2 + (zpos - zpos[c])**2)
            dfct = np.exp(-2*(dist/cutoff)**4) * good_channels
            pdata[c] = np.sum(pdata * dfct)/np.sum(dfct)

    return pdata

def interp_pdata(pdata, rpos, zpos, istep, imethod, aspect_ratio=1.3):
    # interpolation
    # istep = 0.002 [m] radial resolution 
    # imethod = 'cubic'
    # aspect ratio is the ratio between vertical channel spacing and radial channel spacing

    ri = np.arange(np.min(rpos), np.max(rpos) + istep, istep)
    zi = np.arange(np.min(zpos), np.max(zpos) + istep*aspect_ratio, istep*aspect_ratio)
    ri, zi = np.meshgrid(ri, zi)
    pi = griddata((rpos,zpos), pdata, (ri,zi), method=imethod)

    return ri, zi, pi

def nanmedian_filter(A, size=5):
    """
    Perform 2D median filtering on array A, ignoring NaNs, using scipy.ndimage.generic_filter.

    Parameters:
    - A: 2D numpy array
    - size: scalar or tuple of 2 ints, size of the median filter window (must be odd)

    Returns:
    - M: filtered array, same shape as A
    """
    def nanmedian_function(values):
        values = values[~np.isnan(values)]  # Ignore NaNs
        if values.size > 0:
            return np.median(values)
        else:
            return np.nan

    return generic_filter(A, nanmedian_function, size=size, mode='constant', cval=np.nan)