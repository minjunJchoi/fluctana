import numpy as np
from scipy.interpolate import griddata

def fill_bad_channel(pdata, rpos, zpos, good_channels, cutoff):
    # cutoff = 0.003 [m]
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

def interp_pdata(pdata, rpos, zpos, istep, imethod):
    # interpolation
    # istep = 0.002
    # imethod = 'cubic'

    ri = np.arange(np.min(rpos), np.max(rpos), istep)
    zi = np.arange(np.min(zpos), np.max(zpos), istep)
    ri, zi = np.meshgrid(ri, zi)
    pi = griddata((rpos,zpos),pdata,(ri,zi),method=imethod)

    return ri, zi, pi