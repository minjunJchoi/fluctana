import numpy as np
from scipy import signal

def hurst(t, x, bins=30, detrend=1, fitlims=[10,1000], **kwargs):
    # axis
    bsize = int(1.0*len(t)/bins)
    ax = np.floor( 10**(np.arange(1.0, np.log10(bsize), 0.01)) )

    ers = np.zeros((bins, len(ax)))

    for b in range(bins):
        idx1 = b*bsize
        idx2 = idx1 + bsize

        sx = x[idx1:idx2]

        if detrend == 1:
            sx = signal.detrend(sx, type='linear')

        for i in range(len(ax)):
            ls = int( ax[i] ) # length of each sub-region
            ns = int( 1.0*ax[-1]/ls ) # number of sub-region

            delta = np.zeros((ls + 1, 1))
            for j in range(ns):
                jdx1 = j*ls
                jdx2 = jdx1 + ls

                ssx = sx[jdx1:jdx2]

                delta[1:,0] = np.cumsum(ssx) - np.cumsum(np.ones(ls))*sum(ssx)/ls

                r = np.max(delta) - np.min(delta)
                s = np.sqrt(np.sum(ssx**2)/ls - (np.sum(ssx)/ls)**2)

                ers[b,i] = ers[b,i] + r/s/ns

    # time lag axis
    dt = t[1] - t[0]
    tax = ax*dt*1e6 # [us]
    # E RS
    val = np.mean(ers, 0)
    std = np.std(ers, axis=0)

    ptime = tax # time lag [us]
    pdata = val
    # plt.plot(ptime, pdata, '-x')
    fidx = (fitlims[0] <= ptime) * (ptime <= fitlims[1])
    fit = np.polyfit(np.log10(ptime[fidx]), np.log10(pdata[fidx]), 1)
    fit_data = 10**(fit[1])*ptime**(fit[0])
    # plt.plot(ptime, fit_data, 'r')

    # Hurst exponent
    H = fit[0]

    return tax, val, H
