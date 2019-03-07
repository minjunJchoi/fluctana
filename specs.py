import numpy as np
from scipy import signal


def fft_window(tnum, nfft, window, overlap):
    # IN : full length of time series, nfft, window name, overlap ratio
    # OUT : bins, 1 x nfft window function

    # use overlapping
    bins = int(np.fix((int(tnum/nfft) - overlap)/(1.0 - overlap)))

    # window function
    if window == 'rectwin':  # overlap = 0.5
        win = np.ones(nfft)
    elif window == 'hann':  # overlap = 0.5
        win = np.hanning(nfft)
    elif window == 'hamm':  # overlap = 0.5
        win = np.hamming(nfft)
    elif window == 'kaiser':  # overlap = 0.62
        win = np.kaiser(nfft, beta=30)
    elif window == 'HFT248D':  # overlap = 0.84
        z = 2*np.pi/nfft*np.arange(0,nfft)
        win = 1 - 1.985844164102*np.cos(z) + 1.791176438506*np.cos(2*z) - 1.282075284005*np.cos(3*z) + \
            0.667777530266*np.cos(4*z) - 0.240160796576*np.cos(5*z) + 0.056656381764*np.cos(6*z) - \
            0.008134974479*np.cos(7*z) + 0.000624544650*np.cos(8*z) - 0.000019808998*np.cos(9*z) + \
            0.000000132974*np.cos(10*z)

    return bins, win


def fftbins(x, dt, nfft, window, overlap, detrend, full):
    # IN : 1 x tnum data
    # OUT : bins x faxis fftdata
    tnum = len(x)
    bins, win = fft_window(tnum, nfft, window, overlap)
    win_factor = np.mean(win**2)  # window factors

    # make an x-axis #
    ax = np.fft.fftfreq(nfft, d=dt) # full 0~fN -fN~-f1
    if np.mod(nfft, 2) == 0:  # even nfft
        ax = np.hstack([ax[0:int(nfft/2)], -(ax[int(nfft/2)]), ax[int(nfft/2):nfft]])
    if full == 1: # full shift to -fN ~ 0 ~ fN
        ax = np.fft.fftshift(ax)
    else: # half 0~fN
        ax = ax[0:int(nfft/2+1)]

    # make fftdata
    if full == 1: # full shift to -fN ~ 0 ~ fN
        if np.mod(nfft, 2) == 0:  # even nfft
            fftdata = np.zeros((bins, nfft+1), dtype=np.complex_)
        else:  # odd nfft
            fftdata = np.zeros((bins, nfft), dtype=np.complex_)
    else: # half 0 ~ fN
        fftdata = np.zeros((bins, int(nfft/2+1)), dtype=np.complex_)

    for b in range(bins):
        idx1 = int(b*np.fix(nfft*(1 - overlap)))
        idx2 = idx1 + nfft

        sx = x[idx1:idx2]

        if detrend == 1:
            sx = signal.detrend(sx, type='linear')
        sx = signal.detrend(sx, type='constant')  # subtract mean

        sx = sx * win  # apply window function

        # get fft
        SX = np.fft.fft(sx, n=nfft)/nfft  # divide by the length
        if np.mod(nfft, 2) == 0:  # even nfft
            SX = np.hstack([SX[0:int(nfft/2)], np.conj(SX[int(nfft/2)]), SX[int(nfft/2):nfft]])
        if full == 1: # shift to -fN ~ 0 ~ fN
            SX = np.fft.fftshift(SX)
        else: # half 0 ~ fN
            SX = SX[0:int(nfft/2+1)]

        fftdata[b,:] = SX

    return ax, fftdata, win_factor


def cross_power(XX, YY, win_factor):
    # calculate cross power
    # IN : bins x faxis fftdata

    val = np.zeros(XX.shape, dtype=np.complex_)

    bins = len(XX)
    for b in range(bins):
        X = XX[b,:]
        Y = YY[b,:]

        val[b,:] = X*np.matrix.conjugate(Y) / win_factor

    # average over bins
    Pxy = np.mean(val, 0)
    Pxy = np.abs(Pxy).real

    return Pxy


def coherence(XX, YY):
    val = np.zeros(XX.shape, dtype=np.complex_)

    bins = len(XX)
    for b in range(bins):
        X = XX[b,:]
        Y = YY[b,:]

        Pxx = X * np.matrix.conjugate(X)
        Pyy = Y * np.matrix.conjugate(Y)

        val[b,:] = X*np.matrix.conjugate(Y) / np.sqrt(Pxx*Pyy)
        # saturated data gives zero Pxx!!

    # average over bins
    Gxy = np.mean(val, 0)
    Gxy = np.abs(Gxy).real

    return Gxy


def cross_phase(XX, YY):
    val = np.zeros(XX.shape, dtype=np.complex_)

    bins = len(XX)
    for b in range(bins):
        X = XX[b,:]
        Y = YY[b,:]

        Pxx = X * np.matrix.conjugate(X)
        Pyy = Y * np.matrix.conjugate(Y)

        val[b,:] = X*np.matrix.conjugate(Y)

    # average over bins
    Pxy = np.mean(val, 0)
    # result saved in val
    Axy = np.arctan2(Pxy.imag, Pxy.real).real

    return Axy


def xspec(XX, YY, win_factor):
    val = np.zeros(XX.shape)

    bins = len(XX)
    for b in range(bins):
        X = XX[b,:]
        Y = YY[b,:]

        Pxy = X*np.matrix.conjugate(Y) / win_factor

        val[b,:] = np.abs(Pxy).real

    return val
