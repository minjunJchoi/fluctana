import time

import numpy as np
from scipy import signal

import matplotlib.pyplot as plt

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


def fftbins(x, dt, nfft, window, overlap, detrend=0, full=0):
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

        if detrend == 0:
            sx = signal.detrend(sx, type='constant')  # subtract mean
        elif detrend == 1:
            sx = signal.detrend(sx, type='linear')

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


def cwt(x, dt, df, detrend=0, full=1):
    # detrend signal 
    if detrend == 0:
        x = signal.detrend(x, type='constant')  # subtract mean
    elif detrend == 1:
        x = signal.detrend(x, type='linear')

    # make a t-axis
    tnum = len(x)
    nfft = nextpow2(tnum) # power of 2
    t = np.arange(nfft)*dt

    # make a f-axis with constant df
    s0 = 2.0*dt # the smallest scale
    ax = np.arange(0.0, 1.0/(1.03*s0), df) # 1.03 for the Morlet wavelet function

    # scales
    old_settings = np.seterr(divide='ignore')
    sj = 1.0/(1.03*ax)
    np.seterr(**old_settings)
    dj = np.log2(sj/s0) / np.arange(len(sj)) # dj; necessary for reconstruction
    sj[0] = tnum*dt/2.0
    dj[0] = 0 # remove infinity point due to fmin = 0

    # Morlet wavelet function (unnormalized)
    omega0 = 6.0 # nondimensional wavelet frequency
    ts = np.sqrt(2)*sj # e-folding time for Morlet wavelet with omega0 = 6; significance level
    wf0 = lambda eta: np.pi**(-1.0/4) * np.exp(1.0j*omega0*eta) * np.exp(-1.0/2*eta**2)

    # FFT of signal
    X = np.fft.fft(x, n=nfft)/nfft

    # calculate CWT
    snum = len(sj) 
    cwtdata = np.zeros((nfft, snum), dtype=np.complex_)
    for j, s in enumerate(sj):
        # nondimensional time axis at time scale s
        eta = t/s
        # FFT of the normalized wavelet function
        W = np.fft.fft( np.conj( wf0(eta - np.mean(eta))*np.sqrt(dt/s) ) )
        # Wavelet transform at scae s for all n time
        cwtdata[:,j] = np.conj(np.fft.fftshift(np.fft.ifft(X * W) * nfft)) # phase direction correct

    # full size
    if full == 1:
        cwtdata = np.hstack([np.fliplr(np.conj(cwtdata)), cwtdata[:,1:]]) # real x only
        ax = np.hstack([-ax[::-1], ax[1:]])

    return ax, cwtdata[0:tnum,:], dj, ts


def cross_power(XX, YY, win_factor, bidx=0):
    # calculate cross power
    # IN : bins x faxis fftdata
    if type(bidx) is int:
        bidx = np.arange(len(XX))

    val = np.zeros((len(bidx),XX.shape[1]), dtype=np.complex_)

    for i,b in enumerate(bidx):
        X = XX[b,:]
        Y = YY[b,:]

        val[i,:] = X*np.matrix.conjugate(Y) / win_factor

    # average over bins
    Pxy = np.mean(val, 0)
    Pxy = np.abs(Pxy).real

    return Pxy


def coherence(XX, YY, bidx=0):
    if type(bidx) is int:
        bidx = np.arange(len(XX))

    val = np.zeros((len(bidx),XX.shape[1]), dtype=np.complex_)

    for i,b in enumerate(bidx):
        X = XX[b,:]
        Y = YY[b,:]

        Pxx = X * np.matrix.conjugate(X)
        Pyy = Y * np.matrix.conjugate(Y)

        val[i,:] = X*np.matrix.conjugate(Y) / np.sqrt(Pxx*Pyy)
        # saturated data gives zero Pxx!!

    # average over bins
    Gxy = np.mean(val, 0)
    Gxy = np.abs(Gxy).real

    return Gxy


def cross_phase(XX, YY, bidx=0):
    if type(bidx) is int:
        bidx = np.arange(len(XX))

    val = np.zeros((len(bidx),XX.shape[1]), dtype=np.complex_)

    for i,b in enumerate(bidx):
        X = XX[b,:]
        Y = YY[b,:]

        val[i,:] = X*np.matrix.conjugate(Y)

    # average over bins
    Pxy = np.mean(val, 0)
    # result saved in val
    Axy = np.arctan2(Pxy.imag, Pxy.real).real

    return Axy


def correlation(XX, YY, win_factor, bidx=0):
    if type(bidx) is int:
        bidx = np.arange(len(XX))

    nfreq = XX.shape[1]
    val = np.zeros((len(bidx),nfreq), dtype=np.complex_)

    for i,b in enumerate(bidx):
        X = XX[b,:]
        Y = YY[b,:]

        val[i,:] = np.fft.ifftshift(X*np.matrix.conjugate(Y) / win_factor)
        val[i,:] = np.fft.ifft(val[i,:], n=nfreq)*nfreq
        val[i,:] = np.fft.fftshift(val[i,:])

        val[i,:] = np.flip(val[i,:], axis=0)

    # average over bins; return real value
    Cxy = np.mean(val, 0)
    
    return Cxy.real 


def corr_coef(XX, YY, bidx=0):
    if type(bidx) is int:
        bidx = np.arange(len(XX))

    nfreq = XX.shape[1]
    val = np.zeros((len(bidx),nfreq), dtype=np.complex_)

    for i,b in enumerate(bidx):
        X = XX[b,:]
        Y = YY[b,:]

        x = np.fft.ifft(np.fft.ifftshift(X), n=nfreq)*nfreq
        Rxx = np.mean(x**2)
        y = np.fft.ifft(np.fft.ifftshift(Y), n=nfreq)*nfreq
        Ryy = np.mean(y**2)

        val[i,:] = np.fft.ifftshift(X*np.matrix.conjugate(Y))
        val[i,:] = np.fft.ifft(val[i,:], n=nfreq)*nfreq
        val[i,:] = np.fft.fftshift(val[i,:])

        val[i,:] = np.flip(val[i,:], axis=0)/np.sqrt(Rxx*Ryy)

    # average over bins; return real value
    cxy = np.mean(val, 0)
    
    return cxy.real 


def xspec(XX, YY, win_factor):
    bins = len(XX)

    val = np.zeros(XX.shape)

    for b in range(bins):
        X = XX[b,:]
        Y = YY[b,:]

        Pxy = X*np.matrix.conjugate(Y) / win_factor

        val[b,:] = np.abs(Pxy).real

    return val


def bicoherence(XX, YY, bidx=0):
    # ax1 = self.Dlist[dtwo].ax # full -fN ~ fN
    # ax2 = np.fft.ifftshift(self.Dlist[dtwo].ax) # full 0 ~ fN, -fN ~ -f1
    # ax2 = ax2[0:int(nfft/2+1)] # half 0 ~ fN
    if type(bidx) is int:
        bidx = np.arange(len(XX))
    full = len(XX[0,:]) # full length
    half = int(full/2+1) # half length

    # calculate bicoherence
    B = np.zeros((full, half), dtype=np.complex_)
    P12 = np.zeros((full, half))
    P3 = np.zeros((full, half))
    val = np.zeros((full, half))

    for b in bidx:
        X = XX[b,:] # full -fN ~ fN
        Y = YY[b,:] # full -fN ~ fN

        Xhalf = np.fft.ifftshift(X) # full 0 ~ fN, -fN ~ -f1
        Xhalf = Xhalf[0:half] # half 0 ~ fN

        X1 = np.transpose(np.tile(X, (half, 1)))
        X2 = np.tile(Xhalf, (full, 1))
        X3 = np.zeros((full, half), dtype=np.complex_)
        for j in range(half):
            if j == 0:
                X3[0:, j] = Y[j:]
            else:
                X3[0:(-j), j] = Y[j:]

        B = B + X1 * X2 * np.matrix.conjugate(X3) #  complex bin average
        P12 = P12 + (np.abs(X1 * X2).real)**2 # real average
        P3 = P3 + (np.abs(X3).real)**2 # real average

    # val = np.log10(np.abs(B)**2) # bispectrum
    val = (np.abs(B)**2) / P12 / P3 # bicoherence

    # summation over pairs
    sum_val = np.zeros(full)
    for i in range(half):
        if i == 0:
            sum_val = sum_val + val[:,i]
        else:
            sum_val[i:] = sum_val[i:] + val[:-i,i]

    N = np.array([i+1 for i in range(half)] + [half for i in range(full-half)])
    sum_val = sum_val / N # element wise division

    return val, sum_val


def ritz_nonlinear(XX, YY): # need to change range(bins) -> bidx
    # calculate
    bins = len(XX)
    full = len(XX[0,:]) # full length

    kidx = get_kidx(full)

    Aijk = np.zeros((full, full), dtype=np.complex_) # Xo1 Xo2 cXo
    cAijk = np.zeros((full, full), dtype=np.complex_) # cXo1 cXo2 Xo
    Bijk = np.zeros((full, full), dtype=np.complex_) # Yo cXo1 cXo2
    Aij = np.zeros((full, full)) # |Xo1 Xo2|^2

    Ak = np.zeros(full) # Xo cXo
    Bk = np.zeros(full, dtype=np.complex_) # Yo cXo

    for b in range(bins):
        X = XX[b,:] # full -fN ~ fN
        Y = YY[b,:] # full -fN ~ fN

        # make Xi and Xj
        Xi = np.transpose(np.tile(X, (full, 1))) # columns of (-fN ~ fN)
        Xj = np.tile(X, (full, 1)) # rows of (-fN ~ fN)

        # make Xk and Yk
        Xk = np.zeros((full, full), dtype=np.complex_)
        Yk = np.zeros((full, full), dtype=np.complex_)
        for k in range(full):
            idx = kidx[k]
            for n, ij in enumerate(idx):
                Xk[ij] = X[k]
                Yk[ij] = Y[k]

        # do ensemble average
        Aijk = Aijk + Xi * Xj * np.matrix.conjugate(Xk) / bins

        cAijk = cAijk + np.matrix.conjugate(Xi) * np.matrix.conjugate(Xj) * Xk / bins

        Bijk = Bijk + np.matrix.conjugate(Xi) * np.matrix.conjugate(Xj) * Yk / bins

        Aij = Aij + (np.abs(Xi * Xj).real)**2 / bins

        Ak = Ak + (np.abs(X).real)**2 / bins

        Bk = Bk + Y * np.matrix.conjugate(X) / bins

    # Linear transfer function ~ growth rate
    Lk = np.zeros(full, dtype=np.complex_)

    bsum = np.zeros(full, dtype=np.complex_)
    asum = np.zeros(full)
    for k in range(full):
        idx = kidx[k]
        for n, ij in enumerate(idx):
            bsum[k] = bsum[k] + Aijk[ij] * Bijk[ij] / Aij[ij]
            asum[k] = asum[k] + (np.abs(Aijk[ij]).real)**2 / Aij[ij]

    Lk = (Bk - bsum) / (Ak - asum)

    # Quadratic transfer function ~ nonlinear energy transfer rate
    Lkk = np.zeros((full, full), dtype=np.complex_)
    for k in range(full):
        idx = kidx[k]
        for n, ij in enumerate(idx):
            Lkk[ij] = Lk[k]

    Qijk = (Bijk - Lkk * cAijk) / Aij

    return Lk, Qijk, Bk, Aijk


def wit_nonlinear(XX, YY, bidx=0):
    # calculate
    if type(bidx) is int:
        bidx = np.arange(len(XX))
    bins = len(bidx)
    full = len(XX[0,:]) # full length

    kidx = get_kidx(full)

    Lk = np.zeros(full, dtype=np.complex_) # Linear
    Qijk = np.zeros((full, full), dtype=np.complex_) # Quadratic

    print('For stable calculations, bins ({0}) >> full/2 ({1})'.format(bins, full/2))
    for k in range(full):
        idx = kidx[k]

        # construct equations for each k
        U = np.zeros((bins, len(idx)+1), dtype=np.complex_)  # N (number of ensembles) x P (number of pairs + 1)
        V = np.zeros(bins, dtype=np.complex_) # N x 1

        for i,b in enumerate(bidx):
            U[i,0] = XX[b,k]
            for n, ij in enumerate(idx):
                U[i,n+1] = XX[b, ij[0]]*XX[b, ij[1]]

            V[i] = YY[b,k]

        # solution for each k
        H = np.matmul(np.linalg.pinv(U), V)

        Lk[k] = H[0]
        for n, ij in enumerate(idx):
            Qijk[ij] = H[n+1]

    # calculate others for the rates
    Aijk = np.zeros((full, full), dtype=np.complex_) # Xo1 Xo2 cXo
    Bk = np.zeros(full, dtype=np.complex_) # Yo cXo

    # print('DO NOT CALCULATE RATES')
    for b in bidx:
        X = XX[b,:] # full -fN ~ fN
        Y = YY[b,:] # full -fN ~ fN

        # make Xi and Xj
        Xi = np.transpose(np.tile(X, (full, 1))) # columns of (-fN ~ fN)
        Xj = np.tile(X, (full, 1)) # rows of (-fN ~ fN)

        # make Xk and Yk
        Xk = np.zeros((full, full), dtype=np.complex_)
        for k in range(full):
            idx = kidx[k]
            for n, ij in enumerate(idx):
                Xk[ij] = X[k]

        # do ensemble average
        Aijk = Aijk + Xi * Xj * np.matrix.conjugate(Xk) / bins
        Bk = Bk + Y * np.matrix.conjugate(X) / bins

    return Lk, Qijk, Bk, Aijk


def nonlinear_rates(Lk, Qijk, Bk, Aijk, dt):
    ## Linear growth rate and nonlinear energy transfer rates
    # dt = vd / dz
    full = len(Lk)

    kidx = get_kidx(full)

    # Cross phase related terms
    Ek = (Bk / np.abs(Bk))**(-1.0) # Exp[-i(Tk)]
    Tk = np.arctan2(Bk.imag, Bk.real).real

    Ekk = np.zeros((full, full), dtype=np.complex_)
    for k in range(full):
        idx = kidx[k]
        for n, ij in enumerate(idx):
            Ekk[ij] = Ek[k]

    # Linear kernel
    # Gk = (Lk * Ek - 1.0 + 1.0j*Tk) / dz
    # Gk = ( Lk * Exp[-i(dth)] - 1 + i(dth) ) /  dz

    # Linear growth rate
    # gk = vd * Gk.real
    gk = 1.0/dt * (Lk * Ek - 1.0).real

    # Quadratic kernel
    # Mijk = Qijk * Ekk / dz
    # Mijk = Qijk * Exp[-i(dth)] / dz

    # Nonlinear energy transfer rate
    # Tijk = 1.0/2.0 * vd * (Mijk * Aijk).real
    Tijk = 1.0/2.0 * 1.0/dt * (Qijk * Ekk * Aijk).real

    # summed Tijk
    sum_Tijk = np.zeros(full)
    for k in range(full):
        idx = kidx[k]
        for n, ij in enumerate(idx):
            sum_Tijk[k] += Tijk[ij]
            # sum_Tijk[k] += Tijk[ij] / len(idx) # divide by number of pairs?

    return gk, Tijk, sum_Tijk


def nonlinear_ratesJS(Lk, Aijk, Qijk, XX, delta):
    ## Linear growth rate and nonlinear energy transfer rates from JS Kim PoP 96
    # delta = dt or dz
    full = len(Lk)

    kidx = get_kidx(full)

    gk = (np.abs(Lk)**2 - 1)/delta # JSKim 96

    sum_Tijk = np.zeros(full, dtype=np.complex_)
    for k in range(full):
        idx = kidx[k]

        for n, ij in enumerate(idx):
            # XXX = np.mean( XX[:,ij[0]] * XX[:,ij[1]] * np.conjugate(XX[:,k]) ) # same with Aijk[ij]
            sum_Tijk[k] += 2.0*(np.conjugate(Lk[k]) * Qijk[ij] * Aijk[ij] / delta).real

        # # fourth order terms
        # for n, ij in enumerate(idx):
        #     for m, lm in enumerate(idx):
        #         XXXX = np.mean( XX[:,ij[0]] * XX[:,ij[1]] * np.conjugate(XX[:,lm[0]]) * np.conjugate(XX[:,lm[1]]) )
        #         sum_Tijk[k] += Qijk[ij] * np.conjugate(Qijk[lm]) * XXXX / delta

    Tijk = Qijk

    return gk, Tijk, sum_Tijk


def nonlinear_test(ax, XX):
    bins = len(XX)
    full = len(XX[0,:]) # full length

    pN = ax[-1]

    kidx = get_kidx(full)

    # Lk = np.zeros(full, dtype=np.complex_)
    Lk = 1.0 - 0.4*ax**2/pN**2 + 0.8j*ax/pN

    Qijk = np.zeros((full, full), dtype=np.complex_)
    for k in range(full):
        idx = kidx[k]
        for n, ij in enumerate(idx):
            pi = ax[ij[0]]
            pj = ax[ij[1]]
            pk = ax[k]

            Qijk[ij] = 1.0j/(5.0*pN**4)*pi*pj*(pj**2 - pi**2)/(1.0 + pk**2/pN**2)

    # modeled YY from Lk and Qijk
    YY = np.zeros(XX.shape, dtype=np.complex_)
    for b in range(bins):
        for k in range(full):
            YY[b, k] = Lk[k]*XX[b, k]

            idx = kidx[k]
            for n, ij in enumerate(idx):
                YY[b, k] += Qijk[ij]*XX[b, ij[0]]*XX[b, ij[1]]

    return YY, Lk, Qijk


def get_kidx(full):
    half = int(full/2 + 1)
    kidx = []
    for k in range(full):
        idx = []

        if k <= half - 1:
            i = 0
            j = half - 1 + k
        else:
            i = k - half + 1
            j = full - 1

        while j >= i:
            idx.append((i,j))
            i += 1
            j -= 1

        kidx.append(idx)

    return kidx


def nextpow2(i):
    n = 1
    while n < i: n *= 2
    return n
