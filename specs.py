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
    bins = len(XX)

    val = np.zeros(XX.shape, dtype=np.complex_)

    for b in range(bins):
        X = XX[b,:]
        Y = YY[b,:]

        val[b,:] = X*np.matrix.conjugate(Y) / win_factor

    # average over bins
    Pxy = np.mean(val, 0)
    Pxy = np.abs(Pxy).real

    return Pxy


def coherence(XX, YY):
    bins = len(XX)

    val = np.zeros(XX.shape, dtype=np.complex_)

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
    bins = len(XX)

    val = np.zeros(XX.shape, dtype=np.complex_)

    for b in range(bins):
        X = XX[b,:]
        Y = YY[b,:]

        val[b,:] = X*np.matrix.conjugate(Y)

    # average over bins
    Pxy = np.mean(val, 0)
    # result saved in val
    Axy = np.arctan2(Pxy.imag, Pxy.real).real

    return Axy


def xspec(XX, YY, win_factor):
    bins = len(XX)

    val = np.zeros(XX.shape)

    for b in range(bins):
        X = XX[b,:]
        Y = YY[b,:]

        Pxy = X*np.matrix.conjugate(Y) / win_factor

        val[b,:] = np.abs(Pxy).real

    return val


def bicoherence(XX, YY):
    # ax1 = self.Dlist[dtwo].ax # full -fN ~ fN
    # ax2 = np.fft.ifftshift(self.Dlist[dtwo].ax) # full 0 ~ fN, -fN ~ -f1
    # ax2 = ax2[0:int(nfft/2+1)] # half 0 ~ fN

    bins = len(XX)
    full = len(XX[0,:]) # full length
    half = int(full/2+1) # half length

    # calculate bicoherence
    B = np.zeros((full, half), dtype=np.complex_)
    P12 = np.zeros((full, half))
    P3 = np.zeros((full, half))
    val = np.zeros((full, half))

    for b in range(bins):
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

        B = B + X1 * X2 * np.matrix.conjugate(X3) / bins #  complex bin average
        P12 = P12 + (np.abs(X1 * X2).real)**2 / bins # real average
        P3 = P3 + (np.abs(X3).real)**2 / bins # real average

    # val = np.log10(np.abs(B)**2) # bispectrum
    val = (np.abs(B)**2) / P12 / P3 # bicoherence

    return val


def ritz_nonlinear(XX, YY):
    # calculate
    bins = len(XX)
    full = len(XX[0,:]) # full length
    half = int(full/2+1) # half length

    kidx = get_kidx(full)

    Aijk = np.zeros((full, full), dtype=np.complex_) # Xo1 Xo2 cXo
    Bijk = np.zeros((full, full), dtype=np.complex_) # Yo cXo1 cXo2
    Aij = np.zeros((full, full)) # |Xo1 Xo2|^2

    Ak = np.zeros(full) # Xo cXo
    Bk = np.zeros(full, dtype=np.complex_) # Yo cXo

    for b in range(bins):  ####################### fix
        X = XX[b,:] # full -fN ~ fN
        Y = YY[b,:] # full -fN ~ fN

        # ##### test
        # X = ax1 # full -fN ~ fN
        # print('test bin {:d}'.format(b))

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
                # if int(Xk[ij].real) == int(Xi[ij].real + Xj[ij].real): # for test
                #     pass
                # else:
                #     print('uncorrect')
                #     print('k {:g}, Xk {:g}, Xi + Xj {:g}, Xi {:g}, Xj {:g}'.format(k, Xk[ij], Xi[ij] + Xj[ij], Xi[ij], Xj[ij]))
        # plt.imshow(Xk)
        # plt.show()

        # do ensemble average
        Aijk = Aijk + Xi * Xj * np.matrix.conjugate(Xk) / bins

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

    Qijk = (Bijk - Lkk * Aijk) / Aij

    # Cross phase related terms
    Ek = Bk / np.abs(Bk) # Exp[-i(dth)]
    Tk = np.arctan2(Bk.imag, Bk.real).real

    Ekk = np.zeros((full, full), dtype=np.complex_)
    for k in range(full):
        idx = kidx[k]
        for n, ij in enumerate(idx):
            Ekk[ij] = Ek[k]

    ############################## time difference between two measurements (Need to shift signals)
    # dt = 6.582e-6*2/2 # [s]
    dt = 9.607e-6*3/2 # [s]

    ############################### drift velocity
    # vd = 6138.0 # [m/s]
    vd = 4202.0 # [m/s]
    # dt = dx / vd

    # Linear kernel
    Gk = (Lk * Ek - 1.0 + 1.0j*Tk) / dt
    # Gk = ( Lk * Exp[-i(dth)] - 1 + i(dth) ) /  dt

    # Linear growth rate
    gk = vd * Gk.real

    # Quadratic kernel
    Mijk = Qijk * Ekk / dt
    # Mijk = Qijk * Exp[-i(dth)] / dt

    # Nonlinear energy transfer rate
    Tijk = 1.0/2.0 * vd * (Mijk * Aijk).real

    # summed Tijk
    sum_Tijk = np.zeros(full)
    for k in range(full):
        idx = kidx[k]
        for n, ij in enumerate(idx):
            # sum_Tijk[k] += Tijk[ij] ############# divide by number of pairs?
            sum_Tijk[k] += Tijk[ij] / len(idx)

    return gk, Tijk, sum_Tijk


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
