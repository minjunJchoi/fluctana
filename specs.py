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

#   case {15} % nonlinear spectral energy transfer
#         ax1 = Fs/2*linspace(-1,1,nfft); % full -fN~fN axes
#         ax2 = Fs/2*linspace(0,1,nfft/2+1); % half 0~fN axes
#
#         AA = zeros(length(ax1), length(ax2));
#         BB = zeros(length(ax1), length(ax2));
#         XX = zeros(bins, length(ax1), length(ax2));
#         for b = 1:bins
#             Xfull = fftshift(X(b,:)).'; % top -fN~fN bottom  ref
#             Xhalf = X(b, 1:length(ax2)); % left 0~fN right         ref
#             Yfull = fftshift(Y(b,:)).'; % top -fN~fN bottom cmp
#
#             X1 = repmat(Xfull, 1, length(ax2)); % -fN~fN ref
#             X2 = repmat(Xhalf, length(ax1), 1); % 0~fN   ref
#             X3 = zeros(length(ax1), length(ax2)); % f1+f2=f3     ref
#             Y3 = zeros(length(ax1), length(ax2)); % f1+f2=f3     cmp
#             for j = 1:length(ax2)
#                 X3(1:(end-j+1),j) = Xfull(j:end);
#                 Y3(1:(end-j+1),j) = Yfull(j:end);
#             end
#
#             AA(:,:) = AA(:,:) + X1.*X2.*conj(X3)/bins;
#             BB(:,:) = BB(:,:) + X1.*X2.*conj(Y3)/bins;
#
#             XX(b,:,:) = X1.*X2;
#         end
#
# %         fk = 15000; % set frequency of f_k
#         fkaxis = 0:2000:200000;
#         gammak = zeros(size(fkaxis));
#         for fk = 1:length(fkaxis)
#
#             [l,idxs] = nset_fkaxis(fkaxis(fk), ax1, ax2);
#             I = size(idxs,1);
#
#             A = zeros(I,1);
#             B = zeros(I,1);
#             for i = 1:I
#                 A(i,1) = AA(idxs(i,1), idxs(i,2));
#                 B(i,1) = BB(idxs(i,1), idxs(i,2));
#             end
#             btbt = 0;
#             FF = zeros(I,I);
#             for b = 1:bins
#                 F = zeros(I,1);
#                 for i = 1:I
#                     F(i,1) = XX(b, idxs(i,1), idxs(i,2));
#                 end
#                 FF(:,:) = FF(:,:) + F*transpose(conj(F))/bins;
#
#                 Xhalf = X(b, 1:length(ax2)); % 0~fN          ref
#                 Yhalf = Y(b, 1:length(ax2)); % 0~fN          cmp
#                 btbt = btbt + Xhalf(l)*conj(Yhalf(l))/bins;
#             end
#
#             tau = 10e-6;
#             gammak(fk) = 1/tau*(transpose(conj(A))*(FF\A) - transpose(conj(B))*(FF\B))/(btbt - transpose(conj(A))*(FF\A));
#
#         end
#
#         figure;
#         plot(fkaxis/1000, real(gammak),'-o')
# %         figure;
# %         plot(A); hold on; plot(B)
#         figure;
#         imagesc(real(FF))
#
#
#         fprintf('done \n')
#         ax = 0;
#         val = 0;


# function [l,idxs] = nset_fkaxis(fk, ax1, ax2)
# % nonlinear spectral energy transfer f_k axis
#
# l = find(ax2>=fk, 1, 'first');
#     fprintf('fk = %g // l = %d, ax2(l) = %g \n', fk, l, ax2(l));
#
# n = ceil(l/2); % ~half index on ax2
# m = l-n+length(ax2); % ~half indx on ax1
# L = length(ax2) - n; % # of points
#
# idxs = []; % idxs = [m-(0:L)',n+(0:L)']; % ax1, ax2 index for f1+f2 = fk
# for i=0:L
#     if ax1(m-i) <= ax2(n+i)
#         idxs = [idxs; [m-i,n+i]];
#         fprintf('%d : Correct %g + %g = %g \n', i, ax1(m-i), ax2(n+i), ax1(m-i)+ax2(n+i))
#     end
# end
