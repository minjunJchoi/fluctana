#!/usr/bin/env python2.7

# Author : Minjun J. Choi (mjchoi@nfri.re.kr)
#
# Description : Filter data
#
# Acknowledgement : TomRoelandts.com (FirFilter), C. Bowman (ThresholdFftFilter)
#
# Last updated
#  2018.12.15 : version 0.10;

import numpy as np
import h5py
import pywt

import matplotlib.pyplot as plt

import stats as st

def nextpow2(i):
    n = 1
    while n < i: n *= 2
    return n

class FirFilter(object):
    def __init__(self, name, fs, fL, fH, b=0.08):
        self.name = name
        self.fs = fs
        self.fL = fL
        self.fH = fH
        self.b = b

        N = int(np.ceil((4 / b)))
        if not N % 2: N += 1
        self.N = N

        self.filt_coef = np.ones(N)
        if name == 'FIR_pass' and fL == 0:
            self.filt_coef = self.fir_lowpass(float(fH/fs), N)
        elif name == 'FIR_pass' and fH == 0:
            self.filt_coef = self.fir_lowpass(float(fL/fs), N)
        elif name == 'FIR_block':
            self.filt_coef = self.fir_bandblock(fL/fs, fH/fs, N)

    def apply(self, x):
        xlp = np.convolve(x, self.filt_coef)
        if self.name == 'FIR_pass' and self.fH == 0: # high pass filter
            x = x - xlp[int(self.N/2):int(self.N/2 + len(x))] # delay correction
        else:
            x = xlp[int(self.N/2):int(self.N/2 + len(x))] # delay correction

        return x

    def fir_lowpass(self, fc, N):
        n = np.arange(N)

        # Compute sinc filter.
        h = np.sinc(2 * fc * (n - (N - 1) / 2.))
        # Compute Blackman window.
        w = np.blackman(N)
        # Multiply sinc filter with window.
        h = h * w
        # Normalize to get unity gain.
        h = h / np.sum(h)

        return h

    def fir_bandblock(self, fL, fH, N):
        n = np.arange(N)

        # Compute a low-pass filter with cutoff frequency fL.
        hlpf = np.sinc(2 * fL * (n - (N - 1) / 2.))
        hlpf *= np.blackman(N)
        hlpf /= np.sum(hlpf)
        # Compute a high-pass filter with cutoff frequency fH.
        hhpf = np.sinc(2 * fH * (n - (N - 1) / 2.))
        hhpf *= np.blackman(N)
        hhpf /= np.sum(hhpf)
        hhpf = -hhpf
        hhpf[int((N - 1) / 2)] += 1
        # Add both filters.
        h = hlpf + hhpf

        return h


class FftFilter(object):
    def __init__(self, name, fs, fL, fH):
        self.name = name
        self.fs = fs
        self.fL = fL
        self.fH = fH

    def apply(self, x):
        nfft = len(x)
        dt = 1.0/self.fs
        
        # frequency axis
        ax = np.fft.fftfreq(nfft, d=dt)

        # fft
        X = np.fft.fft(x, n=nfft)

        # apply filter (brick)
        if self.name == 'FFT_pass':
            self.filt_coef = (self.fL <= np.abs(ax)) & (np.abs(ax) <= self.fH)
        elif self.name == 'FFT_block':
            self.filt_coef = ~((self.fL <= np.abs(ax)) & (np.abs(ax) <= self.fH))
        X = X * self.filt_coef

        # ifft
        x = np.fft.ifft(X, n=nfft)
        
        return x


class ThresholdFftFilter(object):
    def __init__(self, fs, fL, fH, b=3.0, nbins=100):
        self.fs = fs
        self.fL = fL # white noise range
        self.fH = fH # white noise range
        self.b = b # scale factor 
        self.nbins = nbins

    def apply(self, x):
        nfft = len(x)
        dt = 1.0/self.fs
        
        # frequency axis
        ax = np.fft.fftfreq(nfft, d=dt)

        # fft
        X = np.fft.fft(x, n=nfft)

        # get white noise 
        nidx = (self.fL <= np.abs(ax)) & (np.abs(ax) <= self.fH)
        white_noise = np.abs(X[nidx])

        if np.isnan(white_noise).any():
            return np.zeros(x.shape)

        # get noise characteristics 
        counts, bins = np.histogram(white_noise, bins = self.nbins)
        pdf = counts/np.sum(counts)
        bins_center = (bins[1:] + bins[:-1])/2
        mu = np.sum(bins_center*pdf) 
        threshold = mu*np.sqrt(2/np.pi)

        # determine filter coefficient 
        self.filt_coef = np.zeros(X.shape)
        sidx = np.abs(X) > self.b * threshold 
        self.filt_coef[sidx] = 1 - self.b*threshold / (2*np.abs(X[sidx])) * (np.tanh(12*(self.b*threshold/np.abs(X[sidx]) - 1/2)) + 1)

        # apply filter
        X = X * self.filt_coef

        # ifft
        x = np.fft.ifft(X, n=nfft)
        
        return x


class SvdFilter(object):
    def __init__(self, cutoff=0.9):
        self.cutoff = cutoff

    def apply(self, data, good_channels, verbose=0):
        if np.sum(good_channels) == 0:
            good_channels = self.check_data(data)
        cnum, tnum = data.shape
        
        X = np.zeros((tnum, int(np.sum(good_channels))))
        xm = np.zeros(int(np.sum(good_channels)))

        cnt = 0
        for c in range(cnum):
            if good_channels[c] == 1:
                X[:,cnt] = data[c,:]/np.sqrt(tnum)
                xm[cnt] = np.mean(X[:,cnt])
                X[:,cnt] = X[:,cnt] - xm[cnt]
                cnt += 1

        # Do SVD
        U, s, Vt = np.linalg.svd(X, full_matrices=False)

        # energy of mode and the entropy
        sv = s**2
        E = np.sum(sv)
        pi = sv / E
        nsent = st.ns_entropy(pi)
        print('The normalized Shannon entropy of sv is {:g}'.format(nsent))

        if verbose == 1:
            ax1 = plt.subplot(211)
            ax1.plot(pi)
            ax2 = plt.subplot(212)
            ax2.plot(np.cumsum(sv)/np.sum(sv))
            ax2.axhline(y=self.cutoff, color='r')
            ax1.set_ylabel('SV power')
            ax2.set_ylabel('Cumulated sum')
            ax2.set_xlabel('Mode number')
            plt.show()

        # filtering
        s[np.cumsum(sv)/np.sum(sv) >= self.cutoff] = 0

        # reconstruct
        S = np.diag(s)
        reX = np.dot(U, np.dot(S, Vt))
        # print('reconstructed {:0}'.format(np.allclose(X, reX)))

        cnt = 0
        for c in range(cnum):
            if good_channels[c] == 1:
                data[c,:] = (reX[:,cnt] + xm[cnt])*np.sqrt(tnum)
                cnt += 1

        return data

    def check_data(self, data):
        cnum, tnum = data.shape
        good_channels = np.ones(cnum)

        for c in range(cnum):
            if np.std(data[c,:]) == 0 or ~np.isfinite(np.sum(data[c,:])): # saturated or bad number
                good_channels[c] = 0

        return good_channels


class Wave2dFilter(object):
    def __init__(self, wavename='coif3', alpha=1.0, lim=5):
        self.wavename = wavename
        self.alpha = alpha
        self.lim = lim

    def apply(self, data, verbose=0):
        wavename = self.wavename
        alpha = self.alpha
        lim = self.lim

        dim = min(data.shape)

        if np.abs(dim - nextpow2(dim)) < np.abs(dim - (nextpow2(dim)/2)):
            level = int(np.log2(nextpow2(dim)))
        else:
            level = int(np.log2(nextpow2(dim)/2))

        # wavelet transform
        coeffs = pywt.wavedec2(data, wavename, level=level)

        # set an intial threshold (noise level)
        coef1d = coeffs[0].reshape((coeffs[0].size,))
        for i in range(1,len(coeffs)):
            for j in range(len(coeffs[i])):
                coef1d = np.hstack((coef1d, coeffs[i][j].reshape((coeffs[i][j].size,))))  # ~ size x size
        tlev = np.sum(coef1d**2) 
        e = alpha*np.sqrt(np.var(coef1d)*np.log(coef1d.size))

        # find the noise level via iteration
        old_e = 0.1*e
        new_e = e
        while np.abs(new_e - old_e)/old_e*100 > lim:
            old_e = new_e
            idx = np.abs(coef1d) < old_e
            coef1d = coef1d[idx]
            new_e = alpha*np.sqrt(np.var(coef1d)*np.log(coef1d.size))
        e = old_e

        # get norms of the coherent and incoherent part 
        ilev = np.sum(coef1d**2)
        clev = tlev - ilev
        clev = np.sqrt(clev) 
        ilev = np.sqrt(ilev) 

        # obtain the coherent part    
        idx = np.abs(coeffs[0]) < e
        coeffs[0][idx] = 0
        for i in range(1,len(coeffs)):
            for j in range(len(coeffs[i])):
                idx = np.abs(coeffs[i][j]) < e
                coeffs[i][j][idx] = 0
        coh_data = pywt.waverec2(coeffs, wavename) 

        return coh_data, clev, ilev
