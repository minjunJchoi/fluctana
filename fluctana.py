# Author : Minjun J. Choi (mjchoi@nfri.re.kr)
#
# Description : This code calculates cross power, coherence, cross phase, etc with fusion plasma diagnostics data
#
# Acknowledgement : Dr. S. Zoletnik and Prof. Y.-c. Ghim
#

from scipy import signal
import math
import itertools

import matplotlib.pyplot as plt
from mpl_toolkits.axes_grid1 import make_axes_locatable

import pickle

from kstarecei import *
from kstarmir import *
from kstarmds import *
#from diiiddata import *  # needs pidly

import specs as sp
import stats as st
import massdata as ms
import filtdata as ft

CM = plt.cm.get_cmap('RdYlBu_r')
# CM = plt.cm.get_cmap('spectral')
# CM = plt.cm.get_cmap('YlGn')
# CM = plt.cm.get_cmap('jet')
# CM = plt.cm.get_cmap('hot')


class FluctData(object):
    def __init__(self, shot, clist, time, data, rpos, zpos, apos):
        self.shot = shot
        self.clist = clist
        self.time = time # 1xN [time]
        self.data = data # MxN [channel,time]
        self.rpos = rpos # 1XM [channel]
        self.zpos = zpos # 1XM [channel]
        self.apos = apos # 1XM [channel]

    def get_data(self, trange, norm=1, atrange=[1.0, 1.01], res=0, verbose=1):
        # trim, normalize data
        self.trange = trange

        if norm == 0:
            if verbose == 1: print('Data is not normalized')
        elif norm == 1:
            if verbose == 1: print('Data is normalized by trange average')
        elif norm == 2:
            if verbose == 1: print('Data is normalized by atrange average')

        # trim time
        time, idx1, idx2 = self.time_base(trange)

        if norm == 2:
            _, aidx1, aidx2 = self.time_base(atrange)

        # time series length
        tnum = idx2 - idx1
        # number of channels
        cnum = len(self.clist)

        raw_data = self.data
        data = np.zeros((cnum, tnum))
        for i in range(cnum):
            v = raw_data[i,idx1:idx2]

            if norm == 1:
                v = v/np.mean(v) - 1
            elif norm == 2:
                v = v/np.mean(raw_data[i,aidx1:aidx2]) - 1

            data[i,:] = v

        self.data = data
        self.time = time

        return time, data

    def time_base(self, trange):
        time = self.time

        idx = np.where((time >= trange[0])*(time <= trange[1]))
        idx1 = int(idx[0][0])
        idx2 = int(idx[0][-1]+2)

        return time[idx1:idx2], idx1, idx2


class FluctAna(object):
    def __init__(self):
        self.Dlist = []

    def add_data(self, D, trange, norm=1, atrange=[1.0, 1.01], res=0, verbose=1):

        D.get_data(trange, norm=norm, atrange=atrange, res=res, verbose=verbose)
        self.Dlist.append(D)

    def del_data(self, dnum):
        del self.Dlist[dnum]

    def list_data(self):
        for i in range(len(self.Dlist)):
            print('---- DATA SET # {:d} for [{:0.6f}, {:0.6f}] s ----'.format(i, self.Dlist[i].trange[0], self.Dlist[i].trange[1]))
            cstr = ''
            for j, c in enumerate(self.Dlist[i].clist):
                cstr += '[{:03d}:{:s}]'.format(j, c)
                if np.mod(j+1, 4) == 0 or j == len(self.Dlist[i].clist)-1:
                    print(cstr)
                    cstr = ''
            # print '     # %d size : %s' % (i, self.Dlist[i].data.shape)

    def add_channel(self, dnum, clist):  # re-do fftbins after add channels
        old_clist = self.Dlist[dnum].clist

        # add channels (no duplicates)
        clist = expand_clist(clist)
        clist = [c for c in clist if c not in self.Dlist[dnum].clist]

        # add data
        time, data = self.Dlist[dnum].get_data(self.Dlist[dnum].trange, norm=norm, atrange=atrange, res=res)
        self.Dlist[dnum].data = np.concatenate((self.Dlist[dnum].data, data), axis=0)

        # update clist
        self.Dlist[dnum].clist = old_clist + clist

        self.list_data()

    def del_channel(self, dnum, clist):
        clist = expand_clist(clist)

        for i in range(len(clist)):
            # find index to be deleted
            del_idx = [j for j, s in enumerate(self.Dlist[dnum].clist) if clist[i] in s]

            # delete data
            self.Dlist[dnum].data = np.delete(self.Dlist[dnum].data, del_idx, 0)

            # delete fftdata if it has
            if hasattr(self.Dlist[dnum], 'fftdata'):
                self.Dlist[dnum].fftdata = np.delete(self.Dlist[dnum].fftdata, del_idx, 0)

            # update clist
            self.Dlist[dnum].clist = [self.Dlist[dnum].clist[k] for k in range(len(self.Dlist[dnum].clist)) if k not in del_idx]

        self.list_data()

############################# down sampling #############################

    def downsample(self, dnum, q, verbose=0):
        # down sampling after anti-aliasing filter
        D = self.Dlist[dnum]

        cnum = len(D.data)

        # plot dimension
        if cnum < 4:
            row = cnum
        else:
            row = 4
        col = math.ceil(cnum/row)

        # new time axis
        raw_time = np.copy(D.time)
        tnum = len(D.time)
        idx = np.arange(0, tnum, q)
        D.time = D.time[idx]

        # down sample
        raw_data = np.copy(D.data)
        D.data = np.empty((cnum, len(D.time)))
        for c in range(cnum):
            D.data[c,:] = signal.decimate(raw_data[c,:], q)

            if verbose == 1:
                # plot info
                pshot = D.shot
                pname = D.clist[c]

                # set axes
                if c == 0:
                    plt.subplots_adjust(hspace = 0.5, wspace = 0.3)
                    axes1 = plt.subplot(row,col,c+1)
                    axprops = dict(sharex = axes1, sharey = axes1)
                elif c > 0:
                    plt.subplot(row,col,c+1, **axprops)

                plt.plot(raw_time, raw_data[c,:])
                plt.plot(D.time, D.data[c,:])

                plt.title('#{:d}, {:s}'.format(pshot, pname), fontsize=10)

        plt.show()

        D.fs = round(1/(D.time[1] - D.time[0]))
        print('down sample with q={:d}, fs={:g}'.format(q, D.fs))

############################# data filtering functions #########################

    def filt(self, dnum, name, fL, fH, b=0.08, verbose=0):
        D = self.Dlist[dnum]

        # select filter except svd
        if name[0:3] == 'FIR':
            filter = ft.FirFilter(name, D.fs, fL, fH, b)

        for c in range(len(D.clist)):
            x = np.copy(D.data[c,:])
            D.data[c,:] = filter.apply(x)

        print('dnum {:d} filter {:s} with fL {:g} fH {:g} b {:g}'.format(dnum, name, fL, fH, b))

    def svd_filt(self, dnum, cutoff=0.9, verbose=0):
        D = self.Dlist[dnum]

        svd = ft.SvdFilter(cutoff = cutoff)

        if hasattr(D, 'good_channels'):
            D.data = svd.apply(D.data, D.good_channels, verbose=verbose)
        else:
            D.data = svd.apply(D.data, np.zeros(len(D.clist)), verbose=verbose)

        print('dnum {:d} svd filter with cutoff {:g}'.format(dnum, cutoff))

############################# spectral methods #############################

    def fftbins(self, nfft, window, overlap, detrend, full=0):
        # IN : self, data set number, nfft, window name, detrend or not
        # OUT : bins x N FFT of time series data; frequency axis
        # self.list_data()

        for d, D in enumerate(self.Dlist):
            # get bins and window function
            tnum = len(D.time)
            bins, win = sp.fft_window(tnum, nfft, window, overlap)
            dt = D.time[1] - D.time[0]  # time step

            D.window = window
            D.overlap = overlap
            D.detrend = detrend
            D.bins = bins

            # make fft data
            cnum = len(D.data)
            if full == 1: # full shift to -fN ~ 0 ~ fN
                if np.mod(nfft, 2) == 0:  # even nfft
                    D.spdata = np.zeros((cnum, bins, nfft+1), dtype=np.complex_)
                else:  # odd nfft
                    D.spdata = np.zeros((cnum, bins, nfft), dtype=np.complex_)
            else: # half 0 ~ fN
                D.spdata = np.zeros((cnum, bins, int(nfft/2+1)), dtype=np.complex_)

            for c in range(cnum):
                x = D.data[c,:]
                D.ax, D.spdata[c,:,:], D.win_factor = sp.fftbins(x, dt, nfft, window, overlap, detrend, full)

            # update attributes
            if np.mod(nfft, 2) == 0:
                D.nfft = nfft + 1
            else:
                D.nfft = nfft

            print('dnum {:d} fftbins {:d} with {:s} size {:d} overlap {:g} detrend {:d} full {:d}'.format(d, bins, window, nfft, overlap, detrend, full))

    def cwt(self, df, full=0): ## problem in recovering the signal
        for d, D in enumerate(self.Dlist):
            # make a t-axis
            dt = D.time[1] - D.time[0]  # time step
            tnum = len(D.time)
            nfft = nextpow2(tnum) # power of 2
            t = np.arange(nfft)*dt

            # make a f-axis with constant df
            s0 = 2.0*dt # the smallest scale
            ax = np.arange(0.0, 1.0/(1.03*s0), df)

            # scales
            old_settings = np.seterr(divide='ignore')
            sj = 1.0/(1.03*ax)
            np.seterr(**old_settings)
            dj = np.log2(sj/s0) / np.arange(len(sj)) # dj; necessary for reconstruction
            sj[0] = tnum*dt/2.0
            dj[0] = 0 # remove infinity point due to fmin = 0

            # Morlet wavelet function (unnormalized)
            omega0 = 6.0 # nondimensional wavelet frequency
            ts = np.sqrt(2)*sj # e-folding time for Morlet wavelet with omega0 = 6
            wf0 = lambda eta: np.pi**(-1.0/4) * np.exp(1.0j*omega0*eta) * np.exp(-1.0/2*eta**2)

            cnum = len(D.data)  # number of cmp channels
            snum = len(sj)
            # value dimension
            D.spdata = np.zeros((cnum, tnum, (1+full)*snum-(1*full)), dtype=np.complex_)
            for c in range(cnum):
                x = D.data[c,:]

                # FFT of signal
                X = np.fft.fft(x, n=nfft)/nfft

                # calculate CWT
                cwtdata = np.zeros((nfft, snum), dtype=np.complex_)
                for j, s in enumerate(sj):
                    # nondimensional time axis at time scale s
                    eta = t/s
                    # FFT of the normalized wavelet function
                    W = np.fft.fft( np.conj( wf0(eta - np.mean(eta))*np.sqrt(dt/s) ) )
                    # Wavelet transform at scae s for all n time
                    cwtdata[:,j] = np.fft.fftshift(np.fft.ifft(X * W) * nfft)

                # full size
                if full == 1:
                    cwtdata = np.hstack([np.fliplr(cwtdata.real - 1.0j*cwtdata.imag), cwtdata[:,1:]])

                # return until tnum
                D.spdata[c,:,:] = cwtdata[0:tnum,:]

            if full == 1:
                ax = np.hstack([-ax[::-1], ax[1:]])
            D.ax = ax
            D.cwtdj = dj # for reconstruction
            D.cwtts = ts # for significance level
            D.win_factor = 1.0 

            print('dnum {:d} cwt with Morlet omega0 = 6.0, df {:g}'.format(d, df))

    def cross_power(self, done=0, dtwo=1, tavg=0):
        # IN : data number one (ref), data number two (cmp), etc
        # OUT : x-axis (ax), y-axis (val)

        self.Dlist[dtwo].vkind = 'cross_power'

        rnum = len(self.Dlist[done].data)  # number of ref channels
        cnum = len(self.Dlist[dtwo].data)  # number of cmp channels

        # reference channel names
        self.Dlist[dtwo].rname = []

        # value dimension
        self.Dlist[dtwo].val = np.zeros((cnum, len(self.Dlist[dtwo].ax)))

        # bidx (averaging time window)
        if tavg == 0:
            bidx = np.arange(self.Dlist[dtwo].spdata.shape[1])
        else:
            time = self.Dlist[dtwo].time
            bidx = np.where((np.mean(time) - tavg*1e-6/2 < time)*(time < np.mean(time) + tavg*1e-6/2))[0]

        # calculation loop for multi channels
        for c in range(cnum):
            # reference channel number
            if rnum == 1:
                self.Dlist[dtwo].rname.append(self.Dlist[done].clist[0])
                XX = self.Dlist[done].spdata[0,:,:]
            else:
                self.Dlist[dtwo].rname.append(self.Dlist[done].clist[c])
                XX = self.Dlist[done].spdata[c,:,:]

            YY = self.Dlist[dtwo].spdata[c,:,:]

            if self.Dlist[dtwo].ax[1] < 0: # full range
                self.Dlist[dtwo].val[c,:] = sp.cross_power(XX, YY, self.Dlist[dtwo].win_factor, bidx)
            else: # half
                self.Dlist[dtwo].val[c,:] = 2*sp.cross_power(XX, YY, self.Dlist[dtwo].win_factor, bidx)  # product 2 for half return

    def coherence(self, done=0, dtwo=1, tavg=0):
        # IN : data number one (ref), data number two (cmp), etc
        # OUT : x-axis (ax), y-axis (val)

        self.Dlist[dtwo].vkind = 'coherence'

        rnum = len(self.Dlist[done].data)  # number of ref channels
        cnum = len(self.Dlist[dtwo].data)  # number of cmp channels

        # reference channel names
        self.Dlist[dtwo].rname = []

        # value dimension
        self.Dlist[dtwo].val = np.zeros((cnum, len(self.Dlist[dtwo].ax)))

        # bidx (averaging time window)
        if tavg == 0:
            bidx = np.arange(self.Dlist[dtwo].spdata.shape[1])
        else:
            time = self.Dlist[dtwo].time
            bidx = np.where((np.mean(time) - tavg*1e-6/2 < time)*(time < np.mean(time) + tavg*1e-6/2))[0]

        # calculation loop for multi channels
        for c in range(cnum):
            # reference channel names
            if rnum == 1:
                self.Dlist[dtwo].rname.append(self.Dlist[done].clist[0])
                XX = self.Dlist[done].spdata[0,:,:]
            else:
                self.Dlist[dtwo].rname.append(self.Dlist[done].clist[c])
                XX = self.Dlist[done].spdata[c,:,:]

            YY = self.Dlist[dtwo].spdata[c,:,:]

            self.Dlist[dtwo].val[c,:] = sp.coherence(XX, YY, bidx)

    def cross_phase(self, done=0, dtwo=1, tavg=0):
        # IN : data number one (ref), data number two (cmp)
        # OUT : x-axis (ax), y-axis (val)

        self.Dlist[dtwo].vkind = 'cross_phase'

        rnum = len(self.Dlist[done].data)  # number of ref channels
        cnum = len(self.Dlist[dtwo].data)  # number of cmp channels
        # bins = self.Dlist[dtwo].bins  # number of bins

        # reference channel names
        self.Dlist[dtwo].rname = []

        # distance
        self.Dlist[dtwo].dist = np.zeros(cnum)

        # value dimension
        self.Dlist[dtwo].val = np.zeros((cnum, len(self.Dlist[dtwo].ax)))

        # bidx (averaging time window)
        if tavg == 0:
            bidx = np.arange(self.Dlist[dtwo].spdata.shape[1])
        else:
            time = self.Dlist[dtwo].time
            bidx = np.where((np.mean(time) - tavg*1e-6/2 < time)*(time < np.mean(time) + tavg*1e-6/2))[0]

        # calculation loop for multi channels
        for c in range(cnum):
            # reference channel number and distance between ref and cmp channels
            if rnum == 1:
                self.Dlist[dtwo].rname.append(self.Dlist[done].clist[0])
                self.Dlist[dtwo].dist[c] = np.sqrt((self.Dlist[dtwo].rpos[c] - self.Dlist[done].rpos[0])**2 + \
                (self.Dlist[dtwo].zpos[c] - self.Dlist[done].zpos[0])**2)
                XX = self.Dlist[done].spdata[0,:,:]
            else:
                self.Dlist[dtwo].rname.append(self.Dlist[done].clist[c])
                self.Dlist[dtwo].dist[c] = np.sqrt((self.Dlist[dtwo].rpos[c] - self.Dlist[done].rpos[c])**2 + \
                (self.Dlist[dtwo].zpos[c] - self.Dlist[done].zpos[c])**2)
                XX = self.Dlist[done].spdata[c,:,:]

            YY = self.Dlist[dtwo].spdata[c,:,:]

            self.Dlist[dtwo].val[c,:] = sp.cross_phase(XX, YY, bidx)

    def correlation(self, done=0, dtwo=1):
        # reguire full FFT
        # positive time lag = ref -> cmp
        self.Dlist[dtwo].vkind = 'correlation'

        rnum = len(self.Dlist[done].data)  # number of ref channels
        cnum = len(self.Dlist[dtwo].data)  # number of cmp channels
        bins = self.Dlist[dtwo].bins  # number of bins
        nfft = self.Dlist[dtwo].nfft
        win_factor = self.Dlist[dtwo].win_factor  # window factors

        # reference channel names
        self.Dlist[dtwo].rname = []

        # axes
        fs = round(1/(self.Dlist[dtwo].time[1] - self.Dlist[dtwo].time[0])/1000)*1000.0
        self.Dlist[dtwo].ax = int(nfft/2)*1.0/fs*np.linspace(1,-1,nfft)

        # value dimension
        val = np.zeros((bins, len(self.Dlist[dtwo].ax)), dtype=np.complex_)
        self.Dlist[dtwo].val = np.zeros((cnum, len(self.Dlist[dtwo].ax)))

        # calculation loop for multi channels
        for c in range(cnum):
            # reference channel number
            if rnum == 1:
                self.Dlist[dtwo].rname.append(self.Dlist[done].clist[0])
            else:
                self.Dlist[dtwo].rname.append(self.Dlist[done].clist[c])

            # calculate cross power for each channel and each bins
            for b in range(bins):
                if rnum == 1:  # single reference channel
                    X = self.Dlist[done].spdata[0,b,:]
                else:  # number of ref channels = number of cmp channels
                    X = self.Dlist[done].spdata[c,b,:]

                Y = self.Dlist[dtwo].spdata[c,b,:]

                val[b,:] = np.fft.ifftshift(X*np.matrix.conjugate(Y) / win_factor)
                val[b,:] = np.fft.ifft(val[b,:], n=nfft)*nfft
                val[b,:] = np.fft.fftshift(val[b,:])

            # average over bins
            Cxy = np.mean(val, 0)
            # result saved in val
            self.Dlist[dtwo].val[c,:] = Cxy.real
            # std saved in std

    def corr_coef(self, done=0, dtwo=1):
        # reguire full FFT
        # positive time lag = ref -> cmp
        self.Dlist[dtwo].vkind = 'corr_coef'

        rnum = len(self.Dlist[done].data)  # number of ref channels
        cnum = len(self.Dlist[dtwo].data)  # number of cmp channels
        bins = self.Dlist[dtwo].bins  # number of bins
        nfft = self.Dlist[dtwo].nfft
        win_factor = self.Dlist[dtwo].win_factor  # window factors

        # reference channel names
        self.Dlist[dtwo].rname = []

        # axes
        fs = round(1/(self.Dlist[dtwo].time[1] - self.Dlist[dtwo].time[0])/1000)*1000.0
        self.Dlist[dtwo].ax = int(nfft/2)*1.0/fs*np.linspace(1,-1,nfft)

        # value dimension
        val = np.zeros((bins, len(self.Dlist[dtwo].ax)), dtype=np.complex_)
        self.Dlist[dtwo].val = np.zeros((cnum, len(self.Dlist[dtwo].ax)))

        # calculation loop for multi channels
        for c in range(cnum):
            # reference channel number
            if rnum == 1:
                self.Dlist[dtwo].rname.append(self.Dlist[done].clist[0])
            else:
                self.Dlist[dtwo].rname.append(self.Dlist[done].clist[c])

            # calculate cross power for each channel and each bins
            for b in range(bins):
                if rnum == 1:  # single reference channel
                    X = self.Dlist[done].spdata[0,b,:]
                else:  # number of ref channels = number of cmp channels
                    X = self.Dlist[done].spdata[c,b,:]

                Y = self.Dlist[dtwo].spdata[c,b,:]

                x = np.fft.ifft(np.fft.ifftshift(X), n=nfft)*nfft/np.sqrt(win_factor)
                Rxx = np.mean(x**2)
                y = np.fft.ifft(np.fft.ifftshift(Y), n=nfft)*nfft/np.sqrt(win_factor)
                Ryy = np.mean(y**2)

                val[b,:] = np.fft.ifftshift(X*np.matrix.conjugate(Y) / win_factor)
                val[b,:] = np.fft.ifft(val[b,:], n=nfft)*nfft
                val[b,:] = np.fft.fftshift(val[b,:])

                val[b,:] = val[b,:]/np.sqrt(Rxx*Ryy)

            # average over bins
            cxy = np.mean(val, 0)
            # result saved in val
            self.Dlist[dtwo].val[c,:] = cxy.real
            # std saved in std

    # def xspec(self, done=0, cone=[0], dtwo=1, ctwo=[0], thres=0, **kwargs):
    def xspec(self, done=0, dtwo=1, thres=0, **kwargs):
        # number of cmp channels = number of ref channels
        # add x- and y- cut plot with a given mouse input
        if 'flimits' in kwargs: flimits = kwargs['flimits']*1000
        if 'xlimits' in kwargs: xlimits = kwargs['xlimits']

        self.Dlist[dtwo].vkind = 'xspec'

        cnum = len(self.Dlist[dtwo].data)  # number of cmp channels
        bins = self.Dlist[dtwo].bins  # number of bins
        win_factor = self.Dlist[dtwo].win_factor  # window factors

        # plot dimension
        if cnum < 4:
            row = cnum
        else:
            row = 4
        col = math.ceil(cnum/row)

        # reference channel names
        self.Dlist[dtwo].rname = []

        pshot = self.Dlist[dtwo].shot
        ptime = self.Dlist[dtwo].time
        pfreq = self.Dlist[dtwo].ax/1000

        for c in range(cnum):
            # set axes
            if c == 0:
                plt.subplots_adjust(hspace = 0.5, wspace = 0.3)
                axes1 = plt.subplot(row,col,c+1)
                axprops = dict(sharex = axes1, sharey = axes1)
            else:
                plt.subplot(row,col,c+1, **axprops)

            # reference channel
            rname = self.Dlist[done].clist[c]
            self.Dlist[dtwo].rname.append(rname)
            # cmp channel
            pname = self.Dlist[dtwo].clist[c]
            # pdata
            pdata = np.zeros((bins, len(self.Dlist[dtwo].ax)))  # (full length for calculation)

            # calculate cross power for each channel and each bins
            XX = self.Dlist[done].spdata[c,:,:]
            YY = self.Dlist[dtwo].spdata[c,:,:]

            pdata = sp.xspec(XX, YY, win_factor)

            pdata = np.log10(np.transpose(pdata))

            maxP = np.amax(pdata)
            minP = np.amin(pdata)
            dP = maxP - minP

            # thresholding
            pdata[(pdata < minP + dP*thres)] = -100

            plt.imshow(pdata, extent=(ptime.min(), ptime.max(), pfreq.min(), pfreq.max()), interpolation='none', aspect='auto', origin='lower', cmap=CM)

            plt.clim([minP+dP*0.30, maxP])
            plt.colorbar()

            if 'flimits' in kwargs:  # flimits
                plt.ylim([flimits[0], flimits[1]])
            if 'xlimits' in kwargs:  # xlimits
                plt.ylim([xlimits[0], xlimits[1]])
            else:
                plt.xlim([ptime[0], ptime[-1]])

            chpos = '({:.1f}, {:.1f})'.format(self.Dlist[dtwo].rpos[c]*100, self.Dlist[dtwo].zpos[c]*100) # [cm]
            plt.title('#{:d}, {:s}-{:s} {:s}'.format(pshot, rname, pname, chpos), fontsize=10)
            plt.xlabel('Time [s]')
            plt.ylabel('Frequency [kHz]')

        plt.show()

    def skw(self, done=0, dtwo=1, kstep=0.01, **kwargs):
        # calculate for each pair of done and dtwo and average
        # number of cmp channels = number of ref channels
        # kstep [cm^-1]

        self.Dlist[dtwo].vkind = 'local_SKw'

        rnum = len(self.Dlist[done].data)  # number of ref channels
        cnum = len(self.Dlist[dtwo].data)  # number of cmp channels
        bins = self.Dlist[dtwo].bins  # number of bins
        win_factor = self.Dlist[dtwo].win_factor  # window factors

        # reference channel names
        self.Dlist[dtwo].rname = []

        # distance
        self.Dlist[dtwo].dist = np.zeros(cnum)
        for c in range(cnum):
            self.Dlist[dtwo].dist[c] = np.sqrt((self.Dlist[dtwo].rpos[c] - self.Dlist[done].rpos[c])**2 + \
            (self.Dlist[dtwo].zpos[c] - self.Dlist[done].zpos[c])**2)

        # k-axes
        dmin = self.Dlist[dtwo].dist.min()*100 # [cm]
        kax = np.arange(-np.pi/dmin, np.pi/dmin, kstep) # [cm^-1]
        self.Dlist[dtwo].kax = kax

        nkax = len(kax)
        nfft = len(self.Dlist[dtwo].ax)

        # value dimension
        Pxx = np.zeros((bins, nfft), dtype=np.complex_)
        Pyy = np.zeros((bins, nfft), dtype=np.complex_)
        Kxy = np.zeros((bins, nfft), dtype=np.complex_)
        val = np.zeros((cnum, nkax, nfft), dtype=np.complex_)
        self.Dlist[dtwo].val = np.zeros((nkax, nfft))
        sklw = np.zeros((nkax, nfft), dtype=np.complex_)
        K = np.zeros((cnum, nfft), dtype=np.complex_)
        sigK = np.zeros((cnum, nfft), dtype=np.complex_)

        # calculation loop for multi channels
        for c in range(cnum):
            # reference channel name
            self.Dlist[dtwo].rname.append(self.Dlist[done].clist[c])
            print('pair of {:s} and {:s}'.format(self.Dlist[dtwo].rname[c], self.Dlist[dtwo].clist[c]))

            # calculate auto power and cross phase (wavenumber)
            for b in range(bins):
                X = self.Dlist[done].spdata[c,b,:]
                Y = self.Dlist[dtwo].spdata[c,b,:]

                Pxx[b,:] = X*np.matrix.conjugate(X) / win_factor
                Pyy[b,:] = Y*np.matrix.conjugate(Y) / win_factor
                Pxy = X*np.matrix.conjugate(Y)
                Kxy[b,:] = np.arctan2(Pxy.imag, Pxy.real).real / (self.Dlist[dtwo].dist[c]*100) # [cm^-1]

                # calculate SKw
                for w in range(nfft):
                    idx = (Kxy[b,w] - kstep/2.0 < kax) * (kax < Kxy[b,w] + kstep/2.0)
                    val[c,:,w] = val[c,:,w] + (1.0/bins*(Pxx[b,w] + Pyy[b,w])/2.0) * idx

            # calculate moments
            sklw = val[c,:,:] / np.tile(np.sum(val[c,:,:], 0), (nkax, 1))
            K[c, :] = np.sum(np.transpose(np.tile(kax, (nfft, 1))) * sklw, 0)
            for w in range(nfft):
                sigK[c,w] = np.sqrt(np.sum( (kax - K[c,w])**2 * sklw[:,w] ))

        self.Dlist[dtwo].val[:,:] = np.mean(val, 0).real
        self.Dlist[dtwo].K = np.mean(K, 0)
        self.Dlist[dtwo].sigK = np.mean(sigK, 0)

        pshot = self.Dlist[dtwo].shot
        pfreq = self.Dlist[dtwo].ax/1000
        pdata = self.Dlist[dtwo].val + 1e-10

        pdata = np.log10(pdata)

        plt.imshow(pdata, extent=(pfreq.min(), pfreq.max(), kax.min(), kax.max()), interpolation='none', aspect='auto', origin='lower', cmap=CM)

        plt.colorbar()

        chpos = '({:.1f}, {:.1f})'.format(np.mean(self.Dlist[dtwo].rpos*100), np.mean(self.Dlist[dtwo].zpos*100)) # [cm]
        plt.title('#{:d}, {:s}'.format(pshot, chpos), fontsize=10)
        plt.xlabel('Frequency [kHz]')
        plt.ylabel('Local wavenumber [rad/cm]')

        # plt.plot(pfreq, self.Dlist[dtwo].K, 'k')
        # plt.plot(pfreq, self.Dlist[dtwo].K + self.Dlist[dtwo].sigK, 'r')
        # plt.plot(pfreq, self.Dlist[dtwo].K - self.Dlist[dtwo].sigK, 'r')

        plt.show()

    def bicoherence(self, done=0, dtwo=1, cnl=[0], tavg=0, **kwargs):
        # fftbins full = 1
        # number of cmp channels = number of ref channels
        self.Dlist[dtwo].vkind = 'bicoherence'

        rnum = len(self.Dlist[done].data)  # number of ref channels
        cnum = len(self.Dlist[dtwo].data)  # number of cmp channels

        # plot dimension
        if cnum < 4:
            row = cnum
        else:
            row = 4
        col = math.ceil(cnum/row)

        # reference channel names
        self.Dlist[dtwo].rname = []

        # axes
        ax1 = self.Dlist[dtwo].ax # full -fN ~ fN
        ax2 = np.fft.ifftshift(self.Dlist[dtwo].ax) # full 0 ~ fN, -fN ~ -f1
        ax2 = ax2[0:int(len(ax1)/2+1)] # half 0 ~ fN

        # value dimension
        self.Dlist[dtwo].val = np.zeros((cnum, len(ax1), len(ax2)))
        self.Dlist[dtwo].val2 = np.zeros((cnum, len(ax1)))

        # bidx (averaging time window)
        if tavg == 0:
            bidx = np.arange(self.Dlist[dtwo].spdata.shape[1])
        else:
            time = self.Dlist[dtwo].time
            bidx = np.where((np.mean(time) - tavg*1e-6/2 < time)*(time < np.mean(time) + tavg*1e-6/2))[0]

        # calculation loop for multi channels
        for i, c in enumerate(cnl):
            # reference channel
            if rnum == 1:
                rname = self.Dlist[done].clist[0]
                XX = self.Dlist[done].spdata[0,:,:]
            else:
                rname = self.Dlist[done].clist[c]
                XX = self.Dlist[done].spdata[c,:,:]
            self.Dlist[dtwo].rname.append(rname)

            # cmp channel
            pname = self.Dlist[dtwo].clist[c]
            YY = self.Dlist[dtwo].spdata[c,:,:]

            # calculate bicoherence
            self.Dlist[dtwo].val[c,:,:], self.Dlist[dtwo].val2[c,:] = sp.bicoherence(XX, YY, bidx)

            # plot info
            pshot = self.Dlist[dtwo].shot
            chpos = '({:.1f}, {:.1f})'.format(self.Dlist[dtwo].rpos[c]*100, self.Dlist[dtwo].zpos[c]*100) # [cm]

            # Plot results
            fig, (a1,a2) = plt.subplots(1,2, figsize=(10,6), gridspec_kw = {'width_ratios':[1,1.5]})
            plt.subplots_adjust(hspace = 0.5, wspace = 0.3)

            pax1 = ax1/1000.0 # [kHz]
            pax2 = ax2/1000.0 # [kHz]

            pdata = self.Dlist[dtwo].val[c,:,:]
            pdata2 = self.Dlist[dtwo].val2[c,:]

            im = a1.imshow(pdata, extent=(pax2.min(), pax2.max(), pax1.min(), pax1.max()), interpolation='none', aspect='equal', origin='lower', cmap=CM)
            a1.set_xlabel('F1 [kHz]')
            a1.set_ylabel('F2 [kHz]')
            a1.set_title('The squared bicoherence')
            divider = make_axes_locatable(a1)
            cax = divider.append_axes('right', size='5%', pad=0.05)
            fig.colorbar(im, cax=cax, orientation='vertical')

            a2.plot(pax1, pdata2, 'k')
            a2.set_xlim([0,pax2[-1]])
            a2.set_xlabel('F3 [kHz]')
            a2.set_ylabel('Summed bicoherence (avg)')
            a2.set_title('#{:d}, {:s}-{:s} {:s}'.format(pshot, rname, pname, chpos), fontsize=10)

            plt.show()

    def nonlin_evolution(self, done=0, dtwo=1, delta=1.0, wit=1, js=1, test=0, **kwargs):
        if 'xlimits' in kwargs: xlimits = kwargs['xlimits']

        # rnum = cnum = 1 or 2
        self.Dlist[dtwo].vkind = 'nonlin_rates'

        cnum = len(self.Dlist[dtwo].data)  # number of cmp channels

        # reference channel names
        self.Dlist[dtwo].rname = []

        # distance
        self.Dlist[dtwo].dist = np.zeros(cnum)

        # axes
        ax1 = self.Dlist[dtwo].ax # full -fN ~ fN
        ax2 = np.fft.ifftshift(self.Dlist[dtwo].ax) # full 0 ~ fN, -fN ~ -f1
        ax2 = ax2[0:int(len(ax1)/2+1)] # half 0 ~ fN

        # value dimension
        self.Dlist[dtwo].val = np.zeros((cnum, len(ax1), len(ax2)))

        # obtain XX and YY
        # reference channel
        rname = self.Dlist[done].clist[0]
        XX = self.Dlist[done].spdata[0,:,:]
        self.Dlist[dtwo].rname.append(rname)
        # cmp channel
        pname = self.Dlist[dtwo].clist[0]
        YY = self.Dlist[dtwo].spdata[0,:,:]

        # if cnum == 1:
        #     # reference channel
        #     rname = self.Dlist[done].clist[0]
        #     XX = self.Dlist[done].spdata[0,:,:]
        #     self.Dlist[dtwo].rname.append(rname)
        #     # cmp channel
        #     pname = self.Dlist[dtwo].clist[0]
        #     YY = self.Dlist[dtwo].spdata[0,:,:]
        # else:
        #     # reference channel
        #     rname = self.Dlist[done].clist[0]
        #     self.Dlist[dtwo].rname.append(rname)
        #     XXa = self.Dlist[done].spdata[0,:,:]
        #     XXb = self.Dlist[done].spdata[1,:,:]
        #     print('use {:s} and {:s} for XX'.format(self.Dlist[done].clist[0], self.Dlist[done].clist[1]))
        #     # cmp channel
        #     pname = self.Dlist[dtwo].clist[0]
        #     YYa = self.Dlist[dtwo].spdata[0,:,:]
        #     YYb = self.Dlist[dtwo].spdata[1,:,:]
        #     print('use {:s} and {:s} for YY'.format(self.Dlist[dtwo].clist[0], self.Dlist[dtwo].clist[1]))

        #     # # reconstructed XX
        #     # XXc = np.sqrt(np.abs(XXa * np.matrix.conjugate(XXb)).real) # amplitude of the cross power
        #     # # XXc = np.abs(XXa * np.matrix.conjugate(XXb)) / (np.abs(XXa)*np.abs(XXb)) # coherence
        #     # XXt = (np.arctan2(XXa.imag, XXa.real).real + np.arctan2(XXb.imag, XXb.real).real)/2.0
        #     # XX = XXc * np.cos(XXt) + 1.0j * XXc * np.sin(XXt)
        #     # # reconstructed YY
        #     # YYc = np.sqrt(np.abs(YYa * np.matrix.conjugate(YYb)).real) # amplitude of the cross power
        #     # # YYc = np.abs(YYa * np.matrix.conjugate(YYb)) / (np.abs(YYa)*np.abs(YYb)) # coherence
        #     # YYt = (np.arctan2(YYa.imag, YYa.real).real + np.arctan2(YYb.imag, YYb.real).real)/2.0
        #     # YY = YYc * np.cos(YYt) + 1.0j * YYc * np.sin(YYt)

        #     # reconstruction using sub averg
        #     bins = XXa.shape[0]
        #     sdim = 20
        #     soverlap = 0.0
        #     sbins = int(np.fix((int(bins/sdim) - soverlap)/(1.0 - soverlap)))

        #     XXt = (np.arctan2(XXa.imag, XXa.real).real + np.arctan2(XXb.imag, XXb.real).real)/2.0
        #     XX = np.zeros((sbins, XXa.shape[1]), dtype=np.complex_)

        #     YYt = (np.arctan2(YYa.imag, YYa.real).real + np.arctan2(YYb.imag, YYb.real).real)/2.0
        #     YY = np.zeros((sbins, YYa.shape[1]), dtype=np.complex_)

        #     for b in range(sbins):
        #         idx1 = int(b*np.fix(sdim*(1 - soverlap)))
        #         idx2 = idx1 + sdim

        #         Xc = np.sqrt(np.abs(np.mean(XXa[idx1:idx2,:]*np.matrix.conjugate(XXb[idx1:idx2,:]), 0)))
        #         Xt = np.mean(XXt[idx1:idx2,:], 0)
        #         XX[b,:] = Xc * np.cos(Xt) + 1.0j * Xc * np.sin(Xt)

        #         Yc = np.sqrt(np.abs(np.mean(YYa[idx1:idx2,:]*np.matrix.conjugate(YYb[idx1:idx2,:]), 0)))
        #         Yt = np.mean(YYt[idx1:idx2,:], 0)
        #         YY[b,:] = Yc * np.cos(Yt) + 1.0j * Yc * np.sin(Yt)

        #     # XX = np.zeros(XXa.shape, dtype=np.complex_)
        #     # XXt = (np.arctan2(XXa.imag, XXa.real).real + np.arctan2(XXb.imag, XXb.real).real)/2.0
        #     # for i in range(XXa.shape[0]):
        #     #     si = i
        #     #     ei = i+sdim
        #     #     Xc = np.sqrt(np.abs(np.mean(XXa[si:ei,:]*np.matrix.conjugate(XXb[si:ei,:]), 0)))
        #     #     Xt = np.mean(XXt[si:ei,:])
        #     #     XX[i,:] = Xc * np.cos(Xt) + 1.0j * Xc * np.sin(Xt)

        #     # YY = np.zeros(YYa.shape, dtype=np.complex_)
        #     # YYt = (np.arctan2(YYa.imag, YYa.real).real + np.arctan2(YYb.imag, YYb.real).real)/2.0
        #     # for i in range(YYa.shape[0]):
        #     #     si = i
        #     #     ei = i+sdim
        #     #     Yc = np.sqrt(np.abs(np.mean(YYa[si:ei,:]*np.matrix.conjugate(YYb[si:ei,:]), 0)))
        #     #     Yt = np.mean(YYt[si:ei,:])
        #     #     YY[i,:] = Yc * np.cos(Yt) + 1.0j * Yc * np.sin(Yt)

        #     self.Dlist[done].spdata = np.expand_dims(XX, axis=0)
        #     self.Dlist[dtwo].spdata = np.expand_dims(YY, axis=0)

        # # check bicoherence
        # self.bicoherence(done, dtwo, cnl=[0])
        # return 0

        # modeled data
        if test == 1:
            YY, _, _ = sp.nonlinear_test(ax1, XX)
            print('TEST with MODEL DATA')

        # calculate transfer functions
        if wit == 0:
            print('Ritz method')
            Lk, Qijk, Bk, Aijk = sp.ritz_nonlinear(XX, YY)
        else:
            print('Wit method')
            Lk, Qijk, Bk, Aijk = sp.wit_nonlinear(XX, YY)

        # plot info
        pshot = self.Dlist[dtwo].shot
        chpos = '({:.1f}, {:.1f})'.format(self.Dlist[dtwo].rpos[0]*100, self.Dlist[dtwo].zpos[0]*100) # [cm]

        # Plot results
        fig, (a1,a2) = plt.subplots(2,1, figsize=(6,8), gridspec_kw = {'height_ratios':[1,2]})
        plt.subplots_adjust(hspace = 0.8, wspace = 0.3)

        pax1 = ax1/1000.0 # [kHz]
        pax2 = ax1/1000.0 # [kHz]

        # linear transfer function
        a1.plot(pax1, np.abs(Lk), 'k')
        a1.set_xlabel('Frequency [kHz]')
        a1.set_ylabel('Linear transfer function')
        a1.set_title('#{:d}, {:s}-{:s} {:s}'.format(pshot, rname, pname, chpos), fontsize=10)

        # Nonlinear transfer function
        im = a2.imshow(np.abs(Qijk), extent=(pax2.min(), pax2.max(), pax1.min(), pax1.max()), interpolation='none', aspect='equal', origin='lower', cmap=CM)
        a2.set_xlabel('Frequency [kHz]')
        a2.set_ylabel('Frequency [kHz]')
        a2.set_title('Nonlinear transfer function')
        divider = make_axes_locatable(a2)
        cax = divider.append_axes('right', size='5%', pad=0.05)
        fig.colorbar(im, cax=cax, orientation='vertical')

        plt.show()

        # calculate rates
        if js == 1:
            gk, Tijk, sum_Tijk = sp.nonlinear_ratesJS(Lk, Aijk, Qijk, XX, delta)
        else:
            gk, Tijk, sum_Tijk = sp.nonlinear_rates(Lk, Qijk, Bk, Aijk, dt)

        # Plot results
        fig, (a1,a2,a3) = plt.subplots(3,1, figsize=(6,11), gridspec_kw = {'height_ratios':[1,1,2]})
        plt.subplots_adjust(hspace = 0.5, wspace = 0.3)

        pax1 = ax1/1000.0 # [kHz]
        pax2 = ax1/1000.0 # [kHz]

        # linear growth rate
        a1.plot(pax1, gk, 'k')
        a1.set_xlabel('Frequency [kHz]')
        a1.set_ylabel('Growth rate [a.b.]')
        a1.set_title('#{:d}, {:s}-{:s} {:s}'.format(pshot, rname, pname, chpos), fontsize=10)
        a1.axhline(y=0, ls='--', color='k')
        if 'xlimits' in kwargs: a1.set_xlim([xlimits[0], xlimits[1]])

        # Nonlinear transfer rate
        a2.plot(pax1, sum_Tijk.real, 'k')
        a2.set_xlabel('Frequency [kHz]')
        a2.set_ylabel('Nonlinear transfer rate [a.b.]')
        a2.axhline(y=0, ls='--', color='k')
        if 'xlimits' in kwargs: a2.set_xlim([xlimits[0], xlimits[1]])

        im = a3.imshow(Tijk.real, extent=(pax2.min(), pax2.max(), pax1.min(), pax1.max()), interpolation='none', aspect='equal', origin='lower', cmap=CM)
        a3.set_xlabel('Frequency [kHz]')
        a3.set_ylabel('Frequency [kHz]')
        a3.set_title('Nonlinear transfer function')
        divider = make_axes_locatable(a3)
        cax = divider.append_axes('right', size='5%', pad=0.05)
        fig.colorbar(im, cax=cax, orientation='vertical')

        plt.show()

############################# statistical methods ##############################

    def skplane(self, dnum=0, cnl=[0], detrend=1, verbose=1, **kwargs):
        if 'ylimits' in kwargs: ylimits = kwargs['ylimits']
        if 'xlimits' in kwargs: xlimits = kwargs['xlimits']
        self.Dlist[dnum].vkind = 'skplane'

        cnum = len(self.Dlist[dnum].data)  # number of cmp channels

        # plot dimension
        nch = len(cnl)
        if nch < 4:
            row = nch
        else:
            row = 4
        col = math.ceil(nch/row)

        # data dimension
        self.Dlist[dnum].skew = np.zeros(cnum)
        self.Dlist[dnum].kurt = np.zeros(cnum)

        for i, c in enumerate(cnl):
            t = self.Dlist[dnum].time
            x = self.Dlist[dnum].data[c,:]

            self.Dlist[dnum].skew[c] = st.skewness(t, x, detrend)
            self.Dlist[dnum].kurt[c] = st.kurtosis(t, x, detrend)

            if verbose == 1:
                # plot info
                pshot = self.Dlist[dnum].shot
                pname = self.Dlist[dnum].clist[c]
                chpos = '({:.1f}, {:.1f})'.format(self.Dlist[dnum].rpos[c]*100, self.Dlist[dnum].zpos[c]*100) # [cm]

                # set axes
                if i == 0:
                    plt.subplots_adjust(hspace = 0.5, wspace = 0.3)
                    axes1 = plt.subplot(row,col,i+1)
                    axprops = dict(sharex = axes1, sharey = axes1)
                elif i > 0:
                    plt.subplot(row,col,i+1, **axprops)

                plt.plot(t, x)

                plt.title('#{:d}, {:s} {:s}, Skewness = {:g}, Kurtosis = {:g}'.format(pshot, pname, chpos, self.Dlist[dnum].skew[c], self.Dlist[dnum].kurt[c]), fontsize=10)
                plt.xlabel('Time [s]')

        if verbose == 1:
            plt.show()

            plt.plot(self.Dlist[dnum].skew[cnl], self.Dlist[dnum].kurt[cnl], 'o')
            for i, c in enumerate(cnl):
                plt.annotate(self.Dlist[dnum].clist[c], (self.Dlist[dnum].skew[c], self.Dlist[dnum].kurt[c]))

            sax = np.arange((self.Dlist[dnum].skew[cnl]).min(), (self.Dlist[dnum].skew[cnl]).max(), 0.001)
            kax = 3*sax**2/2 # parabolic relationship for exponential pulse and exponentially distributed pulse amplitudes [Garcia NME 2017]
            plt.plot(sax, kax, 'k')

            plt.xlabel('Skewness')
            plt.ylabel('Kurtosis')

            plt.show()

    def skewness(self, dnum=0, cnl=[0], detrend=1, **kwargs):
        self.Dlist[dnum].vkind = 'skewness'

        cnum = len(self.Dlist[dnum].data)  # number of cmp channels

        # data dimension
        self.Dlist[dnum].val = np.zeros(cnum)

        for i, c in enumerate(cnl):
            t = self.Dlist[dnum].time
            x = self.Dlist[dnum].data[c,:]

            self.Dlist[dnum].val[c] = st.skewness(t, x, detrend)

    def kurtosis(self, dnum=0, cnl=[0], detrend=1, **kwargs):
        self.Dlist[dnum].vkind = 'kurtosis'

        cnum = len(self.Dlist[dnum].data)  # number of cmp channels

        # data dimension
        self.Dlist[dnum].val = np.zeros(cnum)

        for i, c in enumerate(cnl):
            t = self.Dlist[dnum].time
            x = self.Dlist[dnum].data[c,:]

            self.Dlist[dnum].val[c] = st.kurtosis(t, x, detrend)

    def hurst(self, dnum=0, cnl=[0], bins=30, detrend=1, fitlims=[10,1000], **kwargs):
        self.Dlist[dnum].vkind = 'hurst'

        pshot = self.Dlist[dnum].shot
        cnum = len(self.Dlist[dnum].data)  # number of cmp channels

        # axis
        bsize = int(1.0*len(self.Dlist[dnum].time)/bins)
        ax = np.floor( 10**(np.arange(1.0, np.log10(bsize), 0.01)) )

        # data dimension
        self.Dlist[dnum].ers = np.zeros((cnum, len(ax)))
        self.Dlist[dnum].std = np.zeros((cnum, len(ax)))
        self.Dlist[dnum].fit = np.zeros((cnum, len(ax)))
        self.Dlist[dnum].val = np.zeros(cnum)

        for i, c in enumerate(cnl):
            pname = self.Dlist[dnum].clist[c]
            t = self.Dlist[dnum].time
            x = self.Dlist[dnum].data[c,:]

            self.Dlist[dnum].ax, self.Dlist[dnum].ers[c,:], self.Dlist[dnum].std[c,:], \
            self.Dlist[dnum].val[c], self.Dlist[dnum].fit[c,:] = st.hurst(t, x, bins, detrend, fitlims, **kwargs)

    def chplane(self, dnum=0, cnl=[0], d=6, bins=1, verbose=1, **kwargs):
        # CH plane [Rosso PRL 2007]
        # chaotic : moderate C and H, above fBm
        # stochastic : low C and high H, below fBm
        if 'ylimits' in kwargs: ylimits = kwargs['ylimits']
        if 'xlimits' in kwargs: xlimits = kwargs['xlimits']

        self.Dlist[dnum].vkind = 'BP_probability'

        pshot = self.Dlist[dnum].shot
        cnum = len(self.Dlist[dnum].data)  # number of cmp channels

        # plot dimension
        nch = len(cnl)
        if nch < 4:
            row = nch
        else:
            row = 4
        col = math.ceil(nch/row)

        nst = math.factorial(d) # number of possible states

        bsize = int(1.0*len(self.Dlist[dnum].data[0,:])/bins)
        print('For an accurate estimation of the probability, bsize {:g} should be considerably larger than nst {:g}'.format(bsize, nst))

        # data dimension
        self.Dlist[dnum].pi = np.zeros((cnum, nst))
        self.Dlist[dnum].std = np.zeros((cnum, nst))
        self.Dlist[dnum].jscom = np.zeros(cnum)
        self.Dlist[dnum].nsent = np.zeros(cnum)

        for i, c in enumerate(cnl):
            # set axes
            if verbose == 1 and i == 0:
                plt.subplots_adjust(hspace = 0.5, wspace = 0.3)
                axes1 = plt.subplot(row,col,i+1)
                axprops = dict(sharex = axes1)
            elif verbose == 1 and i > 0:
                plt.subplot(row,col,i+1, **axprops)

            pname = self.Dlist[dnum].clist[c]

            x = self.Dlist[dnum].data[c,:]

            self.Dlist[dnum].ax, self.Dlist[dnum].pi[c,:], self.Dlist[dnum].std[c,:] = st.bp_prob(x, d, bins)
            self.Dlist[dnum].jscom[c], self.Dlist[dnum].nsent[c] = st.ch_measure(self.Dlist[dnum].pi[c,:])

            # plot BP probability
            if verbose == 1:
                pax = self.Dlist[dnum].ax
                pdata = self.Dlist[dnum].pi[c,:]

                plt.plot(pax, pdata, '-x')

                chpos = '({:.1f}, {:.1f})'.format(self.Dlist[dnum].rpos[c]*100, self.Dlist[dnum].zpos[c]*100) # [cm]
                plt.title('#{:d}, {:s} {:s}; C={:g}, H={:g}'.format(pshot, pname, chpos, self.Dlist[dnum].jscom[c], self.Dlist[dnum].nsent[c]), fontsize=10)
                plt.xlabel('order number')
                plt.ylabel('BP probability')

                plt.yscale('log')

        if verbose == 1: plt.show()

        # plot CH plane
        if verbose == 1:
            plt.plot(self.Dlist[dnum].nsent[cnl], self.Dlist[dnum].jscom[cnl], 'o')
            for i, c in enumerate(cnl):
                plt.annotate(self.Dlist[dnum].clist[c], (self.Dlist[dnum].nsent[c], self.Dlist[dnum].jscom[c]))

            # complexity limits
            h_one, c_one, h_two, c_two = st.complexity_limits(d)
            plt.plot(h_one, c_one, 'k')
            plt.plot(h_two, c_two, 'k')

            # draw fbm, fgn locus
            c_fbm, h_fbm, c_fgn, h_fgn = st.fmb_fgn_locus(d)
            plt.plot(h_fbm, c_fbm, 'y')
            plt.plot(h_fgn, c_fgn, 'g')

            plt.xlabel('Entropy (H)')
            plt.ylabel('Complexity (C)')

            plt.show()

    def js_complexity(self, dnum=0, cnl=[0], d=6, bins=1, **kwargs):
        self.Dlist[dnum].vkind = 'jscom'

        cnum = len(self.Dlist[dnum].data)  # number of cmp channels

        nst = math.factorial(d) # number of possible states

        bsize = int(1.0*len(self.Dlist[dnum].data[0,:])/bins)
        print('For an accurate estimation of the probability, bsize {:g} should be considerably larger than nst {:g}'.format(bsize, nst))

        # data dimension
        self.Dlist[dnum].pi = np.zeros((cnum, nst))
        self.Dlist[dnum].std = np.zeros((cnum, nst))
        self.Dlist[dnum].val = np.zeros(cnum)

        for i, c in enumerate(cnl):
            x = self.Dlist[dnum].data[c,:]

            self.Dlist[dnum].ax, self.Dlist[dnum].pi[c,:], self.Dlist[dnum].std[c,:] = st.bp_prob(x, d, bins)
            self.Dlist[dnum].val[c] = st.js_complexity(self.Dlist[dnum].pi[c,:])

    def ns_entropy(self, dnum=0, cnl=[0], d=6, bins=1, **kwargs):
        self.Dlist[dnum].vkind = 'nsent'

        cnum = len(self.Dlist[dnum].data)  # number of cmp channels

        nst = math.factorial(d) # number of possible states

        bsize = int(1.0*len(self.Dlist[dnum].data[0,:])/bins)
        print('For an accurate estimation of the probability, bsize {:g} should be considerably larger than nst {:g}'.format(bsize, nst))

        # data dimension
        self.Dlist[dnum].pi = np.zeros((cnum, nst))
        self.Dlist[dnum].std = np.zeros((cnum, nst))
        self.Dlist[dnum].val = np.zeros(cnum)

        for i, c in enumerate(cnl):
            x = self.Dlist[dnum].data[c,:]

            self.Dlist[dnum].ax, self.Dlist[dnum].pi[c,:], self.Dlist[dnum].std[c,:] = st.bp_prob(x, d, bins)
            self.Dlist[dnum].val[c] = st.ns_entropy(self.Dlist[dnum].pi[c,:])

    def intermittency(self, dnum=0, cnl=[0], bins=20, overlap=0.2, qstep=0.3, fitlims=[20.0,100.0], verbose=1, **kwargs):
        # intermittency parameter from multi-fractal analysis [Carreras PoP 2000]
        # this ranges from 0 (mono-fractal) to 1
        # add D fitting later
        if 'ylimits' in kwargs: ylimits = kwargs['ylimits']
        if 'xlimits' in kwargs: xlimits = kwargs['xlimits']

        self.Dlist[dnum].vkind = 'intermittency'

        pshot = self.Dlist[dnum].shot
        cnum = len(self.Dlist[dnum].data)  # number of cmp channels

        self.Dlist[dnum].intmit = np.zeros(cnum)

        for i, c in enumerate(cnl):
            t = self.Dlist[dnum].time
            x = self.Dlist[dnum].data[c,:]

            self.Dlist[dnum].intmit[c] = st.intermittency(t, x, bins, overlap, qstep, fitlims, verbose, **kwargs)

############################# default plot functions ###########################

    def mplot(self, dnum=1, cnl=[0], type='time', show=1, **kwargs):
        if 'ylimits' in kwargs: ylimits = kwargs['ylimits']
        if 'xlimits' in kwargs: xlimits = kwargs['xlimits']

        pshot = self.Dlist[dnum].shot

        # plot dimension
        nch = len(cnl)
        if nch < 8:
            col = nch
        else:
            col = 8
        row = math.ceil(nch/col)

        for i, c in enumerate(cnl):
            # set axes
            if i == 0:
                plt.subplots_adjust(hspace = 0.5, wspace = 0.3)
                axes1 = plt.subplot(row,col,i+1)
                if type == 'time':
                    axprops = dict(sharex = axes1)
                else:
                    axprops = dict(sharex = axes1, sharey = axes1)
            else:
                plt.subplot(row,col,i+1, **axprops)

            pname = self.Dlist[dnum].clist[c]

            # set data
            if type == 'time':
                pbase = self.Dlist[dnum].time
                pdata = self.Dlist[dnum].data[c,:]
            elif type == 'val':
                vkind = self.Dlist[dnum].vkind

                if hasattr(self.Dlist[dnum], 'rname'):
                    rname = self.Dlist[dnum].rname[c]
                else:
                    rname = ''

                # set data
                if vkind in ['skewness','kurtosis']:
                    pdata = self.Dlist[dnum].data[c,:]
                elif vkind == 'hurst':
                    pdata = self.Dlist[dnum].ers[c,:]
                elif vkind in ['jscom','nsent']:
                    pdata = self.Dlist[dnum].pi[c,:]
                else:
                    pdata = self.Dlist[dnum].val[c,:].real

                # set base
                if vkind in ['correlation','corr_coef']:
                    pbase = self.Dlist[dnum].ax*1e6
                elif vkind in ['cross_power','coherence','cross_phase','bicoherence']:
                    pbase = self.Dlist[dnum].ax/1000
                elif vkind in ['skewness','kurtosis']:
                    pbase = self.Dlist[dnum].time
                else:
                    pbase = self.Dlist[dnum].ax

            if type == 'time':
                plt.plot(pbase, pdata)  # plot
            elif type == 'val':
                plt.plot(pbase, pdata, '-x')  # plot

            # aux plot
            if type == 'val':
                if vkind == 'coherence':
                    plt.axhline(y=1/np.sqrt(self.Dlist[dnum].bins), color='r')
                elif vkind == 'hurst':
                    plt.plot(pbase, self.Dlist[dnum].fit[c,:], 'r')
                elif vkind in ['correlation','corr_coef']:
                    hdata = signal.hilbert(pdata)
                    plt.plot(pbase, np.abs(hdata), '--')

            # xy limits
            if 'ylimits' in kwargs:  # ylimits
                plt.ylim([ylimits[0], ylimits[1]])
            if 'xlimits' in kwargs:  # xlimits
                plt.xlim([xlimits[0], xlimits[1]])
            else:
                plt.xlim([pbase[0], pbase[-1]])

            # title
            chpos = '({:.1f}, {:.1f})'.format(self.Dlist[dnum].rpos[c]*100, self.Dlist[dnum].zpos[c]*100) # [cm]
            if type == 'time':
                plt.title('#{:d}, {:s} {:s}'.format(pshot, pname, chpos), fontsize=10)
            elif type == 'val':
                if vkind in ['skewness','kurtosis','hurst','jscom','nsent']:
                    plt.title('#{:d}, {:s} {:s}, {:s} = {:g}'.format(pshot, pname, chpos, vkind, self.Dlist[dnum].val[c]), fontsize=10)
                else:
                    plt.title('#{:d}, {:s}-{:s} {:s}'.format(pshot, rname, pname, chpos), fontsize=10)

            # xy scale
            if type == 'val':
                if vkind in ['hurst']:
                    plt.xscale('log')
                if vkind in ['cross_power','hurst','jscom','nsent']:
                    plt.yscale('log')

            # xy label
            if type == 'time':
                plt.xlabel('Time [s]')
                plt.ylabel('Signal')
            elif type == 'val':
                if vkind in ['cross_power','coherence','cross_phase','bicoherence']:
                    plt.xlabel('Frequency [kHz]')
                    plt.ylabel(vkind)
                elif vkind == 'hurst':
                    plt.xlabel('Time lag [us]')
                    plt.ylabel('R/S')
                elif vkind in ['jscom','nsent']:
                    plt.xlabel('order number')
                    plt.ylabel('BP probability')
                elif vkind in ['correlation','corr_coef']:
                    plt.xlabel('Time lag [us]')
                    plt.ylabel(vkind)
                else:
                    plt.xlabel('Time [s]')
                    plt.ylabel('Signal')

        if show == 1:
            plt.show()

    def oplot(self, dnum, cnl, type='time', **kwargs):
        if 'ylimits' in kwargs: ylimits = kwargs['ylimits']
        if 'xlimits' in kwargs: xlimits = kwargs['xlimits']

        for i, c in enumerate(cnl):
            pname = self.Dlist[dnum].clist[c]

            if type == 'time':
                pbase = self.Dlist[dnum].time
                pdata = self.Dlist[dnum].data[c,:]
            elif type == 'val':
                pbase = self.Dlist[dnum].ax/1000
                pdata = self.Dlist[dnum].val[c,:].real
                rname = self.Dlist[dnum].rname[c]
                if i == 0 and self.Dlist[dnum].vkind == 'coherence':
                    plt.axhline(y=1/np.sqrt(self.Dlist[dnum].bins), color='r')

            plt.plot(pbase, pdata)

            if type == 'time':
                print('dnum {:d} : channel {:s} is plotted'.format(dnum, pname))
            elif type == 'val':
                print('dnum {:d} : calculation {:s}-{:s} is plotted'.format(dnum, rname, pname))

            if 'ylimits' in kwargs: # ylimits
                plt.ylim([ylimits[0], ylimits[1]])
            if 'xlimits' in kwargs: # xlimits
                plt.xlim([xlimits[0], xlimits[1]])
            else:
                plt.xlim([pbase[0], pbase[-1]])

            if type == 'time':
                plt.xlabel('Time [s]')
                plt.ylabel('Signal')
            elif type == 'val' and self.Dlist[dnum].vkind == 'cross_power':
                plt.xlabel('Frequency [kHz]')
                plt.ylabel('Cross power')
                plt.yscale('log')
            elif type == 'val' and self.Dlist[dnum].vkind == 'coherence':
                plt.xlabel('Frequency [kHz]')
                plt.ylabel('Coherence')
            elif type == 'val' and self.Dlist[dnum].vkind == 'cross_phase':
                plt.xlabel('Frequency [kHz]')
                plt.ylabel('Cross phase [rad]')

        plt.show()

    def cplot(self, dnum, snum=0, frange=[0, 100], vlimits=[0, 1], **kwargs):
        if 'ylimits' in kwargs: ylimits = kwargs['ylimits']
        if 'xlimits' in kwargs: xlimits = kwargs['xlimits']
        # calculate mean coherence image
        # or cross power rms image
        # or group velocity image

        vkind = self.Dlist[dnum].vkind

        if vkind in ['cross_power','coherence','cross_phase','bicoherence']:
            # axis
            sbase = self.Dlist[dnum].ax/1000  # [kHz]

            # fidx
            idx = np.where((sbase >= frange[0])*(sbase <= frange[1]))
            idx1 = int(idx[0][0])
            idx2 = int(idx[0][-1]+1)
            sdata = self.Dlist[dnum].val[snum,:]
        elif vkind == 'hurst':
            sbase = self.Dlist[dnum].ax
            sdata = self.Dlist[dnum].ers[snum,:]
        elif vkind in ['jscom','nsent']:
            sbase = self.Dlist[dnum].ax
            sdata = self.Dlist[dnum].pi[snum,:]
        else:
            sbase = self.Dlist[dnum].time
            sdata = self.Dlist[dnum].data[snum,:]

        # data
        if vkind == 'cross_power':  # rms
            pdata = np.sqrt(np.sum(self.Dlist[dnum].val[:,idx1:idx2], 1))
        elif vkind == 'coherence':  # mean coherence
            pdata = np.mean(self.Dlist[dnum].val[:,idx1:idx2], 1)
        elif vkind == 'cross_phase':  # group velocity
            base = self.Dlist[dnum].ax[idx1:idx2]  # [Hz]
            pdata = np.zeros(len(self.Dlist[dnum].val))
            for c in range(len(self.Dlist[dnum].val)):
                data = self.Dlist[dnum].val[c,idx1:idx2]
                pfit = np.polyfit(base, data, 1)
                fitdata = np.polyval(pfit, base)
                pdata[c] = 2*np.pi*self.Dlist[dnum].dist[c]/pfit[0]/1000.0  # [km/s]

                # chisq = np.sum((data - fitdata)**2)
                if c == snum:
                    fbase = base/1000  # [kHz]
                    fdata = fitdata
        else:
            pdata = self.Dlist[dnum].val

        # position
        rpos = self.Dlist[dnum].rpos[:]
        zpos = self.Dlist[dnum].zpos[:]

        # prepare figure
        fig = plt.figure(facecolor='w', figsize=(5,10))
        ax1 = fig.add_axes([0.20, 0.8, 0.65, 0.15])  # [left bottom width height]
        ax2 = fig.add_axes([0.20, 0.1, 0.65, 0.60])
        ax3 = fig.add_axes([0.88, 0.2, 0.03, 0.4])
        axs = [ax1, ax2, ax3]

        # sample plot
        axs[0].plot(sbase, sdata)  # ax1.hold(True)
        if vkind == 'cross_phase':
            axs[0].plot(fbase, fdata)
        if vkind in ['cross_power','coherence','cross_phase']:
            axs[0].axvline(x=sbase[idx1], color='g')
            axs[0].axvline(x=sbase[idx2], color='g')

        if vkind in ['hurst']:
            axs[0].set_xscale('log')
        if vkind in ['cross_power','hurst','jscom','nsent']:
            axs[0].set_yscale('log')

        if 'ylimits' in kwargs: # ylimits
            axs[0].set_ylim([ylimits[0], ylimits[1]])
        if 'xlimits' in kwargs: # xlimits
            axs[0].set_xlim([xlimits[0], xlimits[1]])
        else:
            axs[0].set_xlim([sbase[0], sbase[-1]])

        if vkind in ['cross_power','coherence','cross_phase','bicoherence']:
            axs[0].set_xlabel('Frequency [kHz]')
            axs[0].set_ylabel(vkind)
        elif vkind == 'hurst':
            axs[0].set_xlabel('Time lag [us]')
            axs[0].set_ylabel('R/S')
        elif vkind in ['jscom','nsent']:
            axs[0].set_xlabel('order number')
            axs[0].set_ylabel('BP probability')
        else:
            axs[0].set_xlabel('Time [s]')
            axs[0].set_ylabel('Signal')

        # pdata plot
        sc = axs[1].scatter(rpos, zpos, 500, pdata, marker='s', vmin=vlimits[0], vmax=vlimits[1], cmap=CM)
        axs[1].set_aspect('equal')

        # color bar
        plt.colorbar(sc, cax=axs[2])

        axs[1].set_xlabel('R [m]')
        axs[1].set_ylabel('z [m]')
        if vkind == 'cross_power':
            axs[1].set_title('Cross power rms')
        elif vkind == 'coherence':
            axs[1].set_title('Coherence mean')
        elif vkind == 'cross_phase':
            axs[1].set_title('Group velocity [km/s]')
        else:
            axs[1].set_title(vkind)

        self.Dlist[dnum].pdata = pdata

        plt.show()

    def spec(self, dnum=0, cnl=[0], nfft=512, **kwargs):
        if 'flimits' in kwargs: flimits = kwargs['flimits']*1000
        if 'xlimits' in kwargs: xlimits = kwargs['xlimits']

        fs = self.Dlist[dnum].fs
        nov = nfft*0.9

        for c in cnl:
            pshot = self.Dlist[dnum].shot
            pname = self.Dlist[dnum].clist[c]
            pbase = self.Dlist[dnum].time
            pdata = self.Dlist[dnum].data[c,:]

            pxx, freq, time, cax = plt.specgram(pdata, NFFT=nfft, Fs=fs, noverlap=nov,
                                                xextent=[pbase[0], pbase[-1]], cmap=CM)  # spectrum

            maxP = math.log(np.amax(pxx),10)*10
            minP = math.log(np.amin(pxx),10)*10
            dP = maxP - minP
            plt.clim([minP+dP*0.55, maxP])
            plt.colorbar(cax)

            if 'flimits' in kwargs:  # flimits
                plt.ylim([flimits[0], flimits[1]])
            if 'xlimits' in kwargs:  # xlimits
                plt.ylim([xlimits[0], xlimits[1]])
            else:
                plt.xlim([pbase[0], pbase[-1]])

            plt.title(pname, fontsize=10)  # labeling
            plt.xlabel('Time [s]')
            plt.ylabel('Frequency [Hz]')

            plt.show()

    def iplot(self, dnum, snum=0, vlimits=[-0.1, 0.1], istep=0.002, imethod='cubic', cutoff=0.03, pmethod='scatter', **kwargs):
        # keyboard interactive iplot
        D = self.Dlist[dnum]

        CM = plt.cm.get_cmap('RdYlBu_r')

        c = int(input('automatic, or manual [0,1]: '))
        tidx1 = 0  # starting index
        if c == 0:
            # make axes
            fig = plt.figure(facecolor='w', figsize=(5,10))
            ax1 = fig.add_axes([0.1, 0.77, 0.7, 0.2])  # [left bottom width height]
            ax2 = fig.add_axes([0.1, 0.07, 0.7, 0.60])
            ax3 = fig.add_axes([0.83, 0.07, 0.03, 0.6])
            axs = [ax1, ax2, ax3]

            tstep = int(input('time step [idx]: '))  # jumping index # tstep = 10
            for tidx in range(tidx1, len(D.time), tstep):
                # take data and channel position
                pdata = D.data[:,tidx]
                rpos = D.rpos[:]
                zpos = D.zpos[:]

                # fill bad channel
                pdata = ms.fill_bad_channel(pdata, rpos, zpos, D.good_channels, cutoff)

                # interpolation
                if istep > 0:
                    ri, zi, pi = ms.interp_pdata(pdata, rpos, zpos, istep, imethod)

                # plot
                axs[0].cla()
                axs[1].cla()
                axs[2].cla()
                plt.ion()

                axs[0].plot(D.time, D.data[snum,:])  # ax1.hold(True)
                axs[0].axvline(x=D.time[tidx], color='g')
                if istep > 0:
                    if pmethod == 'scatter':
                        im = axs[1].scatter(ri.ravel(), zi.ravel(), 5, pi.ravel(), marker='s', vmin=vlimits[0], vmax=vlimits[1], cmap=CM, edgecolors='none')
                    elif pmethod == 'contour':
                        im = axs[1].contourf(ri, zi, pi, 50, cmap=CM)
                else:
                    im = axs[1].scatter(rpos, zpos, 500, pdata, marker='s', vmin=vlimits[0], vmax=vlimits[1], cmap=CM, edgecolors='none')
                axs[1].set_aspect('equal')
                plt.colorbar(im, cax=axs[2])

                axs[1].set_xlabel('R [m]')
                axs[1].set_ylabel('z [m]')
                axs[1].set_title('ECE image at t = {:g} sec'.format(D.time[tidx]))

                plt.show()
                plt.pause(0.1)

            plt.ioff()
            plt.close()

        elif c == 1:
            tidx = tidx1
            print('Select a point in the top axes to plot the image')

            # make axes
            fig = plt.figure(facecolor='w', figsize=(5,10))
            ax1 = fig.add_axes([0.1, 0.77, 0.7, 0.2])  # [left bottom width height]
            ax2 = fig.add_axes([0.1, 0.07, 0.7, 0.60])
            ax3 = fig.add_axes([0.83, 0.07, 0.03, 0.6])
            axs = [ax1, ax2, ax3]
            while True:

                # take data and channel position
                pdata = D.data[:,tidx]
                rpos = D.rpos[:]
                zpos = D.zpos[:]

                # fill bad channel
                pdata = ms.fill_bad_channel(pdata, rpos, zpos, D.good_channels, cutoff)

                # interpolation
                if istep > 0:
                    ri, zi, pi = ms.interp_pdata(pdata, rpos, zpos, istep, imethod)

                # plot
                axs[0].cla()
                axs[1].cla()
                axs[2].cla()
                plt.ion()

                axs[0].plot(D.time, D.data[snum,:])  # ax1.hold(True)
                axs[0].axvline(x=D.time[tidx], color='g')
                if istep > 0:
                    if pmethod == 'scatter':
                        im = axs[1].scatter(ri.ravel(), zi.ravel(), 5, pi.ravel(), marker='s', vmin=vlimits[0], vmax=vlimits[1], cmap=CM, edgecolors='none')
                    elif pmethod == 'contour':
                        im = axs[1].contourf(ri, zi, pi, 50, cmap=CM)
                else:
                    im = axs[1].scatter(rpos, zpos, 500, pdata, marker='s', vmin=vlimits[0], vmax=vlimits[1], cmap=CM, edgecolors='none')
                axs[1].set_aspect('equal')
                plt.colorbar(im, cax=axs[2])

                axs[1].set_xlabel('R [m]')
                axs[1].set_ylabel('z [m]')
                axs[1].set_title('ECE image at t = {:g} sec'.format(D.time[tidx]))

                plt.show()

                ## mouse input
                g = plt.ginput(1)[0][0]
                if g >= D.time[0] and g <= D.time[-1]:
                    tidx = np.where(D.time > g)[0][0]
                else:
                    print('Out of the time range')
                    plt.ioff()
                    plt.close()
                    break

                ## keyboard input
                # k = input('set time step [idx][+,-,0]: ')
                # try:
                #     tstep = int(k)
                #     if tstep == 0:
                #         plt.ioff()
                #         plt.close()
                #         break
                # except:
                #     pass

                # if tidx + tstep < len(D.time) - 1 and 0 < tidx + tstep:
                #     tidx = tidx + tstep

                plt.ioff()
            plt.close()

        D.pdata = pdata

############################# test functions ###################################

    def fftbins_bicoh_test(self, nfft, window, overlap, detrend, full=1):
        # self.list_data()

        for dnum in range(len(self.Dlist)):
            # get bins and window function
            tnum = len(self.Dlist[dnum].data[0,:])
            bins, win = sp.fft_window(tnum, nfft, window, overlap)

            # make an x-axis #
            dt = self.Dlist[dnum].time[1] - self.Dlist[dnum].time[0]  # time step
            ax = np.fft.fftfreq(nfft, d=dt) # full 0~fN -fN~-f1
            if np.mod(nfft, 2) == 0:  # even nfft
                ax = np.hstack([ax[0:int(nfft/2)], -(ax[int(nfft/2)]), ax[int(nfft/2):nfft]])
            if full == 1: # full shift to -fN ~ 0 ~ fN
                ax = np.fft.fftshift(ax)
            else: # half 0~fN
                ax = ax[0:int(nfft/2+1)]
            self.Dlist[dnum].ax = ax

            # make fftdata
            cnum = len(self.Dlist[dnum].data)
            if full == 1: # full shift to -fN ~ 0 ~ fN
                if np.mod(nfft, 2) == 0:  # even nfft
                    self.Dlist[dnum].spdata = np.zeros((cnum, bins, nfft+1), dtype=np.complex_)
                else:  # odd nfft
                    self.Dlist[dnum].spdata = np.zeros((cnum, bins, nfft), dtype=np.complex_)
            else: # half 0 ~ fN
                self.Dlist[dnum].spdata = np.zeros((cnum, bins, int(nfft/2+1)), dtype=np.complex_)

            pbs = 2*np.pi*(0.5 - np.random.randn(bins))
            pcs = 2*np.pi*(0.5 - np.random.randn(bins))
            pds = 2*np.pi*(0.5 - np.random.randn(bins))

            for c in range(cnum):
                for b in range(bins):
                    idx1 = int(b*np.fix(nfft*(1 - overlap)))
                    idx2 = idx1 + nfft

                    sx = np.zeros(idx2-idx1)
                    st = self.Dlist[dnum].time[idx1:idx2]

                    # test signal for bicoherence test
                    fb = 54*1000
                    fc = 94*1000
                    fd = fb + fc

                    pb = pbs[b]
                    pc = pcs[b]
                    pd = pds[b] # non-coherent case
                    # pd = pb + pc # coherent case

                    sx = np.cos(2*np.pi*fb*st + pb) + np.cos(2*np.pi*fc*st + pc) + 1/2*np.cos(2*np.pi*fd*st + pd) + 1/2*np.random.randn(len(sx))
                    sx = sx + np.cos(2*np.pi*fb*st + pb)*np.cos(2*np.pi*fc*st + pc) # +,- coupling

                    if detrend == 1:
                        sx = signal.detrend(sx, type='linear')
                    sx = signal.detrend(sx, type='constant')  # subtract mean

                    self.Dlist[dnum].data[c,idx1:idx2] = sx

                    # get fft
                    sx = sx * win  # apply window function
                    fftdata = np.fft.fft(sx, n=nfft)/nfft  # divide by the length
                    if np.mod(nfft, 2) == 0:  # even nfft
                        fftdata = np.hstack([fftdata[0:int(nfft/2)], np.conj(fftdata[int(nfft/2)]), fftdata[int(nfft/2):nfft]])
                    if full == 1: # shift to -fN ~ 0 ~ fN
                        fftdata = np.fft.fftshift(fftdata)
                    else: # half 0 ~ fN
                        fftdata = fftdata[0:int(nfft/2+1)]
                    self.Dlist[dnum].spdata[c,b,:] = fftdata

            # update attributes
            if np.mod(nfft, 2) == 0:
                self.Dlist[dnum].nfft = nfft + 1
            else:
                self.Dlist[dnum].nfft = nfft
            self.Dlist[dnum].window = window
            self.Dlist[dnum].overlap = overlap
            self.Dlist[dnum].detrend = detrend
            self.Dlist[dnum].bins = bins
            self.Dlist[dnum].win_factor = np.mean(win**2)

            print('TEST :: dnum {:d} fftbins {:d} with {:s} size {:d} overlap {:g} detrend {:d} full {:d}'.format(dnum, bins, window, nfft, overlap, detrend, full))

def expand_clist(clist):
    # IN : List of channel names (e.g. 'ECEI_G1201-1208' or 'ECEI_GT1201-1208').
    # OUT : Expanded list (e.g. 'ECEI_G1201', ..., 'ECEI_G1208')

    # KSTAR ECEI
    exp_clist = []
    for c in range(len(clist)):
        if 'ECEI' in clist[c] and len(clist[c]) == 15: # before 2018
            vi = int(clist[c][6:8])
            fi = int(clist[c][8:10])
            vf = int(clist[c][11:13])
            ff = int(clist[c][13:15])

            for v in range(vi, vf+1):
                for f in range(fi, ff+1):
                    exp_clist.append(clist[c][0:6] + '{:02d}{:02d}'.format(v, f))
        elif 'ECEI' in clist[c] and len(clist[c]) == 16: # since 2018
            vi = int(clist[c][7:9])
            fi = int(clist[c][9:11])
            vf = int(clist[c][12:14])
            ff = int(clist[c][14:16])

            for v in range(vi, vf+1):
                for f in range(fi, ff+1):
                    exp_clist.append(clist[c][0:7] + '{:02d}{:02d}'.format(v, f))
        else:
            exp_clist.append(clist[c])
    clist = exp_clist

    return clist


def nextpow2(i):
    n = 1
    while n < i: n *= 2
    return n
