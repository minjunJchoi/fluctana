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
from matplotlib.widgets import Slider, Button
from mpl_toolkits.axes_grid1 import make_axes_locatable

import pickle

from kstarecei import *
from kstarmir import *
from kstarcss import *
from kstarbes import *
from kstarmds import *
#from diiiddata import *  # needs pidly
from diiidbes import *

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
    def __init__(self, shot=None, clist=None, time=None, data=None, rpos=None, zpos=None, apos=None):
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
        time, idx = self.time_base(trange)

        if norm == 2:
            _, aidx = self.time_base(atrange)

        # time series length
        tnum = len(time)
        # number of channels
        cnum = len(self.clist)

        raw_data = self.data
        data = np.zeros((cnum, tnum))
        for i in range(cnum):
            v = raw_data[i,idx]

            if norm == 1:
                v = v/np.mean(v) - 1
            elif norm == 2:
                v = v/np.mean(raw_data[i,aidx]) - 1

            data[i,:] = v

        self.data = data
        self.time = time

        return time, data

    def time_base(self, trange):
        time = self.time

        # get index
        idx = (time >= trange[0])*(time <= trange[1])

        # get fs
        self.fs = round(1/(time[1] - time[0])/1000)*1000.0

        return time[idx], idx


class FluctAna(object):
    def __init__(self):
        self.Dlist = []

    def add_data(self, dev='KSTAR', shot=23359, clist=['ECEI_GT1201'], trange=[6.8,7.0], norm=1, atrange=[1.0, 1.01], res=0, verbose=1, **kwargs):
        # for an arbitrary data
        if 'data' in kwargs:
            time = kwargs['time']
            data = kwargs['data']
            rpos = kwargs['rpos']
            zpos = kwargs['zpos']
            apos = kwargs['apos']
            D = FluctData(shot=shot, clist=clist, time=time, data=data, rpos=rpos, zpos=zpos, apos=apos)
        else:
            if dev == 'KSTAR': # KSTAR data 
                if 'ECEI' in clist[0]:
                    D = KstarEcei(shot=shot, clist=clist)
                elif 'MIR' in clist[0]:
                    D = KstarMir(shot=shot, clist=clist)
                elif 'CSS' in clist[0]:
                    D = KstarCss(shot=shot, clist=clist)
                elif 'BES' in clist[0]:
                    D = KstarBes(shot=shot, clist=clist)
                else:
                    D = KstarMds(shot=shot, clist=clist)
            elif dev == 'DIIID': # DIII-D data 
                if 'BES' in clist[0]:
                    D = DiiidBes(shot=shot, clist=clist)

        D.get_data(trange, norm=norm, atrange=atrange, res=res, verbose=verbose)
        self.Dlist.append(D)

    def add_multi_data(self, dev='KSTAR', shot=23359, clist=['ECEI_GT1201'], time_list=[6.8,6.9], tspan=1e-3, norm=1, res=0, verbose=1, **kwargs):
        # KSTAR diagnostics
        if dev == 'KSTAR':
            if 'ECEI' in clist[0]:
                D = KstarEcei(shot=shot, clist=clist)
            elif 'MIR' in clist[0]:
                D = KstarMir(shot=shot, clist=clist)
            elif 'CSS' in clist[0]:
                D = KstarCss(shot=shot, clist=clist)
            elif 'BES' in clist[0]:
                D = KstarBes(shot=shot, clist=clist)
            else:
                D = KstarMds(shot=shot, clist=clist)
        elif dev == 'DIIID':
            if 'BES' in clist[0]:
                D = DiiidBes(shot=shot, clist=clist)

        D.get_multi_data(time_list=time_list, tspan=tspan, norm=norm, res=res, verbose=verbose)
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
        D = self.Dlist[dnum]

        old_clist = D.clist

        # add channels (no duplicates)
        clist = D.expand_clist(clist)
        clist = [c for c in clist if c not in D.clist]

        # add data
        time, data = D.get_data(D.trange, norm=norm, atrange=atrange, res=res)
        D.data = np.concatenate((D.data, data), axis=0)

        # update clist
        D.clist = old_clist + clist

        self.list_data()

    def del_channel(self, dnum, clist):
        D = self.Dlist[dnum]

        clist = D.expand_clist(clist)

        for i in range(len(clist)):
            # find index to be deleted
            del_idx = [j for j, s in enumerate(D.clist) if clist[i] in s]

            # delete data
            D.data = np.delete(D.data, del_idx, 0)

            # delete fftdata if it has
            if hasattr(D, 'fftdata'):
                D.fftdata = np.delete(D.fftdata, del_idx, 0)

            # update clist
            D.clist = [D.clist[k] for k in range(len(D.clist)) if k not in del_idx]

        self.list_data()

    def ch_pos_correction(self, dnum=0, fname=None, verbose=0):
        D = self.Dlist[dnum]

        # plot original position
        if verbose == 1:
            plt.plot(D.rpos, D.zpos, 'bx')

        # replace with corrected channel position saved in fname
        with open(fname, 'rb') as fin:
            fdata = pickle.load(fin)

            clist_full = fdata[0]
            rpos_full = fdata[1]
            zpos_full = fdata[2]

            if len(rpos_full) == len(D.rpos):
                D.rpos = rpos_full
                D.zpos = zpos_full
            else:
                clist_full = D.expand_clist(clist_full)
                D.rpos = np.array([rpos_full[clist_full.index(i)] for i in D.clist])
                D.zpos = np.array([zpos_full[clist_full.index(i)] for i in D.clist])

        if verbose == 1:
            plt.plot(D.rpos, D.zpos, 'ro')
            plt.gca().set_aspect('equal')
            plt.xlabel('R [m]'); plt.ylabel('z [m]')
            plt.show()

        print('channel position corrected with {:s}'.format(fname))

    def calibration(self, dnum=0, new=0, calib_factor_fname=None, abs_fname=None):
        D = self.Dlist[dnum]

        if new == 0: # use calibration factors saved file
            with open(calib_factor_fname, 'rb') as fin:
                [calib_factor] = pickle.load(fin)

            D.data = D.data * calib_factor

            print('data calibrated with {:s}'.format(calib_factor_fname))
        elif new == 1: # use absolute data saved file (e.g., ecei_pos*.pkl (syndia))
            with open(abs_fname, 'rb') as fin:
                fdata = pickle.load(fin)
                abs_data = fdata[-1]

            raw_data = np.mean(D.data, axis=1)

            calib_factor = np.expand_dims(abs_data/raw_data, axis=1)

            D.data = D.data * calib_factor

            with open(calib_factor_fname, 'wb') as fout:
                pickle.dump([calib_factor], fout)

            print('data calibrated with {:s}'.format(abs_fname))
            print('calibration factors are saved in {:s}'.format(calib_factor_fname))

        if hasattr(D, 'good_channels'):
            D.good_channels = D.good_channels * np.squeeze(~(calib_factor == 0))

############################# down sampling #############################

    def downsample(self, dnum, q, verbose=0, fig=None, axs=None):
        # down sampling after anti-aliasing filter
        D = self.Dlist[dnum]

        cnum = len(D.clist)

        # make axes
        if verbose == 1:
            fig, axs = make_axes(cnum, ptype='mplot', fig=fig, axs=axs, type='time')

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

                axs[c].plot(raw_time, raw_data[c,:])
                axs[c].plot(D.time, D.data[c,:])

                axs[c].set_title('#{:d}, {:s}'.format(pshot, pname), fontsize=10)

        if verbose == 1: plt.show()

        D.fs = round(1/(D.time[1] - D.time[0]))
        print('down sample with q={:d}, fs={:g}'.format(q, D.fs))

    def subsample(self, dnum, twin, tstep):
        # return sub samples along time
        D = self.Dlist[dnum]

        cnum = len(D.clist)

        raw_data = np.copy(D.data)

        # subsample tidx list
        tidx_win = int(D.fs*twin) # buffer window
        tidx_step = int(D.fs*tstep)
        tidx_list = range(int(tidx_win/2), len(D.time)-int(tidx_win/2), tidx_step)

        D.time = D.time[tidx_list]
        D.data = np.empty((cnum, len(D.time)))
        for c in range(cnum):
            D.data[c,:] = raw_data[c,tidx_list]

        D.fs = round(1/(D.time[1] - D.time[0]))
        print('sub sample with twin={:g} us, tstep={:g} us'.format(twin*1e6, tstep*1e6))

############################# data filtering functions #########################

    def ma_filt(self, dnum=0, twin=300e-6, window='rectwin', type='time', demean=0, cut_edge=False): # twin window size in [s]
        # moving average #### add option data or val
        D = self.Dlist[dnum]

        cnum = len(D.clist)

        win_size = int(D.fs*twin) # window size in [idx]

        # window function
        if window == 'rectwin':  # overlap = 0.5
            win = np.ones(win_size)
        elif window == 'hann':  # overlap = 0.5
            win = np.hanning(win_size)
        elif window == 'hamm':  # overlap = 0.5
            win = np.hamming(win_size)
        elif window == 'kaiser':  # overlap = 0.62
            win = np.kaiser(win_size, beta=30)
        elif window == 'HFT248D':  # overlap = 0.84
            z = 2*np.pi/win_size*np.arange(0,win_size)
            win = 1 - 1.985844164102*np.cos(z) + 1.791176438506*np.cos(2*z) - 1.282075284005*np.cos(3*z) + \
                0.667777530266*np.cos(4*z) - 0.240160796576*np.cos(5*z) + 0.056656381764*np.cos(6*z) - \
                0.008134974479*np.cos(7*z) + 0.000624544650*np.cos(8*z) - 0.000019808998*np.cos(9*z) + \
                0.000000132974*np.cos(10*z)

        for c in range(cnum):
            if type == 'time':
                x = D.data[c,:]
            elif type == 'val':
                x = D.val[c,:]

            if demean == 1:
                x = np.convolve(x, win, 'same') / np.sum(win) - np.mean(x)
            else:
                x = np.convolve(x, win, 'same') / np.sum(win)

            if type == 'time':
                D.data[c,:] = x
            elif type == 'val':
                D.val[c,:] = x

        if cut_edge == True:
            D.time = D.time[int(win_size/2):-int(win_size/2)]
            if type == 'time':
                D.data = D.data[:,int(win_size/2):-int(win_size/2)]
            elif type == 'val':
                D.val = D.val[:,int(win_size/2):-int(win_size/2)]

        print('dnum {:d} moving average filter with window {:s} size {:g} [us] demean {:d}'.format(dnum, window, twin*1e6, demean))

    def filt(self, dnum=0, name='FIR_pass', fL=0, fH=10000, b=0.08, nbins=100, verbose=0):
        D = self.Dlist[dnum]

        # select filter except svd
        if name[0:3] == 'FIR':
            freq_filter = ft.FirFilter(name, D.fs, fL, fH, b)
        elif name[0:3] == 'FFT':
            freq_filter = ft.FftFilter(name, D.fs, fL, fH)
        elif name == 'Threshold_FFT':
            freq_filter = ft.ThresholdFftFilter(D.fs, fL, fH, b=b, nbins=nbins)

        for c in range(len(D.clist)):
            x = np.copy(D.data[c,:])
            D.data[c,:] = freq_filter.apply(x)

        print('dnum {:d} filter {:s} with fL {:g} fH {:g} b {:g}'.format(dnum, name, fL, fH, b))

    def svd_filt(self, dnum=0, cutoff=0.9, verbose=0):
        D = self.Dlist[dnum]

        svd_filter = ft.SvdFilter(cutoff = cutoff)

        if hasattr(D, 'good_channels'):
            D.data = svd_filter.apply(D.data, D.good_channels, verbose=verbose)
        else:
            D.data = svd_filter.apply(D.data, np.zeros(len(D.clist)), verbose=verbose)

        print('dnum {:d} svd filter with cutoff {:g}'.format(dnum, cutoff))

    def wave2d_filt(self, dnum=0, row=24, column=8, wavename='coif3', alpha=1.0, lim=5, bcut=0.0, verbose=0):
        D = self.Dlist[dnum]

        wave2d_filter = ft.Wave2dFilter(wavename=wavename, alpha=alpha, lim=lim)

        rpos = D.rpos[:]
        zpos = D.zpos[:]

        tnum = len(D.time)
        D.clev = np.zeros(tnum)
        D.ilev = np.zeros(tnum)

        for i in range(tnum):
            data1d = D.data[:,i]

            # fill bad channel
            if bcut > 0:
                data1d = ms.fill_bad_channel(data1d, rpos, zpos, D.good_channels, bcut)

            # make it 2D
            data2d = data1d.reshape((row,column))

            # apply filter
            data2d, D.clev[i], D.ilev[i] = wave2d_filter.apply(data2d)

            # make it 1D
            D.data[:,i] = data2d.reshape(data1d.shape)

            if verbose == 1 and i%100 ==0: print('wave2d filter: {:d}/{:d} done'.format(i, tnum))

        if verbose == 1:
            plt.plot(D.time, D.clev, 'r')
            plt.plot(D.time, D.ilev, 'b')
            plt.show()

############################# spectral methods #############################

    def fftbins(self, nfft, window, overlap, detrend=0, full=0):
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
                D.nfreq = nfft + 1
            else:
                D.nfreq = nfft

            print('dnum {:d} fftbins {:d} with {:s} size {:d} overlap {:g} detrend {:d} full {:d}'.format(d, bins, window, nfft, overlap, detrend, full))


    def fftmulti(self, window, detrend=0, full=0):
        # IN : self, multi data window name, detrend or not
        # OUT : multi x N FFT of time series data; frequency axis
        # self.list_data()

        for d, D in enumerate(self.Dlist):
            # get window function
            tnum = len(D.multi_time[0,:])
            nfft = tnum
            overlap = 0 
            _, win = sp.fft_window(tnum, nfft, window, overlap)
            dt = D.multi_time[0,1] - D.multi_time[0,0]  # time step

            D.window = window
            D.detrend = detrend

            # make fft data
            bins = len(D.multi_time)
            cnum = len(D.multi_data)
            if full == 1: # full shift to -fN ~ 0 ~ fN
                if np.mod(nfft, 2) == 0:  # even nfft
                    D.spdata = np.zeros((cnum, bins, nfft+1), dtype=np.complex_)
                else:  # odd nfft
                    D.spdata = np.zeros((cnum, bins, nfft), dtype=np.complex_)
            else: # half 0 ~ fN
                D.spdata = np.zeros((cnum, bins, int(nfft/2+1)), dtype=np.complex_)

            for c in range(cnum):
                for t in range(bins):
                    x = D.multi_data[c,t,:]
                    D.ax, D.spdata[c,t,:], D.win_factor = sp.fftbins(x, dt, nfft, window, overlap, detrend, full)

            print(D.multi_time.shape)
            print(D.multi_data.shape)
            print(D.ax.shape)
            print(D.spdata.shape)

            # update attributes
            if np.mod(nfft, 2) == 0:
                D.nfreq = nfft + 1
            else:
                D.nfreq = nfft

            print('dnum {:d} fftbins {:d} with {:s} size {:d} overlap {:g} detrend {:d} full {:d}'.format(d, bins, window, nfft, overlap, detrend, full))

    def cwt(self, df, tavg=0, detrend=0, full=0):
        for d, D in enumerate(self.Dlist):
            # time step
            dt = D.time[1] - D.time[0]

            # bin index
            bidx = np.where((np.mean(D.time) - tavg*1e-6/2 < D.time)*(D.time < np.mean(D.time) + tavg*1e-6/2))[0]

            # make a f-axis with constant df
            s0 = 2.0*dt # the smallest scale
            ax_half = np.arange(0.0, 1.0/(1.03*s0), df) # 1.03 for the Morlet wavelet function

            # value dimension
            cnum = len(D.data)  # number of cmp channels
            bins = len(bidx)
            snum = len(ax_half)
            ncwt = (1+full)*snum-(1*full)
            D.spdata = np.zeros((cnum, bins, ncwt), dtype=np.complex_)
            for c in range(cnum):
                x = D.data[c,:]
                D.ax, cwtdata, D.cwtdj, D.cwtts = sp.cwt(x, dt, df, detrend, full)
                D.spdata[c,:,:] = cwtdata[bidx,:]

            D.win_factor = 1.0
            D.nfreq = ncwt
            D.bins = len(bidx)
            # D.nens = len(D.bidx) / ( D.fs/(2*1.03*ax_half) ) # need corrections

            print('dnum {:d} cwt with Morlet omega0 = 6.0, df {:g}, tavg {:g}, bins {:d}'.format(d, df, tavg, bins))

    def cross_power(self, done=0, dtwo=1):
        # IN : data number one (ref), data number two (cmp), etc
        # OUT : x-axis (ax), y-axis (val)
        Done = self.Dlist[done]
        Dtwo = self.Dlist[dtwo]

        Dtwo.vkind = 'cross_power'

        rnum = len(Done.clist)  # number of ref channels
        cnum = len(Dtwo.clist)  # number of cmp channels

        # reference channel names
        Dtwo.rname = []

        # value dimension
        Dtwo.val = np.zeros((cnum, len(Dtwo.ax)))

        # calculation loop for multi channels
        for c in range(cnum):
            # reference channel number
            if rnum == 1:
                Dtwo.rname.append(Done.clist[0])
                XX = Done.spdata[0,:,:]
            else:
                Dtwo.rname.append(Done.clist[c])
                XX = Done.spdata[c,:,:]

            YY = Dtwo.spdata[c,:,:]

            if Dtwo.ax[1] < 0: # full range
                Dtwo.val[c,:] = sp.cross_power(XX, YY, Dtwo.win_factor)
            else: # half
                Dtwo.val[c,:] = 2*sp.cross_power(XX, YY, Dtwo.win_factor)  # product 2 for half return

    def coherence(self, done=0, dtwo=1):
        # IN : data number one (ref), data number two (cmp), etc
        # OUT : x-axis (ax), y-axis (val)
        Done = self.Dlist[done]
        Dtwo = self.Dlist[dtwo]

        Dtwo.vkind = 'coherence'

        rnum = len(Done.clist)  # number of ref channels
        cnum = len(Dtwo.clist)  # number of cmp channels

        # reference channel names
        Dtwo.rname = []

        # value dimension
        Dtwo.val = np.zeros((cnum, len(Dtwo.ax)))

        # calculation loop for multi channels
        for c in range(cnum):
            # reference channel names
            if rnum == 1:
                Dtwo.rname.append(Done.clist[0])
                XX = Done.spdata[0,:,:]
            else:
                Dtwo.rname.append(Done.clist[c])
                XX = Done.spdata[c,:,:]

            YY = Dtwo.spdata[c,:,:]

            Dtwo.val[c,:] = sp.coherence(XX, YY)

    def cross_phase(self, done=0, dtwo=1):
        # IN : data number one (ref), data number two (cmp)
        # OUT : x-axis (ax), y-axis (val)
        Done = self.Dlist[done]
        Dtwo = self.Dlist[dtwo]

        Dtwo.vkind = 'cross_phase'

        rnum = len(Done.clist)  # number of ref channels
        cnum = len(Dtwo.clist)  # number of cmp channels
        # bins = Dtwo.bins  # number of bins

        # reference channel names
        Dtwo.rname = []

        # distance
        Dtwo.dist = np.zeros(cnum)

        # value dimension
        Dtwo.val = np.zeros((cnum, len(Dtwo.ax)))

        # calculation loop for multi channels
        for c in range(cnum):
            # reference channel number and distance between ref and cmp channels
            if rnum == 1:
                Dtwo.rname.append(Done.clist[0])
                Dtwo.dist[c] = np.sqrt((Dtwo.rpos[c] - Done.rpos[0])**2 + \
                (Dtwo.zpos[c] - Done.zpos[0])**2)
                XX = Done.spdata[0,:,:]
            else:
                Dtwo.rname.append(Done.clist[c])
                Dtwo.dist[c] = np.sqrt((Dtwo.rpos[c] - Done.rpos[c])**2 + \
                (Dtwo.zpos[c] - Done.zpos[c])**2)
                XX = Done.spdata[c,:,:]

            YY = Dtwo.spdata[c,:,:]

            Dtwo.val[c,:] = sp.cross_phase(XX, YY)

    def correlation(self, done=0, dtwo=1):
        # reguire full FFT
        # positive time lag = ref -> cmp
        Done = self.Dlist[done]
        Dtwo = self.Dlist[dtwo]

        Dtwo.vkind = 'correlation'

        rnum = len(Done.clist)  # number of ref channels
        cnum = len(Dtwo.clist)  # number of cmp channels
        bins = Dtwo.bins  # number of bins
        nfreq = Dtwo.nfreq
        win_factor = Dtwo.win_factor  # window factors

        # reference channel names
        Dtwo.rname = []

        # axes
        fs = Dtwo.fs
        Dtwo.ax = int(nfreq/2)*1.0/fs*np.linspace(-1,1,nfreq)

        # distance
        Dtwo.dist = np.zeros(cnum)

        # value dimension
        val = np.zeros((bins, len(Dtwo.ax)), dtype=np.complex_)
        Dtwo.val = np.zeros((cnum, len(Dtwo.ax)))

        # calculation loop for multi channels
        for c in range(cnum):
            # reference channel number
            if rnum == 1:
                Dtwo.rname.append(Done.clist[0])
                Dtwo.dist[c] = np.sqrt((Dtwo.rpos[c] - Done.rpos[0])**2 + \
                (Dtwo.zpos[c] - Done.zpos[0])**2)
                XX = Done.spdata[0,:,:]
            else:
                Dtwo.rname.append(Done.clist[c])
                Dtwo.dist[c] = np.sqrt((Dtwo.rpos[c] - Done.rpos[c])**2 + \
                (Dtwo.zpos[c] - Done.zpos[c])**2)
                XX = Done.spdata[c,:,:]

            YY = Dtwo.spdata[c,:,:]

            Dtwo.val[c,:] = sp.correlation(XX, YY, win_factor)

    def corr_coef(self, done=0, dtwo=1):
        # reguire full FFT
        # positive time lag = ref -> cmp
        Done = self.Dlist[done]
        Dtwo = self.Dlist[dtwo]

        Dtwo.vkind = 'corr_coef'

        rnum = len(Done.clist)  # number of ref channels
        cnum = len(Dtwo.clist)  # number of cmp channels
        bins = Dtwo.bins  # number of bins
        nfreq = Dtwo.nfreq
        win_factor = Dtwo.win_factor  # window factors

        # reference channel names
        Dtwo.rname = []

        # axes
        fs = Dtwo.fs
        Dtwo.ax = int(nfreq/2)*1.0/fs*np.linspace(-1,1,nfreq)

        # distance
        Dtwo.dist = np.zeros(cnum)

        # value dimension
        val = np.zeros((bins, len(Dtwo.ax)), dtype=np.complex_)
        Dtwo.val = np.zeros((cnum, len(Dtwo.ax)))

        # calculation loop for multi channels
        for c in range(cnum):
            # reference channel number
            if rnum == 1:
                Dtwo.rname.append(Done.clist[0])
                Dtwo.dist[c] = np.sqrt((Dtwo.rpos[c] - Done.rpos[0])**2 + \
                (Dtwo.zpos[c] - Done.zpos[0])**2)
                XX = Done.spdata[0,:,:]
            else:
                Dtwo.rname.append(Done.clist[c])
                Dtwo.dist[c] = np.sqrt((Dtwo.rpos[c] - Done.rpos[c])**2 + \
                (Dtwo.zpos[c] - Done.zpos[c])**2)
                XX = Done.spdata[c,:,:]

            YY = Dtwo.spdata[c,:,:]

            Dtwo.val[c,:] = sp.corr_coef(XX, YY)

    def xspec(self, done=0, dtwo=1, cnl='all', thres=0, fig=None, axs=None, show=1, cbar=1, **kwargs):
        # number of cmp channels = number of ref channels
        # add x- and y- cut plot with a given mouse input
        if 'flimits' in kwargs: flimits = kwargs['flimits']
        if 'xlimits' in kwargs: xlimits = kwargs['xlimits']
        Done = self.Dlist[done]
        Dtwo = self.Dlist[dtwo]

        Dtwo.vkind = 'xspec'

        cnum = len(Dtwo.clist)  # number of cmp channels
        bins = Dtwo.bins  # number of bins
        win_factor = Dtwo.win_factor  # window factors

        if cnl == 'all': cnl = range(cnum)

        # reference channel names
        Dtwo.rname = []

        pshot = Dtwo.shot
        ptime = Dtwo.time
        pfreq = Dtwo.ax/1000

        fig, axs = make_axes(len(cnl), ptype='mplot', maxcol=4, type='val', fig=fig, axs=axs)

        for i, c in enumerate(cnl):
            plt.sca(axs[i])

            # reference channel
            rname = Done.clist[c]
            Dtwo.rname.append(rname)
            # cmp channel
            pname = Dtwo.clist[c]
            # pdata
            pdata = np.zeros((bins, len(Dtwo.ax)))  # (full length for calculation)

            # calculate cross power for each channel and each bins
            XX = Done.spdata[c,:,:]
            YY = Dtwo.spdata[c,:,:]

            pdata = sp.xspec(XX, YY, win_factor)

            pdata = np.log10(np.transpose(pdata))

            maxP = np.amax(pdata)
            minP = np.amin(pdata)
            dP = maxP - minP

            # thresholding
            pdata[(pdata < minP + dP*thres)] = -100

            plt.imshow(pdata, extent=(ptime.min(), ptime.max(), pfreq.min(), pfreq.max()), interpolation='none', aspect='auto', origin='lower', cmap=CM)

            plt.clim([minP+dP*0.30, maxP])

            if cbar == 1: plt.colorbar()

            if 'flimits' in kwargs:  # flimits
                plt.ylim([flimits[0], flimits[1]])
            if 'xlimits' in kwargs:  # xlimits
                plt.xlim([xlimits[0], xlimits[1]])
            else:
                plt.xlim([ptime[0], ptime[-1]])

            chpos = '({:.1f}, {:.1f})'.format(Dtwo.rpos[c]*100, Dtwo.zpos[c]*100) # [cm]
            plt.title('#{:d}, {:s}-{:s} {:s}'.format(pshot, rname, pname, chpos), fontsize=10)
            plt.xlabel('Time [s]')
            plt.ylabel('Frequency [kHz]')

        if show == 1: plt.show()

        return fig, axs

    def skw(self, done=0, dtwo=1, kstep=0.01, **kwargs):
        # calculate for each pair of done and dtwo and average
        # number of cmp channels = number of ref channels
        # kstep [cm^-1]
        Done = self.Dlist[done]
        Dtwo = self.Dlist[dtwo]

        Dtwo.vkind = 'local_SKw'

        rnum = len(Done.clist)  # number of ref channels
        cnum = len(Dtwo.clist)  # number of cmp channels
        bins = Dtwo.bins  # number of bins
        win_factor = Dtwo.win_factor  # window factors

        # reference channel names
        Dtwo.rname = []

        # distance
        Dtwo.dist = np.zeros(cnum)
        for c in range(cnum):
            Dtwo.dist[c] = np.sqrt((Dtwo.rpos[c] - Done.rpos[c])**2 + \
            (Dtwo.zpos[c] - Done.zpos[c])**2)

        # k-axes
        dmin = Dtwo.dist.min()*100 # [cm]
        kax = np.arange(-np.pi/dmin, np.pi/dmin, kstep) # [cm^-1]
        Dtwo.kax = kax

        nkax = len(kax)
        nfreq = len(Dtwo.ax)

        # value dimension
        Pxx = np.zeros((bins, nfreq), dtype=np.complex_)
        Pyy = np.zeros((bins, nfreq), dtype=np.complex_)
        Kxy = np.zeros((bins, nfreq), dtype=np.complex_)
        val = np.zeros((cnum, nkax, nfreq), dtype=np.complex_)
        Dtwo.val = np.zeros((nkax, nfreq))
        sklw = np.zeros((nkax, nfreq), dtype=np.complex_)
        K = np.zeros((cnum, nfreq), dtype=np.complex_)
        sigK = np.zeros((cnum, nfreq), dtype=np.complex_)

        # calculation loop for multi channels
        for c in range(cnum):
            # reference channel name
            Dtwo.rname.append(Done.clist[c])
            print('pair of {:s} and {:s}'.format(Dtwo.rname[c], Dtwo.clist[c]))

            # calculate auto power and cross phase (wavenumber)
            for b in range(bins):
                X = Done.spdata[c,b,:]
                Y = Dtwo.spdata[c,b,:]

                Pxx[b,:] = X*np.matrix.conjugate(X) / win_factor
                Pyy[b,:] = Y*np.matrix.conjugate(Y) / win_factor
                Pxy = X*np.matrix.conjugate(Y)
                Kxy[b,:] = np.arctan2(Pxy.imag, Pxy.real).real / (Dtwo.dist[c]*100) # [cm^-1]

                # calculate SKw
                for w in range(nfreq):
                    idx = (Kxy[b,w] - kstep/2.0 < kax) * (kax < Kxy[b,w] + kstep/2.0)
                    val[c,:,w] = val[c,:,w] + (1.0/bins*(Pxx[b,w] + Pyy[b,w])/2.0) * idx

            # calculate moments
            sklw = val[c,:,:] / np.tile(np.sum(val[c,:,:], 0), (nkax, 1))
            K[c, :] = np.sum(np.transpose(np.tile(kax, (nfreq, 1))) * sklw, 0)
            for w in range(nfreq):
                sigK[c,w] = np.sqrt(np.sum( (kax - K[c,w])**2 * sklw[:,w] ))

        Dtwo.val[:,:] = np.mean(val, 0).real
        Dtwo.K = np.mean(K, 0)
        Dtwo.sigK = np.mean(sigK, 0)

        pshot = Dtwo.shot
        pfreq = Dtwo.ax/1000
        pdata = Dtwo.val + 1e-10

        pdata = np.log10(pdata)

        plt.imshow(pdata, extent=(pfreq.min(), pfreq.max(), kax.min(), kax.max()), interpolation='none', aspect='auto', origin='lower', cmap=CM)

        plt.colorbar()

        chpos = '({:.1f}, {:.1f})'.format(np.mean(Dtwo.rpos*100), np.mean(Dtwo.zpos*100)) # [cm]
        plt.title('#{:d}, {:s}'.format(pshot, chpos), fontsize=10)
        plt.xlabel('Frequency [kHz]')
        plt.ylabel('Local wavenumber [rad/cm]')

        # plt.plot(pfreq, Dtwo.K, 'k')
        # plt.plot(pfreq, Dtwo.K + Dtwo.sigK, 'r')
        # plt.plot(pfreq, Dtwo.K - Dtwo.sigK, 'r')

        plt.show()

    def bicoherence(self, done=0, dtwo=1, cnl='all', vlimits=[0,0.3], show=1, **kwargs):
        # fftbins full = 1
        # number of cmp channels = number of ref channels
        if 'xlimits' in kwargs: xlimits = kwargs['xlimits']
        if 'ylimits' in kwargs: ylimits = kwargs['ylimits']

        Done = self.Dlist[done]
        Dtwo = self.Dlist[dtwo]

        Dtwo.vkind = 'bicoherence'

        rnum = len(Done.clist)  # number of ref channels
        cnum = len(Dtwo.clist)  # number of cmp channels

        if cnl == 'all': cnl = range(cnum)

        # plot dimension
        if cnum < 4:
            row = cnum
        else:
            row = 4
        col = math.ceil(cnum/row)

        # reference channel names
        Dtwo.rname = []

        # axes
        ax1 = Dtwo.ax # full -fN ~ fN
        ax2 = np.fft.ifftshift(Dtwo.ax) # full 0 ~ fN, -fN ~ -f1
        ax2 = ax2[0:int(len(ax1)/2+1)] # half 0 ~ fN

        # value dimension
        Dtwo.val = np.zeros((cnum, len(ax1), len(ax2)))
        Dtwo.val2 = np.zeros((cnum, len(ax1)))

        # calculation loop for multi channels
        for i, c in enumerate(cnl):
            # reference channel
            if rnum == 1:
                rname = Done.clist[0]
                XX = Done.spdata[0,:,:]
            else:
                rname = Done.clist[c]
                XX = Done.spdata[c,:,:]
            Dtwo.rname.append(rname)

            # cmp channel
            pname = Dtwo.clist[c]
            YY = Dtwo.spdata[c,:,:]

            # calculate bicoherence
            Dtwo.val[c,:,:], Dtwo.val2[c,:] = sp.bicoherence(XX, YY)

            if show == 1:
                # plot info
                pshot = Dtwo.shot
                chpos = '({:.1f}, {:.1f})'.format(Dtwo.rpos[c]*100, Dtwo.zpos[c]*100) # [cm]

                # Plot results
                fig, (a1,a2) = plt.subplots(1,2, figsize=(10,6), gridspec_kw = {'width_ratios':[1,1.5]})
                plt.subplots_adjust(left = 0.1, right = 0.95, hspace = 0.5, wspace = 0.3)

                pax1 = ax1/1000.0 # [kHz]
                pax2 = ax2/1000.0 # [kHz]

                pdata = Dtwo.val[c,:,:]
                pdata2 = Dtwo.val2[c,:]

                im = a1.imshow(pdata, extent=(pax2.min(), pax2.max(), pax1.min(), pax1.max()), interpolation='none', aspect='equal', origin='lower', vmin=vlimits[0], vmax=vlimits[1], cmap=CM)
                a1.set_xlabel('f1 [kHz]')
                a1.set_ylabel('f2 [kHz]')
                a1.set_title('The squared bicoherence of f3')
                if 'xlimits' in kwargs:  # xlimits
                    a1.set_xlim([xlimits[0], xlimits[1]])
                if 'ylimits' in kwargs:  # xlimits
                    a1.set_ylim([ylimits[0], ylimits[1]])
                divider = make_axes_locatable(a1)
                cax = divider.append_axes('right', size='5%', pad=0.05)
                fig.colorbar(im, cax=cax, orientation='vertical')

                a2.plot(pax1, pdata2, 'k')
                # a2.axhline(y=1/Dtwo.bins, color='r') ## need correction
                a2.set_xlim([0,pax2[-1]])
                a2.set_xlabel('f3 [kHz]')
                a2.set_ylabel('Summed bicoherence (avg)')
                a2.set_title('#{:d}, {:s}-{:s} {:s}'.format(pshot, rname, pname, chpos), fontsize=10)

                plt.show()

    def nonlin_evolution(self, done=0, dtwo=1, delta=1.0, wit=1, js=1, test=0, show=1, check=0, **kwargs):
        if 'xlimits' in kwargs: xlimits = kwargs['xlimits']

        Done = self.Dlist[done]
        Dtwo = self.Dlist[dtwo]

        Dtwo.vkind = 'nonlin_rates'

        # rnum = cnum = 1
        cnum = len(Dtwo.clist)  # number of cmp channels

        # reference channel names
        Dtwo.rname = []

        # distance
        Dtwo.dist = np.zeros(cnum)

        # axes
        ax1 = Dtwo.ax # full -fN ~ fN
        ax2 = np.fft.ifftshift(Dtwo.ax) # full 0 ~ fN, -fN ~ -f1
        ax2 = ax2[0:int(len(ax1)/2+1)] # half 0 ~ fN

        # value dimension
        Dtwo.val = np.zeros((cnum, len(ax1), len(ax2)))

        # obtain XX and YY
        # reference channel
        rname = Done.clist[0]
        XX = Done.spdata[0,:,:]
        Dtwo.rname.append(rname)
        # cmp channel
        pname = Dtwo.clist[0]
        YY = Dtwo.spdata[0,:,:]

        # modeled data
        if test == 1:
            YY, _, _ = sp.nonlinear_test(ax1, XX)
            print('TEST with MODEL DATA')

        # calculate transfer functions
        if wit == 1:
            print('Wit method')
            Lk, Qijk, Bk, Aijk = sp.wit_nonlinear(XX, YY)
        else:
            print('Ritz method')
            Lk, Qijk, Bk, Aijk = sp.ritz_nonlinear(XX, YY)

        if show == 1:
            # plot info
            pshot = Dtwo.shot
            chpos = '({:.1f}, {:.1f})'.format(Dtwo.rpos[0]*100, Dtwo.zpos[0]*100) # [cm]

            # Plot results
            fig, (a1,a2) = plt.subplots(2,1, figsize=(6,8), gridspec_kw = {'height_ratios':[1,2]})
            plt.subplots_adjust(hspace = 0.8, wspace = 0.3)

            pax1 = ax1/1000.0 # [kHz]
            pax2 = ax1/1000.0 # [kHz]

            # linear transfer function
            a1.plot(pax1, Lk.real, 'k')
            a1.set_xlabel('Frequency [kHz]')
            a1.set_ylabel('Linear transfer function')
            a1.set_title('#{:d}, {:s}-{:s} {:s}'.format(pshot, rname, pname, chpos), fontsize=10)

            # Nonlinear transfer function
            im = a2.imshow((np.abs(Qijk)), extent=(pax2.min(), pax2.max(), pax1.min(), pax1.max()), interpolation='none', aspect='equal', origin='lower', cmap=CM)
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
            gk, Tijk, sum_Tijk = sp.nonlinear_rates(Lk, Qijk, Bk, Aijk, delta)

        if show == 1:
            # Plot results
            fig, (a1,a2,a3) = plt.subplots(3,1, figsize=(6,11), gridspec_kw = {'height_ratios':[1,1,2]})
            plt.subplots_adjust(hspace = 0.5, wspace = 0.3)

            pax1 = ax1/1000.0 # [kHz]
            pax2 = ax1/1000.0 # [kHz]

            # linear growth rate
            a1.plot(pax1, gk, 'k')
            a1.set_xlabel('Frequency [kHz]')
            a1.set_ylabel('Growth rate [a.u.]')
            a1.set_title('#{:d}, {:s}-{:s} {:s}'.format(pshot, rname, pname, chpos), fontsize=10)
            a1.axhline(y=0, ls='--', color='k')
            if 'xlimits' in kwargs: a1.set_xlim([xlimits[0], xlimits[1]])

            # Nonlinear transfer rate
            # # limited summed Tijk
            # full = len(ax1)
            # kidx = sp.get_kidx(full)
            # sum_Tijk = np.zeros(full)
            # for k in range(full):
            #     f3 = ax1[k]
            #     idx = kidx[k]
            #     for n, ij in enumerate(idx):
            #         f1 = ax1[ij[0]]
            #         f2 = ax1[ij[1]]
            #         if np.min([np.abs(f1), np.abs(f2), np.abs(f3)]) > 12000:
            #             sum_Tijk[k] += Tijk[ij]
            a2.plot(pax1, sum_Tijk.real, 'k')
            a2.set_xlabel('Frequency [kHz]')
            a2.set_ylabel('Nonlinear transfer rate [a.u.]')
            a2.axhline(y=0, ls='--', color='k')
            if 'xlimits' in kwargs: a2.set_xlim([xlimits[0], xlimits[1]])

            im = a3.imshow(np.sign(Tijk.real)*np.log(np.abs(Tijk.real)), extent=(pax2.min(), pax2.max(), pax1.min(), pax1.max()), interpolation='none', aspect='equal', origin='lower', cmap=CM)
            a3.set_xlabel('Frequency [kHz]')
            a3.set_ylabel('Frequency [kHz]')
            a3.set_title('Nonlinear transfer function')
            divider = make_axes_locatable(a3)
            cax = divider.append_axes('right', size='5%', pad=0.05)
            fig.colorbar(im, cax=cax, orientation='vertical')

            plt.show()

        # check local coherency and gof
        if check == 1:
            Glin, Gquad, Glq = sp.local_coherency(XX, YY, Lk, Qijk, Aijk)
            pax1 = ax1/1000.0
            plt.plot(pax1, Glin, color='b', label='Glin')
            plt.plot(pax1, Gquad, color='r', label='Gquad')
            plt.plot(pax1, Glq, color='g', label='Glq')
            plt.plot(pax1, Glin + Gquad + Glq, color='k', label='total')
            plt.axhline(y = 1, ls='--', color='k')
            plt.xlim([0, pax1[-1]])
            plt.legend()
            plt.show()

            # use another XX and YY from different ensemble (data set) if want to check the model keeps working
            Gyy2 = sp.nonlinear_gof(self.Dlist[done+2].spdata[0,:,:], self.Dlist[dtwo+2].spdata[0,:,:], Lk, Qijk)
            plt.plot(pax1, Gyy2, color='k', label='goodness of fit')
            plt.xlim([0, pax1[-1]])
            plt.legend()
            plt.show()

        # save results
        Dtwo.val = gk # linear growth rate
        Dtwo.val2 = sum_Tijk.real # nonlinear transfer rate

############################# statistical methods ##############################

    def skplane(self, dnum=0, cnl='all', detrend=1, verbose=1, fig=None, axs=None, **kwargs):
        if 'ylimits' in kwargs: ylimits = kwargs['ylimits']
        if 'xlimits' in kwargs: xlimits = kwargs['xlimits']

        D = self.Dlist[dnum]

        if verbose == 1:
            fig, axs = make_axes(len(D.clist), ptype='mplot', fig=fig, axs=axs)

        D.vkind = 'skplane'

        cnum = len(D.clist)  # number of cmp channels

        if cnl == 'all': cnl = range(cnum)

        # data dimension
        D.skew = np.zeros(cnum)
        D.kurt = np.zeros(cnum)

        for i, c in enumerate(cnl):
            t = D.time
            x = D.data[c,:]

            D.skew[c] = st.skewness(x, detrend)
            D.kurt[c] = st.kurtosis(x, detrend)

            if verbose == 1:
                # plot info
                pshot = D.shot
                pname = D.clist[c]
                chpos = '({:.1f}, {:.1f})'.format(D.rpos[c]*100, D.zpos[c]*100) # [cm]

                axs[i].plot(t, x)

                axs[i].set_title('#{:d} {:s} {:s} \n S = {:.3f}, K = {:.3f}'.format(pshot, pname, chpos, D.skew[c], D.kurt[c]), fontsize=10)
                axs[i].set_xlabel('Time [s]')

        if verbose == 1:
            plt.show()

            fig = plt.figure(facecolor='w')
            plt.plot(D.skew[cnl], D.kurt[cnl], 'o')
            for i, c in enumerate(cnl):
                plt.annotate(D.clist[c], (D.skew[c], D.kurt[c]))

            sax = np.arange((D.skew[cnl]).min(), (D.skew[cnl]).max(), 0.001)
            kax = 3 + 3*sax**2/2 # parabolic relationship for exponential pulse and exponentially distributed pulse amplitudes [Garcia NME 2017]
            plt.plot(sax, kax, 'k')

            plt.xlabel('Skewness')
            plt.ylabel('Kurtosis')

            plt.show()

    def skewness(self, dnum=0, cnl='all', detrend=1, **kwargs):
        D = self.Dlist[dnum]

        D.vkind = 'skewness'

        cnum = len(D.clist)  # number of cmp channels

        if cnl == 'all': cnl = range(cnum)

        # data dimension
        D.val = np.zeros(cnum)

        for i, c in enumerate(cnl):
            x = D.data[c,:]

            D.val[c] = st.skewness(x, detrend)

    def kurtosis(self, dnum=0, cnl='all', detrend=1, **kwargs):
        D = self.Dlist[dnum]

        D.vkind = 'kurtosis'

        cnum = len(D.clist)  # number of cmp channels

        if cnl == 'all': cnl = range(cnum)

        # data dimension
        D.val = np.zeros(cnum)

        for i, c in enumerate(cnl):
            x = D.data[c,:]

            D.val[c] = st.kurtosis(x, detrend)

    def hurst(self, dnum=0, cnl='all', bins=30, detrend=1, fitrange=[10,1000], **kwargs):
        D = self.Dlist[dnum]

        D.vkind = 'hurst'

        pshot = D.shot
        cnum = len(D.clist)  # number of cmp channels

        if cnl == 'all': cnl = range(cnum)

        # axis
        bsize = int(1.0*len(D.time)/bins)
        ax = np.floor( 10**(np.arange(1.0, np.log10(bsize), 0.01)) )

        # data dimension
        D.ers = np.zeros((cnum, len(ax)))
        D.std = np.zeros((cnum, len(ax)))
        D.fit = np.zeros((cnum, len(ax)))
        D.val = np.zeros(cnum)

        for i, c in enumerate(cnl):
            pname = D.clist[c]
            t = D.time
            x = D.data[c,:]

            D.ax, D.ers[c,:], D.std[c,:], \
            D.val[c], D.fit[c,:] = st.hurst(t, x, bins, detrend, fitrange, **kwargs)

    def chplane(self, dnum=0, cnl='all', d=6, bins=1, rescale=0, verbose=1, fig=None, axs=None, **kwargs):
        # CH plane [Rosso PRL 2007]
        # chaotic : moderate C and H, above fBm
        # stochastic : low C and high H, below fBm
        if 'ylimits' in kwargs: ylimits = kwargs['ylimits']
        if 'xlimits' in kwargs: xlimits = kwargs['xlimits']

        D = self.Dlist[dnum]

        D.vkind = 'BP_probability'

        pshot = D.shot

        cnum = len(D.clist)  # number of cmp channels

        if cnl == 'all': cnl = range(cnum)

        nst = math.factorial(d) # number of possible states

        bsize = int(1.0*len(D.data[0,:])/bins)
        print('For an accurate estimation of the probability, bsize {:g} should be considerably larger than nst {:g}'.format(bsize, nst))

        # make axes
        if verbose == 1: fig, axs = make_axes(cnum, ptype='mplot', fig=fig, axs=axs, type='val')

        # data dimension
        D.pi = np.zeros((cnum, nst))
        D.std = np.zeros((cnum, nst))
        D.jscom = np.zeros(cnum)
        D.nsent = np.zeros(cnum)

        for i, c in enumerate(cnl):
            pname = D.clist[c]

            x = D.data[c,:]

            D.ax, D.pi[c,:], D.std[c,:] = st.bp_prob(x, d, bins)
            D.jscom[c], D.nsent[c] = st.ch_measure(D.pi[c,:])

            # plot BP probability
            if verbose == 1:
                pax = D.ax
                pdata = D.pi[c,:]

                axs[i].plot(pax, pdata, '-x')

                chpos = '({:.1f}, {:.1f})'.format(D.rpos[c]*100, D.zpos[c]*100) # [cm]
                axs[i].set_title('#{:d} \n {:s} {:s} \n C={:.3f}, H={:.3f}'.format(pshot, pname, chpos, D.jscom[c], D.nsent[c]), fontsize=8)
                axs[i].set_xlabel('Order number')
                axs[i].set_ylabel('BP probability')

                axs[i].set_yscale('log')
                # axs[i].margins(0.01)

        if verbose == 1: plt.show()

        # plot CH plane
        if verbose == 1:
            fig = plt.figure(facecolor='w')

            # get ch boundary lines (min, max, cen)
            h_min, c_min, h_max, c_max, h_cen, c_cen = st.ch_bdry(d)

            # rescale if necessary
            if rescale == 1:
                D.jscom[cnl] = st.complexity_rescale(D.nsent[cnl], D.jscom[cnl], h_min, c_min, h_max, c_max, h_cen, c_cen)
                plt.axhline(y=0, ls='--', color='r')
                plt.ylabel('Rescaled Complexity (C)')
            else:
                plt.plot(h_min, c_min, 'k')
                plt.plot(h_max, c_max, 'k')
                plt.plot(h_cen, c_cen, 'g')
                plt.ylabel('Complexity (C)')
            plt.xlabel('Entropy (H)')

            plt.plot(D.nsent[cnl], D.jscom[cnl], 'o')
            for i, c in enumerate(cnl):
                plt.annotate(D.clist[c], (D.nsent[c], D.jscom[c]))

            plt.show()

    def js_complexity(self, dnum=0, cnl='all', d=5, bins=1, **kwargs):
        D = self.Dlist[dnum]

        D.vkind = 'jscom'

        cnum = len(D.clist)  # number of cmp channels

        if cnl == 'all': cnl = range(cnum)

        nst = math.factorial(d) # number of possible states

        bsize = int(1.0*len(D.data[0,:])/bins)
        print('For an accurate estimation of the probability, bsize {:g} should be considerably larger than nst {:g}'.format(bsize, nst))

        # data dimension
        D.pi = np.zeros((cnum, nst))
        D.std = np.zeros((cnum, nst))
        D.val = np.zeros(cnum)

        for i, c in enumerate(cnl):
            x = D.data[c,:]

            D.ax, D.pi[c,:], D.std[c,:] = st.bp_prob(x, d, bins)
            D.val[c] = st.js_complexity(D.pi[c,:])

    def ns_entropy(self, dnum=0, cnl='all', d=5, bins=1, **kwargs):
        D = self.Dlist[dnum]

        D.vkind = 'nsent'

        cnum = len(D.clist)  # number of cmp channels

        if cnl == 'all': cnl = range(cnum)

        nst = math.factorial(d) # number of possible states

        bsize = int(1.0*len(D.data[0,:])/bins)
        print('For an accurate estimation of the probability, bsize {:g} should be considerably larger than nst {:g}'.format(bsize, nst))

        # data dimension
        D.pi = np.zeros((cnum, nst))
        D.std = np.zeros((cnum, nst))
        D.val = np.zeros(cnum)

        for i, c in enumerate(cnl):
            x = D.data[c,:]

            D.ax, D.pi[c,:], D.std[c,:] = st.bp_prob(x, d, bins)
            D.val[c] = st.ns_entropy(D.pi[c,:])

    def rescaled_complexity(self, dnum=0, cnl='all', d=5, bins=1, **kwargs):
        self.chplane(dnum=dnum, cnl=cnl, d=d, bins=bins, verbose=0)

        D = self.Dlist[dnum]

        cnum = len(D.clist)  # number of cmp channels

        if cnl == 'all': cnl = range(cnum)

        D.val = np.zeros(cnum)

        h_min, c_min, h_max, c_max, h_cen, c_cen = st.ch_bdry(d)
        D.val[cnl] = st.complexity_rescale(D.nsent[cnl], D.jscom[cnl], h_min, c_min, h_max, c_max, h_cen, c_cen)

    def intermittency(self, dnum=0, cnl='all', bins=20, overlap=0.2, qstep=0.3, fitrange=[20.0,100.0], verbose=1, **kwargs):
        # intermittency parameter from multi-fractal analysis [Carreras PoP 2000]
        # this ranges from 0 (mono-fractal) to 1
        # add D fitting later
        if 'ylimits' in kwargs: ylimits = kwargs['ylimits']
        if 'xlimits' in kwargs: xlimits = kwargs['xlimits']

        D = self.Dlist[dnum]

        D.vkind = 'intermittency'

        pshot = D.shot
        cnum = len(D.clist)  # number of cmp channels

        if cnl == 'all': cnl = range(len(D.clist))

        D.intmit = np.zeros(cnum)

        for i, c in enumerate(cnl):
            t = D.time
            x = D.data[c,:]

            D.intmit[c] = st.intermittency(t, x, bins, overlap, qstep, fitrange, verbose, **kwargs)

############################# calculation along time ###########################

    def tcal(self, done=0, dtwo=None, cnl='all', twin=0.005, tstep=0.001, vkind='rescaled_complexity', vpara=None, **kwargs):
        if 'ylimits' in kwargs: ylimits = kwargs['ylimits']
        if 'xlimits' in kwargs: xlimits = kwargs['xlimits']

        Done = self.Dlist[done]
        if dtwo != None:
            Dtwo = self.Dlist[dtwo]

        # tidx list for ax
        tidx_win = int(Done.fs*twin) # buffer window
        tidx_step = int(Done.fs*tstep)
        tidx_list = np.arange(int(tidx_win/2), len(Done.time)-int(tidx_win/2), tidx_step, dtype='int64')
        Done.ax = Done.time[tidx_list]

        cnum = len(Done.clist)  # number of cmp channels

        if cnl == 'all': cnl = range(cnum)

        if vkind == 'rescaled_complexity':
            Done.nsent = np.zeros((cnum, len(Done.ax)))
            Done.jscom = np.zeros((cnum, len(Done.ax)))
        elif vkind == 'peak_stat':
            Done.npeak = np.zeros((cnum, len(Done.ax)))
            Done.mprom = np.zeros((cnum, len(Done.ax)))

        Done.val  = np.zeros((cnum, len(Done.ax)))
        for i, c in enumerate(cnl):
            for j, tidx in enumerate(tidx_list):
                t1idx = tidx - int(tidx_win/2)
                t2idx = tidx + int(tidx_win/2)

                dy = Done.data[c,t1idx:t2idx]

                # pre-processing here
                if vkind in ['cross_power', 'coherence']:
                    dx = Dtwo.data[c,t1idx:t2idx]

                    if vpara['norm'] == 1:
                        dy = dy/np.mean(dy) - 1.0
                        dx = dx/np.mean(dx) - 1.0

                    fax, YY, win_factor = sp.fftbins(dy, 1.0/Done.fs, nfft=vpara['nfft'], window=vpara['window'], overlap=vpara['overlap'], detrend=vpara['detrend'], full=vpara['full'])
                    _, XX, _ = sp.fftbins(dx, 1.0/Done.fs, nfft=vpara['nfft'], window=vpara['window'], overlap=vpara['overlap'], detrend=vpara['detrend'], full=vpara['full'])

                # calculation here
                if vkind == 'rescaled_complexity':
                    _, pi, _ = st.bp_prob(dy, d=vpara['d'], bins=vpara['bins'])
                    Done.jscom[c,j], Done.nsent[c,j] = st.ch_measure(pi) # jscom and nsent
                elif vkind == 'cross_power':
                    fidx = (vpara['f1'] <= fax) & (fax <= vpara['f2'])
                    val = 2*sp.cross_power(XX, YY, win_factor)
                    Done.val[c,j] = np.sqrt(np.sum(val[fidx]))
                elif vkind == 'coherence':
                    fidx = (vpara['f1'] <= fax) & (fax <= vpara['f2'])
                    val = sp.coherence(XX, YY)
                    Done.val[c,j] = np.sum(val[fidx])
                elif vkind == 'peak_stat':
                    pidx, pp = signal.find_peaks(dy, height=(vpara['height_min'],None), width=(None,vpara['width_max']), prominence=(vpara['prom_min'],None))
                    if len(pidx) > 0:
                        Done.npeak[c,j] = len(pidx)
                        Done.mprom[c,j] = np.mean(pp["prominences"])

                print('tcal channel {:d}/{:d} time {:d}/{:d}'.format(c, len(cnl), j+1, len(tidx_list)))

            # post-processing here
            if vkind == 'rescaled_complexity':
                Done.val[c,:] = st.complexity_rescale(Done.nsent[c,:], Done.jscom[c,:], \
                    vpara['h_min'], vpara['c_min'], vpara['h_max'], vpara['c_max'], vpara['h_cen'], vpara['c_cen'])

        if dtwo != None:
            Dtwo.ax = Done.ax
            Dtwo.val = Done.val

        # print(tidx_win, tidx_step)

        # print(tidx_list)
        # print(tidx_list - int(tidx_win/2))
        # print(tidx_list + int(tidx_win/2))

        # print(Done.ax)
        # print(Done.time[tidx_list - int(tidx_win/2)])
        # print(Done.time[tidx_list + int(tidx_win/2)])

############################# default plot functions ###########################

    def mplot(self, dnum=1, cnl='all', type='time', fig=None, axs=None, show=1, **kwargs):
        if 'ylimits' in kwargs: ylimits = kwargs['ylimits']
        if 'xlimits' in kwargs: xlimits = kwargs['xlimits']

        D = self.Dlist[dnum]

        cnum = len(D.clist)  # number of cmp channels

        if cnl == 'all': cnl = range(cnum)

        fig, axs = make_axes(len(cnl), ptype='mplot', fig=fig, axs=axs, type=type)

        pshot = D.shot

        for i, c in enumerate(cnl):
            pname = D.clist[c]

            # set data
            if type == 'time':
                pbase = D.time
                pdata = D.data[c,:]
            elif type == 'val':
                vkind = D.vkind

                if hasattr(D, 'rname'):
                    rname = D.rname[c]
                else:
                    rname = ''

                # set data
                if vkind in ['skewness','kurtosis']:
                    pdata = D.data[c,:]
                elif vkind == 'hurst':
                    pdata = D.ers[c,:]
                elif vkind in ['jscom','nsent']:
                    pdata = D.pi[c,:]
                else:
                    pdata = D.val[c,:].real

                # set base
                if vkind in ['correlation','corr_coef']:
                    pbase = D.ax*1e6
                elif vkind in ['cross_power','coherence','cross_phase','bicoherence']:
                    pbase = D.ax/1000
                elif vkind in ['skewness','kurtosis']:
                    pbase = D.time
                else:
                    pbase = D.ax

            if type == 'time':
                axs[i].plot(pbase, pdata, label='#{:d}, {:s}, [{:g},{:g}]'.format(pshot, pname, D.time[0]*1000, D.time[-1]*1000))  # plot
            elif type == 'val':
                axs[i].plot(pbase, pdata, '-x', label='#{:d}, {:s}-{:s}, [{:g},{:g}]'.format(pshot, rname, pname, D.time[0]*1000, D.time[-1]*1000))  # plot
            if show:
                axs[i].legend(fontsize='x-small')

            # aux plot
            if type == 'val':
                if vkind == 'coherence':
                    axs[i].axhline(y=1/np.sqrt(D.bins), color='r')
                elif vkind == 'hurst':
                    axs[i].plot(pbase, D.fit[c,:], 'r')
                elif vkind in ['correlation','corr_coef']:
                    hdata = signal.hilbert(pdata)
                    axs[i].plot(pbase, np.abs(hdata), '--')

            # xy limits
            if 'ylimits' in kwargs:  # ylimits
                axs[i].set_ylim([ylimits[0], ylimits[1]])
            if 'xlimits' in kwargs:  # xlimits
                axs[i].set_xlim([xlimits[0], xlimits[1]])
            else:
                axs[i].set_xlim([pbase[0], pbase[-1]])

            # title
            chpos = '({:.1f}, {:.1f})'.format(D.rpos[c]*100, D.zpos[c]*100) # [cm]
            if type == 'time':
                axs[i].set_title('#{:d} \n {:s} {:s}'.format(pshot, pname, chpos), fontsize=8)
            elif type == 'val':
                if vkind in ['skewness','kurtosis','hurst','jscom','nsent']:
                    axs[i].set_title('#{:d} \n {:s} {:s} \n {:s} = {:g}'.format(pshot, pname, chpos, vkind, D.val[c]), fontsize=8)
                else:
                    axs[i].set_title('#{:d} \n {:s}-{:s} \n {:s}'.format(pshot, rname, pname, chpos), fontsize=8)

            # xy scale
            if type == 'val':
                if vkind in ['hurst']:
                    axs[i].set_xscale('log')
                if vkind in ['cross_power','hurst','jscom','nsent']:
                    axs[i].set_yscale('log')

            # xy label
            if type == 'time':
                axs[i].set_xlabel('Time [s]')
                axs[i].set_ylabel('Signal')
            elif type == 'val':
                if vkind in ['cross_power','coherence','cross_phase','bicoherence']:
                    axs[i].set_xlabel('Frequency [kHz]')
                    axs[i].set_ylabel(vkind)
                elif vkind == 'hurst':
                    axs[i].set_xlabel('Time lag [us]')
                    axs[i].set_ylabel('R/S')
                elif vkind in ['jscom','nsent']:
                    axs[i].set_xlabel('order number')
                    axs[i].set_ylabel('BP probability')
                elif vkind in ['correlation','corr_coef']:
                    axs[i].set_xlabel('Time lag [us]')
                    axs[i].set_ylabel(vkind)
                else:
                    axs[i].set_xlabel('Time [s]')
                    axs[i].set_ylabel('Signal')

        if show == 1:
            # fig.tight_layout(w_pad=0.3, h_pad=0.3) # not working properly in OMFIT. :(
            plt.show()

        return fig, axs

    def cplot(self, dnum, snum=0, frange=[0, 100], vlimits=None, fig=None, axs=None, show=1, **kwargs):
        if 'ylimits' in kwargs: ylimits = kwargs['ylimits']
        if 'xlimits' in kwargs: xlimits = kwargs['xlimits']
        # calculate summed coherence image
        # or cross power rms image
        # or group velocity image

        D = self.Dlist[dnum]

        vkind = D.vkind

        # sample plot data
        if vkind in ['cross_power','coherence','cross_phase','bicoherence']:
            # axis
            sbase = D.ax/1000  # [kHz]

            # fidx
            idx = np.where((sbase >= frange[0])*(sbase <= frange[1]))
            idx1 = int(idx[0][0])
            idx2 = int(idx[0][-1]+1)
            sdata = D.val[snum,:]
        elif vkind == 'hurst':
            sbase = D.ax
            sdata = D.ers[snum,:]
        elif vkind in ['jscom','nsent']:
            sbase = D.ax
            sdata = D.pi[snum,:]
        else:
            sbase = D.time
            sdata = D.data[snum,:]

        # calculate pdata
        if vkind == 'cross_power':  # rms
            pdata = np.sqrt(np.sum(D.val[:,idx1:idx2], 1))
        elif vkind == 'coherence':  # summed coherence
            pdata = np.sum(D.val[:,idx1:idx2], 1)
        elif vkind == 'cross_phase':  # phase velocity
            cnum = len(D.val)
            base = D.ax[idx1:idx2]  # [Hz]
            pdata = np.zeros(cnum)
            phase_v = np.zeros((cnum, len(base)-1))
            for c in range(cnum):
                data = D.val[c,idx1:idx2]
                # pfit = np.polyfit(base, data, 1)
                # fitdata = np.polyval(pfit, base)
                # chisq = np.sum((data - fitdata)**2)
                # if c == snum:
                #     fbase = base/1000  # [kHz]
                #     fdata = fitdata
                # pdata[c] = 2*np.pi*D.dist[c]/pfit[0]/1000.0  # [km/s]
                phase_v[c,:] = 2*np.pi*D.dist[c]/(data[1:]/base[1:])/1000.0 # [km/s]
                pdata[c] = np.mean(phase_v[c,:])
            D.phase_v = phase_v
        else:
            pdata = D.val

        # remove not finite values
        pidx = np.isfinite(pdata)
        pdata[~pidx] = 0

        # save results
        D.pdata = pdata

        if show == 1:
            # make axes
            fig, axs = make_axes(len(D.clist), ptype='cplot', fig=fig, axs=axs)

            # get position to plot
            rpos = D.rpos[:]
            zpos = D.zpos[:]

            # sample plot
            axs[0].plot(sbase, sdata)  # ax1.hold(True)
            # if vkind == 'cross_phase':
            #     axs[0].plot(fbase, fdata)
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
            if vlimits == None:
                vlimits = [np.mean(pdata) - np.std(pdata), np.mean(pdata) + np.std(pdata)]
            sc = axs[1].scatter(rpos, zpos, 250, pdata, marker='s', vmin=vlimits[0], vmax=vlimits[1], cmap=CM)
            axs[1].set_aspect('equal')
            axs[1].margins(0.01)

            # color bar
            fig.colorbar(sc, cax=axs[2])

            axs[1].set_xlabel('R [m]')
            axs[1].set_ylabel('z [m]')
            if vkind == 'cross_power':
                axs[1].set_title('RMS')
            elif vkind == 'coherence':
                axs[1].set_title('Coherence sum')
            elif vkind == 'cross_phase':
                axs[1].set_title('Phase velocity [km/s]')
            else:
                axs[1].set_title(vkind)

            # fig.tight_layout(w_pad=0.3, h_pad=0.3) # not working properly in OMFIT. :(
            plt.show()

    def spec(self, dnum=0, cnl=[0], nfft=512, **kwargs):
        if 'flimits' in kwargs: flimits = kwargs['flimits']
        if 'xlimits' in kwargs: xlimits = kwargs['xlimits']

        fs = self.Dlist[dnum].fs
        nov = nfft*0.9

        for c in cnl:
            pshot = self.Dlist[dnum].shot
            pname = self.Dlist[dnum].clist[c]
            pbase = self.Dlist[dnum].time
            pdata = self.Dlist[dnum].data[c,:]

            pxx, freq, time, cax = plt.specgram(pdata, NFFT=nfft, Fs=fs, noverlap=nov,
                                                xextent=[pbase[0], pbase[-1]], cmap=CM, detrend='mean')  # spectrum

            maxP = math.log(np.amax(pxx),10)*10
            minP = math.log(np.amin(pxx),10)*10
            dP = maxP - minP
            plt.clim([minP+dP*0.55, maxP])
            plt.colorbar(cax)

            if 'flimits' in kwargs:  # flimits
                plt.ylim([flimits[0]*1000, flimits[1]*1000])
            if 'xlimits' in kwargs:  # xlimits
                plt.xlim([xlimits[0], xlimits[1]])
            else:
                plt.xlim([pbase[0], pbase[-1]])

            plt.title(pname, fontsize=10)  # labeling
            plt.xlabel('Time [s]')
            plt.ylabel('Frequency [Hz]')

            plt.show()

    def iplot(self, dnum, snum=0, c=None, type='time', vlimits=[-0.1, 0.1], istep=0.002, imethod='cubic', bcut=0.03, pmethod='contour', cline=False, fig=None, axs=None, **kwargs):
        # keyboard interactive image plot
        D = self.Dlist[dnum]

        if type == 'time':
            pbase = D.time
        elif type == 'val':
            pbase = D.ax*1e+6
            vkind = D.vkind

        CM = plt.cm.get_cmap('RdYlBu_r')

        if c == None:
            c = int(input('automatic, mouse input, text input [0, 1, 2]: '))
        tidx1 = 0  # starting index
        if c == 0:
            # make axes
            fig, axs = make_axes(len(D.clist), ptype='iplot', fig=fig, axs=axs)

            tstep = int(input('time step [idx]: '))  # jumping index # tstep = 10
            for tidx in range(tidx1, len(pbase), tstep):
                # take data and channel position
                if type == 'time':
                    pdata = D.data[:,tidx]
                    psample = D.data[snum,:]
                elif type == 'val':
                    pdata = D.val[:,tidx]
                    psample = D.val[snum,:]
                rpos = D.rpos[:]
                zpos = D.zpos[:]

                # fill bad channel
                pdata = ms.fill_bad_channel(pdata, rpos, zpos, D.good_channels, bcut)

                # interpolation
                if istep > 0:
                    ri, zi, pi = ms.interp_pdata(pdata, rpos, zpos, istep, imethod)

                # plot
                axs[0].cla()
                axs[1].cla()
                axs[2].cla()
                plt.ion()

                axs[0].plot(pbase, psample)  # ax1.hold(True)
                axs[0].axvline(x=pbase[tidx], color='g')
                if istep > 0:
                    if pmethod == 'scatter':
                        im = axs[1].scatter(ri.ravel(), zi.ravel(), 5, pi.ravel(), marker='s', vmin=vlimits[0], vmax=vlimits[1], cmap=CM, edgecolors='none')
                    elif pmethod == 'contour':
                        if cline: axs[1].contour(ri, zi, pi, 50, vmin=vlimits[0], vmax=vlimits[1], linewidths=0.5, linestyles=cline, colors='k')
                        im = axs[1].contourf(ri, zi, pi, 50, vmin=vlimits[0], vmax=vlimits[1], cmap=CM)
                else:
                    im = axs[1].scatter(rpos, zpos, 500, pdata, marker='s', vmin=vlimits[0], vmax=vlimits[1], cmap=CM, edgecolors='none')
                axs[1].set_aspect('equal')
                plt.colorbar(im, cax=axs[2])

                axs[1].set_xlabel('R [m]')
                axs[1].set_ylabel('z [m]')
                if type == 'time':
                    axs[0].set_xlabel('Time [s]')
                    axs[1].set_title('ECE image at t = {:g} sec'.format(pbase[tidx]))
                elif type == 'val':
                    axs[0].set_xlabel('Time lag [us]')
                    axs[1].set_title('{:s} image at time lag = {:g} us'.format(vkind, pbase[tidx]))

                plt.show()
                plt.pause(0.1)

            plt.ioff()
            plt.close()

        elif c > 0:
            tidx = tidx1
            if c == 1:
                print('Select a point in the top axes to plot the image')

            # make axes
            fig, axs = make_axes(len(D.clist), ptype='iplot', fig=fig, axs=axs)

            while True:
                # take data and channel position
                if type == 'time':
                    pdata = D.data[:,tidx]
                    psample = D.data[snum,:]
                elif type == 'val':
                    pdata = D.val[:,tidx]
                    psample = D.val[snum,:]
                rpos = D.rpos[:]
                zpos = D.zpos[:]

                # fill bad channel
                pdata = ms.fill_bad_channel(pdata, rpos, zpos, D.good_channels, bcut)

                # interpolation
                if istep > 0:
                    ri, zi, pi = ms.interp_pdata(pdata, rpos, zpos, istep, imethod)

                # plot
                axs[0].cla()
                axs[1].cla()
                axs[2].cla()
                plt.ion()

                axs[0].plot(pbase, psample)  # ax1.hold(True)
                axs[0].axvline(x=pbase[tidx], color='g')
                if istep > 0:
                    if pmethod == 'scatter':
                        im = axs[1].scatter(ri.ravel(), zi.ravel(), 5, pi.ravel(), marker='s', vmin=vlimits[0], vmax=vlimits[1], cmap=CM, edgecolors='none')
                    elif pmethod == 'contour':
                        if cline: axs[1].contour(ri, zi, pi, 50, vmin=vlimits[0], vmax=vlimits[1], linewidths=0.5, linestyles=cline, colors='k')
                        im = axs[1].contourf(ri, zi, pi, 50, vmin=vlimits[0], vmax=vlimits[1], cmap=CM)
                else:
                    im = axs[1].scatter(rpos, zpos, 500, pdata, marker='s', vmin=vlimits[0], vmax=vlimits[1], cmap=CM, edgecolors='none')
                axs[1].set_aspect('equal')
                plt.colorbar(im, cax=axs[2])

                axs[1].set_xlabel('R [m]')
                axs[1].set_ylabel('z [m]')
                if type == 'time':
                    axs[0].set_xlabel('Time [s]')
                    axs[1].set_title('ECE image at t = {:g} sec'.format(pbase[tidx]))
                elif type == 'val':
                    axs[0].set_xlabel('Time lag [us]')
                    axs[1].set_title('{:s} image at time lag = {:g} us'.format(vkind, pbase[tidx]))

                plt.show()

                # mouse or text input
                if c == 1:
                    g = plt.ginput(1)[0][0]
                elif c == 2:
                    g = float(input('X value to plot: '))
                    plt.draw()

                if g >= pbase[0] and g <= pbase[-1]:
                    tidx = np.where(pbase + 1e-10 >= g)[0][0]
                else:
                    print('Out of the time range')
                    plt.ioff()
                    plt.close()
                    break

                plt.ioff()
            plt.close()

        D.pdata = pdata

############################# test functions ###################################

    def fftbins_bicoh_test(self, nfft, window, overlap, detrend=0, full=1):
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
                    fb = 50*1000
                    fc = 90*1000
                    fd = fb + fc

                    pb = pbs[b]
                    pc = pcs[b]
                    pd = pds[b] # non-coherent case
                    # pd = pb + pc # coherent case

                    sx = np.cos(2*np.pi*fb*st + pb) + np.cos(2*np.pi*fc*st + pc) + np.cos(2*np.pi*fd*st + pd) + 0.5*np.random.randn(len(sx))
                    # sx = 3*np.random.randn(len(sx))
                    # sx = np.cos(2*np.pi*fb*st + pb) + np.cos(2*np.pi*fc*st + pc) + 1/2*np.cos(2*np.pi*fd*st + pd) + 1/2*np.random.randn(len(sx))
                    # sx = sx + np.cos(2*np.pi*fb*st + pb)*np.cos(2*np.pi*fc*st + pc) # +,- coupling

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
                self.Dlist[dnum].nfreq = nfft + 1
            else:
                self.Dlist[dnum].nfreq = nfft
            self.Dlist[dnum].window = window
            self.Dlist[dnum].overlap = overlap
            self.Dlist[dnum].detrend = detrend
            self.Dlist[dnum].bins = bins
            self.Dlist[dnum].win_factor = np.mean(win**2)

            print('TEST :: dnum {:d} fftbins {:d} with {:s} size {:d} overlap {:g} detrend {:d} full {:d}'.format(dnum, bins, window, nfft, overlap, detrend, full))

# def expand_clist(clist):
#     # IN : List of channel names (e.g. 'ECEI_G1201-1208' or 'ECEI_GT1201-1208').
#     # OUT : Expanded list (e.g. 'ECEI_G1201', ..., 'ECEI_G1208')

#     # KSTAR ECEI
#     exp_clist = []
#     for c in range(len(clist)):
#         if 'ECEI' in clist[c] and len(clist[c]) == 15: # before 2018
#             vi = int(clist[c][6:8])
#             fi = int(clist[c][8:10])
#             vf = int(clist[c][11:13])
#             ff = int(clist[c][13:15])

#             for v in range(vi, vf+1):
#                 for f in range(fi, ff+1):
#                     exp_clist.append(clist[c][0:6] + '{:02d}{:02d}'.format(v, f))
#         elif 'ECEI' in clist[c] and len(clist[c]) == 16: # since 2018
#             vi = int(clist[c][7:9])
#             fi = int(clist[c][9:11])
#             vf = int(clist[c][12:14])
#             ff = int(clist[c][14:16])

#             for v in range(vi, vf+1):
#                 for f in range(fi, ff+1):
#                     exp_clist.append(clist[c][0:7] + '{:02d}{:02d}'.format(v, f))
#         else:
#             exp_clist.append(clist[c])
#     clist = exp_clist

#     return clist


def nextpow2(i):
    n = 1
    while n < i: n *= 2
    return n


def make_axes(cnum, ptype='mplot', maxcol=8, fig=None, axs=None, type='time'):
    # plot dimension
    if cnum < maxcol:
        col = cnum
    else:
        col = maxcol
    row = math.ceil(cnum/col)

    if ptype == 'mplot':
        if fig == None:
            fig = plt.figure(facecolor='w', figsize=(4+col*2,row*3))

        if axs == None:
            axs = ['']*cnum
            for i in range(cnum):
                if i == 0:
                    axs[i] = fig.add_subplot(row,col,i+1)
                    fig.subplots_adjust(left = 0.15-0.1/8*col, right = 0.9+0.07/8*col, bottom = 0.2-0.04*min(4,row), top = 0.8+0.04*min(4,row), hspace = 0.6, wspace = 0.4)
                    if type == 'time':
                        axprops = dict(sharex = axs[i])
                    else:
                        axprops = dict(sharex = axs[i], sharey = axs[i])
                else:
                    axs[i] = fig.add_subplot(row,col,i+1, **axprops)

    elif ptype == 'cplot' or 'iplot':
        if fig == None:
            fig = plt.figure(facecolor='w', figsize=(5,5+int(5/23*row)))

        if axs == None:
            ax0 = fig.add_axes([0.16, 0.82, 0.65, 0.15])  # [left bottom width height]
            ax1 = fig.add_axes([0.16, 0.1, 0.65, 0.58])
            ax2 = fig.add_axes([0.85, 0.25, 0.03, 0.28])
            axs = [ax0, ax1, ax2]

    return fig, axs
