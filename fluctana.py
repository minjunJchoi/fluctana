# Author : Minjun J. Choi (mjchoi@nfri.re.kr)
#
# Description : This code calculates cross power, coherence, cross phase, etc with fusion plasma diagnostics data
#
# Acknowledgement : Dr. S. Zoletnik and Prof. Y.-c. Ghim
#
# Last updated
#  2018.03.23 : version 0.10; even nfft -> odd nfft (for symmetry)


from scipy import signal
import math
import itertools

from kstarecei import *
from kstarmir import *
from kstarmds import *
#from diiiddata import *  # needs pidly

from filtdata import FiltData

import matplotlib.pyplot as plt

#CM = plt.cm.get_cmap('RdYlBu_r')
#CM = plt.cm.get_cmap('spectral')
#CM = plt.cm.get_cmap('YlGn')
CM = plt.cm.get_cmap('jet')


class FluctAna(object):
    def __init__(self):
        self.Dlist = []

    def add_data(self, D, trange, norm=1, atrange=[1.0, 1.01], res=0):

        D.get_data(trange, norm=norm, atrange=atrange, res=res)
        self.Dlist.append(D)

    def del_data(self, dnum):
        del self.Dlist[dnum]

    def list_data(self):
        for i in range(len(self.Dlist)):
            print('---- DATA SET # {:d} for [{:g}, {:g}] s ----'.format(i, self.Dlist[i].trange[0], self.Dlist[i].trange[1]))
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

############################# fft spectral methods #############################

    def fftbins(self, nfft, window, overlap, detrend, full=0):
        # IN : self, data set number, nfft, window name, detrend or not
        # OUT : bins x N FFT of time series data; frequency axis

        # self.list_data()

        for d, D in enumerate(self.Dlist):
            # get bins and window function
            tnum = len(D.data[0,:])
            bins, win = fft_window(tnum, nfft, window, overlap)

            # make an x-axis #
            dt = D.time[1] - D.time[0]  # time step
            ax = np.fft.fftfreq(nfft, d=dt) # full 0~fN -fN~-f1
            if np.mod(nfft, 2) == 0:  # even nfft
                ax = np.hstack([ax[0:int(nfft/2)], -(ax[int(nfft/2)]), ax[int(nfft/2):nfft]])
            if full == 1: # full shift to -fN ~ 0 ~ fN
                ax = np.fft.fftshift(ax)
            else: # half 0~fN
                ax = ax[0:int(nfft/2+1)]
            D.ax = ax

            # make fftdata
            cnum = len(D.data)
            if full == 1: # full shift to -fN ~ 0 ~ fN
                if np.mod(nfft, 2) == 0:  # even nfft
                    D.fftdata = np.zeros((cnum, bins, nfft+1), dtype=np.complex_)
                else:  # odd nfft
                    D.fftdata = np.zeros((cnum, bins, nfft), dtype=np.complex_)
            else: # half 0 ~ fN
                D.fftdata = np.zeros((cnum, bins, int(nfft/2+1)), dtype=np.complex_)

            for c in range(cnum):
                x = D.data[c,:]

                for b in range(bins):
                    idx1 = int(b*np.fix(nfft*(1 - overlap)))
                    idx2 = idx1 + nfft

                    sx = x[idx1:idx2]

                    if detrend == 1:
                        sx = signal.detrend(sx, type='linear')
                    sx = signal.detrend(sx, type='constant')  # subtract mean

                    sx = sx * win  # apply window function

                    # get fft
                    fftdata = np.fft.fft(sx, n=nfft)/nfft  # divide by the length
                    if np.mod(nfft, 2) == 0:  # even nfft
                        fftdata = np.hstack([fftdata[0:int(nfft/2)], np.conj(fftdata[int(nfft/2)]), fftdata[int(nfft/2):nfft]])
                    if full == 1: # shift to -fN ~ 0 ~ fN
                        fftdata = np.fft.fftshift(fftdata)
                    else: # half 0 ~ fN
                        fftdata = fftdata[0:int(nfft/2+1)]
                    D.fftdata[c,b,:] = fftdata

            # update attributes
            if np.mod(nfft, 2) == 0:
                D.nfft = nfft + 1
            else:
                D.nfft = nfft
            D.window = window
            D.overlap = overlap
            D.detrend = detrend
            D.bins = bins
            D.win = win

            print('dnum {:d} fftbins {:d} with {:s} size {:d} overlap {:g} detrend {:d} full {:d}'.format(d, bins, window, nfft, overlap, detrend, full))

    def cross_power(self, done=0, dtwo=1):
        # IN : data number one (ref), data number two (cmp), etc
        # OUT : x-axis (ax), y-axis (val)

        self.Dlist[dtwo].vkind = 'cross_power'

        rnum = len(self.Dlist[done].data)  # number of ref channels
        cnum = len(self.Dlist[dtwo].data)  # number of cmp channels
        bins = self.Dlist[dtwo].bins  # number of bins
        win_factor = np.mean(self.Dlist[dtwo].win**2)  # window factors

        # reference channel names
        self.Dlist[dtwo].rname = []

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
                    X = self.Dlist[done].fftdata[0,b,:]
                else:  # number of ref channels = number of cmp channels
                    X = self.Dlist[done].fftdata[c,b,:]

                Y = self.Dlist[dtwo].fftdata[c,b,:]

                if self.Dlist[dtwo].ax[1] < 0: # full range
                    val[b,:] = X*np.matrix.conjugate(Y) / win_factor
                else: # half
                    val[b,:] = 2*X*np.matrix.conjugate(Y) / win_factor  # product 2 for half return

            # average over bins
            Pxy = np.mean(val, 0)
            # result saved in val
            self.Dlist[dtwo].val[c,:] = np.abs(Pxy).real
            # std saved in std

    def coherence(self, done=0, dtwo=1):
        # IN : data number one (ref), data number two (cmp), etc
        # OUT : x-axis (ax), y-axis (val)

        self.Dlist[dtwo].vkind = 'coherence'

        rnum = len(self.Dlist[done].data)  # number of ref channels
        cnum = len(self.Dlist[dtwo].data)  # number of cmp channels
        bins = self.Dlist[dtwo].bins  # number of bins

        # reference channel names
        self.Dlist[dtwo].rname = []

        # value dimension
        val = np.zeros((bins, len(self.Dlist[dtwo].ax)), dtype=np.complex_)
        self.Dlist[dtwo].val = np.zeros((cnum, len(self.Dlist[dtwo].ax)))

        # calculation loop for multi channels
        for c in range(cnum):
            # reference channel names
            if rnum == 1:
                self.Dlist[dtwo].rname.append(self.Dlist[done].clist[0])
            else:
                self.Dlist[dtwo].rname.append(self.Dlist[done].clist[c])

            # calculate cross power for each channel and each bins
            for b in range(bins):
                if rnum == 1:  # single reference channel
                    X = self.Dlist[done].fftdata[0,b,:]
                else:  # number of ref channels = number of cmp channels
                    X = self.Dlist[done].fftdata[c,b,:]

                Y = self.Dlist[dtwo].fftdata[c,b,:]

                Pxx = X * np.matrix.conjugate(X)
                Pyy = Y * np.matrix.conjugate(Y)

                val[b,:] = X*np.matrix.conjugate(Y) / np.sqrt(Pxx*Pyy)
                # saturated data gives zero Pxx!!

            # average over bins
            Gxy = np.mean(val, 0)
            # results saved in val
            self.Dlist[dtwo].val[c,:] = np.abs(Gxy).real

    def cross_phase(self, done=0, dtwo=1):
        # IN : data number one (ref), data number two (cmp)
        # OUT : x-axis (ax), y-axis (val)

        self.Dlist[dtwo].vkind = 'cross_phase'

        rnum = len(self.Dlist[done].data)  # number of ref channels
        cnum = len(self.Dlist[dtwo].data)  # number of cmp channels
        bins = self.Dlist[dtwo].bins  # number of bins

        # reference channel names
        self.Dlist[dtwo].rname = []

        # distance
        self.Dlist[dtwo].dist = np.zeros(cnum)

        # value dimension
        val = np.zeros((bins, len(self.Dlist[dtwo].ax)), dtype=np.complex_)
        self.Dlist[dtwo].val = np.zeros((cnum, len(self.Dlist[dtwo].ax)))

        # calculation loop for multi channels
        for c in range(cnum):
            # reference channel number and distance between ref and cmp channels
            if rnum == 1:
                self.Dlist[dtwo].rname.append(self.Dlist[done].clist[0])
                self.Dlist[dtwo].dist[c] = np.sqrt((self.Dlist[dtwo].rpos[c] - self.Dlist[done].rpos[0])**2 + \
                (self.Dlist[dtwo].zpos[c] - self.Dlist[done].zpos[0])**2)
            else:
                self.Dlist[dtwo].rname.append(self.Dlist[done].clist[c])
                self.Dlist[dtwo].dist[c] = np.sqrt((self.Dlist[dtwo].rpos[c] - self.Dlist[done].rpos[c])**2 + \
                (self.Dlist[dtwo].zpos[c] - self.Dlist[done].zpos[c])**2)

            # calculate cross power for each channel and each bins
            for b in range(bins):
                if rnum == 1:  # single reference channel
                    X = self.Dlist[done].fftdata[0,b,:]
                else:  # number of ref channels = number of cmp channels
                    X = self.Dlist[done].fftdata[c,b,:]

                Y = self.Dlist[dtwo].fftdata[c,b,:]

                val[b,:] = X*np.matrix.conjugate(Y)

            # average over bins
            Pxy = np.mean(val, 0)
            # result saved in val
            self.Dlist[dtwo].val[c,:] = np.arctan2(Pxy.imag, Pxy.real).real
            # std saved in std

    def correlation(self, done=0, dtwo=1):
        # reguire full FFT
        # positive time lag = cmp is fater than ref (delay in ref)
        self.Dlist[dtwo].vkind = 'correlation'

        rnum = len(self.Dlist[done].data)  # number of ref channels
        cnum = len(self.Dlist[dtwo].data)  # number of cmp channels
        bins = self.Dlist[dtwo].bins  # number of bins
        nfft = self.Dlist[dtwo].nfft
        win_factor = np.mean(self.Dlist[dtwo].win**2)  # window factors

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
                    X = self.Dlist[done].fftdata[0,b,:]
                else:  # number of ref channels = number of cmp channels
                    X = self.Dlist[done].fftdata[c,b,:]

                Y = self.Dlist[dtwo].fftdata[c,b,:]

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
        # positive time lag = cmp is fater than ref (delay in ref)
        self.Dlist[dtwo].vkind = 'corr_coef'

        rnum = len(self.Dlist[done].data)  # number of ref channels
        cnum = len(self.Dlist[dtwo].data)  # number of cmp channels
        bins = self.Dlist[dtwo].bins  # number of bins
        nfft = self.Dlist[dtwo].nfft
        win_factor = np.mean(self.Dlist[dtwo].win**2)  # window factors

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
                    X = self.Dlist[done].fftdata[0,b,:]
                else:  # number of ref channels = number of cmp channels
                    X = self.Dlist[done].fftdata[c,b,:]

                Y = self.Dlist[dtwo].fftdata[c,b,:]

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
        win_factor = np.mean(self.Dlist[dtwo].win**2)  # window factors

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
            for b in range(bins):
                X = self.Dlist[done].fftdata[c,b,:]
                Y = self.Dlist[dtwo].fftdata[c,b,:]

                if pfreq[1] < 0:
                    Pxy = X*np.matrix.conjugate(Y) / win_factor
                else:
                    Pxy = 2*X*np.matrix.conjugate(Y) / win_factor  # product 2 for half return

                pdata[b,:] = np.abs(Pxy).real

            pdata = np.log10(np.transpose(pdata))

            maxP = np.amax(pdata)
            minP = np.amin(pdata)
            dP = maxP - minP

            # thresholding
            pdata[(pdata < minP + dP*thres)] = -100

            plt.imshow(pdata, extent=(ptime.min(), ptime.max(), pfreq.min(), pfreq.max()), interpolation='none', aspect='auto', origin='lower')

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
        win_factor = np.mean(self.Dlist[dtwo].win**2)  # window factors

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
            print(self.Dlist[dtwo].rname[c], self.Dlist[dtwo].clist[c])

            # calculate auto power and cross phase (wavenumber)
            for b in range(bins):
                X = self.Dlist[done].fftdata[c,b,:]
                Y = self.Dlist[dtwo].fftdata[c,b,:]

                Pxx[b,:] = X*np.matrix.conjugate(X) / win_factor
                Pyy[b,:] = Y*np.matrix.conjugate(Y) / win_factor
                Pxy = X*np.matrix.conjugate(Y)
                Kxy[b,:] = np.arctan2(Pxy.imag, Pxy.real).real / (self.Dlist[dtwo].dist[c]*100) # [cm^-1]

                # calculate SKw
                for w in range(nfft):
                    idx = (Kxy[b,w] - kstep/2 < kax) * (kax < Kxy[b,w] + kstep/2)
                    val[c,:,w] = val[c,:,w] + (1/bins*(Pxx[b,w] + Pyy[b,w])/2) * idx

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

    def bicoherence(self, done=0, dtwo=1, **kwargs):
        # fftbins full = 1
        # number of cmp channels = number of ref channels
        self.Dlist[dtwo].vkind = 'bicoherence'

        cnum = len(self.Dlist[dtwo].data)  # number of cmp channels
        nfft = self.Dlist[dtwo].nfft
        bins = self.Dlist[dtwo].bins  # number of bins

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
        ax2 = ax2[0:int(nfft/2+1)] # half 0 ~ fN

        # value dimension
        self.Dlist[dtwo].val = np.zeros((cnum, len(ax1), len(ax2)))

        # calculation loop for multi channels
        for c in range(cnum):
            # reference channel name
            rname = self.Dlist[done].clist[c]
            self.Dlist[dtwo].rname.append(rname)
            # cmp channel name
            pname = self.Dlist[dtwo].clist[c]

            # calculate bicoherence
            B = np.zeros((len(ax1), len(ax2)), dtype=np.complex_)
            P12 = np.zeros((len(ax1), len(ax2)))
            P3 = np.zeros((len(ax1), len(ax2)))

            for b in range(bins):
                X = self.Dlist[done].fftdata[c,b,:] # full -fN ~ fN
                Y = self.Dlist[dtwo].fftdata[c,b,:] # full -fN ~ fN

                Xhalf = np.fft.ifftshift(X) # full 0 ~ fN, -fN ~ -f1
                Xhalf = Xhalf[0:int(nfft/2+1)] # half 0 ~ fN

                X1 = np.transpose(np.tile(X, (len(ax2), 1)))
                X2 = np.tile(Xhalf, (len(ax1), 1))
                X3 = np.zeros((len(ax1), len(ax2)), dtype=np.complex_)
                for j in range(len(ax2)):
                    if j == 0:
                        X3[0:, j] = Y[j:]
                    else:
                        X3[0:(-j), j] = Y[j:]

                B = B + X1 * X2 * np.matrix.conjugate(X3) / bins #  complex bin average
                P12 = P12 + (np.abs(X1 * X2).real)**2 / bins # real average
                P3 = P3 + (np.abs(X3).real)**2 / bins # real average

            # self.Dlist[dtwo].val[c,:,:] = np.log10(np.abs(B)**2) # bispectrum
            self.Dlist[dtwo].val[c,:,:] = (np.abs(B)**2) / P12 / P3 # bicoherence

            # set axes
            if c == 0:
                plt.subplots_adjust(hspace = 0.5, wspace = 0.3)
                axes1 = plt.subplot(row,col,c+1)
                axprops = dict(sharex = axes1, sharey = axes1)
            else:
                plt.subplot(row,col,c+1, **axprops)

            pshot = self.Dlist[dtwo].shot
            pax1 = ax1/1000 # [kHz]
            pax2 = ax2/1000 # [kHz]
            pdata = self.Dlist[dtwo].val[c,:,:]

            plt.imshow(pdata, extent=(pax2.min(), pax2.max(), pax1.min(), pax1.max()), interpolation='none', aspect='equal', origin='lower')

            plt.colorbar()

            chpos = '({:.1f}, {:.1f})'.format(self.Dlist[dtwo].rpos[c]*100, self.Dlist[dtwo].zpos[c]*100) # [cm]
            plt.title('#{:d}, {:s}-{:s} {:s}'.format(pshot, rname, pname, chpos), fontsize=10)
            plt.xlabel('F1 [kHz]')
            plt.ylabel('F2 [kHz]')

            plt.show()

############################# wavelet spectral methods #########################

    def cwt(self, df): ## problem in recovering the signal
        for d, D in enumerate(self.Dlist):
            # make a t-axis
            dt = D.time[1] - D.time[0]  # time step
            tnum = len(D.time)
            nfft = nextpow2(tnum) # power of 2
            t = np.arange(nfft)*dt

            # make a f-axis
            fs = round(1/(dt)/1000)*1000.0 # [Hz]
            s0 = 2/fs # the smallest scale
            fmin = 0 # fmin
            f = np.sign(df) * np.arange(fmin, 1.0/(1.03*s0), np.abs(df))

            # scales
            old_settings = np.seterr(divide='ignore')
            sj = 1.0/(1.03*f)
            np.seterr(**old_settings)
            dj = np.log2(sj/s0) / np.arange(len(sj)) # dj
            dj[0] = 0 # remove infinity point due to fmin = 0

            # Morlet wavelet function (unnormalized)
            omega0 = 6.0 # nondimensional wavelet frequency
            ts = np.sqrt(2)*np.abs(sj) # e-folding time for Morlet wavelet with omega0 = 6
            wf0 = lambda eta: np.pi**(-1.0/4) * np.exp(1.0j*omega0*eta) * np.exp(-1.0/2*eta**2)

            cnum = len(D.data)  # number of cmp channels
            # value dimension
            D.cwtdata = np.zeros((cnum, tnum, len(sj)), dtype=np.complex_)
            for c in range(cnum):
                x = D.data[c,:]

                # FFT of signal
                X = np.fft.fft(x, n=nfft)/nfft

                # calculate
                Wns = np.zeros((nfft, len(sj)), dtype=np.complex_)
                for j, s in enumerate(sj):
                    # nondimensional time axis at scale s
                    eta = t/s
                    # FFT of wavelet function with normalization
                    WF = np.fft.fft(wf0(eta - np.mean(eta))*np.abs(dt/s)) / nfft
                    # Wavelet transform at scae s for all n time
                    Wns[:,j] = np.fft.fftshift(np.fft.ifft(X * WF) * nfft**2)

                # return resized
                D.cwtdata[c,:,:] = Wns[0:tnum,:]

                # plot (not default)
                pshot = D.shot
                pname = D.clist[c]
                ptime = D.time
                pfreq = f/1000.0
                pdata = np.transpose(np.abs(D.cwtdata[c,:,:])**2)

                plt.imshow(pdata, extent=(ptime.min(), ptime.max(), pfreq.min(), pfreq.max()), interpolation='none', aspect='auto', origin='lower')

                chpos = '({:.1f}, {:.1f})'.format(D.rpos[c]*100, D.zpos[c]*100) # [cm]
                plt.title('#{:d}, {:s} {:s}'.format(pshot, pname, chpos), fontsize=10)
                plt.xlabel('Time [s]')
                plt.ylabel('Frequency [kHz]')

                plt.show()

            D.cwtf = f
            D.cwtdf = df
            D.cwtsj = sj
            D.cwtdj = dj
            D.cwtts = ts

############################# statistical methods ##############################

    def hurst(self, dnum=0, cnl=[0], bins=30, detrend=1, fitlims=[10,1000], **kwargs):
        if 'ylimits' in kwargs: ylimits = kwargs['ylimits']
        if 'xlimits' in kwargs: xlimits = kwargs['xlimits']

        pshot = self.Dlist[dnum].shot
        dt = self.Dlist[dnum].time[1] - self.Dlist[dnum].time[0]  # time step
        fs = round(1/(dt)/1000)*1000.0 # [Hz]
        cnum = len(self.Dlist[dnum].data)  # number of cmp channels

        # plot dimension
        nch = len(cnl)
        if nch < 4:
            row = nch
        else:
            row = 4
        col = math.ceil(nch/row)

        # axis
        bsize = int(1.0*len(self.Dlist[dnum].time)/bins)
        ax = np.floor( 10**(np.arange(1.0, np.log10(bsize), 0.01)) )
        self.Dlist[dnum].ax = ax/fs*1e6

        # data dimension
        self.Dlist[dnum].val = np.zeros((cnum, len(ax)))
        self.Dlist[dnum].std = np.zeros((cnum, len(ax)))
        self.Dlist[dnum].hurst = np.zeros(cnum)

        for i, c in enumerate(cnl):
            # set axes
            if i == 0:
                plt.subplots_adjust(hspace = 0.5, wspace = 0.3)
                axes1 = plt.subplot(row,col,i+1)
                axprops = dict(sharex = axes1, sharey = axes1)
            else:
                plt.subplot(row,col,i+1, **axprops)

            pname = self.Dlist[dnum].clist[c]
            x = self.Dlist[dnum].data[c,:]

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

            self.Dlist[dnum].val[c,:] = np.mean(ers, 0)
            self.Dlist[dnum].std[c,:] = np.std(ers, axis=0)

            ptime = self.Dlist[dnum].ax # time lag [us]
            pdata = self.Dlist[dnum].val[c,:]

            plt.plot(ptime, pdata, '-x')

            fidx = (fitlims[0] <= ptime) * (ptime <= fitlims[1])
            fit = np.polyfit(np.log10(ptime[fidx]), np.log10(pdata[fidx]), 1)
            fit_data = 10**(fit[1])*ptime**(fit[0])
            plt.plot(ptime, fit_data, 'r')
            self.Dlist[dnum].hurst[c] = fit[0]

            chpos = '({:.1f}, {:.1f})'.format(self.Dlist[dnum].rpos[c]*100, self.Dlist[dnum].zpos[c]*100) # [cm]
            plt.title('#{:d}, {:s} {:s}; H={:g}'.format(pshot, pname, chpos, fit[0]), fontsize=10)
            plt.xlabel('Time lag [us]')
            plt.ylabel('R/S')

            plt.xscale('log')
            plt.yscale('log')

        plt.show()

    def chplane(self, dnum=0, cnl=[0], d=6, bins=30, verbose=1, **kwargs):
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

        # axis
        nst = math.factorial(d) # number of possible states
        ax = np.arange(nst) + 1 # state number
        self.Dlist[dnum].ax = ax

        bsize = int(1.0*len(self.Dlist[dnum].time)/bins)
        print('For an accurate estimation of the probability, bsize {:g} should be considerably larger than nst {:g}'.format(bsize, nst))

        # possible orders
        orders = np.empty((0,d))
        for p in itertools.permutations(np.arange(d)):
            orders = np.append(orders,np.atleast_2d(p),axis=0)

        # data dimension
        self.Dlist[dnum].val = np.zeros((cnum, nst))
        self.Dlist[dnum].std = np.zeros((cnum, nst))
        self.Dlist[dnum].jscom = np.zeros(cnum)
        self.Dlist[dnum].nsent = np.zeros(cnum)
        self.Dlist[dnum].pment = np.zeros(cnum)

        for i, c in enumerate(cnl):
            # set axes
            if verbose == 1 and i == 0:
                plt.subplots_adjust(hspace = 0.5, wspace = 0.3)
                axes1 = plt.subplot(row,col,i+1)
                axprops = dict(sharex = axes1, sharey = axes1)
            elif verbose == 1 and i > 0:
                plt.subplot(row,col,i+1, **axprops)

            pname = self.Dlist[dnum].clist[c]

            x = self.Dlist[dnum].data[c,:]

            val = np.zeros((nst, bins))

            for b in range(bins):
                idx1 = b*bsize
                idx2 = idx1 + bsize

                sx = x[idx1:idx2]

                # calculate permutation probability
                jnum = len(sx) - d + 1
                for j in range(jnum):
                    ssx = sx[j:(j+d)]

                    sso = np.argsort(ssx)
                    bingo = np.sum(np.abs(orders - np.tile(sso, (nst, 1))), 1) == 0
                    val[bingo, b] = val[bingo, b] + 1.0/jnum

            pi = np.mean(val, 1) # bin averaged pi
            pierr = np.std(val, 1)
            pio = np.argsort(-pi)

            self.Dlist[dnum].val[c,:] = pi[pio]     # bin averaged sorted pi
            self.Dlist[dnum].std[c,:] = pierr[pio]

            # Jensen Shannon complexity, normalized Shannon entropy, Permutation entropy
            self.Dlist[dnum].jscom[c], self.Dlist[dnum].nsent[c], self.Dlist[dnum].pment[c] = complexity_measure(pi, nst)

            # plot BP probability
            if verbose == 1:
                pax = self.Dlist[dnum].ax
                pdata = self.Dlist[dnum].val[c,:]

                plt.plot(pax, pdata, '-x')

                chpos = '({:.1f}, {:.1f})'.format(self.Dlist[dnum].rpos[c]*100, self.Dlist[dnum].zpos[c]*100) # [cm]
                plt.title('#{:d}, {:s} {:s}; C={:g}, H={:g}'.format(pshot, pname, chpos, self.Dlist[dnum].jscom[c], self.Dlist[dnum].nsent[c]), fontsize=10)
                plt.xlabel('order number')
                plt.ylabel('BP probability')

                plt.yscale('log')

        if verbose == 1: plt.show()

        # plot CH plane
        if verbose == 1:
            plt.plot(self.Dlist[dnum].nsent, self.Dlist[dnum].jscom, '-o')

            plt.xlabel('Entropy (H)')
            plt.ylabel('Complexity (C)')

            plt.show()

    def intermittency(self, dnum=0, cnl=[0], bins=20, overlap=0.2, qstep=0.3, fitlims=[20.0,100.0], **kwargs):
        # intermittency parameter from multi-fractal analysis [Carreras PoP 2000]
        # this ranges from 0 (mono-fractal) to 1
        # add D fitting later
        if 'ylimits' in kwargs: ylimits = kwargs['ylimits']
        if 'xlimits' in kwargs: xlimits = kwargs['xlimits']

        self.Dlist[dnum].vkind = 'intermittency'

        pshot = self.Dlist[dnum].shot
        cnum = len(self.Dlist[dnum].data)  # number of cmp channels

        # axis
        qax = np.arange(-2,8,qstep) # order axis
        N = len(self.Dlist[dnum].time)
        Tmax = int( N/(bins - overlap*(bins - 1.0)) ) # minimum bin -> maximum data length
        Tax = np.floor( 10**(np.arange(1, np.log10(Tmax), 0.1)) ) # sub-data length axis
        nTax = Tax/N # normalized axis

        # data dimension
        eTq = np.zeros((cnum, len(Tax), len(qax)))
        K = np.zeros((cnum, len(qax)))
        C = np.zeros((cnum, len(qax)))
        D = np.zeros((cnum, len(qax)))

        for i, c in enumerate(cnl):
            # first axes
            plt.subplots_adjust(hspace = 0.5, wspace = 0.3)
            axes1 = plt.subplot(5,1,1)

            pname = self.Dlist[dnum].clist[c]

            x = self.Dlist[dnum].data[c,:]
            x = signal.detrend(x, type='linear')

            plt.plot(self.Dlist[dnum].time, x)

            ndxe = (x - np.mean(x))**2 / np.mean((x - np.mean(x))**2) # Eq.(7)

            for t, T in enumerate(Tax): # loop over different length T
                bins = int( N/(T - overlap*(T-1)) ) # number of bins with length T

                eT = np.zeros(bins)
                bstep = int(T*(1 - overlap))
                for j in range(bins):
                    idx1 = j*bstep
                    idx2 = int(idx1 + T)

                    eT[j] = np.mean(ndxe[idx1:idx2]) # Eq.(9)

                # calculate moments
                for k, q in enumerate(qax):
                    eTq[c, t, k] = np.mean(eT**(q)) # Eq.(10)

            # second axes
            plt.subplot(5,1,2)
            # calculate K
            for k, q in enumerate(qax):
                plt.plot(nTax, eTq[c,:,k], 'o')

                # fit range
                nT1 = fitlims[0]/N
                nT2 = fitlims[1]/N
                idx = (nT1 < nTax) * (nTax < nT2)

                lx = np.log(nTax[idx])
                ly = np.log(eTq[c,idx,k])

                fit = np.polyfit(lx, ly, 1)
                fit_func = np.poly1d(fit)
                K[c,k] = -fit[0]

                fx = np.arange(nTax.min(), nTax.max(), 1.0/N)
                fy = np.exp(fit_func(np.log(fx)))
                plt.plot(fx, fy)

                plt.axvline(x=nT1, color='r')
                plt.axvline(x=nT2, color='r')

            plt.title('Linear fit of loglog plot is -K(q)')
            plt.xlabel('T/N')
            plt.ylabel('eTq moments')
            plt.xscale('log')
            plt.yscale('log')

            # third axes
            plt.subplot(5,1,3)
            plt.plot(qax, K[c,:], '-o')
            plt.xlabel('q')
            plt.ylabel('K(q)')

            # calculate C and D
            for k, q in enumerate(qax):
                if (0.9 <= q) and (q <= 1.1):
                    Kgrad = np.gradient(K[c,:], qax[1] - qax[0])
                    C[c,k] = Kgrad[k]

                    print('C({:g}) intermittency parameter is {:g}'.format(q, C[c,k]))
                else:
                    C[c,k] = K[c,k] / (q - 1)

                D[c,k] = 1 - C[c,k]

            # fourth axes
            plt.subplot(5,1,4)
            plt.plot(qax, C[c,:], '-o')
            plt.xlabel('q')
            plt.ylabel('C(q)')

            # fifth axes
            plt.subplot(5,1,5)
            plt.plot(qax, D[c,:], '-o')
            plt.xlabel('q')
            plt.ylabel('D(q)')

            plt.show()

############################# data filtering functions #########################

    def filt(self, name, fL, fH, b=0.08):
        # for FIR filters
        for d, D in enumerate(self.Dlist):
            cnum = len(D.data)
            for c in range(cnum):
                x = np.copy(D.data[c,:])
                filter = FiltData(name, D.fs, fL, fH, b)
                D.data[c,:] = filter.apply(x)

            print('dnum {:d} filter {:s} with fL {:g} fH {:g} b {:g}'.format(d, name, fL, fH, b))

############################# default plot functions ###########################

    def mplot(self, dnum=1, cnl=[0], type='time', **kwargs):
        if 'ylimits' in kwargs: ylimits = kwargs['ylimits']
        if 'xlimits' in kwargs: xlimits = kwargs['xlimits']

        pshot = self.Dlist[dnum].shot

        # plot dimension
        nch = len(cnl)
        if nch < 4:
            row = nch
        else:
            row = 4
        col = math.ceil(nch/row)

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
                if self.Dlist[dnum].vkind == 'correlation' or self.Dlist[dnum].vkind == 'corr_coef':
                    pbase = self.Dlist[dnum].ax*1e6
                else:
                    pbase = self.Dlist[dnum].ax/1000
                pdata = self.Dlist[dnum].val[c,:].real
                rname = self.Dlist[dnum].rname[c]
                if self.Dlist[dnum].vkind == 'coherence':
                    plt.axhline(y=1/np.sqrt(self.Dlist[dnum].bins), color='r')

            plt.plot(pbase, pdata)  # plot

            if 'ylimits' in kwargs:  # ylimits
                plt.ylim([ylimits[0], ylimits[1]])
            if 'xlimits' in kwargs:  # xlimits
                plt.xlim([xlimits[0], xlimits[1]])
            else:
                plt.xlim([pbase[0], pbase[-1]])

            chpos = '({:.1f}, {:.1f})'.format(self.Dlist[dnum].rpos[c]*100, self.Dlist[dnum].zpos[c]*100) # [cm]
            if type == 'time':
                plt.title('#{:d}, {:s} {:s}'.format(pshot, pname, chpos), fontsize=10)
            elif type == 'val':
                plt.title('#{:d}, {:s}-{:s} {:s}'.format(pshot, rname, pname, chpos), fontsize=10)

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
            elif type == 'val' and self.Dlist[dnum].vkind == 'correlation':
                plt.xlabel('Time lag [us]')
                plt.ylabel('Correlation')
            elif type == 'val' and self.Dlist[dnum].vkind == 'corr_coef':
                plt.xlabel('Time lag [us]')
                plt.ylabel('Corr. coef.')

        plt.show()

    def oplot(self, dnum, cnl, type='time', **kwargs):
        if 'ylimits' in kwargs: ylimits = kwargs['ylimits']
        if 'xlimits' in kwargs: xlimits = kwargs['xlimits']

        for c in cnl:
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

    def spec(self, dnum, cnl, nfft=2048, **kwargs):
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

    def iplot(self, dnum, snum=0, vlimits=[-0.1, 0.1], **kwargs):
        # keyboard interactive iplot
        # (intp='none', climits=[-0.1,0.1], **kwargs)

        # data filtering

        c = raw_input('automatic, or manual [a,m]: ')
        tidx1 = 0  # starting index
        if c == 'a':
            # make axes
            fig = plt.figure(facecolor='w', figsize=(5,10))
            ax1 = fig.add_axes([0.1, 0.75, 0.7, 0.2])  # [left bottom width height]
            ax2 = fig.add_axes([0.1, 0.1, 0.7, 0.60])
            ax3 = fig.add_axes([0.83, 0.1, 0.03, 0.6])
            axs = [ax1, ax2, ax3]

            tstep = int(input('time step [idx]: '))  # jumping index # tstep = 10
            for tidx in range(tidx1, len(self.Dlist[dnum].time), tstep):
                # prepare data
                pdata = self.Dlist[dnum].data[:,tidx]

                # position
                rpos = self.Dlist[dnum].rpos[:]
                zpos = self.Dlist[dnum].zpos[:]

                # plot
                axs[0].cla()
                axs[1].cla()
                axs[2].cla()
                plt.ion()

                axs[0].plot(self.Dlist[dnum].time, self.Dlist[dnum].data[snum,:])  # ax1.hold(True)
                axs[0].axvline(x=self.Dlist[dnum].time[tidx], color='g')
                sc = axs[1].scatter(rpos, zpos, 500, pdata, marker='s', vmin=vlimits[0], vmax=vlimits[1], cmap=CM)
                axs[1].set_aspect('equal')
                plt.colorbar(sc, cax=axs[2])

                axs[1].set_xlabel('R [m]')
                axs[1].set_ylabel('z [m]')
                axs[1].set_title('ECE image')

                plt.show()
                plt.pause(0.1)

            plt.ioff()
            plt.close()

        elif c == 'm':
            tidx = tidx1
            while True:
                # make axes
                fig = plt.figure(facecolor='w', figsize=(5,10))
                ax1 = fig.add_axes([0.1, 0.75, 0.7, 0.2])  # [left bottom width height]
                ax2 = fig.add_axes([0.1, 0.1, 0.7, 0.60])
                ax3 = fig.add_axes([0.83, 0.1, 0.03, 0.6])
                axs = [ax1, ax2, ax3]

                # prepare data
                pdata = self.Dlist[dnum].data[:,tidx]

                # position
                rpos = self.Dlist[dnum].rpos[:]
                zpos = self.Dlist[dnum].zpos[:]

                # plot
                axs[0].cla()
                axs[1].cla()
                axs[2].cla()
                plt.ion()

                axs[0].plot(self.Dlist[dnum].time, self.Dlist[dnum].data[snum,:])  # ax1.hold(True)
                axs[0].axvline(x=self.Dlist[dnum].time[tidx], color='g')
                sc = axs[1].scatter(rpos, zpos, 500, pdata, marker='s', vmin=vlimits[0], vmax=vlimits[1], cmap=CM)
                axs[1].set_aspect('equal')
                plt.colorbar(sc, cax=axs[2])

                axs[1].set_xlabel('R [m]')
                axs[1].set_ylabel('z [m]')
                axs[1].set_title('ECE image')

                plt.show()

                k = raw_input('set time step [idx][+,-,0]: ')
                try:
                    tstep = int(k)
                    if tstep == 0:
                        plt.ioff()
                        plt.close()
                        break
                except:
                    pass

                if tidx + tstep < len(self.Dlist[dnum].time) - 1 and 0 < tidx + tstep:
                    tidx = tidx + tstep

                plt.ioff()
                plt.close()

        self.Dlist[dnum].pdata = pdata

    def cplot(self, dnum, snum=0, frange=[0, 100], vlimits=[0, 1], **kwargs):
        if 'ylimits' in kwargs: ylimits = kwargs['ylimits']
        if 'xlimits' in kwargs: xlimits = kwargs['xlimits']
        # calculate mean coherence image
        # or cross power rms image
        # or group velocity image

        # axis
        pbase = self.Dlist[dnum].ax/1000  # [kHz]

        # fidx
        idx = np.where((pbase >= frange[0])*(pbase <= frange[1]))
        idx1 = int(idx[0][0])
        idx2 = int(idx[0][-1]+1)

        # data
        if self.Dlist[dnum].vkind == 'cross_power':  # rms
            pdata = np.sqrt(np.sum(self.Dlist[dnum].val[:,idx1:idx2], 1))
        elif self.Dlist[dnum].vkind == 'coherence':  # mean coherence
            pdata = np.mean(self.Dlist[dnum].val[:,idx1:idx2], 1)
        elif self.Dlist[dnum].vkind == 'cross_phase':  # group velocity
            base = self.Dlist[dnum].ax[idx1:idx2]  # [Hz]
            pdata = np.zeros(len(self.Dlist[dnum].val))
            for c in range(len(self.Dlist[dnum].val)):
                data = self.Dlist[dnum].val[c,idx1:idx2]
                pfit = np.polyfit(base, data, 1)
                fitdata = np.polyval(pfit, base)
                pdata[c] = 2*np.pi*self.Dlist[dnum].dist[c]/pfit[0]/1000.0  # [km/s]

                # chisq = np.sum((data - fitdata)**2)
                if c == snum:
                    sbase = base/1000  # [kHz]
                    sdata = fitdata

        # position
        rpos = self.Dlist[dnum].rpos[:]
        zpos = self.Dlist[dnum].zpos[:]

        # prepare figure
        fig = plt.figure(facecolor='w', figsize=(5,10))
        ax1 = fig.add_axes([0.1, 0.75, 0.7, 0.2])  # [left bottom width height]
        ax2 = fig.add_axes([0.1, 0.1, 0.7, 0.60])
        ax3 = fig.add_axes([0.83, 0.1, 0.03, 0.6])
        axs = [ax1, ax2, ax3]

        # sample plot
        axs[0].plot(pbase, self.Dlist[dnum].val[snum,:])  # ax1.hold(True)
        if self.Dlist[dnum].vkind == 'cross_phase':
            axs[0].plot(sbase, sdata)
        axs[0].axvline(x=pbase[idx1], color='g')
        axs[0].axvline(x=pbase[idx2], color='g')

        if self.Dlist[dnum].vkind == 'cross_power':
            axs[0].set_yscale('log')
        if 'ylimits' in kwargs: # ylimits
            axs[0].set_ylim([ylimits[0], ylimits[1]])
        if 'xlimits' in kwargs: # xlimits
            axs[0].set_xlim([xlimits[0], xlimits[1]])
        else:
            axs[0].set_xlim([pbase[0], pbase[-1]])

        # pdata plot
        sc = axs[1].scatter(rpos, zpos, 500, pdata, marker='s', vmin=vlimits[0], vmax=vlimits[1], cmap=CM)
        axs[1].set_aspect('equal')

        # color bar
        plt.colorbar(sc, cax=axs[2])

        axs[1].set_xlabel('R [m]')
        axs[1].set_ylabel('z [m]')
        if self.Dlist[dnum].vkind == 'cross_power':
            axs[1].set_title('Cross power rms')
        elif self.Dlist[dnum].vkind == 'coherence':
            axs[1].set_title('Coherence mean')
        elif self.Dlist[dnum].vkind == 'cross_phase':
            axs[1].set_title('Group velocity [km/s]')

        self.Dlist[dnum].pdata = pdata

        plt.show()

############################# test functions ###################################

    def fftbins_bicoh_test(self, nfft, window, overlap, detrend, full=1):
        # self.list_data()

        for dnum in range(len(self.Dlist)):
            # get bins and window function
            tnum = len(self.Dlist[dnum].data[0,:])
            bins, win = fft_window(tnum, nfft, window, overlap)

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
                    self.Dlist[dnum].fftdata = np.zeros((cnum, bins, nfft+1), dtype=np.complex_)
                else:  # odd nfft
                    self.Dlist[dnum].fftdata = np.zeros((cnum, bins, nfft), dtype=np.complex_)
            else: # half 0 ~ fN
                self.Dlist[dnum].fftdata = np.zeros((cnum, bins, int(nfft/2+1)), dtype=np.complex_)

            pbs = 2*np.pi*(0.5 - np.random.randn(bins))
            pcs = 2*np.pi*(0.5 - np.random.randn(bins))
            pds = 2*np.pi*(0.5 - np.random.randn(bins))

            for c in range(cnum):
                x = self.Dlist[dnum].data[c,:]

                for b in range(bins):
                    idx1 = int(b*np.fix(nfft*(1 - overlap)))
                    idx2 = idx1 + nfft

                    sx = x[idx1:idx2]
                    st = self.Dlist[dnum].time[idx1:idx2]

                    # test signal for bicoherence test
                    fb = 54*1000
                    fc = 94*1000
                    fd = fb + fc

                    pb = pbs[b]
                    pc = pcs[b]
                    # pd = pb + pc # coherent case
                    pd = pds[b] # non-coherent case

                    sx = np.cos(2*np.pi*fb*st + pb) + np.cos(2*np.pi*fc*st + pc) + 1/2*np.cos(2*np.pi*fd*st + pd) + 1/2*np.random.randn(len(sx))
                    sx = sx + np.cos(2*np.pi*fb*st + pb)*np.cos(2*np.pi*fc*st + pc)

                    if detrend == 1:
                        sx = signal.detrend(sx, type='linear')
                    sx = signal.detrend(sx, type='constant')  # subtract mean

                    sx = sx * win  # apply window function

                    # get fft
                    fftdata = np.fft.fft(sx, n=nfft)/nfft  # divide by the length
                    if np.mod(nfft, 2) == 0:  # even nfft
                        fftdata = np.hstack([fftdata[0:int(nfft/2)], np.conj(fftdata[int(nfft/2)]), fftdata[int(nfft/2):nfft]])
                    if full == 1: # shift to -fN ~ 0 ~ fN
                        fftdata = np.fft.fftshift(fftdata)
                    else: # half 0 ~ fN
                        fftdata = fftdata[0:int(nfft/2+1)]
                    self.Dlist[dnum].fftdata[c,b,:] = fftdata

            # update attributes
            if np.mod(nfft, 2) == 0:
                self.Dlist[dnum].nfft = nfft + 1
            else:
                self.Dlist[dnum].nfft = nfft
            self.Dlist[dnum].window = window
            self.Dlist[dnum].overlap = overlap
            self.Dlist[dnum].detrend = detrend
            self.Dlist[dnum].bins = bins
            self.Dlist[dnum].win = win

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


def nextpow2(i):
    n = 1
    while n < i: n *= 2
    return n


def complexity_measure(pi, nst):
    # complexity, entropy measure with a given BP probability
    pinz = pi[pi != 0]
    spi = np.sum(-pinz * np.log2(pinz)) # permutation entropy
    pe = np.ones(nst)/nst
    spe = np.sum(-pe * np.log2(pe))
    pieh = (pi + pe)/2
    spieh = np.sum(-pieh * np.log2(pieh))
    hpi = spi/np.log2(nst) # normalized Shannon entropy

    # Jensen Shannon complexity
    jscom = -2*(spieh - spi/2 - spe/2)/((nst + 1.0)/nst*np.log2(nst+1) - 2*np.log2(2*nst) + np.log2(nst))*hpi
    nsent = hpi
    pment = spi

    return jscom, nsent, pment
