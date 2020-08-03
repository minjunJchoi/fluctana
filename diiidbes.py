#!/usr/bin/env python2.7

# Author : Minjun J. Choi (mjchoi@nfri.re.kr)
#
# Description : This code reads the DIIID BES data (h5 format)
#
# Acknowledgement : Dr. Y. Nam and Prof. G.S. Yun
#
import os
import numpy as np
import h5py

MNUM = 10000000  # totla number of samples in an ECEI channel
VN = 16  # number of vertical arrays


class DiiidBes(object):
    def __init__(self, shot, clist):
        self.shot = shot

        self.data_path = '/home/mjchoi/DIIID/besdata/'

        self.clist = expand_clist(clist)

        # file name
        self.fname = "{:s}{:06d}/BES.{:06d}.h5".format(self.data_path, shot, shot)

        if os.path.exists(self.fname):
            print('BES file = {}'.format(self.fname))
        
        # get channel posistion
        self.channel_position()

    def get_data(self, trange, norm=0, atrange=[1.0, 1.01], res=0, verbose=1):
        self.trange = trange

        # norm = 0 : no normalization
        # norm = 1 : normalization by trange average
        # norm = 2 : normalization by atrange average
        # norm = 3 : normalization by SU
        # res  = 0 : no resampling

        if norm == 0:
            if verbose == 1: print('Data is not normalized BES')
        elif norm == 1:
            if verbose == 1: print('Data is normalized by trange std BES')
        elif norm == 2:
            if verbose == 1: print('Data is normalized by atrange std BES')
        elif norm == 3:
            if verbose == 1: print('Data is normalized by BESSU signal')

        # get time base
        time, idx1F, idx2F, timeS, idx1S, idx2S = self.time_base(trange)
        if norm == 2:
            _, _, _, _, aidx1S, aidx2S = self.time_base(atrange)

        # get data
        with h5py.File(self.fname, 'r') as f:
            # time series length
            tnum = idx2F - idx1F

            # number of channels
            cnum = len(self.clist)

            data = np.zeros((cnum, tnum))
            for i, cname in enumerate(self.clist):
                # node name
                nodeF = 'BESFU{:02d}'.format(int(cname[3:5]))
                nodeS = 'BESSU{:02d}'.format(int(cname[3:5]))

                # raw data
                vF = f[nodeF][idx1F:idx2F]
                vS = f[nodeS][idx1S:idx2S]

                # print(nodeF, f['tFU'][idx1F:idx2F])
                # print(nodeS, f['tSU'][idx1S:idx2S])
                
                # sampling up SU using interpolation
                vS = np.interp(time, timeS, vS)

                # get offset for SU
                offset = np.mean(f[nodeS][0:25])

                # full signal
                v = vF + vS - offset

                # normalization
                if norm == 1:
                    v = v/np.mean(v) - 1
                elif norm == 2:
                    avS = f[nodeS][aidx1S:aidx2S]
                    v = v/(np.mean(avS) - offset) - 1
                elif norm == 3:
                    v = v/(vS - offset) - 1

                # stack data
                data[i][:] = v

            self.data = data

        return time, data

    def time_base(self, trange):
        with h5py.File(self.fname, 'r') as f:
            fulltimeF = f['tFU'][:]/1000.0 # [ms] -> [s]
            fulltimeS = f['tSU'][:]/1000.0 # [ms] -> [s]

            # get sampling frequency
            self.fs = round(1/(fulltimeF[1] - fulltimeF[0])/1000)*1000.0

        idx = np.where((fulltimeF >= trange[0])*(fulltimeF <= trange[1]))
        idx1F = int(idx[0][0])
        idx2F = int(idx[0][-1]+1)

        idx = np.where((fulltimeS >= trange[0])*(fulltimeS <= trange[1]))
        idx1S = int(idx[0][0])
        idx2S = int(idx[0][-1]+1)

        self.time = fulltimeF[idx1F:idx2F]

        return fulltimeF[idx1F:idx2F], idx1F, idx2F, fulltimeS[idx1S:idx2S], idx1S, idx2S

    def channel_position(self):
        # get self.rpos, self.zpos, self.apos

        cnum = len(self.clist)
        self.rpos = np.zeros(cnum)  # R [m] of each channel
        self.zpos = np.zeros(cnum)  # z [m] of each channel
        self.apos = np.zeros(cnum)  # angle [rad] of each channel
        # for c in range(0, cnum):
        #     # vn = int(self.clist[c][(self.cnidx1):(self.cnidx1+2)])
        #     # fn = int(self.clist[c][(self.cnidx1+2):(self.cnidx1+4)])

        #     self.rpos[c] = 0
        #     self.zpos[c], self.apos[c] = 0, 0


def expand_clist(clist):
    # IN : List of channel names (e.g. 'BES10-14')
    # OUT : Expanded list (e.g. 'BES10', ..., 'BES14')

    # DIIID BES
    exp_clist = []
    for c in range(len(clist)):
        if 'BES' in clist[c] and len(clist[c]) == 8:
            ni = int(clist[c][3:5])
            nf = int(clist[c][6:8])

            for n in range(ni, nf+1):
                exp_clist.append(clist[c][0:3] + '{:02d}'.format(n))
        else:
            exp_clist.append(clist[c])
    clist = exp_clist

    return clist
