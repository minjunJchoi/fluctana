#!/usr/bin/env python2.7

# Author : Minjun J. Choi (mjchoi@nfri.re.kr)
#
# Description : This code reads the KSTAR MIR data via the iKSTAR server
#
# Acknowledgement : Dr. Y. Nam and Prof. G.S. Yun
#

import numpy as np
import h5py

MNUM = 10000000  # totla number of samples in an ECEI channel
VN = 16  # number of vertical arrays


class KstarMir(object):
    def __init__(self, shot, clist):
        self.shot = shot

        if 12272 < shot and shot < 14942:
            self.data_path = '/eceidata/exp_2015/'
        elif 14941 < shot and shot < 17356:
            self.data_path = '/eceidata2/exp_2016/'
        elif 17963 < shot and shot < 19392:
            self.data_path = '/eceidata2/exp_2017/'
        elif 19391 < shot:
            self.data_path = '/eceidata2/exp_2018/'

        self.clist = expand_clist(clist)

        # file name
        self.fname = "{:s}{:06d}/MIR.{:06d}.h5".format(self.data_path, shot, shot)

        # get attributes
        with h5py.File(self.fname, 'r') as f:
            # get attributes
            dset = f['MIR']
            self.tt = dset.attrs['TriggerTime'] # in [s]
            self.toff = self.tt[0]+0.001
            self.fs = dset.attrs['SampleRate'][0]*1000.0  # in [Hz] same sampling rate
            self.bt = dset.attrs['TFcurrent']*0.0995556  # [kA] -> [T]
            self.mfl = dset.attrs['MFL']
            self.mirh = dset.attrs['MIRH']
            self.mirf = dset.attrs['MIRF']
            self.lo = dset.attrs['MLo']
            self.rf1 = dset.attrs['MRF1']
            self.rf2 = dset.attrs['MRF2']
            self.rf3 = dset.attrs['MRF3']
            self.rf4 = dset.attrs['MRF4']

            print('MIR file = {}'.format(self.fname))
        
        # get channel posistion
        self.channel_position()

    def get_data(self, trange, norm=0, atrange=[1.0, 1.01], res=0, verbose=1):
        self.trange = trange

        # norm = 0 : no normalization
        # norm = 1 : normalization by trange average
        # norm = 2 : normalization by atrange average
        # res  = 0 : no resampling
        if norm == 0:
            if verbose == 1: print('Data is not normalized MIR')
        elif norm == 1:
            if verbose == 1: print('Data is normalized by trange std MIR')
        elif norm == 2:
            if verbose == 1: print('Data is normalized by atrange std MIR')

        # get time base
        time, idx1, idx2 = self.time_base(trange)
        if norm == 2:
            atime, aidx1, aidx2 = self.time_base(atrange)

        # get data
        with h5py.File(self.fname, 'r') as f:
            # time series length
            tnum = idx2 - idx1

            # number of channels
            cnum = len(self.clist)

            data = np.zeros((cnum, tnum))
            for i in range(0, cnum):

                vn = int(self.clist[i][4:6])
                fn = int(self.clist[i][6:8])

                inode = 'MD{:02d}{:02d}'.format(1 + (fn-1)*2,vn)
                qnode = 'MD{:02d}{:02d}'.format(fn*2,vn)

                inode = "/MIR/" + inode + "/Voltage"
                qnode = "/MIR/" + qnode + "/Voltage"

                iv = f[inode][idx1:idx2]/10000.0
                qv = f[qnode][idx1:idx2]/10000.0

                # remove offset
                iv = iv - np.mean(iv)
                qv = qv - np.mean(qv)

                if norm == 1:
                    iv = iv/np.std(iv)
                    qv = qv/np.std(qv)
                elif norm == 2:
                    iav = f[inode][aidx1:aidx2]/10000.0
                    qav = f[qnode][aidx1:aidx2]/10000.0
                    iv = iv/np.std(iav)
                    qv = qv/np.std(qav)

                # complex iv, qv
                data[i][:] = iv + 1.0j*qv
                print('TRY pre-filtering of iv and qv (threshold fft) and return iv + 1.0j*qv')

            self.data = data

        return time, data

    def time_base(self, trange):
        # using self.tt, self.fs; get self.time
        tt = self.tt
        fs = self.fs

        if len(tt) == 2:
            pl = tt[1] - tt[0] + 0.1
            tt = [tt[0], pl, tt[1]]

        fulltime = []
        for i in range(0, len(tt)/3):
            t0 = tt[i*3]
            pl = tt[i*3+1]
            t1 = tt[i*3+2]
            cnt = 0
            for ti in np.arange(t0, t1, pl):
                cnt = cnt + 1
                if cnt % 2 == 0: continue
                if ti+pl > t1:
                    fulltime = np.append(fulltime,np.arange(ti, t1, 1/fs))
                else:
                    fulltime = np.append(fulltime,np.arange(ti, ti+pl, 1/fs))
                if len(fulltime) > MNUM:
                    break
            if len(fulltime) > MNUM:
                break

        fulltime = fulltime[0:(MNUM+1)]

        idx = np.where((fulltime >= trange[0])*(fulltime <= trange[1]))
        idx1 = int(idx[0][0])
        idx2 = int(idx[0][-1]+1)

        self.time = fulltime[idx1:idx2]

        return fulltime[idx1:idx2], idx1, idx2

    def channel_position(self):
        # get self.rpos, self.zpos, self.apos

        cnum = len(self.clist)
        self.rpos = np.zeros(cnum)  # R [m] of each channel
        self.zpos = np.zeros(cnum)  # z [m] of each channel
        self.apos = np.zeros(cnum)  # angle [rad] of each channel
        for c in range(0, cnum):
            # vn = int(self.clist[c][(self.cnidx1):(self.cnidx1+2)])
            # fn = int(self.clist[c][(self.cnidx1+2):(self.cnidx1+4)])

            self.rpos[c] = 0
            self.zpos[c], self.apos[c] = 0, 0


def expand_clist(clist):
    # IN : List of channel names (e.g. 'MIR_0101-1604')
    # OUT : Expanded list (e.g. 'MIR_0101', ..., 'MIR_1604')

    # KSTAR MIR
    exp_clist = []
    for c in range(len(clist)):
        if 'MIR' in clist[c] and len(clist[c]) == 13:
            vi = int(clist[c][4:6])
            fi = int(clist[c][6:8])
            vf = int(clist[c][9:11])
            ff = int(clist[c][11:13])

            for v in range(vi, vf+1):
                for f in range(fi, ff+1):
                    exp_clist.append(clist[c][0:4] + '{:02d}{:02d}'.format(v, f))
        else:
            exp_clist.append(clist[c])
    clist = exp_clist

    return clist
