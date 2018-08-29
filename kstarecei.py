#!/usr/bin/env python2.7

# Author : Minjun J. Choi (mjchoi@nfri.re.kr)
#
# Description : This code reads the KSTAR ECEI data in iKSTAR server
#
# Acknowledgement : Dr. Y. Nam and Prof. G.S. Yun
#
# Last updated
#  2018.02.15 : version 0.10; cold resonance positions

import numpy as np
import h5py

from kei import KstarEceiInfo

ENUM = 5000000  # totla number of samples in an ECEI channel
VN = 24  # number of vertical arrays


class KstarEcei(KstarEceiInfo):
    def get_data(self, trange, norm=1, atrange=[1.0, 1.01], res=0):
        self.trange = trange

        # norm = 0 : no normalization
        # norm = 1 : normalization by trange average
        # norm = 2 : normalization by atrange average
        # res  = 0 : no resampling
        if norm == 0:
            print 'data is not normalized'
        elif norm == 1:
            print 'data is normalized by trange average'
        elif norm == 2:
            print 'data is normalized by atrange average'

        # get time base
        time, idx1, idx2, oidx1, oidx2 = self.time_base(trange)
        if norm == 2:
            atime, aidx1, aidx2, aoidx1, aoidx2 = self.time_base(atrange)

        # get data
        with h5py.File(self.fname, 'r') as f:
            # time series length
            tnum = idx2 - idx1

            # number of channels
            cnum = len(self.clist)

            data = np.zeros((cnum, tnum))
            for i in range(0, cnum):
                node = "/ECEI/" + self.clist[i] + "/Voltage"

                ov = f[node][oidx1:oidx2]/10000.0
                v = f[node][idx1:idx2]/10000.0

                v = v - np.mean(ov)
                if norm == 1:
                    v = v/np.mean(v) - 1
                elif norm == 2:
                    av = f[node][aidx1:aidx2]/10000.0
                    v = v/np.mean(av) - 1

                data[i][:] = v

            self.data = data

        # get channel posistion
        self.channel_position()

        return time, data

    def time_base(self, trange):
        # using self.tt, self.toff, self.fs; get self.time
        tt = self.tt
        toff = self.toff
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
                if len(fulltime) > ENUM:
                    break
            if len(fulltime) > ENUM:
                break

        fulltime = fulltime[0:(ENUM+1)]

        idx = np.where((fulltime >= trange[0])*(fulltime <= trange[1]))
        idx1 = int(idx[0][0])
        idx2 = int(idx[0][-1]+2)

        if toff < 0:
            oidx = np.where((fulltime >= toff)*(fulltime <= toff+0.01))
            oidx1 = int(oidx[0][0])
            oidx2 = int(oidx[0][-1]+2)
        else:
            print '#### offset from end in KstarEcei.time_base ####'
            oidx1 = int(ENUM - 0.01*fs)
            oidx2 = int(ENUM - 1)

        self.time = fulltime[idx1:idx2]

        return fulltime[idx1:idx2], idx1, idx2, oidx1, oidx2

    def channel_position(self):
        # get self.rpos, self.zpos, self.apos

        cnum = len(self.clist)
        self.rpos = np.zeros(cnum)  # R [m] of each channel
        self.zpos = np.zeros(cnum)  # z [m] of each channel
        self.apos = np.zeros(cnum)  # angle [rad] of each channel
        for c in range(0, cnum):
            vn = int(self.clist[c][(self.cnidx1):(self.cnidx1+2)])
            fn = int(self.clist[c][(self.cnidx1+2):(self.cnidx1+4)])

            # assume cold resonance
            self.rpos[c] = 1.80*27.99*self.hn*self.bt/((fn - 1)*0.9 + 2.6 + self.lo)  # this assumes bt ~ 1/R
            # bt should be bt[vn][fn]

            # get vertical position and angle at rpos
            self.zpos[c], self.apos[c] = self.beam_path(self.rpos[c], vn)


    def beam_path(self, rpos, vn):
        # IN : shot, device name, R posistion [m], vertical channel number
        # OUT : a ray vertical position and angle at rpos [m] [rad]
        # this will find a ray vertical position and angle at rpos [m]
        # ray starting from the array box posistion

        abcd = self.get_abcd(self.sf, self.sz, rpos)

        # vertical position from the reference axis (vertical center of all lens, z=0 line) at ECEI array box
        zz = (np.arange(VN,0,-1) - 12.5)*14  # [mm]
        # angle against the reference axis at ECEI array box
        aa = np.zeros(np.size(zz))

        # vertical posistion and angle at rpos
        za = np.dot(abcd, [zz, aa])
        zpos = za[0][vn-1]/1000  # zpos [m]
        apos = za[1][vn-1]  # angle [rad] positive means the (z+) up-directed (divering from array to plasma)

        return zpos, apos
