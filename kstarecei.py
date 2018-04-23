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

DIR = '/home/mjchoi/KSTAR/ECEI_data/'
ENUM = 5000000  # totla number of samples in an ECEI channel
VN = 24  # number of vertical arrays


class KstarEcei(object):
    def __init__(self):
        pass

    def get_data(self, shot, trange, clist, norm=1, atrange=[1.0, 1.01], res=0):
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

        global DIR

        self.shot = shot
        self.trange = trange        
        self.clist = clist

        # set data folder
        if 14941 < shot and shot < 17356:
            DIR = '/eceidata2/exp_2016/'
        elif 17963 < shot and shot < 19391:
            DIR = '/eceidata2/exp_2017/'

        # device [L, H, or G]
        dev = clist[0][5]

        # get time base
        time, idx1, idx2, oidx1, oidx2 = self.time_base(shot, dev, trange)
        if norm == 2:
            atime, aidx1, aidx2, aoidx1, aoidx2 = self.time_base(shot, dev, atrange)

        # get data
        fname = "{:s}{:06d}/ECEI.{:06d}.{:s}FS.h5".format(DIR, shot, shot, dev)
        with h5py.File(fname, 'r') as f:
            # time series length
            tnum = idx2 - idx1

            # number of channels
            cnum = len(clist)

            data = np.zeros((cnum, tnum))
            for i in range(0, cnum):
                node = "/ECEI/" + clist[i] + "/Voltage"
                
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
        self.channel_position(shot, dev, clist)

        return time, data

    def time_base(self, shot, dev, trange):
        # set self.toff, self.fs, self.time 

        fname = "{:s}{:06d}/ECEI.{:06d}.{:s}FS.h5".format(DIR, shot, shot, dev)
        with h5py.File(fname, 'r') as f:
            # get attributes
            dset = f['ECEI']
            tt = dset.attrs['TriggerTime']
            self.toff = tt[0]
            self.fs = dset.attrs['SampleRate'][0]*1000.0  # in Hz

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
                        fulltime = np.append(fulltime,np.arange(ti, t1, 1/self.fs))
                    else:
                        fulltime = np.append(fulltime,np.arange(ti, ti+pl, 1/self.fs))
                    if len(fulltime) > ENUM:
                        break
                if len(fulltime) > ENUM:
                    break

            fulltime = fulltime[0:(ENUM+1)]

            idx = np.where((fulltime >= trange[0])*(fulltime <= trange[1]))
            idx1 = int(idx[0][0])
            idx2 = int(idx[0][-1]+2)

            if self.toff < 0:
                oidx = np.where((fulltime >= self.toff)*(fulltime <= self.toff+0.01))
                oidx1 = int(oidx[0][0])
                oidx2 = int(oidx[0][-1]+2)
            else:
                print '#### offset from end in KstarEcei.time_base ####'
                oidx1 = int(ENUM - 0.01*self.fs)
                oidx2 = int(ENUM - 1)

            self.time = fulltime[idx1:idx2]

            return fulltime[idx1:idx2], idx1, idx2, oidx1, oidx2

    def channel_position(self, shot, dev, clist):
        # set self.rpos, self.zpos, self.apos

        fname = "{:s}{:06d}/ECEI.{:06d}.{:s}FS.h5".format(DIR, shot, shot, dev)
        with h5py.File(fname, 'r') as f:
            # get attributes
            dset = f['ECEI']
            bt = dset.attrs['TFcurrent']*0.0995556  # [kA] -> [T]
            mode = dset.attrs['Mode']
            if mode is 'O':
                hn = 1  # harmonic number
            elif mode is 'X':
                hn = 2
            lo = dset.attrs['LoFreq']

            cnum = len(clist)
            self.rpos = np.zeros(cnum)  # R [m] of each channel
            self.zpos = np.zeros(cnum)  # z [m] of each channel
            self.apos = np.zeros(cnum)  # angle [rad] of each channel
            for c in range(0, cnum):
                vn = int(clist[c][6:8])
                fn = int(clist[c][8:10])

                # assume cold resonance
                self.rpos[c] = 1.80*27.99*hn*bt/((fn - 1)*0.9 + 2.6 + lo)  # this assumes bt ~ 1/R
                # bt should be bt[vn][fn] 

                # get vertical position and angle at rpos
                self.zpos[c], self.apos[c] = beam_path(shot, dev, self.rpos[c], vn)


def beam_path(shot, dev, rpos, vn):
    # IN : shot, device name, R posistion [m], vertical channel number
    # OUT : a ray vertical position and angle at rpos [m] [rad]
    # this will find a ray vertical position and angle at rpos [m] 
    # ray starting from the array box posistion

    fname = "{:s}{:06d}/ECEI.{:06d}.{:s}FS.h5".format(DIR, shot, shot, dev)
    with h5py.File(fname, 'r') as f:
        # get attributes
        dset = f['ECEI']        
        sf = dset.attrs['LensFocus']
        sz = dset.attrs['LensZoom']

        rpos = rpos*1000  # [m] -> [mm] # for rpos = [1600:50:2300]

        # ABCD matrix for LFS, HFS, GFS
        if dev is 'L':
            sp = 3350 - rpos

            abcd = np.array([[1,250+sp],[0,1]]).dot(
                   np.array([[1,0],[(1.52-1)/(-730),1.52]])).dot(
                   np.array([[1,135],[0,1]])).dot(
                   np.array([[1,0],[(1-1.52)/(2700*1.52),1/1.52]])).dot(
                   np.array([[1,1265-sz],[0,1]])).dot(
                   np.array([[1,0],[(1.52-1)/1100,1.52]])).dot(
                   np.array([[1,40],[0,1]])).dot(
                   np.array([[1,0],[(1-1.52)/(-1100*1.52),1/1.52]])).dot(
                   np.array([[1,sz],[0,1]])).dot(
                   np.array([[1,0],[0,1.52]])).dot(
                   np.array([[1,65],[0,1]])).dot(
                   np.array([[1,0],[(1-1.52)/(800*1.52),1/1.52]])).dot(
                   np.array([[1,710-sf+140],[0,1]])).dot(
                   np.array([[1,0],[(1.52-1)/(-1270),1.52]])).dot(
                   np.array([[1,90],[0,1]])).dot(
                   np.array([[1,0],[(1-1.52)/(1270*1.52),1/1.52]])).dot(
                   np.array([[1,539+35+sf],[0,1]]))
        elif dev is 'H':
            sp = 3350 - rpos

            abcd = np.array([[1,250+sp],[0,1]]).dot(
                   np.array([[1,0],[(1.52-1)/(-730),1.52]])).dot(
                   np.array([[1,135],[0,1]])).dot(
                   np.array([[1,0],[(1-1.52)/(2700*1.52),1/1.52]])).dot(
                   np.array([[1,1265-sz],[0,1]])).dot(
                   np.array([[1,0],[(1.52-1)/1100,1.52]])).dot(
                   np.array([[1,40],[0,1]])).dot(
                   np.array([[1,0],[(1-1.52)/(-1100*1.52),1/1.52]])).dot(
                   np.array([[1,sz],[0,1]])).dot(
                   np.array([[1,0],[0,1.52]])).dot(
                   np.array([[1,65],[0,1]])).dot(
                   np.array([[1,0],[(1-1.52)/(800*1.52),1/1.52]]))
            if shot > 12297:  # since 2015 campaign
                abcd = abcd.dot(
                   np.array([[1,520-sf+590-9.2],[0,1]])).dot(
                   np.array([[1,0],[(1.52-1)/(-1100),1.52]])).dot(
                   np.array([[1,88.4],[0,1]])).dot(
                   np.array([[1,0],[(1-1.52)/(1100*1.52),1/1.52]])).dot(
                   np.array([[1,446+35+sf-9.2],[0,1]]))
            else:
                abcd = abcd.dot(
                   np.array([[1,520-sf+590],[0,1]])).dot(
                   np.array([[1,0],[(1.52-1)/(-1400),1.52]])).dot(
                   np.array([[1,70],[0,1]])).dot(
                   np.array([[1,0],[(1-1.52)/(1400*1.52),1/1.52]])).dot(
                   np.array([[1,446+35+sf],[0,1]]))
        elif dev is 'G':
            sp = 3150 - rpos

            abcd = np.array([[1,1350-sz+sp],[0,1]]).dot(
                   np.array([[1,0],[0,1.545]])).dot(
                   np.array([[1,100],[0,1]])).dot(
                   np.array([[1,0],[(1-1.545)/(900*1.545),1/1.545]])).dot(
                   np.array([[1,1430-sf+660+sz+470],[0,1]])).dot(
                   np.array([[1,0],[0,1.545]])).dot(
                   np.array([[1,70],[0,1]])).dot(
                   np.array([[1,0],[(1-1.545)/(800*1.545),1/1.545]])).dot(
                   np.array([[1,sf-470],[0,1]])).dot(
                   np.array([[1,0],[0,1.545]])).dot(
                   np.array([[1,80],[0,1]])).dot(
                   np.array([[1,0],[(1-1.545)/(800*1.545),1/1.545]])).dot(
                   np.array([[1,390],[0,1]]))

        # vertical position from the reference axis (vertical center of all lens, z=0 line) at ECEI array box
        zz = (np.arange(VN,0,-1) - 12.5)*14  # [mm]
        # angle against the reference axis at ECEI array box
        aa = np.zeros(np.size(zz))

        # vertical posistion and angle at rpos
        za = np.dot(abcd, [zz, aa])
        zpos = za[0][vn-1]/1000  # zpos [m]
        apos = za[1][vn-1]  # angle [rad] positive means the (z+) up-directed (divering from array to plasma)

        return zpos, apos
