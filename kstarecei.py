#!/usr/bin/env python2.7

# Author : Minjun J. Choi (mjchoi@nfri.re.kr)
#
# Description : This code reads the KSTAR ECEI data via the iKSTAR server
#
# Acknowledgement : Dr. Y. Nam and Prof. G.S. Yun
#
# Last updated
#  2018.02.15 : version 0.10; cold resonance positions

import numpy as np
import h5py

import matplotlib.pyplot as plt

import filtdata as ft

ENUM = 5000000  # totla number of samples in an ECEI channel
VN = 24  # number of vertical arrays


class KstarEcei(object):
    def __init__(self, shot, clist):
        self.shot = shot

        if 5073 < shot and shot < 6393:
            self.data_path = '/eceidata/exp_2011/ECEI/DATA_H5/'
        elif 7065 < shot and shot < 8225:
            self.data_path = '/eceidata/exp_2012/'
        elif 8639 < shot and shot < 9427:
            self.data_path = '/eceidata/exp_2013/'
        elif 9741 < shot and shot < 11723:
            self.data_path = '/eceidata/exp_2014/'
        elif 12272 < shot and shot < 14942:
            self.data_path = '/eceidata/exp_2015/'
        elif 14941 < shot and shot < 17356:
            self.data_path = '/eceidata2/exp_2016/'
        elif 17963 < shot and shot < 19392:
            self.data_path = '/eceidata2/exp_2017/'
        elif 19391 < shot and shot < 21779:
            self.data_path = '/eceidata2/exp_2018/'
        elif 21778 < shot and shot < 24100:
            self.data_path = '/eceidata2/exp_2019/'
        elif 24100 < shot and shot < 27400:
            self.data_path = '/eceidata2/exp_2020/'
        elif 27400 < shot:
            self.data_path = '/eceidata2/exp_2021/'

        self.clist = expand_clist(clist)

        if shot < 19392:
            self.cnidx1 = 6
            self.dev = self.clist[0][5]
        else:
            self.cnidx1 = 7
            self.dev = self.clist[0][5:7]

        # file name
        if shot < 19392:
            self.fname = "{:s}{:06d}/ECEI.{:06d}.{:s}FS.h5".format(self.data_path, shot, shot, self.dev)
        else:
            self.fname = "{:s}{:06d}/ECEI.{:06d}.{:s}.h5".format(self.data_path, shot, shot, self.dev)

        # get attributes
        with h5py.File(self.fname, 'r') as f:
            # get attributes
            dset = f['ECEI']
            self.tt = dset.attrs['TriggerTime'] # in [s]
            self.toff = self.tt[0]+0.001
            self.fs = dset.attrs['SampleRate'][0]*1000.0  # in [Hz] same sampling rate
            self.itf = dset.attrs['TFcurrent']*1.0e3  # [A]
            try:
                self.mode = dset.attrs['Mode'].strip().decode()
                if self.mode == 'O':
                    self.hn = 1  # harmonic number
                elif self.mode == 'X':
                    self.hn = 2
            except:
                print('#### no Mode attribute in file, default: 2nd X-mode ####')
                self.mode = 'X'
                self.hn = 2
            self.lo = dset.attrs['LoFreq']
            self.sf = dset.attrs['LensFocus']
            self.sz = dset.attrs['LensZoom']

            print('ECEI file = {}'.format(self.fname))

        # data quality
        self.good_channels = np.ones(len(self.clist))
        self.offlev = np.zeros(len(self.clist))
        self.offstd = np.zeros(len(self.clist))
        self.siglev = np.zeros(len(self.clist))
        self.sigstd = np.zeros(len(self.clist))

        # get channel posistion
        self.channel_position()

    def get_data(self, trange, norm=1, atrange=[1.0, 1.01], res=0, verbose=1):
        self.trange = trange

        # norm = 0 : no normalization
        # norm = 1 : normalization by trange average
        # norm = 2 : normalization by atrange average
        # res  = 0 : no resampling
        if norm == 0:
            if verbose == 1: print('Data is not normalized ECEI')
        elif norm == 1:
            if verbose == 1: print('Data is normalized by trange average ECEI')
        elif norm == 2:
            if verbose == 1: print('Data is normalized by atrange average ECEI')
        elif norm == 3:
            if verbose == 1: print('Data is normalized by the low pass signal')

        # get time base
        time, idx1, idx2, oidx1, oidx2 = self.time_base(trange)

        if norm == 2:
            _, aidx1, aidx2, _, _ = self.time_base(atrange)

        # get data
        with h5py.File(self.fname, 'r') as f:
            # time series length
            tnum = idx2 - idx1

            # number of channels
            cnum = len(self.clist)

            data = np.zeros((cnum, tnum))
            for i in range(cnum):
                node = "/ECEI/" + self.clist[i] + "/Voltage"

                ov = f[node][oidx1:oidx2]/10000.0
                v = f[node][idx1:idx2]/10000.0

                self.offlev[i] = np.median(ov)
                self.offstd[i] = np.std(ov)

                v = v - self.offlev[i]

                self.siglev[i] = np.median(v)
                self.sigstd[i] = np.std(v)

                if norm == 1:
                    v = v/np.mean(v) - 1
                elif norm == 2:
                    av = f[node][aidx1:aidx2]/10000.0
                    v = v/(np.mean(av) - self.offlev[i]) - 1
                elif norm == 3:
                    base_filter = ft.FftFilter('FFT_pass', self.fs, 0, 10)
                    base = base_filter.apply(v).real
                    v = v/base - 1

                data[i,:] = v

            self.data = data

        self.time = time

        # check data quality
        self.find_bad_channel()

        return time, data

    def time_base(self, trange):
        # using self.tt, self.toff, self.fs; get self.time
        tt = self.tt
        toff = self.toff
        fs = self.fs

        if int(len(tt)) == 2:
            pl = tt[1] - tt[0] + 0.1
            tt = [tt[0], pl, tt[1]]

        fulltime = []
        for i in range(0, int(len(tt)/3)):
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

        fulltime = fulltime[0:ENUM]

        idx = np.where((fulltime >= trange[0])*(fulltime <= trange[1]))
        idx1 = int(idx[0][0])
        idx2 = int(idx[0][-1]+1)

        if toff < 0:
            oidx = np.where((fulltime >= toff)*(fulltime <= toff+0.01))
        else:
            print('#### offset from end in KstarEcei.time_base ####')
            oidx = np.where((fulltime >= fulltime[-1]-0.01)*(fulltime <= fulltime[-1]))
        oidx1 = int(oidx[0][0])
        oidx2 = int(oidx[0][-1]+1)

        return fulltime[idx1:idx2], idx1, idx2, oidx1, oidx2

    def find_bad_channel(self):
        # auto-find bad 
        for c in range(len(self.clist)):
            # check signal level
            if self.siglev[c] > 0.01:
                ref = 100*self.offstd[c]/self.siglev[c]
            else:
                ref = 100            
            if ref > 30:
                self.good_channels[c] = 0
                print('LOW signal level channel {:s}, ref = {:g}%, siglevel = {:g} V'.format(self.clist[c], ref, self.siglev[c]))
            
            # check bottom saturation
            if self.offstd[c] < 0.001:
                self.good_channels[c] = 0
                print('SAT offset data  channel {:s}, offstd = {:g}%, offlevel = {:g} V'.format(self.clist[c], self.offstd[c], self.offlev[c]))

            # check top saturation.               
            if self.sigstd[c] < 0.001:
                self.good_channels[c] = 0
                print('SAT signal data  channel {:s}, offstd = {:g}%, siglevel = {:g} V'.format(self.clist[c], self.sigstd[c], self.siglev[c]))

    def channel_position(self):
        # get self.rpos, self.zpos, self.apos
        # NEED corrections using syndia
        
        me = 9.109e-31        # electron mass
        e = 1.602e-19       # charge
        mu0 = 4*np.pi*1e-7  # permeability
        ttn = 56*16         # total TF coil turns

        cnum = len(self.clist)
        self.rpos = np.zeros(cnum)  # R [m] of each channel
        self.zpos = np.zeros(cnum)  # z [m] of each channel
        self.apos = np.zeros(cnum)  # angle [rad] of each channel
        for c in range(0, cnum):
            vn = int(self.clist[c][(self.cnidx1):(self.cnidx1+2)])
            fn = int(self.clist[c][(self.cnidx1+2):(self.cnidx1+4)])

            # assume cold resonance with Bt ~ 1/R
            self.rpos[c] = self.hn*e*mu0*ttn*self.itf/((2*np.pi)**2*me*((fn - 1)*0.9 + 2.6 + self.lo)*1e9)

            # get vertical position and angle at rpos
            self.zpos[c], self.apos[c] = self.beam_path(self.rpos[c], vn)

    def show_ch_position(self):
        fig, (a1) = plt.subplots(1,1, figsize=(6,6))
        a1.plot(self.rpos, self.zpos, 'o')
        for c, cname in enumerate(self.clist):
            a1.annotate(cname[5:], (self.rpos[c], self.zpos[c]))
        a1.set_title('ABCD positions (need corrections from syndia)')
        a1.set_xlabel('R [m]')
        a1.set_ylabel('z [m]')
        plt.show()

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

    def get_abcd(self, sf, sz, Rinit):
        # ABCD matrix
        if self.dev == 'L':
            sp = 3350 - Rinit*1000  # [m] -> [mm]
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
        elif self.dev == 'H':
            sp = 3350 - Rinit*1000
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
            if self.shot > 12297:  # since 2015 campaign
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
        elif self.dev == 'G':
            sp = 3150 - Rinit*1000
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
        elif self.dev == 'GT':
            sp = 2300 - Rinit*1000
            abcd = np.array([[1,sp+(1954-sz)],[0,1]]).dot(
                   np.array([[1,0],[(1.52-1)/(-1000),1.52]])).dot(
                   np.array([[1,160],[0,1]])).dot(
                   np.array([[1,0],[(1-1.52)/(1000*1.52),1/1.52]])).dot(
                   np.array([[1,2280-(1954+160-sz)],[0,1]])).dot(
                   np.array([[1,0],[(1.52-1)/1000,1.52]])).dot(
                   np.array([[1,20],[0,1]])).dot(
                   np.array([[1,0],[0,1/1.52]])).dot(
                   np.array([[1,4288-(2280+20)-sf],[0,1]])).dot(
                   np.array([[1,0],[(1.52-1)/(-1200),1.52]])).dot(
                   np.array([[1,140],[0,1]])).dot(
                   np.array([[1,0],[(1-1.52)/(1200*1.52),1/1.52]])).dot(
                   np.array([[1,4520-(4288+140-sf)],[0,1]])).dot(
                   np.array([[1,0],[0,1.52]])).dot(
                   np.array([[1,30],[0,1]])).dot(
                   np.array([[1,0],[0,1/1.52]])).dot(
                   np.array([[1,4940-(4520+30)],[0,1]]))
        elif self.dev == 'GR':
            sp = 2300 - Rinit*1000
            abcd = np.array([[1,sp+(1954-sz)],[0,1]]).dot(
                   np.array([[1,0],[(1.52-1)/(-1000),1.52]])).dot(
                   np.array([[1,160],[0,1]])).dot(
                   np.array([[1,0],[(1-1.52)/(1000*1.52),1/1.52]])).dot(
                   np.array([[1,2280-(1954+160-sz)],[0,1]])).dot(
                   np.array([[1,0],[(1.52-1)/1000,1.52]])).dot(
                   np.array([[1,20],[0,1]])).dot(
                   np.array([[1,0],[0,1/1.52]])).dot(
                   np.array([[1,4288-(2280+20)-sf],[0,1]])).dot(
                   np.array([[1,0],[(1.52-1)/(-1200),1.52]])).dot(
                   np.array([[1,140],[0,1]])).dot(
                   np.array([[1,0],[(1-1.52)/(1200*1.52),1/1.52]])).dot(
                   np.array([[1,4520-(4288+140-sf)],[0,1]])).dot(
                   np.array([[1,0],[0,1.52]])).dot(
                   np.array([[1,30],[0,1]])).dot(
                   np.array([[1,0],[0,1/1.52]])).dot(
                   np.array([[1,4940-(4520+30)],[0,1]]))
        elif self.dev == 'HT':
            sp = 2300 - Rinit*1000
            abcd = np.array([[1,sp+2586],[0,1]]).dot(
                   np.array([[1,0],[0,1.52]])).dot(
                   np.array([[1,140],[0,1]])).dot(
                   np.array([[1,0],[(1-1.52)/(770*1.52),1/1.52]])).dot(
                   np.array([[1,4929-(2586+140)-sz],[0,1]])).dot(
                   np.array([[1,0],[(1.52-1)/(1200),1.52]])).dot(
                   np.array([[1,20],[0,1]])).dot(
                   np.array([[1,0],[(1-1.52)/(-1200*1.52),1/1.52]])).dot(
                   np.array([[1,5919-(4929+20-sz)-sf],[0,1]])).dot(
                   np.array([[1,0],[(1.52-1)/(-1300),1.52]])).dot(
                   np.array([[1,130],[0,1]])).dot(
                   np.array([[1,0],[(1-1.52)/(1300*1.52),1/1.52]])).dot(
                   np.array([[1,6489-(5919+130-sf)],[0,1]])).dot(
                   np.array([[1,0],[0,1.52]])).dot(
                   np.array([[1,25.62],[0,1]])).dot(
                   np.array([[1,0],[0,1/1.52]])).dot(
                   np.array([[1,7094.62-(6489+25.62)],[0,1]]))

        return abcd


def expand_clist(clist):
    # IN : List of channel names (e.g. 'ECEI_G1201-1208' or 'ECEI_GT1201-1208').
    # OUT : Expanded list (e.g. 'ECEI_G1201', ..., 'ECEI_G1208')

    # KSTAR ECEI
    exp_clist = []
    for c in range(len(clist)):
        if len(clist[c]) < 15:
            exp_clist.append(clist[c])
            continue
        elif 'ECEI' in clist[c] and len(clist[c]) == 15: # before 2018
            vi = int(clist[c][6:8])
            fi = int(clist[c][8:10])
            vf = int(clist[c][11:13])
            ff = int(clist[c][13:15])
            ip = 6
        elif 'ECEI' in clist[c] and len(clist[c]) == 16: # since 2018
            vi = int(clist[c][7:9])
            fi = int(clist[c][9:11])
            vf = int(clist[c][12:14])
            ff = int(clist[c][14:16])
            ip = 7
        
        for v in range(vi, vf+1):
            for f in range(fi, ff+1):
                exp_clist.append(clist[c][0:ip] + '{:02d}{:02d}'.format(v, f))

    clist = exp_clist

    return clist
