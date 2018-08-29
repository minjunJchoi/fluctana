#!/usr/bin/env python2.7

# Author : Minjun J. Choi (mjchoi@nfri.re.kr)

import h5py
import numpy as np

class KstarEceiInfo(object):
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
        elif 19391 < shot:
            self.data_path = '/eceidata2/exp_2018/'

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
            self.bt = dset.attrs['TFcurrent']*0.0995556  # [kA] -> [T]
            self.mode = dset.attrs['Mode'].strip()
            if self.mode is 'O':
                self.hn = 1  # harmonic number
            elif self.mode is 'X':
                self.hn = 2
            self.lo = dset.attrs['LoFreq']
            self.sf = dset.attrs['LensFocus']
            self.sz = dset.attrs['LensZoom']

            print 'ECEI file = {}'.format(self.fname)

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
            abcd = np.array([[1,1350],[0,1]])
        elif self.dev == 'GR':
            abcd = np.array([[1,1350],[0,1]])
        elif self.dev == 'HT':
            abcd = np.array([[1,1350],[0,1]])

        return abcd


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
                    exp_clist.append(clist[c][0:6] + '%02d' % v + '%02d' % f)
        elif 'ECEI' in clist[c] and len(clist[c]) == 16: # since 2018
            vi = int(clist[c][7:9])
            fi = int(clist[c][9:11])
            vf = int(clist[c][12:14])
            ff = int(clist[c][14:16])

            for v in range(vi, vf+1):
                for f in range(fi, ff+1):
                    exp_clist.append(clist[c][0:7] + '%02d' % v + '%02d' % f)
        else:
            exp_clist.append(clist[c])
    clist = exp_clist

    return clist
