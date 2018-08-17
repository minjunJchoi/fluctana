#!/usr/bin/env python2.7

# Author : Minjun J. Choi (mjchoi@nfri.re.kr)

import h5py

class KstarEceiInfo(object):
    def __init__(self, shot, clist):
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

        if shot < 19392:
            self.cnidx1 = 6
            self.dev = clist[0][5]
        else:
            self.cnidx1 = 7
            self.dev = clist[0][5:7]

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
            self.fs = dset.attrs['SampleRate'][0]*1000.0  # in [Hz]
            self.bt = dset.attrs['TFcurrent']*0.0995556  # [kA] -> [T]
            self.mode = dset.attrs['Mode'].strip()
            if self.mode is 'O':
                self.hn = 1  # harmonic number
            elif self.mode is 'X':
                self.hn = 2
            self.lo = dset.attrs['LoFreq']
            self.sf = dset.attrs['LensFocus']
            self.sz = dset.attrs['LensZoom']

    def get_abcd(self, Rinit):
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
