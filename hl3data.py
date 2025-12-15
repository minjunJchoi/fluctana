"""
Author: Minjun J. Choi (mjchoi@kfe.re.kr)

Description: This code reads the HL-3 data saved in hdf5 format

Acknowledgement: Dr. M. Jiang, Dr. Y. Zhang, Mr. Y. Zhiyi
"""

import os
import numpy as np
import h5py

import matplotlib.pyplot as plt

import filtdata as ft

# HL3 data path 
HL3_PATH = '/home/users/mjchoi/data/HL-3/' # on nKSTAR
if not os.path.exists(HL3_PATH):
    HL3_PATH = '/Users/mjchoi/Work/data/HL-3/' # on local machine

class Hl3Data():
    def __init__(self, shot, clist):

        self.shot = shot

        self.clist = self.expand_clist(clist)

        # hdf5 file name
        self.fname = "{:s}{:06d}/HL3.{:06d}.h5".format(HL3_PATH, shot, shot)

        # get channel posistions
        self.channel_position()

        # data quality
        self.good_channels = np.ones(len(self.clist))
        self.offlev = np.zeros(len(self.clist))
        self.offstd = np.zeros(len(self.clist))
        self.siglev = np.zeros(len(self.clist))
        self.sigstd = np.zeros(len(self.clist))

        self.time = None
        self.data = None

    def get_data(self, trange, norm=1, atrange=[1.0, 1.01], res=0, verbose=1):
        self.trange = trange

        if norm == 0:
            if verbose == 1: print('Data is not normalized')
        elif norm == 1:
            if verbose == 1: print('Data is normalized by trange average')
        elif norm == 2:
            if verbose == 1: print('Data is normalized by atrange average')
        elif norm == 3:
            if verbose == 1: print('Data is normalized by the low pass signal')

        if os.path.exists(self.fname):
            # read data from hdf5
            with h5py.File(self.fname, 'r') as fin:
                print('Read HL3 data from hdf5 file')

                # read time base and get tidx 
                if self.clist[0].startswith('ECEI_'):
                    self.time = np.squeeze(fin.get('ECEI/TIME'))
                elif self.clist[0].startswith('BES'):
                    self.time = np.squeeze(fin.get('BES/TIME'))
                elif self.clist[0].startswith('DMW_ECE1'):
                    self.time = np.squeeze(fin.get('DMW_ECE/TIME'))
                elif self.clist[0].startswith('MDR_'):
                    self.time = np.squeeze(fin.get('MDR_ECE/TIME'))
                else:
                    self.time = np.squeeze(fin.get(f'{self.clist[0]}/TIME'))
                
                if self.time[-1] > 1000: # unit of ms
                    self.time = self.time/1000 # [ms] -> [s]

                # get sampling frequency 
                self.fs = round(1/(self.time[1] - self.time[0])/1000)*1000.0

                # get tidx for signal, offset, and atrange
                idx1 = round((max(trange[0],self.time[0]) + 1e-8 - self.time[0])*self.fs) 
                idx2 = round((min(trange[1],self.time[-1]) - 1e-8 - self.time[0])*self.fs)

                oidx1 = round((2.8 + 1e-8 - self.time[0])*self.fs) 
                oidx2 = round((2.88 - 1e-8 - self.time[0])*self.fs)

                aidx1 = round((max(atrange[0],self.time[0]) + 1e-8 - self.time[0])*self.fs) 
                aidx2 = round((min(atrange[1],self.time[-1]) - 1e-8 - self.time[0])*self.fs)

                # get time for trange
                self.time = self.time[idx1:idx2+1]

                # get data
                for i, cname in enumerate(self.clist):
                    # load data
                    ov = fin.get(f'{cname}/DATA')[0][oidx1:oidx2+1] # mydp.m returns (1, N) shape data
                    v = fin.get(f'{cname}/DATA')[0][idx1:idx2+1]

                    self.offlev[i] = np.median(ov)
                    self.offstd[i] = np.std(ov)

                    v = v - self.offlev[i]

                    self.siglev[i] = np.median(v)
                    self.sigstd[i] = np.std(v)

                    # normalization 
                    if norm == 1:
                        v = v/np.mean(v) - 1
                    elif norm == 2:
                        av = fin.get(f'{cname}/DATA')[0][aidx1:aidx2+1] # atrange signal [int32] -> [V]
                        v = v/(np.mean(av) - self.offlev[i]) - 1
                    elif norm == 3:
                        base_filter = ft.FirFilter('FIR_pass', self.fs, 0, 10, 0.01)
                        base = base_filter.apply(v).real
                        v = v/base - 1

                    # expand dimension - concatenate
                    v = np.expand_dims(v, axis=0)
                    if self.data is None:
                        self.data = v
                    else:
                        self.data = np.concatenate((self.data, v), axis=0)

        # check data quality
        self.find_bad_channel()

        return self.time, self.data

    # def get_multi_data(self, time_list=None, tspan=1e-3, norm=0, res=0, verbose=1):
    #     if norm == 0:
    #         if verbose == 1: print('Data is not normalized')
    #     elif norm == 1:
    #         if verbose == 1: print('Data is normalized by time average')

    #     self.time_list = time_list

    #     # open file   
    #     with h5py.File(self.fname, 'r') as fin:

    #         # get fs
    #         full_time = np.squeeze(fin.get('TIME'))
    #         self.fs = round(1/(full_time[1] - full_time[0])/1000)*1000.0        

    #         # get data size
    #         idx1 = round((time_list[0] - tspan/2 + 1e-8 - full_time[0])*self.fs) 
    #         idx2 = round((time_list[0] + tspan/2 - 1e-8 - full_time[0])*self.fs) 
    #         tnum = len(full_time[idx1:idx2+1])

    #         # get multi time and data 
    #         self.multi_time = np.zeros((len(time_list), tnum))
    #         self.multi_data = np.zeros((len(self.clist), len(time_list), tnum))

    #         for i, cname in enumerate(self.clist):
    #             for j, tp in enumerate(time_list):
    #                 # get tidx 
    #                 idx1 = round((tp - tspan/2 + 1e-8 - full_time[0])*self.fs) 
    #                 idx2 = round((tp + tspan/2 - 1e-8 - full_time[0])*self.fs) 

    #                 # load time
    #                 if i == 0:
    #                     self.multi_time[j,:] = full_time[idx1:idx2+1]

    #                 # load data
    #                 v = np.squeeze(fin.get(cname)[idx1:idx2+1])

    #                 # normalize by std if norm == 1
    #                 if norm == 1:
    #                     v = v/np.mean(v) - 1

    #                 # add data 
    #                 self.multi_data[i,j,:] = v

    #             print('Data added', cname)

    #     return self.multi_time, self.multi_data

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
        
        cnum = len(self.clist)
        self.rpos = np.zeros(cnum)  # R [m] of each channel
        self.zpos = np.zeros(cnum)  # z [m] of each channel
        self.apos = np.zeros(cnum)  # angle [rad] of each channel

        if os.path.exists(self.fname):
            # open file to read channel positions
            with h5py.File(self.fname, 'r') as fin:
                for c, cname in enumerate(self.clist):                    
                    dset = fin[cname]
                    self.rpos[c] = dset.attrs['RPOS'] # [m]
                    self.zpos[c] = dset.attrs['ZPOS'] # [m]
                    self.apos[c] = dset.attrs['APOS'] # [rad]

    def show_ch_position(self):
        fig, (a1) = plt.subplots(1,1, figsize=(6,6))
        a1.plot(self.rpos, self.zpos, 'o')
        for c, cname in enumerate(self.clist):
            if self.clist[0].startswith('ECEI_'):
                a1.annotate(cname[5:], (self.rpos[c], self.zpos[c]))
            elif self.clist[0].startswith('BES'):
                a1.annotate(cname[3:], (self.rpos[c], self.zpos[c]))
        a1.set_title('Positions')
        a1.set_xlabel('R [m]')
        a1.set_ylabel('z [m]')
        plt.show()

    def expand_clist(self, clist):
        # IN : List of channel names (e.g. 'ECEI_1201-1208').
        # OUT : Expanded list (e.g. 'ECEI_1201', ..., 'ECEI_1208')

        # HL-3 ECEI
        if clist[0].startswith('ECEI_'):
            exp_clist = []

            for cname in clist:
                if len(cname) < 12:
                    exp_clist.append(cname)
                else: 
                    vi = int(cname[5:7])
                    fi = int(cname[7:9])
                    vf = int(cname[10:12])
                    ff = int(cname[12:14])
            
                    for v in range(vi, vf+1):
                        for f in range(fi, ff+1):
                            exp_clist.append(cname[0:5] + '{:02d}{:02d}'.format(v, f))

            clist = exp_clist

        # HL-3 BES
        if clist[0].startswith('BES'):
            exp_clist = []

            for cname in clist:
                if len(cname) < 6:
                    exp_clist.append(cname)
                else: 
                    i = int(cname[3:5])
                    f = int(cname[6:8])
            
                    for j in range(i, f+1):
                        exp_clist.append(cname[0:3] + '{:02d}'.format(j))

            clist = exp_clist            

        return clist


def expand_clist(self, clist):
    # IN : List of channel names (e.g. 'ECEI_1201-1208').
    # OUT : Expanded list (e.g. 'ECEI_1201', ..., 'ECEI_1208')

    # HL-3 ECEI
    if clist[0].startswith('ECEI_'):
        exp_clist = []

        for cname in clist:
            if len(cname) < 12:
                exp_clist.append(cname)
            else: 
                vi = int(cname[5:7])
                fi = int(cname[7:9])
                vf = int(cname[10:12])
                ff = int(cname[12:14])
        
                for v in range(vi, vf+1):
                    for f in range(fi, ff+1):
                        exp_clist.append(cname[0:5] + '{:02d}{:02d}'.format(v, f))

        clist = exp_clist

    # HL-3 BES
    if clist[0].startswith('BES'):
        exp_clist = []

        for cname in clist:
            if len(cname) < 6:
                exp_clist.append(cname)
            else: 
                i = int(cname[3:5])
                f = int(cname[6:8])
        
                for j in range(i, f+1):
                    exp_clist.append(cname[0:3] + '{:02d}'.format(j))

        clist = exp_clist

    return clist