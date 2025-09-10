"""
Author : Minjun J. Choi (mjchoi@kfe.re.kr)

Description : This code reads the KSTAR BES data from the KSTAR MDSplus server

Acknowledgement : Dr. J.W. Kim

Last updated
2024.11.26 : version 0.10; cold resonance positions
"""

import filtdata as ft

from MDSplus import Connection
# from MDSplus import DisconnectFromMds
# from MDSplus._mdsshr import MdsException

import os
import numpy as np
import matplotlib.pyplot as plt
import h5py

# BES tree
BES_TREE = 'KSTAR'
BES_PATH = '/home/users/mjchoi/bes_data/' # on nKSTAR
BES_PATH = '/UKSTAR_HOME/mjchoi/data/KSTAR/bes_data/' # on uKSTAR
# BES_PATH = '/Users/mjchoi/Work/data/KSTAR/bes_data/' # on local machine

# on uKSTAR
class KstarBes(Connection):
    def __init__(self, shot, clist, savedata):
        super(KstarBes,self).__init__('mdsr.kstar.kfe.re.kr:8005')  # call __init__ in Connection

# on local machine
# class KstarBes(object):
#     def __init__(self, shot, clist):

        self.shot = shot

        self.clist = self.expand_clist(clist)

        self.good_channels = np.ones(len(self.clist))
        
        # hdf5 file name
        self.fname = "{:s}{:06d}/BES.{:06d}.h5".format(BES_PATH, shot, shot)
        
        # get channel posistions
        self.channel_position()            

        # save MDSplus data in hdf5 format if necessary
        if os.path.exists(self.fname) == False and savedata == True:
            print('Reformat BES data to hdf5 file') 
            self.reformat_hdf5()

        self.time = None
        self.data = None

    def reformat_hdf5(self):
        # make directory if necessary
        path = '{:s}{:06d}'.format(BES_PATH, self.shot)
        if os.path.exists(path) == False:
            os.makedirs(path)        

        # open MDSplus tree
        self.openTree(BES_TREE, self.shot)

        # get full clist 
        clist = self.expand_clist(['BES_0101-0416'])

        # get time base
        time_node = 'dim_of(\BES_0101:FOO)'
        time = self.get(time_node).data()

        with h5py.File(self.fname, 'w') as fout:
            fout.create_dataset('TIME', data=time)

            for c, cname in enumerate(clist):
                # get and add data after multiply 1e6 and transform it into int32
                data = self.get(f'\{cname}:FOO').data()
                data = (data * 1e6).astype(np.int32)
                fout.create_dataset(cname, data=data, compression="gzip", compression_opts=5)

                # add channel positions as attributes
                dset = fout[cname]
                dset.attrs['RPOS'] = self.get(f'\{cname}:RPOS') / 1000 # [mm] -> [m]
                dset.attrs['ZPOS'] = self.get(f'\{cname}:VPOS') / 1000 # [mm] -> [m]

                print('Added', cname)

        print('Saved', self.fname)

        # close tree
        self.closeTree(BES_TREE, self.shot)

        # disconnect 
        self.disconnect()

    def get_data(self, trange, norm=0, atrange=[1.0, 1.1], res=0, verbose=1):
        if norm == 0:
            if verbose == 1: print('Data is not normalized')
        elif norm == 1:
            if verbose == 1: print('Data is normalized by trange {:g}-{:g} average'.format(trange[0],trange[1]))

        self.trange = trange

        if os.path.exists(self.fname):
            # read data from hdf5
            with h5py.File(self.fname, 'r') as fin:
                print('Read BES data from hdf5 file')

                # read time base and get tidx 
                self.time = np.array(fin.get('TIME'))

                # get sampling frequency 
                self.fs = round(1/(self.time[1] - self.time[0])/1000)*1000.0

                # subsample 
                idx1 = round((max(trange[0],self.time[0]) + 1e-8 - self.time[0])*self.fs) 
                idx2 = round((min(trange[1],self.time[-1]) - 1e-8 - self.time[0])*self.fs)

                aidx1 = round((max(atrange[0],self.time[0]) + 1e-8 - self.time[0])*self.fs) 
                aidx2 = round((min(atrange[1],self.time[-1]) - 1e-8 - self.time[0])*self.fs)                

                self.time = self.time[idx1:idx2+1]

                # get data
                for i, cname in enumerate(self.clist):
                    # load data
                    v = np.array(fin.get(cname)[idx1:idx2+1]) / 1e6 # [int32] -> [V]

                    # normalize by std if norm == 1
                    if norm == 1:
                        v = v/np.mean(v) - 1
                    elif norm == 2:
                        av = np.array(fin.get(cname)[aidx1:aidx2+1]) / 1e6 # [int32] -> [V] atrange signal
                        v = v/np.mean(av) - 1
                    elif norm == 3:
                        base_filter = ft.FftFilter('FFT_pass', self.fs, 0, 10)
                        base = base_filter.apply(v).real
                        v = v/base - 1

                    # expand dimension - concatenate
                    v = np.expand_dims(v, axis=0)
                    if self.data is None:
                        self.data = v
                    else:
                        self.data = np.concatenate((self.data, v), axis=0)
        else: 
            print('Load BES data from MDSplus server')
            ################################# setTimeContext cannot be used ofr BES data
            # load data from MDSplus
            self.openTree(BES_TREE, self.shot)

            # time
            tnode = f'dim_of(\{self.clist[0]}:FOO)'
            self.time = self.get(tnode).data()
            self.fs = round(1/(self.time[1] - self.time[0])/1000)*1000.0

            # subsample 
            idx1 = round((max(trange[0],self.time[0]) + 1e-8 - self.time[0])*self.fs) 
            idx2 = round((min(trange[1],self.time[-1]) - 1e-8 - self.time[0])*self.fs)

            aidx1 = round((max(atrange[0],self.time[0]) + 1e-8 - self.time[0])*self.fs) 
            aidx2 = round((min(atrange[1],self.time[-1]) - 1e-8 - self.time[0])*self.fs)                

            self.time = self.time[idx1:idx2+1]

            # data
            for i, cname in enumerate(self.clist):
                dnode = f'\{cname}:FOO'
                full_v = self.get(dnode).data()
                v = full_v[idx1:idx2+1]

                # normalization 
                if norm == 1:
                    v = v/np.mean(v) - 1
                elif norm == 2:
                    av = full_v[aidx1:aidx2+1]
                    v = v/np.mean(av) - 1
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

                print('Loaded', cname)

            # close tree
            self.closeTree(BES_TREE, self.shot)

            # disconnect 
            self.disconnect()

        return self.time, self.data

    def get_multi_data(self, time_list=None, tspan=1e-3, norm=0, res=0, verbose=1):
        if norm == 0:
            if verbose == 1: print('Data is not normalized')
        elif norm == 1:
            if verbose == 1: print('Data is normalized by time average')

        self.time_list = time_list

        # open file   
        with h5py.File(self.fname, 'r') as fin:

            # get fs
            full_time = np.array(fin.get('TIME'))
            self.fs = round(1/(full_time[1] - full_time[0])/1000)*1000.0        

            # get data size
            # idx1 = round((time_list[0] - tspan/2 + 1e-8 - full_time[0])*self.fs) 
            # # idx2 = round((time_list[0] + tspan/2 - 1e-8 - full_time[0])*self.fs) 
            # idx2 = idx1 + int(tspan*self.fs)
            tnum = int(tspan*self.fs)

            # get multi time and data 
            self.multi_time = np.zeros((len(time_list), tnum))
            self.multi_data = np.zeros((len(self.clist), len(time_list), tnum))

            for i, cname in enumerate(self.clist):
                for j, tp in enumerate(time_list):
                    # get tidx 
                    idx1 = round((tp - tspan/2 + 1e-8 - full_time[0])*self.fs) 
                    # idx2 = round((tp + tspan/2 - 1e-8 - full_time[0])*self.fs) 
                    idx2 = idx1 + tnum

                    # load time
                    if i == 0:
                        self.multi_time[j,:] = full_time[idx1:idx2]

                    # load data
                    v = np.array(fin.get(cname)[idx1:idx2]) / 1e6 # [int32] -> [V]

                    # normalize by std if norm == 1
                    if norm == 1:
                        v = v/np.mean(v) - 1

                    # add data 
                    self.multi_data[i,j,:] = v

                print('Data added', cname)

        return self.multi_time, self.multi_data

    def channel_position(self):
        # get channel position from MDSplus server
        cnum = len(self.clist)
        self.rpos = np.zeros(cnum)  # R [m]
        self.zpos = np.zeros(cnum)  # z [m]
        self.apos = np.zeros(cnum)  # angle [rad]

        if os.path.exists(self.fname):
            # open file   
            with h5py.File(self.fname, 'r') as fin:
                for c, cname in enumerate(self.clist):
                    # self.rpos[c] = np.array(fin.get(cname+'_RPOS')) / 1000 # [mm] -> [m]
                    # self.zpos[c] = np.array(fin.get(cname+'_ZPOS')) / 1000 # [mm] -> [m]
                    
                    dset = fin[cname]
                    self.rpos[c] = dset.attrs['RPOS'] # [m]
                    self.zpos[c] = dset.attrs['ZPOS'] # [m]
        else:
            # open MDSplus tree
            self.openTree(BES_TREE, self.shot)

            # rpos and zpos
            for c, cname in enumerate(self.clist):
                self.rpos[c] = self.get(f'\{cname}:RPOS') / 1000 # [mm] -> [m]
                self.zpos[c] = self.get(f'\{cname}:VPOS') / 1000 # [mm] -> [m]

            # close tree
            self.closeTree(BES_TREE, self.shot)

    def expand_clist(self, clist):
        # IN : List of channel names (e.g. 'BES_0101-0416')
        # OUT : Expanded list (e.g. 'BES_0101', 'BES_0102', ... 'BES_0416')

        # KSTAR BES
        exp_clist = []
        for c in range(len(clist)):
            if 'BES' in clist[c] and len(clist[c]) == 13:
                vi = int(clist[c][4:6])
                fi = int(clist[c][6:8])
                vf = int(clist[c][9:11])
                ff = int(clist[c][11:13])
                ip = 4

                for v in range(vi, vf+1):
                    for f in range(fi, ff+1):
                        exp_clist.append(clist[c][0:ip] + '{:02d}{:02d}'.format(v, f))
            else:
                exp_clist.append(clist[c])
        clist = exp_clist

        return clist
