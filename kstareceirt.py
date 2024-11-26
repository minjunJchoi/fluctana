"""
Author : Minjun J. Choi (mjchoi@kfe.re.kr)

Description : This code reads the KSTAR ECEI data for shot number > 35000 (2024 fall campaign ~)

Acknowledgement : Dr. J. Lee

Last updated
2024.11.26 : version 0.10; cold resonance positions
"""

import os
import numpy as np
import h5py

import matplotlib.pyplot as plt

import filtdata as ft

from MDSplus import Connection

# ECEI tree 
ECEI_TREE = 'KSTAR'
ECEI_PATH = '/home/mjchoi/data/KSTAR/ecei_data/' # on ukstar
# ECEI_PATH = '/Users/mjchoi/Work/data/KSTAR/ecei_data/' # on local machine

# on iKSTAR/uKSTAR
class KstarEceiRemote(Connection):
    def __init__(self, shot, clist):
        super(KstarEceiRemote,self).__init__('mdsr.kstar.kfe.re.kr:8005')  # call __init__ in Connection

# on local machine
# class KstarEcei(object):
#     def __init__(self, shot, clist):

        self.shot = shot

        self.data_path = ECEI_PATH

        self.clist = self.expand_clist(clist)

        self.cnidx1 = 7
        self.dev = self.clist[0][5:7]

        # file name
        self.fname = "{:s}{:06d}/ECEI.{:06d}.{:s}.h5".format(self.data_path, shot, shot, self.dev)

        # if files do not exist, read the data from KSTAR MDSplus
        if os.path.exists(self.fname) == False:
            print('reformat ECEI data to hdf5 file') 
            self.reformat_hdf5()

        # data quality
        self.good_channels = np.ones(len(self.clist))
        self.offlev = np.zeros(len(self.clist))
        self.offstd = np.zeros(len(self.clist))
        self.siglev = np.zeros(len(self.clist))
        self.sigstd = np.zeros(len(self.clist))

        # get channel posistion
        self.channel_position()

        self.time = None
        self.data = None

    def reformat_hdf5(self):
        # make directory if necessary
        path = '{:s}/{:06d}'.format(self.data_path, self.shot)
        if os.path.exists(path) == False:
            os.makedirs(path)        
        self.fname = "{:s}{:06d}/ECEI.{:06d}.{:s}.h5".format(self.data_path, self.shot, self.shot, self.dev)

        # open MDSplus tree
        self.openTree(ECEI_TREE, self.shot)

        # get full clist 
        clist = self.expand_clist([f'ECEI_{self.dev}0101-2408'])

        # get time base
        time_node = f'dim_of(\ECEI_{self.dev}0101:FOO)'
        time = self.get(time_node).data()

        with h5py.File(self.fname, 'w') as fout:
            fout.create_dataset('TIME', data=time)

            for cname in clist:
                # get and add data 
                data = self.get('\{:s}:FOO'.format(cname)).data()
                fout.create_dataset(cname, data=data)
                
                print('added', cname)

        print('saved', self.fname)        

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

        # get data
        with h5py.File(self.fname, 'r') as fin:
            # read time base and get tidx 
            self.time = np.array(fin.get('TIME'))

            # get sampling frequency 
            self.fs = round(1/(self.time[1] - self.time[0])/1000)*1000.0

            # get tidx for signal, offset, and atrange
            idx1 = round((max(trange[0],self.time[0]) + 1e-8 - self.time[0])*self.fs) 
            idx2 = round((min(trange[1],self.time[-1]) + 1e-8 - self.time[0])*self.fs)

            oidx1 = round((-0.08 + 1e-8 - self.time[0])*self.fs) 
            oidx2 = round((-0.02 + 1e-8 - self.time[0])*self.fs)

            aidx1 = round((max(atrange[0],self.time[0]) + 1e-8 - self.time[0])*self.fs) 
            aidx2 = round((min(atrange[1],self.time[-1]) + 1e-8 - self.time[0])*self.fs)

            # get time for trange
            self.time = self.time[idx1:idx2]

            # get data
            for i, cname in enumerate(self.clist):
                # load data
                ov = np.array(fin.get(cname)[oidx1:oidx2]) # offset
                v = np.array(fin.get(cname)[idx1:idx2]) # signal
                
                self.offlev[i] = np.median(ov)
                self.offstd[i] = np.std(ov)

                v = v - self.offlev[i]

                self.siglev[i] = np.median(v)
                self.sigstd[i] = np.std(v)

                # normalization 
                if norm == 1:
                    v = v/np.mean(v) - 1
                elif norm == 2:
                    av = np.array(fin.get(cname)[aidx1:aidx2]) # atrange signal
                    v = v/(np.mean(av) - self.offlev[i]) - 1
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
    #         full_time = np.array(fin.get('TIME'))
    #         self.fs = round(1/(full_time[1] - full_time[0])/1000)*1000.0        

    #         # get data size
    #         idx1 = round((time_list[0] - tspan/2 + 1e-8 - full_time[0])*self.fs) 
    #         idx2 = round((time_list[0] + tspan/2 + 1e-8 - full_time[0])*self.fs) 
    #         tnum = len(full_time[idx1:idx2])

    #         # get multi time and data 
    #         self.multi_time = np.zeros((len(time_list), tnum))
    #         self.multi_data = np.zeros((len(self.clist), len(time_list), tnum))

    #         for i, cname in enumerate(self.clist):
    #             for j, tp in enumerate(time_list):
    #                 # get tidx 
    #                 idx1 = round((tp - tspan/2 + 1e-8 - full_time[0])*self.fs) 
    #                 idx2 = round((tp + tspan/2 + 1e-8 - full_time[0])*self.fs) 

    #                 # load time
    #                 if i == 0:
    #                     self.multi_time[j,:] = full_time[idx1:idx2]

    #                 # load data
    #                 v = np.array(fin.get(cname)[idx1:idx2])

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
        
        me = 9.109e-31        # electron mass
        e = 1.602e-19       # charge
        mu0 = 4*np.pi*1e-7  # permeability
        ttn = 56*16         # total TF coil turns

        self.hn = 2 ###########
        self.itf = 18000 ############# toroidal field coil current [A]
        self.lo = 75 ############### [GHz]
        self.sf = 498 ########## [mm]
        self.sz = 250 ########### [mm]

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
        zz = (np.arange(24,0,-1) - 12.5)*14  # [mm]
        # angle against the reference axis at ECEI array box
        aa = np.zeros(np.size(zz))

        # vertical posistion and angle at rpos
        za = np.dot(abcd, [zz, aa])
        zpos = za[0][vn-1]/1000  # zpos [m]
        apos = za[1][vn-1]  # angle [rad] positive means the (z+) up-directed (divering from array to plasma)

        return zpos, apos

    def get_abcd(self, sf, sz, Rinit):
        # ABCD matrix
        if self.dev == 'GT':
            sp = 2300 - Rinit*1000
            abcd = np.array([[1,sp+(2025-sz)],[0,1]]).dot(
                   np.array([[1,0],[(1.52-1)/(-1000*1),1.52/1]])).dot(
                   np.array([[1,160],[0,1]])).dot(
                   np.array([[1,0],[(1-1.52)/(1000*1.52),1/1.52]])).dot(
                   np.array([[1,2280-(2025+160-sz)],[0,1]])).dot(
                   np.array([[1,0],[(1.52-1)/(1000*1),1.52/1]])).dot(
                   np.array([[1,20],[0,1]])).dot(
                   np.array([[1,0],[0,1/1.52]])).dot(
                   np.array([[1,(4343-sf)-(2280+20)],[0,1]])).dot(
                   np.array([[1,0],[(1.52-1)/(-1200*1),1.52/1]])).dot(
                   np.array([[1,140],[0,1]])).dot(
                   np.array([[1,0],[(1-1.52)/(1200*1.52),1/1.52]])).dot(
                   np.array([[1,4520-(4343+140-sf)],[0,1]])).dot(
                   np.array([[1,0],[0,1.52/1]])).dot(
                   np.array([[1,30],[0,1]])).dot(
                   np.array([[1,0],[0,1/1.52]])).dot(
                   np.array([[1,4940-(4520+30)],[0,1]]))
        elif self.dev == 'GR':
            sp = 2300 - Rinit*1000
            abcd = np.array([[1,sp+(2025-sz)],[0,1]]).dot(
                   np.array([[1,0],[(1.52-1)/(-1000),1.52/1]])).dot(
                   np.array([[1,160],[0,1]])).dot(
                   np.array([[1,0],[(1-1.52)/(1000*1.52),1/1.52]])).dot(
                   np.array([[1,2280-(2025+160-sz)],[0,1]])).dot(
                   np.array([[1,0],[(1.52-1)/(1000*1),1.52/1]])).dot(
                   np.array([[1,20],[0,1]])).dot(
                   np.array([[1,0],[0,1/1.52]])).dot(
                   np.array([[1,4343-(2280+20)-sf],[0,1]])).dot(
                   np.array([[1,0],[(1.52-1)/(-1200),1.52/1]])).dot(
                   np.array([[1,140],[0,1]])).dot(
                   np.array([[1,0],[(1-1.52)/(1200*1.52),1/1.52]])).dot(
                   np.array([[1,4520-(4343+140-sf)],[0,1]])).dot(
                   np.array([[1,0],[0,1.52]])).dot(
                   np.array([[1,30],[0,1]])).dot(
                   np.array([[1,0],[0,1/1.52]])).dot(
                   np.array([[1,4940-(4520+30)],[0,1]]))
        elif self.dev == 'HT':
            sp = 2300 - Rinit*1000
            abcd = np.array([[1,sp+2553.01],[0,1]]).dot(
                   np.array([[1,0],[(1.526-1)/(-695*1),1.526/1]])).dot(
                   np.array([[1,150],[0,1]])).dot(
                   np.array([[1,0],[0,1/1.526]])).dot(
                   np.array([[1,4500.41-(2553.01+150)-sz],[0,1]])).dot(
                   np.array([[1,0],[0,1.526/1]])).dot(
                   np.array([[1,40],[0,1]])).dot(
                   np.array([[1,0],[(1-1.526)/(-515*1.526),1/1.526]])).dot(
                   np.array([[1,6122.41-(4500.41+40-sz)-sf],[0,1]])).dot(
                   np.array([[1,0],[0,1.52/1]])).dot(
                   np.array([[1,150],[0,1]])).dot(
                   np.array([[1,0],[(1-1.526)/(630*1.526),1/1.526]])).dot(
                   np.array([[1,6478.41-(6122.41+150-sf)],[0,1]])).dot(
                   np.array([[1,0],[0,1.526/1]])).dot(
                   np.array([[1,40],[0,1]])).dot(
                   np.array([[1,0],[0,1/1.526]])).dot(
                   np.array([[1,7161.01-(6478.41+40)],[0,1]]))

        return abcd

    def expand_clist(self, clist):
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