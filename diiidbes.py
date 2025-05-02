# Author : Minjun J. Choi (mjchoi@kfe.re.kr)
#
# Description : This code reads and saves the DIIID BES data (h5 format)

import os
import numpy as np
import h5py
try: 
    import omfit_classes.omfit_mds as om
except ImportError:
    om = None

BES_PATH = '/cscratch/share/choiminjun/bes_data/' # omega or local path

class DiiidBes(object):
    def __init__(self, shot, clist, savedata=False):
        self.shot = shot

        self.clist = self.expand_clist(clist)

        # BES hdf5 file name
        self.fname = "{:s}{:06d}/BES.{:06d}.h5".format(BES_PATH, shot, shot)

        # get channel posistions
        self.channel_position()

        # save MDSplus data in hdf5 format if necessary
        if os.path.exists(self.fname) == False and savedata == True:
            print('Reformat BES data to hdf5 file') 
            self.reformat_hdf5()

        # data quality
        self.good_channels = np.ones(len(self.clist))
        self.offlev = np.zeros(len(self.clist))
        self.offstd = np.zeros(len(self.clist))
        self.siglev = np.zeros(len(self.clist))
        self.sigstd = np.zeros(len(self.clist))

        self.time = None
        self.data = None

    def reformat_hdf5(self):
        # make directory if necessary
        path = '/cscratch/share/choiminjun/bes_data/{:06d}'.format(self.shot)
        if os.path.exists(path) == False:
            os.makedirs(path)

        # get fast time base
        nodeF = 'BESFU{:02d}'.format(int(self.clist[0][4:6]))
        temp = om.OMFITmdsValue(server='DIII-D', shot=self.shot, TDI=nodeF)
        timeF = temp.dim_of(0)/1000.0 # [ms] -> [s]

        # get slow time base
        nodeS = 'BESSU{:02d}'.format(int(self.clist[0][4:6]))
        temp = om.OMFITmdsValue(server='DIII-D', shot=self.shot, TDI=nodeS)
        timeS = temp.dim_of(0)/1000.0 # [ms] -> [s]        

        with h5py.File(self.fname, 'w') as fout:
            fout.create_dataset('TIMEFU', data=timeF)
            fout.create_dataset('TIMESU', data=timeS)

            for c, cname in enumerate(self.clist):
                # get and add data after multiply 1e6 and transform it into int32
                nodeF = 'BESFU{:02d}'.format(int(cname[4:6]))
                temp = om.OMFITmdsValue(server='DIII-D', shot=self.shot, TDI=nodeF)
                dataF = temp.data()
                dataF = (dataF * 1e6).astype(np.int32)
                fout.create_dataset(nodeF, data=dataF, compression="gzip", compression_opts=5)

                nodeS = 'BESSU{:02d}'.format(int(cname[4:6]))
                temp = om.OMFITmdsValue(server='DIII-D', shot=self.shot, TDI=nodeS)
                dataS = temp.data()
                dataS = (dataS * 1e6).astype(np.int32)
                fout.create_dataset(nodeS, data=dataS, compression="gzip", compression_opts=5)                
                
                dset = fout[nodeF] # channel positions will be saved undr BESFUXX nodes
                dset.attrs['RPOS'] = self.rpos[c]
                dset.attrs['ZPOS'] = self.zpos[c]
                dset.attrs['APOS'] = self.apos[c]

                print('Added', cname)
                
        print('Saved', self.fname)        

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
            if verbose == 1: print('Data is normalized by trange average')
        elif norm == 2:
            if verbose == 1: print('Data is normalized by atrange average')
        elif norm == 3:
            if verbose == 1: print('Data is normalized by SU signal')

        if os.path.exists(self.fname):
            # read data from hdf5
            with h5py.File(self.fname, 'r') as fin:
                print('Read BES data from hdf5 file')
                # read FU and SU time base and get tidx 
                timeF = np.array(fin.get('TIMEFU'))
                timeS = np.array(fin.get('TIMESU'))

                # get time base
                self.time, idx1F, idx2F, oidx1F, oidx2F, timeS, idx1S, idx2S, oidx1S, oidx2S = self.time_base(timeF, timeS, trange)
                if norm == 2:
                    _, _, _, _, _, _, aidx1S, aidx2S, _, _ = self.time_base(timeF, timeS, atrange)

                # get data
                for i, cname in enumerate(self.clist):
                    # node name
                    nodeF = 'BESFU{:02d}'.format(int(cname[4:6]))
                    nodeS = 'BESSU{:02d}'.format(int(cname[4:6]))

                    # raw data
                    vF = np.array(fin.get(nodeF)[idx1F:idx2F+1]) / 1e6 # signal [int32] -> [V]
                    ovF = np.array(fin.get(nodeF)[oidx1F:oidx2F+1]) / 1e6 # offset [int32] -> [V]
                    vS = np.array(fin.get(nodeS)[idx1S:idx2S+1]) / 1e6 # signal [int32] -> [V]
                    ovS = np.array(fin.get(nodeS)[oidx1S:oidx2S+1]) / 1e6 # offset [int32] -> [V]

                    self.offlev[i] = np.median(ovS) + np.median(ovF)
                    self.offstd[i] = np.sqrt(np.std(ovS)**2 + np.std(ovF)**2)

                    # sampling up SU using interpolation
                    vS = np.interp(self.time, timeS, vS)

                    # full signal
                    v = vF + vS - self.offlev[i]

                    self.siglev[i] = np.mean(v)
                    self.sigstd[i] = np.std(v)

                    # normalization
                    if norm == 1:
                        v = v/np.mean(v) - 1
                    elif norm == 2:
                        avS = np.array(fin.get(nodeS)[aidx1S:aidx2S+1]) / 1e6 # signal [int32] -> [V]
                        v = v/(np.mean(avS) - self.offlev[i]) - 1
                    elif norm == 3:
                        v = v/(vS - self.offlev[i]) - 1

                    # expand dimension - concatenate
                    v = np.expand_dims(v, axis=0)
                    if i == 0:
                        self.data = v
                    else:
                        self.data = np.concatenate((self.data, v), axis=0)
        else:
            print('Load BES data from MDSplus server')
            
            # get fast time base
            nodeF = 'BESFU{:02d}'.format(int(self.clist[0][4:6]))
            temp = om.OMFITmdsValue(server='DIII-D', shot=self.shot, TDI=nodeF)
            timeF = temp.dim_of(0)/1000.0 # [ms] -> [s]

            # get slow time base
            nodeS = 'BESSU{:02d}'.format(int(self.clist[0][4:6]))
            temp = om.OMFITmdsValue(server='DIII-D', shot=self.shot, TDI=nodeS)
            timeS = temp.dim_of(0)/1000.0 # [ms] -> [s]        
            
            # get time base
            self.time, idx1F, idx2F, oidx1F, oidx2F, timeS, idx1S, idx2S, oidx1S, oidx2S = self.time_base(timeF, timeS, trange)
            if norm == 2:
                _, _, _, _, _, _, aidx1S, aidx2S, _, _ = self.time_base(timeF, timeS, atrange)            
            
            # get data
            for i, cname in enumerate(self.clist):
                # get and add data after multiply 1e6 and transform it into int32
                nodeF = 'BESFU{:02d}'.format(int(cname[4:6]))
                temp = om.OMFITmdsValue(server='DIII-D', shot=self.shot, TDI=nodeF)
                dataF = temp.data()

                nodeS = 'BESSU{:02d}'.format(int(cname[4:6]))
                temp = om.OMFITmdsValue(server='DIII-D', shot=self.shot, TDI=nodeS)
                dataS = temp.data()

                # raw data
                vF = dataF[idx1F:idx2F+1]
                ovF = dataF[oidx1F:oidx2F+1]
                vS = dataS[idx1S:idx2S+1]
                ovS = dataS[oidx1S:oidx2S+1]

                self.offlev[i] = np.median(ovS) + np.median(ovF)
                self.offstd[i] = np.sqrt(np.std(ovS)**2 + np.std(ovF)**2)

                # sampling up SU using interpolation
                vS = np.interp(self.time, timeS, vS)

                # full signal
                v = vF + vS - self.offlev[i]

                self.siglev[i] = np.mean(v)
                self.sigstd[i] = np.std(v)

                # normalization
                if norm == 1:
                    v = v/np.mean(v) - 1
                elif norm == 2:
                    avS = dataS[aidx1S:aidx2S+1]
                    v = v/(np.mean(avS) - self.offlev[i]) - 1
                elif norm == 3:
                    v = v/(vS - self.offlev[i]) - 1

                # expand dimension - concatenate
                v = np.expand_dims(v, axis=0)
                if i == 0:
                    self.data = v
                else:
                    self.data = np.concatenate((self.data, v), axis=0)

        return self.time, self.data

    def time_base(self, timeF, timeS, trange):
        # sampling frequency
        self.fs = round(1/(timeF[1] - timeF[0])/1000)*1000.0
        fsS = round(1/(timeS[1] - timeS[0])/1000)*1000.0

        # get tidx for signals, offset (from SU)
        idx1F = round((max(trange[0],timeF[0]) + 1e-8 - timeF[0])*self.fs) 
        idx2F = round((min(trange[1],timeF[-1]) - 1e-8 - timeF[0])*self.fs)
        oidx1F = round((max(0.25,timeF[0]) + 1e-8 - timeF[0])*self.fs) 
        oidx2F = round((min(0.27,timeF[-1]) - 1e-8 - timeF[0])*self.fs)
        idx1S = round((max(trange[0],timeS[0]) + 1e-8 - timeS[0])*fsS) 
        idx2S = round((min(trange[1],timeS[-1]) - 1e-8 - timeS[0])*fsS)
        oidx1S = round((max(-0.05,timeS[0]) + 1e-8 - timeS[0])*fsS) 
        oidx2S = round((min(0,timeS[-1]) - 1e-8 - timeS[0])*fsS)

        return timeF[idx1F:idx2F+1], idx1F, idx2F, oidx1F, oidx2F, timeS[idx1S:idx2S+1], idx1S, idx2S, oidx1S, oidx2S

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

        cnum = len(self.clist)
        self.rpos = np.zeros(cnum)  # R [m] of each channel
        self.zpos = np.zeros(cnum)  # z [m] of each channel
        self.apos = np.zeros(cnum)  # angle [rad] of each channel

    def expand_clist(self, clist):
        # IN : List of channel names (e.g. 'BES_10-14')
        # OUT : Expanded list (e.g. 'BES_10', ..., 'BES_14')

        exp_clist = []

        for c in range(len(clist)):
            if len(clist[c]) < 8:
                exp_clist.append(clist[c])
            else:
                ni = int(clist[c][4:6])
                nf = int(clist[c][7:9])

                for n in range(ni, nf+1):
                    exp_clist.append(clist[c][0:4] + '{:02d}'.format(n))
                
        clist = exp_clist

        return clist


def expand_clist(self, clist):
    # IN : List of channel names (e.g. 'BES_10-14')
    # OUT : Expanded list (e.g. 'BES_10', ..., 'BES_14')

    exp_clist = []

    for c in range(len(clist)):
        if len(clist[c]) < 8:
            exp_clist.append(clist[c])
        else:
            ni = int(clist[c][4:6])
            nf = int(clist[c][7:9])

            for n in range(ni, nf+1):
                exp_clist.append(clist[c][0:4] + '{:02d}'.format(n))
            
    clist = exp_clist

    return clist