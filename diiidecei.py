"""
Author: Minjun J. Choi (mjchoi@kfe.re.kr)

Description: This code reads and saves the DIII-D ECEI data (h5 format)

Acknowledgement: Dr. G. Yu
"""

import os
import numpy as np
import h5py
import filtdata as ft
try: 
    import omfit_classes.omfit_mds as om
except ImportError:
    om = None

ECEI_PATH = '/Users/mjchoi/Desktop/ecei_data/' # omega or local path

class DiiidEcei():
    def __init__(self, shot, clist, savedata=False):
        self.shot = shot

        self.clist = self.expand_clist(clist)

        self.dev = self.clist[0][0:3]

        # ECEI hdf5 file name LFS or HFS
        self.fname = "{:s}{:06d}/ECEI.{:06d}.{:s}.h5".format(ECEI_PATH, shot, shot, self.dev)

        # get channel posistions
        self.channel_position()

        # save MDSplus data in hdf5 format if necessary
        if os.path.exists(self.fname) == False and savedata == True:
            print('Reformat ECEI data to hdf5 file') 
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
        path = '/cscratch/share/choiminjun/ecei_data/{:06d}'.format(self.shot)
        if os.path.exists(path) == False:
            os.makedirs(path)

        # get time base
        temp = om.OMFITmdsValue(server='DIII-D', shot=self.shot, TDI=f'PTDATA2("{self.clist[0]}",{self.shot})')
        time = temp.dim_of(0)/1000.0 # [ms] -> [s]

        with h5py.File(self.fname, 'w') as fout:
            fout.create_dataset('TIME', data=time)

            for c, cname in enumerate(self.clist):
                # get and add data after multiply 1e6 and transform it into int32
                temp = om.OMFITmdsValue(server='DIII-D', shot=self.shot, TDI=f'PTDATA2("{cname}",{self.shot})')
                data = temp.data()
                data = (data * 1e6).astype(np.int32)
                fout.create_dataset(cname, data=data, compression="gzip", compression_opts=5)
                
                dset = fout[cname]
                dset.attrs['RPOS'] = self.rpos[c]
                dset.attrs['ZPOS'] = self.zpos[c]
                dset.attrs['APOS'] = self.apos[c]

                print('Added', cname)
                
            fout.create_dataset(self.dev, shape=(0,))
            dset = fout[self.dev]
            dset.attrs['TFcurrent'] = self.itf/1000.0 # [A] -> [kA]
            dset.attrs['TriggerTime'] = time[0] # [sec]
            dset.attrs['SampleRate'] = round(1/(time[1] - time[0])/1000)*1000.0 # [Hz]
            dset.attrs['HarmNum'] = self.hn # harmonic number
            dset.attrs['LoFreq'] = self.lo # [GHz]
            dset.attrs['LensFocus'] = self.sf # [mm]
            dset.attrs['LensZoom'] = self.sz # [mm]

        print('Saved', self.fname)        

    def get_data(self, trange, norm=0, atrange=[1.0, 1.1], res=0, verbose=1):
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

        if os.path.exists(self.fname):
            # read data from hdf5
            with h5py.File(self.fname, 'r') as fin:
                print('Read ECEI data from hdf5 file')
                # read time base and get tidx 
                self.time = np.array(fin.get('TIME'))

                # get sampling frequency 
                self.fs = round(1/(self.time[1] - self.time[0])/1000)*1000.0

                # get tidx for signal, offset, and atrange
                idx1 = round((max(trange[0],self.time[0]) + 1e-8 - self.time[0])*self.fs) 
                idx2 = round((min(trange[1],self.time[-1]) - 1e-8 - self.time[0])*self.fs)

                oidx1 = round((max(-0.04,self.time[0]) + 1e-8 - self.time[0])*self.fs) 
                oidx2 = round((min(-0.01,self.time[-1]) - 1e-8 - self.time[0])*self.fs)

                aidx1 = round((max(atrange[0],self.time[0]) + 1e-8 - self.time[0])*self.fs) 
                aidx2 = round((min(atrange[1],self.time[-1]) - 1e-8 - self.time[0])*self.fs)

                # get time for trange
                self.time = self.time[idx1:idx2+1]

                # get data
                for i, cname in enumerate(self.clist):
                    # load data
                    ov = np.array(fin.get(cname)[oidx1:oidx2+1]) / 1e6 # offset [int32] -> [V]
                    v = np.array(fin.get(cname)[idx1:idx2+1]) / 1e6 # signal [int32] -> [V]
                    
                    self.offlev[i] = np.median(ov)
                    self.offstd[i] = np.std(ov)

                    v = v - self.offlev[i]

                    self.siglev[i] = np.median(v)
                    self.sigstd[i] = np.std(v)

                    # normalization 
                    if norm == 1:
                        v = v/np.mean(v) - 1
                    elif norm == 2:
                        av = np.array(fin.get(cname)[aidx1:aidx2+1]) / 1e6 # atrange signal [int32] -> [V]
                        v = v/(np.mean(av) - self.offlev[i]) - 1
                    elif norm == 3:
                        base_filter = ft.FirFilter('FIR_pass', self.fs, 0, 10, 0.01)
                        base = base_filter.apply(v).real
                        v = v/base - 1

                    # expand dimension - concatenate
                    v = np.expand_dims(v, axis=0)
                    if i == 0:
                        self.data = v
                    else:
                        self.data = np.concatenate((self.data, v), axis=0)
        else:
            print('Load ECEI data from MDSplus server')

            # time 
            temp = om.OMFITmdsValue(server='DIII-D', shot=self.shot, TDI=f'PTDATA2("{self.clist[0]}",{self.shot})')
            self.time = temp.dim_of(0)/1000.0 # [ms] -> [s]

            # sampling frequency
            self.fs = round(1/(self.time[1] - self.time[0])/1000)*1000.0

            # get tidx for signal, offset, and atrange
            idx1 = round((max(trange[0],self.time[0]) + 1e-8 - self.time[0])*self.fs) 
            idx2 = round((min(trange[1],self.time[-1]) - 1e-8 - self.time[0])*self.fs)

            oidx1 = round((max(-0.04,self.time[0]) + 1e-8 - self.time[0])*self.fs) 
            oidx2 = round((min(-0.01,self.time[-1]) - 1e-8 - self.time[0])*self.fs)

            aidx1 = round((max(atrange[0],self.time[0]) + 1e-8 - self.time[0])*self.fs) 
            aidx2 = round((min(atrange[1],self.time[-1]) - 1e-8 - self.time[0])*self.fs)

            # get time for trange
            self.time = self.time[idx1:idx2+1]

            # data
            for i, cname in enumerate(self.clist):
                temp = om.OMFITmdsValue(server='DIII-D', shot=self.shot, TDI=f'PTDATA2("{cname}",{self.shot})')
                temp_data = temp.data()

                ov = temp_data[oidx1:oidx2+1]
                v = temp_data[idx1:idx2+1]
                
                self.offlev[i] = np.median(ov)
                self.offstd[i] = np.std(ov)

                v = v - self.offlev[i]

                self.siglev[i] = np.median(v)
                self.sigstd[i] = np.std(v)

                # normalization 
                if norm == 1:
                    v = v/np.mean(v) - 1
                elif norm == 2:
                    av = temp_data[aidx1:aidx2+1]
                    v = v/(np.mean(av) - self.offlev[i]) - 1
                elif norm == 3:
                    base_filter = ft.FftFilter('FFT_pass', self.fs, 0, 10)
                    base = base_filter.apply(v).real
                    v = v/base - 1

                # expand dimension - concatenate
                v = np.expand_dims(v, axis=0)
                if i == 0:
                    self.data = v
                else:
                    self.data = np.concatenate((self.data, v), axis=0)

                print('Loaded', cname)

        # check data quality
        self.find_bad_channel()

        return self.time, self.data
    
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

    def channel_position(self):  # Needs updates ####################
        cnum = len(self.clist)
        self.rpos = np.zeros(cnum)  # R [m]
        self.zpos = np.zeros(cnum)  # z [m]
        self.apos = np.zeros(cnum)  # angle [rad]

        if os.path.exists(self.fname):
            # open file   
            with h5py.File(self.fname, 'r') as fin:
                for c, cname in enumerate(self.clist):                    
                    dset = fin[cname]
                    self.rpos[c] = dset.attrs['RPOS'] # [m]
                    self.zpos[c] = dset.attrs['ZPOS'] # [m]
                    self.apos[c] = dset.attrs['APOS'] # [rad]
        else:
            # arbitrary positions
            for c, cname in enumerate(self.clist):
                self.zpos[c] = float(cname[3:5])
                self.rpos[c] = 9.0 - float(cname[5:7])

            # hn_node = '\\{0}::TOP.ECEI_{1}:{2}_MODE'.format(ECEI_TREE, self.dev, self.dev) 
            self.hn = 2
            # itf_node = '\\{0}::TOP:{1}'.format(ECEI_TREE, 'ECEI_I_TF') 
            self.itf = 19*1000 # [A]
            # lo_node = '\\{0}::TOP.ECEI_{1}:{2}_LOFREQ'.format(ECEI_TREE, self.dev, self.dev) 
            self.lo = 73 # [GHz]
            # sf_node = '\\{0}::TOP.ECEI_{1}:{2}_LENSFOCUS'.format(ECEI_TREE, self.dev, self.dev) 
            self.sf = 250 # [mm]
            # sz_node = '\\{0}::TOP.ECEI_{1}:{2}_LENSZOOM'.format(ECEI_TREE, self.dev, self.dev) 
            self.sz = 500 # [mm]

    def expand_clist(self, clist):
        # IN : List of channel names (e.g. 'LFS1201-1208').
        # OUT : Expanded list (e.g. 'LFS1201', ..., 'LFS1208')

        exp_clist = []

        for cname in clist:
            if len(cname) < 8:
                exp_clist.append(cname)
            else: 
                vi = int(cname[3:5])
                fi = int(cname[5:7])
                vf = int(cname[8:10])
                ff = int(cname[10:12])
        
                for v in range(vi, vf+1):
                    for f in range(fi, ff+1):
                        exp_clist.append(cname[0:3] + '{:02d}{:02d}'.format(v, f))

        clist = exp_clist      

        return clist


def expand_clist(self, clist):
    # IN : List of channel names (e.g. 'LFS1201-1208').
    # OUT : Expanded list (e.g. 'LFS1201', ..., 'LFS1208')

    exp_clist = []

    for cname in clist:
        if len(cname) < 8:
            exp_clist.append(cname)
        else: 
            vi = int(cname[3:5])
            fi = int(cname[5:7])
            vf = int(cname[8:10])
            ff = int(cname[10:12])
    
            for v in range(vi, vf+1):
                for f in range(fi, ff+1):
                    exp_clist.append(cname[0:3] + '{:02d}{:02d}'.format(v, f))

    clist = exp_clist

    return clist

if __name__ == "__main__":
    pass

    # g = DiiiEcei(shot=174964,clist=['LFS0301'])
    # g.get_data(trange=[0,10])
    # plt.plot(g.time, g.data[0,:], color='k')
    # plt.show()
    # g.close

# DisconnectFromMds(g.socket)
