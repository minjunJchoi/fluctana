"""
Author: Minjun J. Choi (mjchoi@kfe.re.kr)

Description: This code reads the DIII-D data using OMFITmdsValue

Acknowledgement: Dr. L. Bardoczi
"""

import numpy as np
# import pidly
import traceback
try: 
    import omfit_classes.omfit_mds as om
except ImportError:
    om = None

#### VAR to NODE
# TECE01--TECE40 : calibrated ECE
# Bt : toroidal field
# DENSITY : plasma density from EFIT01
# TAUE : energy confinement time
# BETAN : normalized beta
# CERQROTCT1--T48 : toroidal velocity from CER

VAR_NODE = {'ne':'DENR0F', 'Da04':'FS04', 'NBI_15L':'PINJ_15L', 'NBI_15R':'PINJ_15R', 'n1_amp':'n1rms', 'n2_amp':'n2rms'}

class DiiidData():
    def __init__(self, shot, clist):
        self.shot = shot

        self.clist = self.expand_clist(clist)

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

    def get_data(self, trange, norm=0, atrange=[1.0, 1.1], res=0, verbose=1):
        if norm == 0:
            if verbose == 1: print('Data is not normalized')
        elif norm == 1:
            if verbose == 1: print('Data is normalized by trange average')
        elif norm == 2:
            if verbose == 1: print('Data is normalized by atrange average')

        self.trange = trange

        # # open idl
        # idl = pidly.IDL('/fusion/usc/opt/idl/idl84/bin/idl')

        # --- loop starts --- #
        clist_temp = self.clist.copy()
        for i, cname in enumerate(clist_temp):

            # set node
            if cname in VAR_NODE:
                node ='{:s}'.format(VAR_NODE[cname])
            else:
                node = cname

            # load data
            try:
                #idl.pro('gadat,time,data,/alldata',node,self.shot,XMIN=self.trange[0]*1000.0,XMAX=self.trange[1]*1000.0)
                # idl.pro('gadat2,time,data,/alldata',node,self.shot,XMIN=self.trange[0]*1000.0,XMAX=self.trange[1]*1000.0)
                temp = om.OMFITmdsValue(server='DIII-D', shot=self.shot, TDI=node)

                if i == 0:
                    # self.time, v = idl.time, idl.data
                    self.time, v = temp.dim_of(0), temp.data()

                    # [ms] -> [s]
                    self.time = self.time/1000.0
                    self.fs = round(1/(self.time[1] - self.time[0])/1000)*1000

                    # data resize
                    idx = np.where((self.time >= trange[0])*(self.time <= trange[1]))
                    idx1 = int(idx[0][0])
                    idx2 = int(idx[0][-1]+1)
                    self.time = self.time[idx1:idx2]
                    v = v[idx1:idx2]
                else:
                    # _, v = idl.time, idl.data
                    _, v = temp.dim_of(0), temp.data()
                    v = v[idx1:idx2]

                if verbose == 1: print("Read {:d} - {:s} (number of data points = {:d})".format(self.shot, node, len(v)))

                if norm == 1:
                    v = v/np.mean(v) - 1

                # expand dimension - concatenate
                v = np.expand_dims(v, axis=0)
                if i == 0:
                    self.data = v
                else:
                    self.data = np.concatenate((self.data, v), axis=0)
            except Exception as e:
                self.clist.remove(cname)
                self.rpos[i] = -1
                if verbose == 1: 
                    print('Failed {:d} : {:s}. {:s} is removed'.format(self.shot, node, cname))
                    print(f"Error: {e}")
                    traceback.print_exc()                      
        # --- loop ends --- #

        # # close idl
        # idl.close()

        # remove positions of excluded channels
        cidx = self.rpos >= 0
        self.rpos = self.rpos[cidx]
        self.zpos = self.zpos[cidx]
        self.apos = self.apos[cidx]

        # # check data quality
        # self.find_bad_channel()

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

        # ECE
        if self.clist[0].startswith('ece'):
            R0 = 1.67 # [m]
            m = 2
            Bt = 1.9 # [T]
            ece_freq = np.concatenate((np.arange(83.5, 98.5 + 1, 1), np.arange(98.5, 113.5 + 1, 1), np.arange(115.5, 129.5 + 2, 2))) # RF frequency (center [GHz])
            ece_rpos = 27.99*m*Bt*R0/ece_freq # [m]

            for c, cname in enumerate(self.clist):
                self.rpos[c] = ece_rpos[int(int(cname[3:]) - 1)]

    def expand_clist(self, clist):
        # IN : List of channel names (e.g. 'LFS1201-1208').
        # OUT : Expanded list (e.g. 'LFS1201', ..., 'LFS1208')
        pass

        return clist


def expand_clist(self, clist):
    # IN : List of channel names (e.g. '').
    # OUT : Expanded list (e.g. '', ..., '')

    pass

    return clist

if __name__ == "__main__":
    pass

    # g = DiiidData(shot=174964,clist=['ne'])
    # g.get_data(trange=[0,10])
    # plt.plot(g.time, g.data[0,:], color='k')
    # plt.show()
    # g.close

# DisconnectFromMds(g.socket)
