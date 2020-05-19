# Author : Minjun J. Choi (mjchoi@nfri.re.kr)
#
# Description : This code reads the DIII-D data using gadat.pro code
#
# Last updated
#  2018.10.08 :

import numpy as np
import matplotlib.pyplot as plt
import pidly

#### VAR to NODE
# TECE01--TECE40 : calibrated ECE
# Bt : toroidal field
# DENSITY : plasma density from EFIT01
# TAUE : energy confinement time
# BETAN : normalized beta
# CERQROTCT1--T48 : toroidal velocity from CER

VAR_NODE = {'ne':'DENR0F', 'Da04':'FS04', 'NBI_15L':'PINJ_15L', 'NBI_15R':'PINJ_15R', 'n1_amp':'n1rms', 'n2_amp':'n2rms'}

class DiiidData():
    def __init__(self, shot ,clist):
        self.shot = shot
        self.clist = clist

    def get_data(self, trange, norm=0, atrange=[1.0, 1.1], res=0, verbose=1):
        if norm == 0:
            if verbose == 1: print('Data is not normalized')
        elif norm == 1:
            if verbose == 1: print('Data is normalized by trange average')
        elif norm == 2:
            if verbose == 1: print('Data is normalized by atrange average')

        self.trange = trange

        # open idl
        idl = pidly.IDL('/fusion/usc/opt/idl/idl84/bin/idl')

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
                idl.pro('gadat2,time,data,/alldata',node,self.shot,XMIN=self.trange[0]*1000.0,XMAX=self.trange[1]*1000.0)
                time, v = idl.time, idl.data
                if verbose == 1: print("Read {:d} - {:s} (number of data points = {:d})".format(self.shot, node, len(v)))
            except:
                self.clist.remove(cname)
                if verbose == 1: print("Failed   {:s}".format(node))
                continue

            # [ms] -> [s]
            time = time/1000.0

            # set data size
            idx = np.where((time >= trange[0])*(time <= trange[1]))
            idx1 = int(idx[0][0])
            idx2 = int(idx[0][-1]+1)
            time = time[idx1:idx2]
            v = v[idx1:idx2]

            if norm == 1:
                v = v/np.mean(v) - 1

            # expand dimension - concatenate
            v = np.expand_dims(v, axis=0)
            if i == 0:
                data = v
            else:
                data = np.concatenate((data, v), axis=0)
        # --- loop ends --- #

        self.time = time
        self.fs = round(1/(time[1] - time[0])/1000)*1000
        self.data = data

        # get channel position
        self.channel_position()

        # close idl
        idl.close()

        return time, data

    def channel_position(self):  # Needs updates ####################
        cnum = len(self.clist)
        self.rpos = np.arange(cnum)  # R [m]
        self.zpos = np.zeros(cnum)  # z [m]
        self.apos = np.arange(cnum)  # angle [rad]
        for c in range(cnum):
            # Mirnov coils
            # ECE
            pass


if __name__ == "__main__":
    pass

    # g = DiiidData(shot=174964,clist=['ne'])
    # g.get_data(trange=[0,10])
    # plt.plot(g.time, g.data[0,:], color='k')
    # plt.show()
    # g.close

# DisconnectFromMds(g.socket)
