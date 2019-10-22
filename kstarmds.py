# Author : Minjun J. Choi (mjchoi@nfri.re.kr)
#
# Description : This code reads the KSTAR MDSplus server data
#
# Acknowledgement : Special thanks to Dr. Y.M. Jeon
#
# Last updated
#  2018.03.23 : version 0.10; Mirnov data resampling

from MDSplus import Connection
# from MDSplus import DisconnectFromMds
# from MDSplus._mdsshr import MdsException

import numpy as np
import matplotlib.pyplot as plt

from kstardata import ep_pos
from kstardata import ece_pos
from kstardata import mc_pos
from kstardata import ts_pos
from kstardata import tor_field

VAR_NODE = {'NBI11':'NB11_pnb', 'NBI12':'NB12_pnb', 'NBI13':'NB13_pnb', 'ECH':'ECH_VFWD1', 'ECCD':'EC1_RFFWD1',
            'ICRF':'ICRF_FWD', 'LHCD':'LH1_AFWD', 'GASI':'I_GFLOW_IN:FOO', 'GASK':'K_GFLOW_IN:FOO', 'SMBI':'SM_VAL_OUT:FOO',
            'Ip':'RC03', 'neAVGM':'NE_INTER01', 'ECE05':'ECE05', 'ECE35':'ECE35', 'Rp':'LMSR',
            'Zp':'LMSZ', 'VOL':'VOLUME', 'KAP':'KAPPA', 'BETAp':'BETAP', 'BETAn':'BETAN', 'q95':'q95', 'Li':'LI3',
            'GASG':'G_GFLOW_IN:FOO', 'WTkp':'WTOT_KAPPA', 'WTdlm':'WTOT_DLM03', 'DaT11':'TOR_HA11', 'DaT10':'TOR_HA10',
            'DaP02':'POL_HA02', 'DaP04':'POL_HA04', 'neAVGF':'NE_INTER02',
            'RMP_T3':'PCRMPTBULI', 'RMP_T4':'PCRMPTFULI', 'RMP_T1':'PCRMPTJULI', 'RMP_T2':'PCRMPTNULI',
            'RMP_M3':'PCRMPMBULI', 'RMP_M4':'PCRMPMFULI', 'RMP_M1':'PCRMPMJULI', 'RMP_M2':'PCRMPMNULI',
            'RMP_B3':'PCRMPBBULI', 'RMP_B4':'PCRMPBFULI', 'RMP_B1':'PCRMPBJULI', 'RMP_B2':'PCRMPBNULI'}

# nodes in PCS_KSTAR tree
PCS_TREE = ['LMSR', 'LMSZ', 'PCITFMSRD', 'PCRMPTBULI', 'PCRMPTFULI', 'PCRMPTJULI', 'PCRMPTNULI',
            'PCRMPMBULI', 'PCRMPMFULI', 'PCRMPMJULI', 'PCRMPMNULI', 'PCRMPBBULI', 'PCRMPBFULI', 'PCRMPBJULI', 'PCRMPBNULI']

# nodes in CSS tree
CSS_TREE = ['CSS_I{:02d}:FOO'.format(i) for i in range(1,5)] + ['CSS_Q{:02d}:FOO'.format(i) for i in range(1,5)]

# nodes in EFIT01 or EFIT02
EFIT_TREE = ['VOLUME', 'KAPPA', 'BETAP', 'BETAN', 'q95', 'LI3', 'WMHD']

# nodes need postprocessing
POST_NODE = {'ECH_VFWD1':'/1000', 'EC1_RFFWD1':'/1000', 'LH1_AFWD':'/200', 'SM_VAL_OUT:FOO':'/5',
            'RC03':'*(-1)/1000', 'NE_INTER01':'/1.9', 'VOLUME':'/10', 'KAPPA':'-1',
            'NE_INTER02':'/2.7'}

# nodes NOT support segment reading in 2018
NSEG_NODE = ['NB11_pnb', 'NB12_pnb', 'ECH_VFWD1'] # etc

class KstarMds(Connection):
    def __init__(self, shot ,clist):
        # from iKSTAR
        super(KstarMds,self).__init__('172.17.250.23:8005')  # call __init__ in Connection
        # from opi to CSS Host PC
        # super(KstarMds,self).__init__('172.17.102.69:8000')  # call __init__ in Connection
        self.shot = shot
        self.clist = clist

    def get_data(self, trange, norm=0, atrange=[1.0, 1.1], res=0):
        if norm == 0:
            print('data is not normalized')
        elif norm == 1:
            print('data is normalized by trange average')
        elif norm == 2:
            print('data is normalized by atrange average')

        self.trange = trange

        # open tree
        tree = find_tree(self.clist[0])
        try:
            self.openTree(tree,self.shot)
        except:
            print('Failed to open tree {:s}'.format(tree))
            time, data = None, None
            return time, data

        # --- loop starts --- #
        clist_temp = self.clist[:]
        for i, cname in enumerate(clist_temp):

            # get MDSplus node from channel name
            if cname in VAR_NODE:
                node = VAR_NODE[cname]
            else:
                node = cname

            # resampling, time node
            if res != 0:
                snode = 'resample(\{:s},{:f},{:f},{:f})'.format(node,self.trange[0],self.trange[1],res)  # resampling
                tnode = 'dim_of(resample(\{:s},{:f},{:f},{:f}))'.format(node,self.trange[0],self.trange[1],res)  # resampling
            else:
                snode = 'setTimeContext({:f},{:f},*),\{:s}'.format(self.trange[0],self.trange[1],node)
                tnode = 'setTimeContext({:f},{:f},*),dim_of(\{:s})'.format(self.trange[0],self.trange[1],node)

            if 'ECE' in clist_temp[0]: # do not use setTimeContext for ECE
                snode = 'setTimeContext(*,*,*),\{:s}'.format(node)
                tnode = 'setTimeContext(*,*,*),dim_of(\{:s})'.format(node)

            # post processing for data
            if node in POST_NODE:
                pnode = POST_NODE[node]
            else:
                pnode = ''

            # data node name
            dnode = snode + pnode

            try:
                # get data
                time = self.get(tnode).data()
                v = self.get(dnode).data()
                print('Read {:d} : {:s} (number of data points = {:d})'.format(self.shot, dnode, len(v)))
            except:
                self.clist.remove(cname)
                print('Failed {:d} : {:s}. {:s} is removed'.format(self.shot, dnode, cname))
                continue

            if tree == 'EFIT01':
                time = time*0.001

            # set data size
            idx = np.where((time >= trange[0])*(time <= trange[1]))
            idx1 = int(idx[0][0])
            idx2 = int(idx[0][-1]+2)
            time = time[idx1:idx2]
            v = v[idx1:idx2]

            # normalize if necessary
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
        self.fs = round(1/(time[1] - time[0])/1000)*1000.0
        self.data = data

        # get channel position
        self.channel_position()

        # get measurement error
        self.meas_error()

        # close tree
        self.closeTree(tree, self.shot)

        return time, data

    def channel_position(self):  # Needs updates ####################
        # get channel position either from MDSplus server or kstardata
        cnum = len(self.clist)
        self.rpos = np.arange(cnum, dtype=np.float64)  # R [m]
        self.zpos = np.zeros(cnum)  # z [m]
        self.apos = np.arange(cnum, dtype=np.float64)  # angle [rad]

        for c in range(cnum):
            try:
                # set rnode 
                if 'CES' == self.clist[0][0:3]: # CES
                    rnode = '\{:s}RT{:}'.format(self.clist[c][:4],self.clist[c][6:])
                elif 'ECE' == self.clist[0][0:3]: # ECE
                    rnode = '\{:s}:RPOS2ND'.format(self.clist[c])
                else:
                    rnode = ''

                # try to read from MDSplus node
                rpos = self.get(rnode).data()
                if hasattr(rpos, "__len__"):
                    self.rpos[c] = self.get(rnode).data()[0]
                else:
                    self.rpos[c] = rpos

                print('channel position read from MDSplus rnode {:s}'.format(rnode))
            except:
                # get channel positions from kstardata
                if 'ECE' == self.clist[0][0:3]: # ECE 2nd harmonics cold resonance
                    if c == 0: bt = tor_field.get_bt(self.shot)
                    ece_rpos = ece_pos.get_ece_pos(self.shot, bt)
                    self.rpos[c] = ece_rpos[self.clist[c][0:5]]
                elif 'TS' == self.clist[0][0:2]:
                    ts_core_rpos, ts_edge_rpos = ts_pos.get_ts_pos(self.shot)
                    dTS = self.clist[c][3:7]
                    prec = self.clist[c].split(':')[0]
                    if len(prec) == 8:
                        cTS = int(prec[7:8])
                    elif len(prec) == 9:
                        cTS = int(prec[7:9])
                    if dTS == 'CORE':
                        self.rpos[c] = ts_core_rpos[cTS-1]/1000.0
                    elif dTS == 'EDGE':
                        self.rpos[c] = ts_edge_rpos[cTS-1]/1000.0
                elif 'EP' == self.clist[0][0:2]:
                    ep_rpos, ep_zpos = ep_pos.get_ep_pos()
                    self.rpos[c] = ep_rpos[self.clist[c][0:4]]
                    self.zpos[c] = ep_zpos[self.clist[c][0:4]]
                    self.apos[c] = float(self.clist[c][2:4])
                elif 'MC1' == self.clist[0][0:3]:
                    mc1t_apos, mc1p_apos = mc_pos.get_mc_pos()
                    if 'MC1T' == self.clist[0][0:4]:
                        self.apos[c] = mc1t_apos[self.clist[c]]
                    elif 'MC1P' == self.clist[0][0:4]:
                        self.apos[c] = mc1p_apos[self.clist[c]]
                
                print('channel position from kstardata'.format(rnode))

    def meas_error(self):  # Needs updates ####################
        # read from MDSplus node
        cnum = len(self.clist)
        self.err = np.zeros(cnum)  # measurement error
        for c in range(cnum):
            if 'CES' in self.clist[0]: # CES
                enode = '\{:s}:err_bar'.format(self.clist[c])

            try:
                self.err[c] = np.mean(self.get(enode).data())
            except:
                pass

def find_tree(cname):
    # cname -> node
    if cname in VAR_NODE:
        node = VAR_NODE[cname]
    else:
        node = cname

    # find tree
    if node in PCS_TREE:
        tree = 'PCS_KSTAR'
    elif node in CSS_TREE:
        tree = 'CSS'
    elif node in EFIT_TREE:
        tree = 'EFIT01'
    else:
        tree = 'KSTAR'

    return tree


if __name__ == "__main__":
    pass

    # g = KstarMds(shot=17245,clist=['neAVGM'])
    # g.get_data(trange=[0,10])
    # plt.plot(g.time, g.data[0,:], color='k')
    # plt.show()
    # g.close

# DisconnectFromMds(g.socket)
