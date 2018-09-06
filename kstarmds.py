# Author : Minjun J. Choi (mjchoi@nfri.re.kr)
#
# Description : This code reads the KSTAR MDSplus server data
#
# Acknowledgement : Special thanks to Dr. Y.M. Jeon
#
# Last updated
#  2018.03.23 : version 0.10; Mirnov data resampling

from MDSplus import Connection
from MDSplus import DisconnectFromMds
from MDSplus._mdsshr import MdsException

import numpy as np
import matplotlib.pyplot as plt

VAR_NODE = {'NBI11':'NB11_pnb', 'NBI12':'NB12_pnb', 'NBI13':'NB13_pnb', 'ECH':'ECH_VFWD1', 'ECCD':'EC1_RFFWD1',
            'ICRF':'ICRF_FWD', 'LHCD':'LH1_AFWD', 'GASI':'I_GFLOW_IN:FOO', 'GASK':'K_GFLOW_IN:FOO', 'SMBI':'SM_VAL_OUT:FOO',
            'Ip':'RC03', 'neAVGM':'NE_INTER01', 'ECE05':'ECE05', 'ECE35':'ECE35', 'Rp':'LMSR',
            'Zp':'LMSZ', 'VOL':'VOLUME', 'KAP':'KAPPA', 'BETAp':'BETAP', 'BETAn':'BETAN', 'q95':'q95', 'Li':'LI3',
            'GASG':'G_GFLOW_IN:FOO', 'WTkp':'WTOT_KAPPA', 'WTdlm':'WTOT_DLM03', 'DaT11':'TOR_HA11', 'DaT10':'TOR_HA10',
            'DaP02':'POL_HA02', 'DaP04':'POL_HA04', 'neAVGF':'NE_INTER02',
            'RMP_T4':'PCRMPTBULI', 'RMP_T3':'PCRMPTFULI', 'RMP_T2':'PCRMPTJULI', 'RMP_T1':'PCRMPTNULI',
            'RMP_M4':'PCRMPMBULI', 'RMP_M3':'PCRMPMFULI', 'RMP_M2':'PCRMPMJULI', 'RMP_M1':'PCRMPMNULI',
            'RMP_B4':'PCRMPBBULI', 'RMP_B3':'PCRMPBFULI', 'RMP_B2':'PCRMPBJULI', 'RMP_B1':'PCRMPBNULI'}

# nodes in PCS_KSTAR tree
PCS_TREE = ['LMSR', 'LMSZ', 'PCRMPTBULI', 'PCRMPTFULI', 'PCRMPTJULI', 'PCRMPTNULI',
            'PCRMPMBULI', 'PCRMPMFULI', 'PCRMPMJULI', 'PCRMPMNULI', 'PCRMPBBULI', 'PCRMPBFULI', 'PCRMPBJULI', 'PCRMPBNULI']

# nodes in CSS tree
CSS_TREE = ['CSS_I%02d:FOO' % i for i in range(1,5)] + ['CSS_Q%02d:FOO' % i for i in range(1,5)]

# nodes in EFIT01 or EFIT02
EFIT_TREE = ['VOLUME', 'KAPPA', 'BETAP', 'BETAN', 'q95', 'LI3', 'WMHD']

# nodes need postprocessing
POST_NODE = {'ECH_VFWD1':'/1000', 'EC1_RFFWD1':'/1000', 'LH1_AFWD':'/200', 'SM_VAL_OUT:FOO':'/5',
            'RC03':'*(-1)/1000', 'NE_INTER01':'/1.9', 'LMSR':'-1.8', 'VOLUME':'/10', 'KAPPA':'-1',
            'NE_INTER02':'/2.7'}

# nodes support segment reading
SEG_NODE = ['ECE%02d' % i for i in range(2,150)] + ['MC1T%02d' % i for i in range(1,25)] + \
            ['MC1P%02d' % i for i in range(1,25)] + ['LM%02d' % i for i in range(1,5)] + \
            ['TOR_HA%02d' % i for i in range(1,25)] + ['POL_HA%02d' % i for i in range(1,25)]
SEG_NODE = SEG_NODE + ['NB11_pnb', 'NB12_pnb', 'NB13_pnb', 'ECH_VFWD1', 'ec1_rffwd1', 'I_GFLOW_IN:FOO', 'K_GFLOW_IN:FOO',
            'SM_VAL_OUT:FOO', 'G_GFLOW_IN:FOO', 'RC03']

# nodes need resampling
RES_NODE = ['MC1T%02d' % i for i in range(1,25)] + ['MC1P%02d' % i for i in range(1,25)]


class KstarMds(Connection):
    def __init__(self):
        # from iKSTAR
        # super(KstarMds,self).__init__('172.17.250.23:8005')  # call __init__ in Connection
        # from opi to CSS Host PC
        super(KstarMds,self).__init__('172.17.102.69:8000')  # call __init__ in Connection

    def get_data(self, shot, trange, clist, norm=0, atrange=[1.0, 1.1], res=0):
        if norm == 0:
            print 'data is not normalized'
        elif norm == 1:
            print 'data is normalized by trange average'
        elif norm == 2:
            print 'data is normalized by atrange average'

        self.shot = shot
        self.trange = trange
        self.clist = clist

        # open tree
        tree = find_tree(clist[0])
        try:
            self.openTree(tree,self.shot)
        except:
            print "Failed to open tree %s" % tree
            time, data = None, None
            return time, data

        # --- loop starts --- #
        for i, cname in enumerate(self.clist):

            # get MDSplus node from channel name
            if cname in VAR_NODE:
                node = '%s' % (VAR_NODE[cname])
            else:
                node = cname

            # segment loading
            if node in SEG_NODE:
                if node in RES_NODE and res != 0:
                    snode = 'resample(\%s, %f, %f, %f)' % (node,self.trange[0],self.trange[1],res)  # segment loading with resampling
                else:
                    snode = '\%s[%g:%g]' % (node,self.trange[0],self.trange[1])  # segment loading
            else:
                snode = node

            # post processing
            if node in POST_NODE:
                pnode = POST_NODE[node]
            else:
                pnode = ''
            node = snode + pnode

            # load data
            expr = '[dim_of({0}), {0}]'.format(node)
            try:
                time, v = self.get(expr).data()
                print "Read %s (number of data points = %d)" % (node, len(v))
            except:
                time, v = None, None
                print "Failed   %s" % node

            # set data size
            idx = np.where((time >= trange[0])*(time <= trange[1]))
            idx1 = int(idx[0][0])
            idx2 = int(idx[0][-1]+2)
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
        self.channel_position(shot,clist)

        # close tree
        self.closeTree(tree,self.shot)

        return time, data

    def channel_position(self, shot, clist):  # Needs updates ####################
        cnum = len(clist)
        self.rpos = np.arange(cnum)  # R [m]
        self.zpos = np.zeros(cnum)  # z [m]
        self.apos = np.arange(cnum)  # angle [rad]
        for c in range(cnum):
            # Mirnov coils
            # ECE
            pass


def find_tree(cname):
    # cname -> node
    if cname in VAR_NODE:
        node = '%s' % (VAR_NODE[cname])
    else:
        node = cname

    # find tree
    if node in PCS_TREE:
        tree = 'PCS_KSTAR'
    elif node in CSS:
        tree = 'CSS'
    elif node in EFIT_TREE:
        tree = 'EFIT01'
    else:
        tree = 'KSTAR'

    return tree


if __name__ == "__main__":
    pass

    # g = KstarMds()
    # g.get_data(shot=17245,trange=[0,10],clist=['neAVGM'])
    # plt.plot(g.time, g.data[0,:], color='k')
    # plt.show()
    # g.close

# DisconnectFromMds(g.socket)
