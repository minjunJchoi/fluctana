# Author : Minjun J. Choi (mjchoi@kfe.re.kr)
#
# Description : This code reads the KSTAR BES data from the KSTAR MDSplus server
#
# Acknowledgement : Special thanks to Dr. J.W. Kim
#

from MDSplus import Connection
# from MDSplus import DisconnectFromMds
# from MDSplus._mdsshr import MdsException

import numpy as np
import matplotlib.pyplot as plt

# BES tree
BES_TREE = 'KSTAR'

class KstarBes(Connection):
    def __init__(self, shot, clist):
        # from iKSTAR/uKSTAR
        super(KstarBes,self).__init__('172.17.100.200:8005')  # call __init__ in Connection

        self.shot = shot

        self.clist = self.expand_clist(clist)

        self.channel_position()

        self.time = None
        self.data = None

    def get_data(self, trange, norm=0, atrange=[1.0, 1.1], res=0, verbose=1):
        if norm == 0:
            if verbose == 1: print('Data is not normalized')
        elif norm == 1:
            if verbose == 1: print('Data is normalized by trange {:g}-{:g} average'.format(trange[0],trange[1]))
        elif norm == 2:
            if verbose == 1: print('Data is normalized by atrange {:g}-{:g} average'.format(atrange[0],atrange[1]))

        self.trange = trange

        # open tree
        try:
            self.openTree(BES_TREE, self.shot)
            if verbose == 1: print('Open the tree {:s} to get data {:s}'.format(BES_TREE, self.clist[0]))
        except:
            if verbose == 1: print('Failed to open the tree {:s} to get data {:s}'.format(BES_TREE, self.clist[0]))
            return self.time, self.data

        # --- loop starts --- #
        clist_temp = self.clist[:]
        for i, cname in enumerate(clist_temp):

            # get MDSplus node from channel name
            node = cname + ':FOO'

            # sub sampling node
            # time_node = 'setTimeContext({:f},{:f},*),dim_of(\{:s})'.format(self.trange[0],self.trange[1],node)
            # data_node = 'setTimeContext({:f},{:f},*),\{:s}'.format(self.trange[0],self.trange[1],node) # offset subtracted already
            time_node = 'dim_of(resample(\{:s},{:f},{:f},2e-6))'.format(node,self.trange[0],self.trange[1])
            data_node = 'resample(\{:s},{:f},{:f},2e-6)'.format(node,self.trange[0],self.trange[1]) # offset subtracted already

            if norm == 2:
                # adata_node = 'setTimeContext({:f},{:f},*),\{:s}'.format(atrange[0],atrange[1],node) # offset subtracted already
                adata_node = 'resample(\{:s},{:f},{:f},2e-6)'.format(node,atrange[0],atrange[1]) # offset subtracted already

            try:
                # load time
                if self.data is None:
                    self.time = self.get(time_node).data()
                    # get fs
                    self.fs = round(1/(self.time[1] - self.time[0])/1000)*1000.0

                # load data
                v = self.get(data_node).data()
                if verbose == 1: print('Read {:d} : {:s} (number of data points = {:d})'.format(self.shot, node, len(v)))

                # normalize by std if norm == 1
                if norm == 1:
                    v = v/np.mean(v) - 1
                elif norm == 2:
                    av = self.get(adata_node).data()
                    v = v/np.mean(av) - 1

                # expand dimension - concatenate
                v = np.expand_dims(v, axis=0)
                if self.data is None:
                    self.data = v
                else:
                    self.data = np.concatenate((self.data, v), axis=0)

            except:
                self.clist.remove(cname)
                if hasattr(self, 'rpos'):
                    self.rpos[i] = -1
                if verbose == 1: print('Failed {:d} : {:s} {:s} is removed'.format(self.shot, data_node, cname))
        # --- loop ends --- #

        # remove positions of bad channels
        if hasattr(self, 'rpos'):
            cidx = self.rpos >= 0
            self.rpos = self.rpos[cidx]
            self.zpos = self.zpos[cidx]
            self.apos = self.apos[cidx]

        # close tree
        self.closeTree(BES_TREE, self.shot)

        return self.time, self.data

    def get_multi_data(self, time_list=None, tspan=1e-3, norm=0, res=0, verbose=1):
        if norm == 0:
            if verbose == 1: print('Data is not normalized')
        elif norm == 1:
            if verbose == 1: print('Data is normalized by time average')

        self.time_list = time_list

        # open tree
        try:
            self.openTree(BES_TREE, self.shot)
            if verbose == 1: print('Open the tree {:s} to get data {:s}'.format(BES_TREE, self.clist[0]))
        except:
            if verbose == 1: print('Failed to open the tree {:s} to get data {:s}'.format(BES_TREE, self.clist[0]))
            return self.multi_time, self.multi_data

        # check data size and get fs
        # test_node = 'setTimeContext({:f},{:f},*),dim_of(\BES_0101:FOO)'.format(time_list[0]-tspan/2,time_list[0]+tspan/2)
        test_node = 'dim_of(resample(\BES_0101:FOO,{:f},{:f},2e-6))'.format(time_list[0]-tspan/2,time_list[0]+tspan/2)
        test_data = self.get(test_node).data()
        # get fs
        self.fs = round(1/(test_data[1] - test_data[0])/1000)*1000.0        

        # --- loop starts --- # assuming all good channels 
        self.multi_time = np.zeros((len(time_list), len(test_data)))
        self.multi_data = np.zeros((len(self.clist), len(time_list), len(test_data)))

        for i, cname in enumerate(self.clist):
            # get MDSplus node from channel name
            node = cname + ':FOO'

            for j, tp in enumerate(time_list):
                # sub sampling node
                # time_node = 'setTimeContext({:f},{:f},*),dim_of(\{:s})'.format(tp-tspan/2,tp+tspan/2,node)
                time_node = 'dim_of(resample(\{:s},{:f},{:f},2e-6))'.format(node,tp-tspan/2,tp+tspan/2)
                # data_node = 'setTimeContext({:f},{:f},*),\{:s}'.format(tp-tspan/2,tp+tspan/2,node) # offset subtracted already
                data_node = 'resample(\{:s},{:f},{:f},2e-6)'.format(node,tp-tspan/2,tp+tspan/2) # offset subtracted already

                try:
                    # load time
                    if i == 0:
                        self.multi_time[j,:] = self.get(time_node).data()

                    # load data
                    v = self.get(data_node).data()
                    if verbose == 1: print('Read {:d} : {:s} (number of data points = {:d})'.format(self.shot, node, len(v)))

                    # normalize by std if norm == 1
                    if norm == 1:
                        v = v/np.mean(v) - 1

                    # expand dimension - concatenate
                    self.multi_data[i,j,:] = v

                except:
                    print('Failed to get data {:d} : {:s} {:s}'.format(self.shot, data_node, cname))
        # --- loop ends --- #

        # close tree
        self.closeTree(BES_TREE, self.shot)

        return self.multi_time, self.multi_data

    def channel_position(self):
        # get channel position from MDSplus server
        cnum = len(self.clist)
        self.rpos = np.zeros(cnum)  # R [m]
        self.zpos = np.zeros(cnum)  # z [m]
        self.apos = np.zeros(cnum)  # angle [rad]

        # open tree
        self.openTree(BES_TREE, self.shot)
        print('OPEN MDS tree {:s} to read rpos'.format(BES_TREE))
        
        # read rnode from MDSplus
        cnum = len(self.clist)
        for c in range(cnum):
            # set r,z node
            rnode = '\{:s}:RPOS'.format(self.clist[c]) 
            znode = '\{:s}:VPOS'.format(self.clist[c]) 

            # read r,z node
            self.rpos[c] = self.get(rnode) / 1000 # [mm] -> [m]
            self.zpos[c] = self.get(znode) / 1000 # [mm] -> [m]
        
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
