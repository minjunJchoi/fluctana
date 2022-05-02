# Author : Minjun J. Choi (mjchoi@nfri.re.kr)
#
# Description : This code reads the KSTAR CSS data from the KSTAR MDSplus server
#
# Acknowledgement : Special thanks to Dr. W. Lee, Mr. D.J. Lee, and Mr. T.G. Lee
#

from MDSplus import Connection
# from MDSplus import DisconnectFromMds
# from MDSplus._mdsshr import MdsException

import numpy as np
import matplotlib.pyplot as plt

# nodes in CSS tree
CSS_TREE = ['CSS_I{:02d}:FOO'.format(i) for i in range(1,5)] + ['CSS_Q{:02d}:FOO'.format(i) for i in range(1,5)]

class KstarCss(Connection):
    def __init__(self, shot, clist):
        # from iKSTAR
        super(KstarCss,self).__init__('172.17.100.200:8005')  # call __init__ in Connection
        # from opi to CSS Host PC
        # super(KstarMds,self).__init__('172.17.102.69:8000')  # call __init__ in Connection
        self.shot = shot
        
        self.clist = self.expand_clist(clist)

        self.channel_position()

        self.time = None
        self.data = None

    def get_data(self, trange, norm=0, atrange=[1.0, 1.1], res=0, verbose=1):
        if norm == 0:
            if verbose == 1: print('Data is not normalized {:s}'.format(self.clist[0]))
        elif norm == 1:
            if verbose == 1: print('Data is normalized by trange std {:s}'.format(self.clist[0]))
        # elif norm == 2:
        #     if verbose == 1: print('Data is normalized by atrange std {:s}'.format(self.clist[0]))

        self.trange = trange

        # open tree
        tree = 'CSS'
        try:
            self.openTree(tree,self.shot)
            if verbose == 1: print('Open the tree {:s} to get data {:s}'.format(tree, self.clist[0]))
        except:
            if verbose == 1: print('Failed to open the tree {:s} to get data {:s}'.format(tree, self.clist[0]))
            return self.time, self.data

        # --- loop starts --- #
        clist_temp = self.clist[:]
        for i, cname in enumerate(clist_temp):

            # get MDSplus node from channel name
            inode = cname[0:4] + 'I{:02d}:FOO'.format(int(cname[4:6]))
            qnode = cname[0:4] + 'Q{:02d}:FOO'.format(int(cname[4:6]))

            # resampling, time node
            if res != 0:
                tnode = 'dim_of(resample(\{:s},{:f},{:f},{:f}))'.format(inode,self.trange[0],self.trange[1],res)  # resampling                
                inode = 'resample(\{:s},{:f},{:f},{:f})'.format(inode,self.trange[0],self.trange[1],res)  # resampling
                qnode = 'resample(\{:s},{:f},{:f},{:f})'.format(qnode,self.trange[0],self.trange[1],res)  # resampling
            else:
                tnode = 'setTimeContext({:f},{:f},*),dim_of(\{:s})'.format(self.trange[0],self.trange[1],inode)
                inode = 'setTimeContext({:f},{:f},*),\{:s}'.format(self.trange[0],self.trange[1],inode)
                qnode = 'setTimeContext({:f},{:f},*),\{:s}'.format(self.trange[0],self.trange[1],qnode)

            try:
                # load data
                iv = self.get(inode).data()
                qv = self.get(qnode).data()
                if verbose == 1: print('Read {:d} : {:s}, {:s} (number of data points = {:d})'.format(self.shot, inode, qnode, len(iv)))

                # load time
                if self.data is None:
                    self.time = self.get(tnode).data()
                    # get fs
                    self.fs = round(1/(self.time[1] - self.time[0])/1000)*1000.0

                # remove offset
                iv = iv - np.mean(iv)
                qv = qv - np.mean(qv)

                # normalize by std if norm == 1
                if norm == 1:
                    iv = iv/np.std(iv)
                    qv = qv/np.std(qv)

                # make v from iv, qv
                v = iv + 1.0j*qv
                # print('TRY pre-filtering of iv and qv (threshold fft) and return iv + 1.0j*qv')

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
                if verbose == 1: print('Failed {:d} : {:s}, {:s}. {:s} is removed'.format(self.shot, inode, qnode, cname))
        # --- loop ends --- #

        # remove positions of bad channels
        if hasattr(self, 'rpos'):
            cidx = self.rpos >= 0
            self.rpos = self.rpos[cidx]
            self.zpos = self.zpos[cidx]
            self.apos = self.apos[cidx]

        # close tree
        self.closeTree(tree, self.shot)

        return self.time, self.data

    def channel_position(self):  # Needs updates ####################
        # get channel position either from MDSplus server or kstardata
        cnum = len(self.clist)
        self.rpos = np.arange(cnum, dtype=np.float64)  # R [m]
        self.zpos = np.zeros(cnum)  # z [m]
        self.apos = np.arange(cnum, dtype=np.float64)  # angle [rad]

        # need to read from google document
        # try: 
        #     print('Read channel position {:s}'.format(self.clist[0]))
        # except:
        #     print('Failed to read the channel position from MDSplus {:s}'.format(self.clist[0]))
        #     print('Try to get the position from kstardata {:s}'.format(self.clist[0]))

    def expand_clist(self, clist):
        # IN : List of channel names (e.g. 'CSS_01-04')
        # OUT : Expanded list (e.g. 'CSS_01', 'CSS_02', 'CSS_03', 'CSS_04')

        # KSTAR CSS
        exp_clist = []
        for c in range(len(clist)):
            if 'CSS' in clist[c] and len(clist[c]) == 9:
                vi = int(clist[c][4:6])
                vf = int(clist[c][7:9])

                for v in range(vi, vf+1):
                    exp_clist.append(clist[c][0:4] + '{:02d}'.format(v))
            else:
                exp_clist.append(clist[c])
        clist = exp_clist

        return clist


class NoPosMdsError(Exception):
    def __init__(self, msg='No position in MDSplus server'):
        self.msg = msg

    def __str__(self):
        return self.msg


if __name__ == "__main__":
    pass

    # g = KstarMds(shot=17245,clist=['neAVGM'])
    # g.get_data(trange=[0,10])
    # plt.plot(g.time, g.data[0,:], color='k')
    # plt.show()
    # g.close

# DisconnectFromMds(g.socket)
