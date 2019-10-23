import sys, os
sys.path.append(os.pardir)
import kstarmds
import numpy as np

def get_bt(shot):
    print('Read Bt from MDSplus')
    B = kstarmds.KstarMds(shot, ['PCITFMSRD'])
    _, data = B.get_data(trange=[3,6], norm=0)
    bt = np.mean(data)*0.0995556*0.001

    return bt