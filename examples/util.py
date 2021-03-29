import sys, os
sys.path.insert(0, os.pardir)
from fluctana import *
import filtdata as ft
import math

def ece_channel_selection(shot, Rrange):
    # select channels to read
    clist_temp = ['ECE{:02d}'.format(i) for i in range(1,77)]
    if shot <= 13728:
        clist_temp = ['ECE{:02d}'.format(i) for i in range(1,49)]
    M = KstarMds(shot=shot, clist=clist_temp)
    # print(M.rpos)
    # print(M.clist)    
    M.rpos[np.isnan(M.rpos)] = 0.0 # zero for nan channels
    idx = np.where((M.rpos >= Rrange[0]) * (M.rpos <= Rrange[-1]))[0]
    # M.clist = ['{:s}'.format(clist_temp[i]) for i in idx]
    # M.rpos = M.rpos[idx]
    # print(M.rpos)
    # print(M.clist)
    clist = ['{:s}'.format(clist_temp[i]) for i in idx]

    return clist
