import sys, os
sys.path.append(os.pardir)
from fluctana import *

import pickle
import math

shot = int(sys.argv[1]) # 19158
trange = eval(sys.argv[2]) # [5.5,5.6]
clist1 = sys.argv[3].split(',') # ECEI_G1201-1208
clist2 = sys.argv[4].split(',') # ECEI_G1301-1308

# add data
A = FluctAna()
if clist1[0][0:4] == 'ECEI':
    A.add_data(KstarEcei(shot=shot, clist=clist1), trange=trange, norm=1)
    A.add_data(KstarEcei(shot=shot, clist=clist2), trange=trange, norm=1)
else:
    A.add_data(KstarMds(shot=shot, clist=clist1), trange=trange, norm=0)
    A.add_data(KstarMds(shot=shot, clist=clist2), trange=trange, norm=0)

# list data
A.list_data()

## do fft 
A.fftbins(nfft=512,window='hann',overlap=0.5,detrend=0)

## calculate coherence
A.coherence(done=0,dtwo=1)

## plot the result
A.mplot(dnum=1,cnl=range(len(A.Dlist[1].clist)),type='val',ylimits=[0,1])

