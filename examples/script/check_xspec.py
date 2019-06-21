import sys, os
sys.path.append(os.pardir)
from fluctana import *

import pickle
import math

shot = int(sys.argv[1]) # 20896
trange = eval(sys.argv[2]) # [1,10]
clist1 = sys.argv[3].split(',')
clist2 = sys.argv[4].split(',')

# add data
A = FluctAna()
if clist1[0][0:4] == 'ECEI':
    A.add_data(KstarEcei(shot=shot, clist=clist1), trange=trange, norm=0)
    A.add_data(KstarEcei(shot=shot, clist=clist2), trange=trange, norm=0)
else:
    A.add_data(KstarMds(shot=shot, clist=clist1), trange=trange, norm=0)
    A.add_data(KstarMds(shot=shot, clist=clist2), trange=trange, norm=0)

# list data
A.list_data()

## xspec
# frequency resolution ~ sampling frequency / nfft
nfft = 2048
# temporal resolution 
overlap = 0.5 # finest overlap = (nfft-1.0)/nfft
# for full frequency range, full=1 (e.g. MIR). else full=0.
A.fftbins(nfft=nfft, window='kaiser', overlap=overlap, detrend=0, full=0)
# calculate the cross power between done and dtwo channels.
A.xspec(done=0,dtwo=1,thres=0.1)

print(A.Dlist[0].apos)
print(A.Dlist[1].apos)