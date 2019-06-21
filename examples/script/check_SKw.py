import sys, os
sys.path.append(os.pardir)
from fluctana import *

import pickle
import math

shot = int(sys.argv[1]) # shot
trange = eval(sys.argv[2]) # 
clist1 = sys.argv[3].split(',')  
clist2 = sys.argv[4].split(',') 

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

## calculate local wavenumber-frequency spectra using channel pairs of done and dtwo
A.skw(done=0, dtwo=1, kstep=0.01)
