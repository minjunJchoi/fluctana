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

## do fft for full range for bicoherence
A.fftbins(nfft=512,window='hann',overlap=0.5,detrend=0,full=1)
# A.fftbins(nfft=512,window='kaiser',overlap=0.8,detrend=0,full=1)

# Plot 2D 
A.bicoherence(done=0,dtwo=1)

# Plot 1D
A.bicoherence(done=0,dtwo=1,sum=1)

