import sys, os
sys.path.append(os.pardir)
from fluctana import *

import pickle
import math

shot = int(sys.argv[1]) # 19158
trange = eval(sys.argv[2]) # [5.5,5.6]
trange1 = trange
trange2 = np.array(trange) - 2.67e-06*0.0 # for 19134 [6.5,6.53]
clist1 = sys.argv[3].split(',') # ECEI_G1201-1208
clist2 = sys.argv[4].split(',') # ECEI_G1301-1308

# add data
A = FluctAna()
if clist1[0][0:4] == 'ECEI':
    A.add_data(KstarEcei(shot=shot, clist=clist1), trange=trange1, norm=1)
    A.add_data(KstarEcei(shot=shot, clist=clist2), trange=trange2, norm=1)
else:
    A.add_data(KstarMds(shot=shot, clist=clist1), trange=trange1, norm=0)
    A.add_data(KstarMds(shot=shot, clist=clist2), trange=trange2, norm=0)

# list data
A.list_data()

# filtering
# A.filt('FIR_pass',0,75000,b=0.01,verbose=1) 

## do fft for full range for bicoherence
# A.fftbins(nfft=512,window='hann',overlap=0.5,detrend=0,full=1)
A.fftbins(nfft=512,window='kaiser',overlap=0.8,detrend=0,full=1)

# Ritz's nonlinear energy transfer
# A.ritz_nonlinear(done=0,dtwo=1)

# Modified Ritz's nonlinear energy transfer
A.ritz_mod_nonlinear(done=0,dtwo=1,cnl=[0])

