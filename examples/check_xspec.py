import sys, os
sys.path.append(os.pardir)
from fluctana import *

# HOW TO RUN
# ./python3 check_xspec.py 10186 [15,16] ECEI_L1303 ECEI_L1403

shot = int(sys.argv[1]) 
trange = eval(sys.argv[2]) 
clist1 = sys.argv[3].split(',') 
clist2 = sys.argv[4].split(',') 

# call fluctana
A = FluctAna()

# add data
A.add_data(dev='KSTAR', shot=shot, clist=clist1, trange=trange, norm=1)
A.add_data(dev='KSTAR', shot=shot, clist=clist2, trange=trange, norm=1)

# list data
A.list_data()

# xspec parameters 
# frequency resolution ~ sampling frequency / nfft
nfft = 8192
# temporal resolution 
overlap = 0.8 # finest overlap = (nfft-1.0)/nfft
# for full frequency range, full=1 (e.g. MIR). Else full=0.
A.fftbins(nfft=nfft, window='kaiser', overlap=overlap, detrend=0, full=0)

# calculate the cross power spectrogram using data sets done and dtwo; thres = threshold for significance
A.xspec(done=0,dtwo=1,thres=0.1)