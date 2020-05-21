import sys, os
sys.path.append(os.pardir)
from fluctana import *

# HOW TO RUN
# ./python3 check_bicoherence.py 10186 [15.9,16] ECEI_L1303 ECEI_L1303

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

# do fft for full range for bicoherence
A.fftbins(nfft=512,window='kaiser',overlap=0.8,detrend=0,full=1)

# do cwt; full = 0 (0 ~ fN) or 1 (-fN ~ fN)
# A.cwt(df=5000,full=1,tavg=1000)

# calculate the bicoherence using data in data sets done and dtwo and show results
A.bicoherence(done=0,dtwo=1)