import sys, os
sys.path.insert(0, os.pardir)
from fluctana import *

# HOW TO RUN
# python3 check_SKw.py 19158 [5.5,5.6] ECEI_G1003-1403 ECEI_G1103-1503
# 
# This calculates the local S(K,w). K along the direction of pairs

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

# do fft 
A.fftbins(nfft=512,window='hann',overlap=0.5,detrend=0)

# do cwt; full = 0 (0 ~ fN) or 1 (-fN ~ fN)
# A.cwt(df=5000,full=0,tavg=1000)

# calculate local wavenumber-frequency spectra using channel pairs of done and dtwo and plot
A.skw(done=0, dtwo=1, kstep=0.01)
