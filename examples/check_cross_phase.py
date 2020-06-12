import sys, os
sys.path.insert(0, os.pardir)
from fluctana import *

# HOW TO RUN
# python3 check_cross_phase.py 10186 [15.9,16] ECEI_L1303 ECEI_L1403

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

# do fft; full = 0 (0 ~ fN) or 1 (-fN ~ fN)
A.fftbins(nfft=256,window='hann',overlap=0.5,detrend=0,full=0)

# do cwt; full = 0 (0 ~ fN) or 1 (-fN ~ fN)
# A.cwt(df=5000,full=1,tavg=1000)

# calculate cross phase using data sets done and dtwo. results are saved in A.Dlist[dtwo].val
A.cross_phase(done=0,dtwo=1)

# plot the results; dnum = data set number, cnl = channel number list to plot
A.mplot(dnum=1,cnl=range(len(A.Dlist[1].clist)),type='val',ylimits=[-3.14,3.14])

# calculate phase velocity
A.cplot(dnum=1,snum=0,frange=[20,40],vlimits=[-10,10])

# phase velocity
# print(A.Dlist[1].pdata)
# distance between channels
# print(A.Dlist[1].dist)
# time 
# print(A.Dlist[1].dist/(A.Dlist[1].pdata*1000))