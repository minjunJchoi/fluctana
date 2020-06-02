import sys, os
sys.path.append(os.pardir)
from fluctana import *

# HOW TO RUN
# python3 check_hurst.py 10186 [15.9,16] ECEI_L1303-1305

shot = int(sys.argv[1]) 
trange = eval(sys.argv[2])
clist = sys.argv[3].split(',')

# call fluctana
A = FluctAna()

# add data
A.add_data(dev='KSTAR', shot=shot, clist=clist, trange=trange, norm=1)

# list data
A.list_data()

# calculate H using data set dnum. results are saved in A.Dlist[dtwo].val; fitrange = fitting range of time lag in us.
A.hurst(dnum=0, cnl=range(len(A.Dlist[0].clist)), bins=200, detrend=1, fitrange=[100,1000])

# plot the results; dnum = data set number, cnl = channel number list to plot
A.mplot(dnum=0, cnl=range(len(A.Dlist[0].clist)), type='val')

# plot over plane; 
# A.cplot(dnum=0, snum=0, vlimits=[0.5,1])