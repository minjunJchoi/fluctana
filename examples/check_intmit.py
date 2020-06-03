import sys, os
sys.path.append(os.pardir)
from fluctana import *

# HOW TO RUN
# python3 check_intmit.py 10186 [15.1,15.17] ECEI_L1303

shot = int(sys.argv[1]) 
trange = eval(sys.argv[2])
clist = sys.argv[3].split(',')

# call fluctana
A = FluctAna()

# add data
A.add_data(dev='KSTAR', shot=shot, clist=clist, trange=trange, norm=0)

# list data
A.list_data()

# calculate using data set dnum. results are saved in A.Dlist[dtwo].val; fitlims = fitting range of time lag in us.
A.intermittency(dnum=0, cnl=[0], bins=50, overlap=0.2, qstep=0.3, fitrange=[20.0,100.0], verbose=1)


