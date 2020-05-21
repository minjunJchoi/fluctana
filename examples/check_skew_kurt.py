import sys, os
sys.path.append(os.pardir)
from fluctana import *

# HOW TO RUN
# ./python3 check_skew_kurt.py 10186 [15.9,16] ECEI_L1303-1305

shot = int(sys.argv[1]) 
trange = eval(sys.argv[2])
clist = sys.argv[3].split(',')

# call fluctana
A = FluctAna()

# add data
A.add_data(dev='KSTAR', shot=shot, clist=clist, trange=trange, norm=1)

# list data
A.list_data()

# calculate skewness and kurtosis (flatness) using data set dnum and plot over SK plane; dnum = data set, cnl = channel number list
A.skplane(dnum=0, cnl=range(len(A.Dlist[0].clist)), detrend=1, verbose=1)

# skewness only
# A.skewness(dnum=0, cnl=range(len(A.Dlist[0].clist)), detrend=1, verbose=1)
# A.mplot(dnum=0, cnl=range(len(A.Dlist[0].clist)), type='val')
# A.cplot(dnum=0, snum=0, vlimits=[-0.5,0.5])

# kurtosis only
# A.kurtosis(dnum=0, cnl=range(len(A.Dlist[0].clist)), detrend=1, verbose=1)
# A.mplot(dnum=0, cnl=range(len(A.Dlist[0].clist)), type='val')
# A.cplot(dnum=0, snum=0, vlimits=[-0.5,0.5])