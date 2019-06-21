import sys, os
sys.path.append(os.pardir)
from fluctana import *

# HOW TO RUN
# ./python3 check_ch.py 10186 [15.9,16] ECEI_L1301-L1307

shot = int(sys.argv[1]) 
trange = eval(sys.argv[2])
clist = sys.argv[3].split(',')

# call fluctana
A = FluctAna()

# add data
A.add_data(KstarEcei(shot=shot, clist=clist), trange=trange, norm=1)

# list data
A.list_data()

# down sampling with the order q if necessary 
A.downsample(q=10, verbose=1)

# calculate CH values using data set dnum and plot over CH plane; dnum = data set, cnl = channel number list, d = dimension
A.chplane(dnum=0, cnl=range(len(A.Dlist[0].clist)), d=6, verbose=1)

# complexity only
# A.js_complexity(dnum=0, cnl=range(len(A.Dlist[0].clist)), d=6, bins=1)
# A.mplot(dnum=0, cnl=range(len(A.Dlist[0].clist)), type='val')
# A.cplot(dnum=0, snum=0, vlimits=[0,1.0])

# entropy only
# A.ns_entropy(dnum=0, cnl=range(len(A.Dlist[0].clist)), d=6, bins=1)
# A.mplot(dnum=0, cnl=range(len(A.Dlist[0].clist)), type='val')
# A.cplot(dnum=0, snum=0, vlimits=[0,1.0])