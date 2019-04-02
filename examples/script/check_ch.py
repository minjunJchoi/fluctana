import sys, os
sys.path.append(os.pardir)
from fluctana import *

shot = int(sys.argv[1]) # shot
trange = eval(sys.argv[2])
clist = sys.argv[3].split(',')

A = FluctAna()
if clist[0][0:4] == 'ECEI':
    A.add_data(KstarEcei(shot=shot, clist=clist), trange=trange, norm=1)
else:
    A.add_data(KstarMds(shot=shot, clist=clist), trange=trange, norm=0)

# list data
A.list_data()

## down sampling
A.downsample(q=10, verbose=0)

## filter
# A.filt('FIR_pass',0,15000,b=0.01,verbose=1) 
# A.filt('FIR_pass',5000,0,b=0.01,verbose=0) # smaller b is sharper
# A.filt('FIR_pass',0,50000,b=0.01,verbose=0) # smaller b is sharper
# A.filt('FIR_pass',10000,0,b=0.01,verbose=1)
# A.svd_filt(cutoff=0.9, verbose=0)

# A.mplot(dnum=0, cnl=range(len(A.Dlist[0].clist)), type='time')

## over plane
A.chplane(dnum=0, cnl=range(len(A.Dlist[0].clist)), d=6, bins=1, verbose=1)

## complexity only
# A.js_complexity(dnum=0, cnl=range(len(A.Dlist[0].clist)), d=6, bins=1)
# A.mplot(dnum=0, cnl=range(len(A.Dlist[0].clist)), type='val')
# A.cplot(dnum=0, snum=0, vlimits=[0,1.0])

## entropy only
# A.ns_entropy(dnum=0, cnl=range(len(A.Dlist[0].clist)), d=6, bins=1)
# A.mplot(dnum=0, cnl=range(len(A.Dlist[0].clist)), type='val')
# A.cplot(dnum=0, snum=0, vlimits=[0,1.0])