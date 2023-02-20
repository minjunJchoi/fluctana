import sys, os
sys.path.insert(0, os.pardir)
from fluctana import *

## ===================== Calculate cross power rms ================
shot = 10186
trange = [15,16]
clist1 = ['ECEI_L1303']
clist2 = ['ECEI_L1403']
vkind = 'cross_power'
twin = 50e-3; tstep = 10e-3
vpara = {'nfft':512, 'window':'hann', 'overlap':0.5, 'detrend':0, 'full':0, 'norm':1, 'f1':1000, 'f2':50000, 'clim':0}

# read not normalized data      
A = FluctAna()
A.add_data(dev='KSTAR', shot=shot, clist=clist1, trange=trange, norm=0)
A.add_data(dev='KSTAR', shot=shot, clist=clist2, trange=trange, norm=0)

# calculation along time 
A.tcal(done=0, dtwo=1, twin=twin, tstep=tstep, vkind=vkind, vpara=vpara)

# take result 
taxis = A.Dlist[0].ax
cpower = A.Dlist[0].val[0,:]

# plot result 
plt.plot(taxis, cpower)
plt.show()

