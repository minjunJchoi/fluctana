import sys, os
sys.path.insert(0, os.pardir)
from fluctana import *
import argparse

## HOW TO RUN
## python3 check_cross_power_rms_in_time.py -shot 10186 -trange 15 16 -ref_clist ECEI_L1303 -cmp_clist ECEI_L1403

parser = argparse.ArgumentParser(description="Cross power rms in time")
parser.add_argument("-shot", type=int, default=10186, help="Shot number")
parser.add_argument("-trange", nargs=2, type=float, default=[15, 16], help="Time range [ms]")
parser.add_argument("-ref_clist", nargs='+', type=str, default=['ECEI_L1303'], help="Reference channel list")
parser.add_argument("-cmp_clist", nargs='+', type=str, default=['ECEI_L1403'], help="Comparison channel list")
a = parser.parse_args()

vkind = 'cross_power'
twin = 50e-3; tstep = 10e-3
vpara = {'nfft':512, 'window':'hann', 'overlap':0.5, 'detrend':0, 'full':0, 'norm':1, 'f1':1000, 'f2':50000, 'clim':0}

# read not normalized data      
A = FluctAna()
A.add_data(dev='KSTAR', shot=a.shot, clist=a.ref_clist, trange=a.trange, norm=0)
A.add_data(dev='KSTAR', shot=a.shot, clist=a.cmp_clist, trange=a.trange, norm=0)

# calculation along time 
A.tcal(done=0, dtwo=1, twin=twin, tstep=tstep, vkind=vkind, vpara=vpara)

# take result 
taxis = A.Dlist[0].ax
cpower = A.Dlist[0].val[0,:]

# plot result 
plt.plot(taxis, cpower)
plt.xlabel('Time [sec]')
plt.ylabel('Cross power RMS')
plt.title('Shot {:d}, Ref {:s}, Cmp {:s}'.format(a.shot, a.ref_clist[0], a.cmp_clist[0]))
plt.show()

