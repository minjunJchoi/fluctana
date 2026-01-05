import sys, os
sys.path.insert(0, os.pardir)
from fluctana import *
import argparse

## HOW TO RUN
## python3 check_spectrum.py -slist 10186 -tlist 15.1 16.1 -twin 0.1 -ref_clist ECEI_L1303-1403 -cmp_clist ECEI_L1403-1503 --co

parser = argparse.ArgumentParser(description="Coherence, Cross power, Cross phase")
parser.add_argument("-slist", nargs='+', type=int, default=[10186], help="Shot number list")
parser.add_argument("-tlist", nargs='+', type=float, default=[15.95, 17], help="Time points [sec] list")
parser.add_argument("-twin", type=float, default=0.1, help="Time window [sec]")
parser.add_argument("-ref_clist", nargs='+', type=str, default=['ECEI_L1303-1403'], help="Reference channel list")
parser.add_argument("-cmp_clist", nargs='+', type=str, default=['ECEI_L1403-1503'], help="Comparison channel list")
parser.add_argument("--co", action="store_true", help="Coherence")
parser.add_argument("--pw", action="store_true", help="Cross power")
parser.add_argument("--ph", action="store_true", help="Cross phase")
a = parser.parse_args()


# Read data 
A = FluctAna()
for shot in a.slist:    
    for time_point in a.tlist:
        trange = [time_point - a.twin/2, time_point + a.twin/2]
        A.add_data(dev='KSTAR', shot=shot, clist=a.ref_clist, trange=trange, norm=1)
        A.add_data(dev='KSTAR', shot=shot, clist=a.cmp_clist, trange=trange, norm=1)


# # do fft; full = 0 (0 ~ fN) or 1 (-fN ~ fN)
A.fftbins(nfft=256,window='hann',overlap=0.5,detrend=0,full=0)

# # do cwt; full = 0 (0 ~ fN) or 1 (-fN ~ fN)
# # A.cwt(df=5000,full=0,tavg=2000)

# calculate coherence using data sets done and dtwo. results are saved in A.Dlist[dtwo].val
if a.co:
    for dnum in range(1, len(A.Dlist), 2):
        A.coherence(done=dnum - 1, dtwo=dnum)
        if dnum < len(A.Dlist) - 1: # before last loop
            fig, axs = A.mplot(dnum=dnum, type='val', show=0)
        else:
            A.mplot(dnum=dnum, type='val', show=1, fig=fig, axs=axs)

# calculate cross_power using data sets done and dtwo. results are saved in A.Dlist[dtwo].val
if a.pw:
    for dnum in range(1, len(A.Dlist), 2):
        A.cross_power(done=dnum - 1, dtwo=dnum)
        if dnum < len(A.Dlist) - 1: # before last loop
            fig, axs = A.mplot(dnum=dnum, type='val', show=0)
        else:
            A.mplot(dnum=dnum, type='val', show=1, fig=fig, axs=axs)

# calculate cross_phase using data sets done and dtwo. results are saved in A.Dlist[dtwo].val
if a.ph:
    for dnum in range(1, len(A.Dlist), 2):
        A.cross_phase(done=dnum - 1, dtwo=dnum)
        if dnum < len(A.Dlist) - 1: # before last loop
            fig, axs = A.mplot(dnum=dnum, type='val', show=0)
        else:
            A.mplot(dnum=dnum, type='val', show=1, fig=fig, axs=axs)
