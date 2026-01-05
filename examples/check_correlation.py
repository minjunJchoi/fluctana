import sys, os
sys.path.insert(0, os.pardir)
from fluctana import *
import argparse

## HOW TO RUN
## python3 check_correlation.py -slist 10186 -tlist 15.1 16.1 -twin 0.1 -ref_clist ECEI_L1303-1403 -cmp_clist ECEI_L1403-1503 -flimits 10 50 --cc

parser = argparse.ArgumentParser(description="Correlation or correlation coefficient")
parser.add_argument("-slist", nargs='+', type=int, default=[10186], help="Shot number list")
parser.add_argument("-tlist", nargs='+', type=float, default=[15.95, 17], help="Time points [sec] list")
parser.add_argument("-twin", type=float, default=0.1, help="Time window [sec]")
parser.add_argument("-ref_clist", nargs='+', type=str, default=['ECEI_L1303-1403'], help="Reference channel list")
parser.add_argument("-cmp_clist", nargs='+', type=str, default=['ECEI_L1403-1503'], help="Comparison channel list")
parser.add_argument("-flimits", nargs=2, type=float, default=[10, 50], help="Frequency range")
parser.add_argument("--cc", action="store_true", help="Correlation coefficient")
a = parser.parse_args()


# Read data 
A = FluctAna()
dnum = 0
for shot in a.slist:    
    for time_point in a.tlist:
        trange = [time_point - a.twin/2, time_point + a.twin/2]
        A.add_data(dev='KSTAR', shot=shot, clist=a.ref_clist, trange=trange, norm=1)
        A.add_data(dev='KSTAR', shot=shot, clist=a.cmp_clist, trange=trange, norm=1)

        # band pass filter 
        A.filt(dnum=len(A.Dlist)-2,name='FFT_pass',fL=a.flimits[0]*1000,fH=a.flimits[1]*1000)
        A.filt(dnum=len(A.Dlist)-1,name='FFT_pass',fL=a.flimits[0]*1000,fH=a.flimits[1]*1000)


# # do fft; full = 0 (0 ~ fN) or 1 (-fN ~ fN)
A.fftbins(nfft=512,window='hann',overlap=0.5,detrend=0,full=1)

# calculate correlation using data sets done and dtwo. results are saved in A.Dlist[dtwo].val
for dnum in range(1, len(A.Dlist), 2):   
    if a.cc:
        A.corr_coef(done=dnum - 1, dtwo=dnum)
    else:
        A.correlation(done=dnum - 1, dtwo=dnum)
    if dnum < len(A.Dlist) - 1: # before last loop
        fig, axs = A.mplot(dnum=dnum, type='val', show=0)
    else:
        A.mplot(dnum=dnum, type='val', show=1, fig=fig, axs=axs)
