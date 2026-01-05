import sys, os
sys.path.insert(0, os.pardir)
from fluctana import *
import argparse


## HOW TO RUN
## python3 check_nonlin.py -shot 10186 -trange 15.85 16 -dt 5e-6 -ref_clist ECEI_L1403 -cmp_clist ECEI_L1403

parser = argparse.ArgumentParser(description="Nonlinear energy transfer")
parser.add_argument("-shot", type=int, default=10186, help="Shot number")
parser.add_argument("-trange", nargs=2, type=float, default=[15.9, 16], help="Time range [ms]")
parser.add_argument("-dt", type=float, default=0, help="Time difference [s]")
parser.add_argument("-ref_clist", nargs='+', type=str, default=['ECEI_L1303'], help="Reference channel list")
parser.add_argument("-cmp_clist", nargs='+', type=str, default=['ECEI_L1303'], help="Comparison channel list")
a = parser.parse_args()


# parameters
wit = 1
js = 0
if a.dt == 0: a.dt = 0.00001 # when using different pairs, take arbitrary time; dt = d/vd

print(a.dt)

# call fluctana
A = FluctAna()

# add data
A.add_data(dev='KSTAR', shot=a.shot, clist=a.ref_clist, trange=a.trange, norm=1)
A.add_data(dev='KSTAR', shot=a.shot, clist=a.cmp_clist, trange=[a.trange[0] + a.dt, a.trange[1] + a.dt], norm=1)

# list data
A.list_data()

# do fft; full = 1 (-fN ~ fN)
A.fftbins(nfft=256,window='kaiser',overlap=0.8,detrend=0,full=1)

# calculate and plot
A.nonlin_evolution(done=0,dtwo=1,delta=a.dt,wit=wit,js=js,xlimits=[0,100])

