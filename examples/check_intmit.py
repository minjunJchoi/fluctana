import sys, os
sys.path.insert(0, os.pardir)
from fluctana import *
import argparse


## HOW TO RUN
## python3 check_intmit.py -shot 10186 -trange 15.9 16 -clist ECEI_L1303

parser = argparse.ArgumentParser(description="Hurst exponents")
parser.add_argument("-shot", type=int, default=10186, help="shot number")
parser.add_argument("-clist", nargs='+', type=str, default=['ECEI_L1303'], help="Channel list")
parser.add_argument("-trange", nargs=2, type=float, default=[15.9, 16], help="time range [sec]")
a = parser.parse_args()


# call fluctana
A = FluctAna()

# add data
A.add_data(dev='KSTAR', shot=a.shot, clist=a.clist, trange=a.trange, norm=0)

# list data
A.list_data()

# calculate using data set dnum. results are saved in A.Dlist[dtwo].val; fitlims = fitting range of time lag in us.
A.intermittency(dnum=0, bins=50, overlap=0.2, qstep=0.2, fitrange=[10.0, 100.0], verbose=1)


