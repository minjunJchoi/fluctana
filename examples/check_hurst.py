import sys, os
sys.path.insert(0, os.pardir)
from fluctana import *
import argparse 


## HOW TO RUN
## python3 check_hurst.py -shot 37389 -trange 7.6 7.95 -clist ECE31 -fitrange 10 10000
## python3 check_hurst.py -shot 37389 -trange 7.6 7.95 -clist ECE31 -fitrange 10 10000 -qrange 0.5 5.0

parser = argparse.ArgumentParser(description="Hurst exponents")
parser.add_argument("-shot", type=int, default=37389, help="shot number")
parser.add_argument("-clist", nargs='+', type=str, default=['ECE31'], help="Channel list")
parser.add_argument("-trange", nargs=2, type=float, default=[7.6, 7.95], help="time range [sec]")
parser.add_argument("-qrange", nargs=2, type=float, default=None, help="order")
parser.add_argument("-fitrange", nargs=2, type=float, default=[10, 10000], help="fit time range [us]")
parser.add_argument("-bins", type=int, default=20, help="number of bins")
a = parser.parse_args()


# call fluctana
A = FluctAna()

# add data
A.add_data(dev='KSTAR', shot=a.shot, clist=a.clist, trange=a.trange, norm=1)

# list data
A.list_data()

# calculate H using data set dnum. results are saved in A.Dlist[dtwo].val; fitrange = fitting range of time lag in us.
# qrange = order range for structure function method
# if qrange is None, R/S method is performed
A.hurst(dnum=0, bins=20, detrend=0, fitrange=a.fitrange, qrange=a.qrange)

# plot the results; dnum = data set number, cnl = channel number list to plot
A.mplot(dnum=0, type='val')
