import sys, os
sys.path.insert(0, os.pardir)
from fluctana import *
import argparse


# HOW TO RUN
# python3 check_skew_kurt.py -shot 10186 -trange 15.9 16 -clist ECEI_L1303-1305

parser = argparse.ArgumentParser(description="Skewness and Kurtosis analysis")
parser.add_argument("-shot", type=int, default=10186, help="Shot number")
parser.add_argument("-trange", nargs=2, type=float, default=[15.9, 16], help="Time range [ms]")
parser.add_argument("-clist", nargs='+', type=str, default=['ECEI_L1303-1305'], help="Channel list")
a = parser.parse_args()



# call fluctana
A = FluctAna()

# add data
A.add_data(dev='KSTAR', shot=a.shot, clist=a.clist, trange=a.trange, norm=1)

# list data
A.list_data()

# calculate skewness and kurtosis (flatness) using data set dnum and plot over SK plane; dnum = data set, cnl = channel number list
A.skplane(dnum=0, detrend=1, verbose=1)

# skewness only
# A.skewness(dnum=0, detrend=1, verbose=1)
# A.mplot(dnum=0, type='val')
# A.cplot(dnum=0, snum=0, vlimits=[-0.5,0.5])

# kurtosis only
# A.kurtosis(dnum=0, detrend=1, verbose=1)
# A.mplot(dnum=0, type='val')
# A.cplot(dnum=0, snum=0, vlimits=[-0.5,0.5])