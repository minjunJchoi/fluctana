import sys, os
sys.path.insert(0, os.pardir)
from fluctana import *
import argparse

## HOW TO RUN
## python3 check_ch.py -shot 10186 -trange 15.9 16 -clist ECEI_L1303-1306 -q 1 -d 6 -bins 1 ## high frequency components are stochastic
## python3 check_ch.py -shot 10186 -trange 15.9 16 -clist ECEI_L1303-1306 -q 2 -d 6 -bins 1 ## low frequency components are chaotic 

parser = argparse.ArgumentParser(description="Complexity Entropy analysis")
parser.add_argument("-shot", type=int, default=10186, help="Shot number")
parser.add_argument("-trange", nargs=2, type=float, default=[15.9, 16], help="Time range [ms]")
parser.add_argument("-clist", nargs='+', type=str, default=['ECEI_L1303-1306'], help="Channel list")
parser.add_argument("-q", type=int, default=2, help="Order q for downsampling")
parser.add_argument("-d", type=int, default=6, help="Dimension d for BP probability calculation")
parser.add_argument("-bins", type=int, default=1, help="Number of bins for CH calculation")
a = parser.parse_args()

# call fluctana
A = FluctAna()

# add data
A.add_data(dev='KSTAR', shot=a.shot, clist=a.clist, trange=a.trange, norm=1)

# list data
A.list_data()

# # check auto-correlation time
# A.fftbins(nfft=256,window='hann',overlap=0.5,detrend=0,full=1)
# A.corr_coef(done=0, dtwo=0)

# down sampling with the order q if necessary 
if a.q > 1:
    A.downsample(dnum=0, q=a.q, verbose=1)

# calculate CH values using data set dnum and plot over CH plane; dnum = data set, cnl = channel number list, d = dimension
A.chplane(dnum=0, d=a.d, bins=a.bins, verbose=1)

# # complexity only
# A.js_complexity(dnum=0, d=a.d, bins=a.bins)
# A.mplot(dnum=0, type='val')
# A.cplot(dnum=0, snum=0, vlimits=[0,1.0])

# # entropy only
# A.ns_entropy(dnum=0, d=a.d, bins=a.bins)
# A.mplot(dnum=0, type='val')
# A.cplot(dnum=0, snum=0, vlimits=[0,1.0])