import sys, os
sys.path.insert(0, os.pardir)
from fluctana import *
import argparse

## HOW TO RUN
## python3 check_bicoherence.py -shot 10186 -trange 15.9 16 -ref_clist ECEI_L1303 -cmp_clist ECEI_L1303

parser = argparse.ArgumentParser(description="Bicoherence")
parser.add_argument("-shot", type=int, default=10186, help="Shot number")
parser.add_argument("-trange", nargs=2, type=float, default=[15.9, 16], help="Time range [ms]")
parser.add_argument("-ref_clist", nargs='+', type=str, default=['ECEI_L1303'], help="Reference channel list")
parser.add_argument("-cmp_clist", nargs='+', type=str, default=['ECEI_L1303'], help="Comparison channel list")
a = parser.parse_args()


# call fluctana
A = FluctAna()

# add data
A.add_data(dev='KSTAR', shot=a.shot, clist=a.ref_clist, trange=a.trange, norm=1)
A.add_data(dev='KSTAR', shot=a.shot, clist=a.cmp_clist, trange=a.trange, norm=1)

# list data
A.list_data()

# do fft for full range for bicoherence
A.fftbins(nfft=512,window='kaiser',overlap=0.8,detrend=0,full=1)

# do cwt; full = 0 (0 ~ fN) or 1 (-fN ~ fN)
# A.cwt(df=5000,full=1,tavg=1000)

# calculate the bicoherence using data in data sets done (ref) and dtwo (cmp) and show results
A.bicoherence(done=0, dtwo=1)