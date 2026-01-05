import sys, os
sys.path.insert(0, os.pardir)
from fluctana import *
import argparse

## HOW TO RUN
## This calculates the local S(K,w). K along the direction of pairs
## python3 check_SKw.py -shot 19158 -trange 5.5 5.6 -ref_clist ECEI_G1003-1403 -cmp_clist ECEI_G1103-1503

parser = argparse.ArgumentParser(description="Local S(K,w)")
parser.add_argument("-shot", type=int, default=19158, help="Shot number")
parser.add_argument("-trange", nargs=2, type=float, default=[5.5, 5.6], help="Time range [ms]")
parser.add_argument("-ref_clist", nargs='+', type=str, default=['ECEI_G1003-1403'], help="Reference channel list")
parser.add_argument("-cmp_clist", nargs='+', type=str, default=['ECEI_G1103-1503'], help="Comparison channel list")
a = parser.parse_args()


# call fluctana
A = FluctAna()

# add data
A.add_data(dev='KSTAR', shot=a.shot, clist=a.ref_clist, trange=a.trange, norm=1)
A.add_data(dev='KSTAR', shot=a.shot, clist=a.cmp_clist, trange=a.trange, norm=1)

# list data
A.list_data()

# do fft 
A.fftbins(nfft=512,window='hann',overlap=0.5,detrend=0)

# do cwt; full = 0 (0 ~ fN) or 1 (-fN ~ fN)
# A.cwt(df=5000,full=0,tavg=1000)

# calculate local wavenumber-frequency spectra using channel pairs of done and dtwo and plot
A.skw(done=0, dtwo=1, kstep=0.01)
