import sys, os
sys.path.insert(0, os.pardir)
from fluctana import *
import argparse


## HOW TO RUN
## python3 check_spectrogram.py -shot 10186 -trange 15 16 -ref_clist ECEI_L1303 -cmp_clist ECEI_L1403

parser = argparse.ArgumentParser(description="Cross-spectrogram analysis")
parser.add_argument("-shot", type=int, default=10186, help="Shot number")
parser.add_argument("-trange", nargs=2, type=float, default=[15, 16], help="Time range [ms]")
parser.add_argument("-ref_clist", nargs='+', type=str, default=['ECEI_L1303'], help="Reference channel list")
parser.add_argument("-cmp_clist", nargs='+', type=str, default=['ECEI_L1403'], help="Comparison channel list")
a = parser.parse_args()

# call fluctana
A = FluctAna()

# add data
A.add_data(dev='KSTAR', shot=a.shot, clist=a.ref_clist, trange=a.trange, norm=1)
A.add_data(dev='KSTAR', shot=a.shot, clist=a.cmp_clist, trange=a.trange, norm=1)

# list data
A.list_data()

# xspec parameters 
# frequency resolution ~ sampling frequency / nfft
nfft = 2048
# temporal resolution 
overlap = 0.8 # finest overlap = (nfft-1.0)/nfft
# for full frequency range, full=1 (e.g. MIR). Else full=0.
A.fftbins(nfft=nfft, window='kaiser', overlap=overlap, detrend=0, full=0)

# calculate the cross power spectrogram using data sets done and dtwo; thres = threshold for significance
A.xspec(done=0,dtwo=1,thres=0.1)

