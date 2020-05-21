import sys, os
sys.path.append(os.pardir)
from fluctana import *

# HOW TO RUN
# ./python3 check_spec.py 10186 [15.9,16] ECEI_L1303-1305

shot = int(sys.argv[1]) 
trange = eval(sys.argv[2])
clist = sys.argv[3].split(',')

# call fluctana
A = FluctAna()

# add data
A.add_data(dev='KSTAR', shot=shot, clist=clist, trange=trange, norm=1)

# list data
A.list_data()

# plot spectrogram of data set dnum; cnl = channel number list to plot
A.spec(dnum=0, cnl=range(len(A.Dlist[0].clist)), nfft=2048, flimits=[0,150])
