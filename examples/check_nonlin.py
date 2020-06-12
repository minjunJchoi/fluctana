import sys, os
sys.path.insert(0, os.pardir)
from fluctana import *

# HOW TO RUN
# python3 check_nonlin.py 10186 [15.85,16,5e-6] ECEI_L1403 ECEI_L1403

shot = int(sys.argv[1]) 
trange = eval(sys.argv[2]) 
dt = trange[2]
trange1 = np.array(trange[0:2])
trange2 = np.array(trange[0:2]) + dt
clist1 = sys.argv[3].split(',') 
clist2 = sys.argv[4].split(',') 

# parameters
wit = 1
js = 0
if dt == 0: dt = 0.00001 # when using different pairs, take arbitrary time; dt = d/vd

# call fluctana
A = FluctAna()

# add data
A.add_data(dev='KSTAR', shot=shot, clist=clist1, trange=trange1, norm=1)
A.add_data(dev='KSTAR', shot=shot, clist=clist2, trange=trange2, norm=1)

# list data
A.list_data()

# do fft; full = 1 (-fN ~ fN)
A.fftbins(nfft=256,window='kaiser',overlap=0.8,detrend=0,full=1)

# calculate and plot
A.nonlin_evolution(done=0,dtwo=1,delta=dt,wit=wit,js=js,xlimits=[0,100])

