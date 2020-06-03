import sys, os
sys.path.append(os.pardir)
from fluctana import *

# HOW TO RUN
# python3 check_ch.py 10186 [15.9,16] ECEI_L1301-1307 2 5 1

shot = int(sys.argv[1]) 
trange = eval(sys.argv[2])
clist = sys.argv[3].split(',')
q = int(sys.argv[4])
d = int(sys.argv[5])
bins = int(sys.argv[6])
nstd = 0

# call fluctana
A = FluctAna()

# add data
A.add_data(dev='KSTAR', shot=shot, clist=clist, trange=trange, norm=1)

# list data
A.list_data()

# # check auto-correlation time
# A.fftbins(nfft=256,window='hann',overlap=0.5,detrend=0,full=1)
# A.corr_coef(done=0, dtwo=0)

# down sampling with the order q if necessary 
if q > 1:
    A.downsample(dnum=0, q=q, verbose=1)

if nstd == 1:
    plt.plot(A.Dlist[0].data[0,:])
    for c in range(len(A.Dlist[0].clist)):
        dy = A.Dlist[0].data[c,:]
        A.Dlist[0].data[c,:] = (dy - np.mean(dy))**2 / np.mean((dy - np.mean(dy))**2)
    plt.plot(A.Dlist[0].data[0,:])
    plt.show()

# calculate CH values using data set dnum and plot over CH plane; dnum = data set, cnl = channel number list, d = dimension
A.chplane(dnum=0, cnl=range(len(A.Dlist[0].clist)), d=d, bins=bins, verbose=1)

# complexity only
# A.js_complexity(dnum=0, cnl=range(len(A.Dlist[0].clist)), d=6, bins=1)
# A.mplot(dnum=0, cnl=range(len(A.Dlist[0].clist)), type='val')
# A.cplot(dnum=0, snum=0, vlimits=[0,1.0])

# entropy only
# A.ns_entropy(dnum=0, cnl=range(len(A.Dlist[0].clist)), d=6, bins=1)
# A.mplot(dnum=0, cnl=range(len(A.Dlist[0].clist)), type='val')
# A.cplot(dnum=0, snum=0, vlimits=[0,1.0])