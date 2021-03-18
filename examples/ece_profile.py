import sys, os
sys.path.insert(0, os.pardir)
from fluctana import *
from util import ece_channel_selection

# python3 ece_profile.py shotnumber [t1, t2]

Rrange = [1.9,2.25] 
npts = len(sys.argv) - 1

A = FluctAna()
for i in range(int(npts/2)):
    shot = int(sys.argv[i*2+1])
    trange = eval(sys.argv[i*2+2])

    # select channels to read
    clist = ece_channel_selection(shot, Rrange)

    A.add_data(dev='KSTAR', shot=shot, clist=clist, trange=trange, norm=0, res=0.0001)

A.list_data()

# ## temporal evolution in R,t space 
# ## low pass filter 
# A.filt(0,'FIR_pass',0,10000,0.01) # smaller b is sharper
# x = A.Dlist[0].time
# y = A.Dlist[0].rpos
# fig = plt.figure(figsize=(12,12))
# ## multiple plots
# for i in range(len(A.Dlist[0].clist)):
#     v = A.Dlist[0].data[i,:]
#     v = 2*v/(np.max(v) - np.min(v))
#     plt.plot(x, y[i]*100 + v, 'k')
# ## imagesc plot
# # plt.imshow(A.Dlist[0].data, extent=(x[0], x[-1], y[-1], y[0]), aspect='auto')
# # plt.colorbar()
# plt.xlabel('Time [s]')
# plt.ylabel('Radial position [m]')
# plt.title('ECE Te [keV]')
# plt.show()

## compare multiple profiles
fig = plt.figure(figsize=(7,7))
for i in range(int(npts/2)):
    shot = int(sys.argv[i*2+1])
    trange = eval(sys.argv[i*2+2])
    tag = 'shot {:d}, t=[{:g},{:g}]'.format(shot, trange[0],trange[1])

    x = A.Dlist[i].rpos
    y = np.mean(A.Dlist[i].data,axis=1)/1000.0
    #yerr = A.Dlist[i].err/1000.0
    
    #plt.errorbar(x, y, yerr=yerr, fmt='-o')
    plt.plot(x, y, marker='o', label=tag)

plt.title('ECE Te [keV]')
plt.xlabel('R [m]')
plt.legend()
plt.show()
