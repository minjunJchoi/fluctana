import sys, os
sys.path.append(os.pardir)
from fluctana import *

# python3 ts_profile.py 22531 [3.5,3.6] 22531 [5,5.1] NE

NCORE = 14
TORN = sys.argv[-1] # TE or NE
Rrange = [1.3.,2.3] 
npts = len(sys.argv) - 1

A = FluctAna()
for i in range(int(npts/2)):
    shot = int(sys.argv[i*2+1])
    if shot < 21779:
        NEDGE = 13
    else:
        NEDGE = 17
    trange = eval(sys.argv[i*2+2])

    # select channels to read
    clist_temp = ['TS_CORE{:d}:CORE{:d}_{:s}'.format(i,i,TORN) for i in range(1,NCORE+1)]
    clist_temp += ['TS_EDGE{:d}:EDGE{:d}_{:s}'.format(i,i,TORN) for i in range(1,NEDGE+1)]
    M = KstarMds(shot=shot, clist=clist_temp)
    M.rpos[np.isnan(M.rpos)] = 0.0 # zero for nan channels
    idx = np.where((M.rpos >= Rrange[0]) * (M.rpos <= Rrange[-1]))[0]
    M.clist = ['{:s}'.format(clist_temp[i]) for i in idx]
    M.rpos = M.rpos[idx]
    print(M.rpos)
    print(M.clist)

    A.add_data(M, trange=trange, norm=0)

A.list_data()

## compare multiple profiles
fig = plt.figure(figsize=(7,7))
for i in range(int(npts/2)):
    shot = int(sys.argv[i*2+1])
    trange = eval(sys.argv[i*2+2])
    tag = 'shot {:d}, t=[{:g},{:g}]'.format(shot, trange[0],trange[1])
    
    x = A.Dlist[i].rpos
    if TORN == 'NE':
        y = np.mean(A.Dlist[i].data,axis=1)/1e19
    elif TORN == 'TE':
        y = np.mean(A.Dlist[i].data,axis=1)/1000.0
    #yerr = A.Dlist[i].err/1000.0    

    plt.plot(x, y, marker='o', label=tag)
    #plt.errorbar(x, y, yerr=yerr, fmt='-o')

if TORN == 'NE':
    plt.title('TS ne [1e19 m^-3]')
elif TORN == 'TE':
    plt.title('TS Te [keV]')
plt.xlabel('R [m]')
plt.legend()
plt.show()

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
# plt.title('TS')
# plt.show()
