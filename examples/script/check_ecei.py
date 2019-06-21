import sys, os
sys.path.append(os.pardir)
from fluctana import *
import math


CM = plt.cm.get_cmap('hot')

shot = int(sys.argv[1]) # 19158
trange = eval(sys.argv[2]) # [5,6]
clist = sys.argv[3].split(',') # ECEI_G0101-2408
if len(sys.argv) == 5:
    fflag = sys.argv[4] # spec or time
else:
    fflag = 'time'
if fflag == 'spec':
    norm = 1
else:
    norm = 0

# Load modules
A = FluctAna()

# select channels
ecei = KstarEcei(shot=shot, clist=clist)
## check channel positions 
# ecei.show_ch_position()

# Add data
A.add_data(ecei, trange=trange, norm=norm) # shot and time range

# list data
A.list_data()

# plot data
if fflag == 'time':
    A.mplot(dnum=0, cnl=range(len(A.Dlist[0].clist)), type='time')
elif fflag == 'spec':
    A.spec(dnum=0,cnl=range(len(A.Dlist[0].clist)), nfft=2048)

# #A.spec(dnum=0,cnum=[0,1,2])

# # plot dimension
# nch = len(A.Dlist[0].data)
# row = 8.0
# col = math.ceil(nch/row)

# # iteration for plots
# fig = plt.figure(figsize=(12,12))
# for i in range(1,nch+1):
#     # define axes
#     if i == 1:
#         ax1 = plt.subplot(row,col,i)
#         plt.title(clist)
#         if fflag == 'spec':
#             axprops = dict(sharex = ax1, sharey = ax1)
#         else:
#             axprops = dict(sharex = ax1)
#     else:
#         plt.subplot(row,col,i, **axprops)
    
#     # check node name
#     print(i, A.Dlist[0].clist[i-1], A.Dlist[0].rpos[i-1], A.Dlist[0].zpos[i-1])
#     x = A.Dlist[0].time
#     y = A.Dlist[0].data[i-1,:]

#     # do plot
#     if fflag == 'spec':
#         fs = round(1/(x[1] - x[0])/1000.0)*1000
#         y = y - np.mean(y)
#         Pxx, freqs, bins, im = plt.specgram(y, NFFT=2048, Fs=fs, noverlap=2048*0.9, xextent=[x[0],x[-1]], cmap=CM)
#         plt.ylim([0, 250000])
#         maxP = math.log(np.amax(Pxx+1e-14),10)*10
#         minP = math.log(np.amin(Pxx+1e-14),10)*10
#         dP = maxP - minP
#         plt.clim([(minP+dP*0.15), maxP])
#         #plt.clim([-120, -20])
#         chpos = '{:.1f}:{:.1f}'.format(A.Dlist[0].rpos[i-1]*100, A.Dlist[0].zpos[i-1]*100)
#         plt.ylabel(chpos)
#     else:
#         plt.plot(x, y, color='k')
#         chpos = '{:.1f}:{:.1f}'.format(A.Dlist[0].rpos[i-1]*100, A.Dlist[0].zpos[i-1]*100)
#         plt.ylabel(chpos)

# plt.xlim([x[0], x[-1]])
# plt.xlabel('Time [s]')
# plt.show()

