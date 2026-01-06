import sys, os
sys.path.insert(0, os.pardir)
from fluctana import *
import argparse


# =============================== ECE, CES Time trace  =====================
parser = argparse.ArgumentParser(description="ECE CES time traces")
parser.add_argument("-shot", type=int, help="Shot number")
parser.add_argument("-trange", nargs=2, type=float, help="Time range [sec]")
parser.add_argument("-Rrange", nargs=2, type=float, default=[1.78, 2.2], help="R range [m]")
parser.add_argument("-scale", nargs=2, type=float, help="Scale factors for rescaling data; first for ECE, second for CES", default=[1.0, 1.0])
a = parser.parse_args()

def ece_channel_selection(shot, Rrange):
    if shot <= 14386: 
        clist_temp = ['ECE{:02d}'.format(i) for i in range(1,49)]
    else:        
        clist_temp = ['ECE{:02d}'.format(i) for i in range(1,77)]

    M = KstarMds(shot=shot, clist=clist_temp)
    M.rpos[np.isnan(M.rpos)] = 0.0 # zero for nan channels
    idx = np.where((M.rpos >= Rrange[0]) * (M.rpos <= Rrange[-1]))[0]
    clist = ['{:s}'.format(clist_temp[i]) for i in idx]

    return clist

def ces_channel_selection(shot, Rrange):

    # select channels to read
    clist_temp = ['CES_TI{:02d}'.format(i) for i in range(1,33)]
    M = KstarMds(shot=shot, clist=clist_temp)
    M.rpos[np.isnan(M.rpos)] = 0.0 # zero for nan channels
    idx = np.where((M.rpos >= Rrange[0]) * (M.rpos <= Rrange[-1]))[0]
    clist = ['{:s}'.format(clist_temp[i]) for i in idx]

    return clist

## Get data
A = FluctAna()

## Load ECE data
ece_clist = ece_channel_selection(a.shot, a.Rrange)
A.add_data(dev='KSTAR', shot=a.shot, clist=ece_clist, trange=a.trange, norm=0, res=1e-5)
A.filt(dnum=0, name='FFT_pass', fL=0*1000, fH=3*1000) # 0--3 kHz filtering
# base_filter = ft.FftFilter('FFT_pass', A.Dlist[0].fs, 0, 5)

## Load CES data
ces_clist = ces_channel_selection(a.shot, a.Rrange)
A.add_data(dev='KSTAR', shot=a.shot, clist=ces_clist, trange=a.trange, norm=0, res=0)
# A.filt(dnum=1, name='FFT_pass', fL=0*1000, fH=3*1000) # 0--3 kHz not necessary for CES data

# Plot data
fig, axs = plt.subplots(1, 2, figsize=(14, 7), sharex=True, sharey=True)
for d in range(len(A.Dlist)):
    if a.scale[d] == 0 or A.Dlist[d].data is None or len(A.Dlist[d].time) == 0:
        continue

    time = A.Dlist[d].time
    rpos = A.Dlist[d].rpos * 100 # [m] -> [cm]
    data = A.Dlist[d].data

    for i in range(data.shape[0]):       
        # if d == 0:
        #     # Ignore slow base line variation
        #     base = base_filter.apply(data[i,:])
        #     data[i,:] = data[i,:]/base - 1.0 
        #     sigma = np.std(data[i,150:-150])
        # else:
        #     data[i,:] = data[i,:] - np.nanmean(data[i,:])
        #     sigma = np.nanstd(data[i,:])
        
        data[i,:] = data[i,:] - np.nanmean(data[i,:])
        sigma = np.nanstd(data[i,:])
                
        data[i,:] = a.scale[d] * data[i,:] / sigma  # normalize by std dev

        axs[d].plot(time, data[i,:] + rpos[i], color='k')
        axs[d].text(time[len(data[i,:])//2], rpos[i], A.Dlist[d].clist[i], fontsize=8, color='red')
axs[0].set_xlim(a.trange)
axs[0].set_ylim([a.Rrange[0]*100 - 2, a.Rrange[1]*100 + 2])
axs[0].set_title(f'{a.shot} -- ECE')
axs[1].set_xlim(a.trange)
axs[1].set_ylim([a.Rrange[0]*100 - 2, a.Rrange[1]*100 + 2])
axs[1].set_title(f'{a.shot} -- CES')
axs[0].set_xlabel('Time [s]')
axs[1].set_xlabel('Time [s]')
axs[0].set_ylabel('R [cm]')
axs[1].set_ylabel('R [cm]')
plt.tight_layout()
plt.show()


