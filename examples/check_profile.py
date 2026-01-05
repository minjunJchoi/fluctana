import sys, os
sys.path.insert(0, os.pardir)
from fluctana import *
import argparse

## HOW TO RUN
## python3 check_profile.py -slist 26040 -tlist 3.9 4.5 -twin 0.05 -Rrange 1.7 2.225 --ti --vt --te --ts --ne

parser = argparse.ArgumentParser(description="CES profile")
parser.add_argument("-slist", nargs='+', type=int, default=[22289], help="Shot number list")
parser.add_argument("-tlist", nargs='+', type=float, default=[3.5, 4.5], help="Time points [sec] list")
parser.add_argument("-twin", type=float, default=0.05, help="Time window [sec]")
parser.add_argument("-Rrange", nargs=2, type=float, default=[1.7, 2.225], help="Radial range [m]")
parser.add_argument("--ti", action="store_true", help="Plot Ti from CES")
parser.add_argument("--vt", action="store_true", help="Plot Vt from CES")
parser.add_argument("--te", action="store_true", help="Plot Te from ECE")
parser.add_argument("--ts", action="store_true", help="Plot Te from Thomson scattering")
parser.add_argument("--ne", action="store_true", help="Plot Ne from Thomson scattering")
a = parser.parse_args()

def ces_channel_selection(shot):
    clist_temp = ['CES_{:s}{:02d}'.format('TI', i) for i in range(1,33)]
    M = KstarMds(shot=shot, clist=clist_temp)
    M.rpos[np.isnan(M.rpos)] = 0.0 # zero for nan channels
    idx = np.where((M.rpos >= a.Rrange[0]) * (M.rpos <= a.Rrange[-1]))[0]
    clist = ['{:s}'.format(clist_temp[i]) for i in idx]

    return clist

def ece_channel_selection(shot):
    if shot <= 14386: 
        clist_temp = ['ECE{:02d}'.format(i) for i in range(1,49)]
    else:        
        clist_temp = ['ECE{:02d}'.format(i) for i in range(1,77)]

    M = KstarMds(shot=shot, clist=clist_temp)
    M.rpos[np.isnan(M.rpos)] = 0.0 # zero for nan channels
    idx = np.where((M.rpos >= a.Rrange[0]) * (M.rpos <= a.Rrange[-1]))[0]
    clist = ['{:s}'.format(clist_temp[i]) for i in idx]

    return clist

def ts_channel_selection(shot):
    NCORE = 14
    if shot < 21779:
        NEDGE = 13
    else:
        NEDGE = 17
    clist_temp = ['TS_CORE{:d}:CORE{:d}_{:s}'.format(i,i,'TE') for i in range(1,NCORE+1)]
    clist_temp += ['TS_EDGE{:d}:EDGE{:d}_{:s}'.format(i,i,'TE') for i in range(1,NEDGE+1)]
    M = KstarMds(shot=shot, clist=clist_temp)
    idx = np.where((M.rpos >= a.Rrange[0]) * (M.rpos <= a.Rrange[-1]))[0]
    clist = ['{:s}'.format(clist_temp[i]) for i in idx]

    return clist


# Make axes 
num_axes = 0
if a.ti: num_axes += 1
if a.vt: num_axes += 1
if a.te: num_axes += 1
if a.ts: num_axes += 1
if a.ne: num_axes += 1
fig, axs = plt.subplots(1, num_axes, figsize=(4*num_axes, 5), sharex=True)
if num_axes == 1:
    axs = [axs]


# Read data and plot
for shot in a.slist:
    if a.ti or a.vt: 
        ti_clist = ces_channel_selection(shot)
        vt_clist = [s.replace('TI','VT') for s in ti_clist]
    if a.te:
        te_clist = ece_channel_selection(shot)
    if a.ts or a.ne:
        ts_clist = ts_channel_selection(shot)
        ne_clist = [s.replace('TE','NE') for s in ts_clist]
        
    for time_point in a.tlist:
        trange = [time_point - a.twin/2, time_point + a.twin/2]
        A = FluctAna()
        if a.ti:
            A.add_data(dev='KSTAR', shot=shot, clist=ti_clist, trange=trange, norm=0)
        if a.vt:
            A.add_data(dev='KSTAR', shot=shot, clist=vt_clist, trange=trange, norm=0)
        if a.te:
            A.add_data(dev='KSTAR', shot=shot, clist=te_clist, trange=trange, norm=0, res=0.0001)
        if a.ts:
            A.add_data(dev='KSTAR', shot=shot, clist=ts_clist, trange=trange, norm=0)
        if a.ne:
            A.add_data(dev='KSTAR', shot=shot, clist=ne_clist, trange=trange, norm=0)

        for dnum, D in enumerate(A.Dlist):
            tag = f'shot {shot}, t={time_point} sec'
            axs[dnum].errorbar(A.Dlist[dnum].rpos, np.mean(A.Dlist[dnum].data,axis=1), yerr=A.Dlist[dnum].err, fmt='-o', label=tag)
            if 'TI' in D.clist[0]:
                axs[dnum].set_ylabel('CES Ti [eV]')
            if 'VT' in D.clist[0]:
                axs[dnum].set_ylabel('CES Vtor [km/s]')
            if 'ECE' in D.clist[0]:
                axs[dnum].set_ylabel('ECE Te [eV]')
            if 'TE' in D.clist[0]:
                axs[dnum].set_ylabel('TS Te [eV]')
            if 'NE' in D.clist[0]:
                axs[dnum].set_ylabel('TS Ne [1e19 m^-3]')
            axs[dnum].set_xlabel('R [m]')
            axs[dnum].legend()
plt.tight_layout()
plt.show()


