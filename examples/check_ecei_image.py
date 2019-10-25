import sys, os
sys.path.append(os.pardir)
from fluctana import *

## HOW TO RUN
# ./python3
# > import ecei_image as ei
# For data before 2018 L,H,G
# > ei.play(22568,[8,9],L,[20,25],[-0.1,0.1])
# For data since 2018 GT,GR,HT
# > ei.play(22568,[8,9],GT,[20,25],[-0.1,0.1])

def play(shot, trange, dname, flimits=[0,20], vlimits=[-0.1,0.1]):
    # call fluctana
    A = FluctAna()
    # add data
    clist = ['ECEI_{:s}0101-2408'.format(dname)]
    A.add_data(KstarEcei(shot, clist), trange, norm=1)

    # band pass filter  # OK
    A.filt(0,'FIR_pass',flimits[0]*1000.0,0,0.01) # smaller b is sharper
    A.filt(0,'FIR_pass',0,flimits[1]*1000.0,0.01) # smaller b is sharper

    # A.iplot(dnum=0,snum=0,vlimits=[-0.1,0.1],istep=0.002,imethod='cubic',cutoff=0.03,pmethod='scatter')
    A.iplot(dnum=0,snum=0,vlimits=vlimits)

if __name__ == "__main__":
    play(shot=22289,trange=[8.2,8.3],dname='GT',flimits=[13,17],vlimits=[-0.05,0.05])