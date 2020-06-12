import sys, os
sys.path.insert(0, os.pardir)
from fluctana import *

## HOW TO RUN
# ipython3
# > import check_ecei_image as ei
# > ei.play(shot=22289,trange=[2.716,2.718],dname='GT',flimits=[5,9],vlimits=[-0.05,0.05])

def play(shot, trange, dname, flimits=[0,20], vlimits=[-0.1,0.1]):
    # call fluctana
    A = FluctAna()
    # add data
    clist = ['ECEI_{:s}0101-2408'.format(dname)]
    A.add_data(dev='KSTAR', shot=shot, clist=clist, trange=trange, norm=1)

    # band pass filter  # OK
    A.filt(0,'FIR_pass',flimits[0]*1000.0,0,0.01) # smaller b is sharper
    A.filt(0,'FIR_pass',0,flimits[1]*1000.0,0.01) # smaller b is sharper

    # A.iplot(dnum=0,snum=0,vlimits=[-0.1,0.1],istep=0.002,imethod='cubic',cutoff=0.03,pmethod='scatter')
    A.iplot(dnum=0,snum=0,type='time',vlimits=vlimits)

if __name__ == "__main__":
    play(shot=22289,trange=[2.716,2.718],dname='GT',flimits=[5,9],vlimits=[-0.05,0.05])