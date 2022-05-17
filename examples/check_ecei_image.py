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

    # ## FIR band pass/block filter  
    # A.filt(0,'FIR_pass',flimits[0]*1000.0,0,0.01) # smaller b is sharper
    # A.filt(0,'FIR_pass',0,flimits[1]*1000.0,0.01) # smaller b is sharper

    ## (brick) FFT band pass/block filter  
    A.filt(dnum=0,name='FFT_pass',fL=flimits[0]*1000,fH=flimits[1]*1000)

    # ## Threshold FFT filter (fL--fH : white noise range)
    # A.filt(dnum=0,name='Threshold_FFT',fL=flimits[0]*1000,fH=flimits[1]*1000,b=3,nbins=100)

    # ## SVD filter
    # A.svd_filt(dnum=0, cutoff=0.9)

    # A.iplot(dnum=0,snum=0,vlimits=[-0.1,0.1],istep=0.002,imethod='cubic',cutoff=0.03,pmethod='scatter')
    A.iplot(dnum=0,snum=0,type='time',vlimits=vlimits)

if __name__ == "__main__":
    play(shot=22289,trange=[2.716,2.718],dname='GT',flimits=[5,9],vlimits=[-0.05,0.05])