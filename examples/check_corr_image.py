import sys, os
sys.path.insert(0, os.pardir)
from fluctana import *

## HOW TO RUN
# ipython3
# > import check_corr_image as ci
# > ci.play(shot=22289,trange=[2.716,2.718],ref='ECEI_GT1003',clist=['ECEI_GT0901-1208'],flimits=[5,9],vlimits=[-0.05,0.05])

def play(shot, trange, ref, clist, flimits=[0,20], vlimits=[-0.1,0.1]):
    # call fluctana
    A = FluctAna()
    
    # add data
    A.add_data(dev='KSTAR', shot=shot, clist=[ref], trange=trange, norm=1)
    A.add_data(dev='KSTAR', shot=shot, clist=clist, trange=trange, norm=1)

    # band pass filter  # OK
    A.filt(0,'FIR_pass',flimits[0]*1000.0,0,0.01) # smaller b is sharper
    A.filt(0,'FIR_pass',0,flimits[1]*1000.0,0.01) # smaller b is sharper

    # A.iplot(dnum=0,snum=0,vlimits=[-0.1,0.1],istep=0.002,imethod='cubic',cutoff=0.03,pmethod='scatter')
    A.fftbins(nfft=512,window='hann',overlap=0.5,detrend=0,full=1)
    A.corr_coef(done=0, dtwo=1)
    A.iplot(dnum=1,snum=0,type='val',vlimits=vlimits,pmethod='contour')

if __name__ == "__main__":
    play(shot=22289,trange=[2.716,2.718],ref='ECEI_GT1003',clist=['ECEI_GT0901-1208'],flimits=[5,9],vlimits=[-0.15,0.15])