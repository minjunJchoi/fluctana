import sys, os
sys.path.insert(0, os.pardir)
from fluctana import *

A = FluctAna()

shot = 20896
trange = [3,4]

A.add_data(dev='KSTAR', shot=shot, clist=['MC1T06'], trange=trange, norm=0)

# calculate spectrum before filter and plot
A.fftbins(nfft=2048,window='hann',overlap=0.5,detrend=1)
A.cross_power(done=0,dtwo=0)

plt.subplot(211)
plt.plot( A.Dlist[0].time, A.Dlist[0].data[0,:])
plt.subplot(212)
plt.plot( A.Dlist[0].ax/1000.0, A.Dlist[0].val[0,:].real )


# A.filt includes FIR filter as follows

# low pass filter; f < 15000 Hz
# A.filt(dnum=0, name='FIR_pass',fL=0,fH=150000,b=0.01,verbose=0) 

# high pass filter; 10000 < f
# A.filt(dnum=0, name='FIR_pass',fL=100000,fH=0,b=0.01,verbose=0)

# band pass filter; 50000 < f < 50000
A.filt(dnum=0, name='FIR_pass',fL=50000,fH=0,b=0.01,verbose=0) 
A.filt(dnum=0, name='FIR_pass',fL=0,fH=500000,b=0.01,verbose=0) 

# band block filter; f < 50000 and f > 150000
# A.filt(dnum=0, name='FIR_block',fL=50000,fH=150000,0.01,verbose=0) # smaller b is sharper

# calculate spectrum after filter and plot
A.fftbins(nfft=2048,window='hann',overlap=0.5,detrend=1)
A.cross_power(done=0,dtwo=0)

plt.subplot(211)
plt.plot( A.Dlist[0].time, A.Dlist[0].data[0,:])
plt.subplot(212)
plt.plot( A.Dlist[0].ax/1000.0, A.Dlist[0].val[0,:].real )
plt.yscale('log')
plt.xlabel('Frequency [kHz]')

plt.show()

