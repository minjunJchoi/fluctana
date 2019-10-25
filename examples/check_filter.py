import sys, os
sys.path.append(os.pardir)
from fluctana import *

A = FluctAna()

shot = 20896
trange = [3,4]

A.add_data(KstarMds(shot=shot, clist=['MC1T06']), trange=trange, norm=0)

# calculate spectrum before filter and plot
A.fftbins(nfft=2048,window='hann',overlap=0.5,detrend=1)
A.cross_power(done=0,dtwo=0)

plt.subplot(211)
plt.plot( A.Dlist[0].time, A.Dlist[0].data[0,:])
plt.subplot(212)
plt.plot( A.Dlist[0].ax/1000.0, A.Dlist[0].val[0,:].real )


# Use A.filt to apply FIR filter for data set number = 0

# low pass filter; f < 15000 Hz
# A.filt(0,'FIR_pass',0,150000,b=0.01,verbose=0) 

# high pass filter; 10000 < f
# A.filt(0,'FIR_pass',100000,0,b=0.01,verbose=0)

# band pass filter; 5000 < f < 50000
# A.filt(0,'FIR_pass',50000,0,b=0.01,verbose=0) 
# A.filt(0,'FIR_pass',0,500000,b=0.01,verbose=0) 

# band block filter; f < 50000 and f > 150000
# A.filt(0,'FIR_block',50000,150000,0.01,verbose=0) # smaller b is sharper


## you can also try to use A.svd_filt 
# A.svd_filt(0, cutoff=0.9)


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

