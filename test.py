from fluctana import *

# Load modules
C = FluctAna()

# test signal
fs = 3000.0
t = np.arange(0,1,1/fs)

A = KstarMir(12345, ['ECEI_GT1201'])
data = np.zeros((1, len(t)), dtype=np.complex_)
x1 = signal.chirp(t,300,t[-1],1300,method='quadratic') + np.random.randn(len(t))/100
data[0][:] = x1 - np.mean(x1)
A.data = data
A.time = t
A.rpos = np.zeros(1)
A.zpos = np.zeros(1)
C.Dlist.append(A)

B = KstarMir(12345, ['ECEI_GR1201'])
data = np.zeros((1, len(t)), dtype=np.complex_)
x2 = np.exp(2.0j*np.pi*100*np.cos(2*np.pi*2*t)) + np.random.randn(len(t))/100
data[0][:] = x2 - np.mean(x2)
B.data = data
B.time = t
B.rpos = np.zeros(1)
B.zpos = np.zeros(1)
C.Dlist.append(B)

# fft
nfft = 256
overlap = (nfft-1.0)/nfft
detrend = 0
full = 1
C.fftbins(nfft, 'kaiser', overlap, detrend, full=full)

# calculate
thres = 0.0
C.xspec(thres=thres)

# print(C.Dlist[1].ax)

# plt.show()
