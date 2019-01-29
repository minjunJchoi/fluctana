from fluctana import *

# Load modules
C = FluctAna()

# ## test xspec with test signals
# fs = 3000.0
# t = np.arange(0,1,1/fs)
#
# A = KstarMir(12345, ['ECEI_GT1201'])
# data = np.zeros((1, len(t)), dtype=np.complex_)
# # x1 = signal.chirp(t,300,t[-1],1300,method='quadratic')# + np.random.randn(len(t))/100
# x1 = np.cos(2*np.pi*2*t + np.pi/8)# + np.random.randn(len(t))/100
# data[0][:] = x1 - np.mean(x1)
# A.data = data
# A.time = t
# A.rpos = np.zeros(1)
# A.zpos = np.zeros(1)
# C.Dlist.append(A)
#
# B = KstarMir(12345, ['ECEI_GR1201'])
# data = np.zeros((1, len(t)), dtype=np.complex_)
# # x2 = np.exp(2.0j*np.pi*100*np.cos(2*np.pi*2*t))# + np.random.randn(len(t))/100
# x2 = np.cos(2*np.pi*2*t)# + np.random.randn(len(t))/100
# data[0][:] = x2 - np.mean(x2)
# B.data = data
# B.time = t
# B.rpos = np.zeros(1)
# B.zpos = np.zeros(1)
# C.Dlist.append(B)
#
# # xspec
# nfft = 256
# overlap = (nfft-1.0)/nfft
# detrend = 0
# full = 1
# C.fftbins(nfft, 'kaiser', overlap, detrend, full=full)
# thres = 0.0
# C.xspec(thres=thres)
#
# ## corr
# nfft = 2999
# overlap = 0.5
# detrend = 0
# full = 1
# C.fftbins(nfft, 'hann', overlap, detrend, full=full)
# C.corr_coef()
# # C.mplot(dnum=1, cnl=[0], type='time')
# C.mplot(dnum=1, cnl=[0], type='val')


# ## xspec test with ECEI data
# shot = 10186
# trange = [15.7,15.85]
# norm = 1
# # ref data
# clist = ['ECEI_L1303']
# C.add_data(KstarEcei(shot=shot, clist=clist), trange=trange, norm=norm) # shot and time range
# # cmp data
# clist = ['ECEI_L1403']
# C.add_data(KstarEcei(shot=shot, clist=clist), trange=trange, norm=norm) # shot and time range
# # fft
# C.fftbins(nfft=745, window='hann', overlap=0.5, detrend=0, full=0)
# # calculate with default options (single channel)
# C.coherence()
# # plot
# C.mplot(dnum=1, cnl=[0], type='val', ylimits=[0,1])
# # for xspec
# C.fftbins(nfft=256, window='kaiser', overlap=0.5, detrend=0, full=1)
# # xspec
# C.xspec(thres=0.5)


## Local SKw
shot = 10186
trange = [15.7,15.85]
norm = 1
# ref data
clist = ['ECEI_L1003', 'ECEI_L1103', 'ECEI_L1203', 'ECEI_L1303', 'ECEI_L1403']
C.add_data(KstarEcei(shot=shot, clist=clist), trange=trange, norm=norm) # shot and time range
# cmp data
clist = ['ECEI_L1103', 'ECEI_L1203', 'ECEI_L1303', 'ECEI_L1403', 'ECEI_L1503']
C.add_data(KstarEcei(shot=shot, clist=clist), trange=trange, norm=norm) # shot and time range
# fft
C.fftbins(nfft=745, window='hann', overlap=0.5, detrend=0, full=1)
# calculate with default options (single channel)
C.skw()


# print(C.Dlist[0].tt, C.Dlist[0].toff, C.Dlist[0].bt, C.Dlist[0].fs, C.Dlist[0].mode, C.Dlist[0].lo, C.Dlist[0].sz, C.Dlist[0].sf)
