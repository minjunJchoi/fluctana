from fluctana import *

# Load modules
C = FluctAna()



# ## xspec with test signals
# fs = 3000.0
# t = np.arange(0,1,1/fs)
#
# A = KstarEcei(10186, ['ECEI_L1201'])
# data = np.zeros((1, len(t)), dtype=np.complex_)
# x1 = signal.chirp(t,300,t[-1],1300,method='quadratic')# + np.random.randn(len(t))/100
# # x1 = np.cos(2*np.pi*2*t + np.pi/8)# + np.random.randn(len(t))/100
# data[0][:] = x1 - np.mean(x1)
# A.data = data
# A.time = t
# A.rpos = np.zeros(1)
# A.zpos = np.zeros(1)
# C.Dlist.append(A)
#
# B = KstarEcei(10186, ['ECEI_L1201'])
# data = np.zeros((1, len(t)), dtype=np.complex_)
# x2 = np.exp(2.0j*np.pi*100*np.cos(2*np.pi*2*t))# + np.random.randn(len(t))/100
# # x2 = np.cos(2*np.pi*2*t)# + np.random.randn(len(t))/100
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
# C.cross_power()
# # C.mplot(dnum=1, cnl=[0], type='time')
# C.mplot(dnum=1, cnl=[0], type='val')


# ## xspec with ECEI data
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
# C.cross_phase()
# # plot
# C.mplot(dnum=1, cnl=[0], type='val', ylimits=[-3,3])
# # for xspec
# C.fftbins(nfft=256, window='kaiser', overlap=0.5, detrend=0, full=1)
# # xspec
# C.xspec(thres=0.5)


# ## Local SKw
# shot = 10186
# trange = [15.7,15.85]
# norm = 1
# # ref data
# clist = ['ECEI_L1003', 'ECEI_L1103', 'ECEI_L1203', 'ECEI_L1303', 'ECEI_L1403']
# C.add_data(KstarEcei(shot=shot, clist=clist), trange=trange, norm=norm) # shot and time range
# # cmp data
# clist = ['ECEI_L1103', 'ECEI_L1203', 'ECEI_L1303', 'ECEI_L1403', 'ECEI_L1503']
# C.add_data(KstarEcei(shot=shot, clist=clist), trange=trange, norm=norm) # shot and time range
# # fft
# C.fftbins(nfft=745, window='hann', overlap=0.5, detrend=0, full=1)
# # calculate with default options (single channel)
# C.skw()
# # C.cross_phase()
# # C.mplot(dnum=1, cnl=[3], type='val')


# ## Bicoherence
# shot = 10186
# trange = [15.1,15.2]
# norm = 1
# # ref data
# clist = ['ECEI_L1303']
# C.add_data(KstarEcei(shot=shot, clist=clist), trange=trange, norm=norm) # shot and time range
# # cmp data
# clist = ['ECEI_L1403']
# C.add_data(KstarEcei(shot=shot, clist=clist), trange=trange, norm=norm) # shot and time range
# # fft
# # C.fftbins_bicoh_test(nfft=249, window='hann', overlap=0.5, detrend=0, full=1)
# C.fftbins(nfft=989, window='hann', overlap=0.5, detrend=0, full=1)
# # calculate
# # C.bicoherence(done=0,dtwo=0) # for test
# # C.bicoherence(done=0,dtwo=0,sum=1) # for test
# C.bicoherence(done=0,dtwo=1)
# C.bicoherence(done=0,dtwo=1,sum=1)


# ## Wavelet
# A = KstarEcei(10186, ['ECEI_L1201'])
# fs = 500000.0
# t = np.arange(0,0.0007,1/fs)
# data = np.zeros((1, len(t)), dtype=np.complex_)
# x1 = signal.chirp(t,10000,t[-1],130000,method='quadratic')# + np.random.randn(len(t))/100
# data[0][:] = x1 - np.mean(x1)
# A.data = data
# A.time = t
# A.rpos = np.zeros(1)
# A.zpos = np.zeros(1)
# C.Dlist.append(A)
#
# C.cwt(df=3000)


# ## Hurst number
# shot = 10186
# trange = [15.8,15.9]
# norm = 0
# clist = ['ECEI_L1303']
# C.add_data(KstarEcei(shot=shot, clist=clist), trange=trange, norm=norm) # shot and time range
# C.hurst(cnl=[0],bins=100,fitlims=[100,1000])
# # C.hurst(cnl=[0],bins=100,fitlims=[100,1000],verbose=0)


# ## BP probability (Rosso PRL 2007) ## add fBm, fGn lines
# shot = 10186
# trange = [15.01,15.02]
# norm = 0
# clist = ['ECEI_L1303', 'ECEI_L1403']
# C.add_data(KstarEcei(shot=shot, clist=clist), trange=trange, norm=norm) # shot and time range
# C.chplane(cnl=[0,1], d=6, bins=1)


# ## Multi-fractal analysis (Carreras PoP 2000)
# shot = 10186
# trange = [15.1,15.17]
# norm = 0
# clist = ['ECEI_L1303']
# C.add_data(KstarEcei(shot=shot, clist=clist), trange=trange, norm=norm) # shot and time range
# C.intermittency()


# # ## Down sampling
# shot = 10186
# trange = [15.01,15.02]
# norm = 0
# clist = ['ECEI_L1303']
# C.add_data(KstarEcei(shot=shot, clist=clist), trange=trange, norm=norm) # shot and time range
# plt.plot(C.Dlist[0].time, C.Dlist[0].data[0,:])
# C.downsample(q=5)
# plt.plot(C.Dlist[0].time, C.Dlist[0].data[0,:], 'o')
# plt.show()


## Nonlinear energy transfer [Ritz]
shot = 10186
trange = [15.1,15.11]
norm = 1
# ref data
clist = ['ECEI_L1303']
C.add_data(KstarEcei(shot=shot, clist=clist), trange=trange, norm=norm) # shot and time range
# cmp data
clist = ['ECEI_L1403']
C.add_data(KstarEcei(shot=shot, clist=clist), trange=trange, norm=norm) # shot and time range
# fft
C.fftbins(nfft=16, window='hann', overlap=0.5, detrend=0, full=1)
# calculate
C.ritz_nonlinear(done=0,dtwo=1)




## SVD


## Threshold FFT


## High order moments




## Wavelet bicoherence





## Transfer entropy


# print(C.Dlist[0].tt, C.Dlist[0].toff, C.Dlist[0].bt, C.Dlist[0].fs, C.Dlist[0].mode, C.Dlist[0].lo, C.Dlist[0].sz, C.Dlist[0].sf)
