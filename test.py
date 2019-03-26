from fluctana import *

# Load modules
C = FluctAna()

############################## Down sampling ##############################
# shot = 10186
# trange = [15.01,15.02]
# norm = 0
# clist = ['ECEI_L1303']
# C.add_data(KstarEcei(shot=shot, clist=clist), trange=trange, norm=norm) # shot and time range
# plt.plot(C.Dlist[0].time, C.Dlist[0].data[0,:])
# C.downsample(q=5)
# plt.plot(C.Dlist[0].time, C.Dlist[0].data[0,:], 'o')
# plt.show()


############################## FIR filters ##############################
# shot = 18569
# trange = [3,4.5]
# # add data
# C.add_data(KstarMds(shot=shot, clist=['EP54:FOO']), trange=trange, norm=0)
# # estimate spectrum before filter
# C.fftbins(nfft=256,window='hann',overlap=0.5,detrend=1)
# C.cross_power(done=0,dtwo=0)
# # plot time series and its power spectrum result
# ax1 = plt.subplot(211)
# ax1.plot( C.Dlist[0].time, C.Dlist[0].data[0,:])
# ax2 = plt.subplot(212)
# ax2.plot( C.Dlist[0].ax/1000.0, C.Dlist[0].val[0,:].real )
#
# ### low pass filter ###
# #C.filt('FIR_pass',0,50000,0.01) # smaller b is sharper
# ### high pass filter ###
# # C.filt('FIR_pass',50000,0,0.01)
# ### band pass filter  ###
# C.filt('FIR_pass',30000,0,0.01) # smaller b is sharper
# C.filt('FIR_pass',0,50000,0.01) # smaller b is sharper
# ### band block filter  ###
# # C.filt('FIR_block',0,1000,0.01) # smaller b is sharper
#
# # estimate spectrum after filter
# C.fftbins(nfft=256,window='hann',overlap=0.5,detrend=1)
# C.cross_power(done=0,dtwo=0)
# # plot filtered time series and its power spectrum results
# ax1.plot( C.Dlist[0].time, C.Dlist[0].data[0,:])
# ax2.plot( C.Dlist[0].ax/1000.0, C.Dlist[0].val[0,:].real )
#
# ax1.set_xlabel('Time [s]')
# ax2.set_yscale('log')
# ax2.set_xlabel('Frequency [kHz]')
# plt.show()

############################## SVD filter ##############################
# shot = 10186
# trange = [15.9,16]
# norm = 0
# # ECEI channels from 0101 to 2408 (all)
# clist = ['ECEI_H0101-2408']
# # add data
# C.add_data(KstarEcei(shot=shot, clist=clist), trange=trange, norm=norm) # shot and time range
# old_data = np.copy(C.Dlist[0].data[7,:])
# # svd filter with cutoff
# C.svd_filt(cutoff=0.9, verbose=1)
# # plot a single data before and after filter
# plt.plot(C.Dlist[0].time, old_data)
# plt.plot(C.Dlist[0].time, C.Dlist[0].data[7,:], 'o')
# plt.show()

############################## Threshold FFT ##############################


############################## cross power ########################


############################## coherence ########################


############################## cross phase ########################


############################## correlation ########################


############################## correlation coefficient ########################


############################## cross power spectrogram ########################
### xspec with test signals
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


### xspec with ECEI data
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


############################## Local SKw ########################
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


###################### Bicoherence (+ summed bicoherence) ######################
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


###################### Ritz nonlinear energy transfer ######################
# shot = 10186
# trange = [15.1,15.11]
# norm = 1
# # ref data
# clist = ['ECEI_L1303']
# C.add_data(KstarEcei(shot=shot, clist=clist), trange=trange, norm=norm) # shot and time range
# # cmp data
# clist = ['ECEI_L1403']
# C.add_data(KstarEcei(shot=shot, clist=clist), trange=trange, norm=norm) # shot and time range
# # fft
# C.fftbins(nfft=16, window='hann', overlap=0.5, detrend=0, full=1)
# # calculate
# C.ritz_nonlinear(done=0,dtwo=1)


###################### continuous wavelet transform ######################
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


###################### Skewness ######################


###################### Kurtosis ######################


###################### Hurst exponent ######################
# shot = 10186
# trange = [15.8,15.9]
# norm = 0
# clist = ['ECEI_L1303']
# C.add_data(KstarEcei(shot=shot, clist=clist), trange=trange, norm=norm) # shot and time range
# C.hurst(cnl=[0],bins=100,fitlims=[100,1000])
# # C.hurst(cnl=[0],bins=100,fitlims=[100,1000],verbose=0)


###################### CH plane [Rosso PRL 2007] ######################
# shot = 10186
# trange = [15.01,15.02]
# norm = 0
# clist = ['ECEI_L1303', 'ECEI_L1403']
# C.add_data(KstarEcei(shot=shot, clist=clist), trange=trange, norm=norm) # shot and time range
# C.chplane(cnl=[0,1], d=6, bins=1)


###################### Multi-fractal analysis [Carreras PoP 2000] ##############
# shot = 10186
# trange = [15.1,15.17]
# norm = 0
# clist = ['ECEI_L1303']
# C.add_data(KstarEcei(shot=shot, clist=clist), trange=trange, norm=norm) # shot and time range
# C.intermittency()






## Wavelet bicoherence

## Wavelet nonlinear energy transfer

## Transfer entropy
