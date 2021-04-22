import sys, os
sys.path.insert(0, os.pardir)
from fluctana import *

## ===================== Calculate rescaled complexity ================
shot = 26027 
trange = [2.0,2.8]
clist = ['ECEI_HT1101-1108']  

# read ECEI data (not normalized)      
A = FluctAna()
A.add_data(dev='KSTAR', shot=shot, clist=clist, trange=trange, norm=0)

# ECEI channel positions
rpos = A.Dlist[0].rpos
zpos = A.Dlist[0].zpos

# calculation parameters 
twin = 0.005; q = 1; d = 5; bins = 1; cnum = len(A.Dlist[0].clist)
nst = math.factorial(d)

# calculation time points 
taxis = np.arange(trange[0]+0.01,trange[1]-0.01,0.001) 

# results 
entp = np.zeros((cnum, len(taxis)))
comp = np.zeros((cnum, len(taxis)))
rs_comp = np.zeros((cnum, len(taxis)))

for i, t in enumerate(taxis):
    tidx = (t - twin/2 < A.Dlist[0].time) & (A.Dlist[0].time < t + twin/2)

    for j in range(cnum):
        ndy = A.Dlist[0].data[j,tidx]

        # calculate BP probability 
        _,val,_ = st.bp_prob(ndy, d=d, bins=bins)
        
        # calculate
        comp[j, i], entp[j, i] = st.ch_measure(val)
 
    print('{:d}/{:d} bsize {:d} >> nst {:d} done'.format(i+1,len(taxis),len(ndy),nst))

# rescale complexity
h_min, c_min, h_max, c_max, h_cen, c_cen = st.ch_bdry(d)
for j in range(cnum):
   rs_comp[j,:] = st.complexity_rescale(entp[j,:], comp[j,:], h_min, c_min, h_max, c_max, h_cen, c_cen)

# save result
fname = 'res_ch_{:d}_{:s}_q{:d}_d{:d}_{:d}ms.pkl'.format(shot, clist[0], q, d, int(twin*1000))
with open(fname, 'wb') as fout:
   pickle.dump([taxis, rpos, zpos, entp, comp, rs_comp], fout)


# #### ========================= Plot results  
# shot = 26027  
# trange = [2.0,2.8]
# clist = ['ECEI_HT1101-1108'] 
# q = 1
# d = 5
# twin = 0.005

# # load result
# fname = 'res_ch_{:d}_{:s}_q{:d}_d{:d}_{:d}ms.pkl'.format(shot, clist[0], q, d, int(twin*1000))
# with open(fname, 'rb') as fin:
#     [taxis, rpos, zpos, entp, comp, rs_comp]= pickle.load(fin)

# # # use difference
# # rs_comp = rs_comp - np.mean(rs_comp[:,0:300],axis=1,keepdims=True)

# fig, (ax1,ax2,ax3) = plt.subplots(3,1, figsize=(8,6), gridspec_kw = {'height_ratios':[1,1,1]}, sharex = True)
# ax12 = ax1.twinx()
# ax22 = ax2.twinx()
# ax33 = fig.add_axes([0.87, 0.11, 0.03, 0.17])
# # ax1.tick_params(bottom=True, top=True)
# fig.subplots_adjust(right = 0.85, left = 0.1)

# # D_alpha
# A = FluctAna()
# A.add_data(dev='KSTAR', shot=shot, clist=['pol_HA06'], trange=trange, norm=0)
# A.filt(dnum=0,name='FIR_pass',fL=0,fH=5000,b=0.01)
# ax1.plot(A.Dlist[0].time, A.Dlist[0].data[0,:]*1.0e-21, color='k')
# ax1.set_ylabel('D_alpha [arb. units]')

# # RMP field
# A = FluctAna()
# A.add_data(dev='KSTAR', shot=shot, clist=['PCRMPMNULI'], trange=trange, norm=0)
# ax12.plot(A.Dlist[0].time, np.abs(A.Dlist[0].data[0,:])*1.0e-3, color='r')
# ax12.set_ylabel('IVCC [kA/turn]')
# ax12.set_title('shot {:d}'.format(shot))

# # ne
# A = FluctAna()
# A.add_data(dev='KSTAR', shot=shot, clist=['ne_tci01'], trange=trange, norm=0)
# A.filt(dnum=0,name='FIR_pass',fL=0,fH=50,b=0.01)
# ax2.plot(A.Dlist[0].time, A.Dlist[0].data[0,:], color='k')
# ax2.set_ylabel('ne [1e19 m-3]')

# # Te
# A = FluctAna()
# A.add_data(dev='KSTAR', shot=shot, clist=['ECE57'], trange=trange, norm=0)
# A.filt(dnum=0,name='FIR_pass',fL=0,fH=100,b=0.01)
# ax22.plot(A.Dlist[0].time, A.Dlist[0].data[0,:]*1.0e-3, color='r')
# ax22.set_ylabel('Te [keV]')

# # plot rescaled complexity 
# xi, yi = np.meshgrid(taxis, rpos)
# cf = ax3.contourf(xi, yi*100, rs_comp, 15, cmap="RdBu_r")
# # for i in range(len(rmid)):
# #     ax3.axhline(y=rpos[i]*100, color='k', linestyle=':')
# plt.colorbar(cf, cax=ax33)
# ax3.set_xlabel('Time [s]')
# ax3.set_ylabel('R [cm]')
# ax33.set_title('  Rescaled\n  complexity', fontsize=8)

# # plot appear 
# ax1.set_xlim([trange[0]+0.01,trange[1]-0.01])
# plt.show()


