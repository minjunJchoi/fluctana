import sys, os
sys.path.insert(0, os.pardir)
from fluctana import *
import argparse


## HOW TO RUN
## python3 check_corr_image.py -shot 22289 -trange 2.716 2.718 -tstep 10 -ref_channel ECEI_GT1003 -cmp_clist ECEI_GT0601-1408 -vlimits -1 1 -flimits 5 9

parser = argparse.ArgumentParser(description="ECEI")
parser.add_argument("-shot", type=int, default=22289, help="Shot number")
parser.add_argument("-trange", nargs=2, type=float, default=[2.716, 2.718], help="Time range [ms]")
parser.add_argument("-tstep", type=int, default=10, help="Movie time step [idx]")
parser.add_argument("-ref_channel", type=str, default='ECEI_GT1003', help="Reference channel")
parser.add_argument("-cmp_clist", nargs='+', type=str, default=['ECEI_GT0601-1408'], help="Comparison channel list")
parser.add_argument("-vlimits", nargs=2, type=float, default=[-1, 1], help="Color range")
parser.add_argument("-flimits", nargs=2, type=float, default=[5, 9], help="Frequency range")
parser.add_argument("-efit", type=str, default=None, help="EFIT tree name")
parser.add_argument("--save", action="store_true", help="Save images")
a = parser.parse_args()


# call fluctana
A = FluctAna()

# add data
A.add_data(dev='KSTAR', shot=a.shot, clist=[a.ref_channel], trange=a.trange, norm=1)
A.add_data(dev='KSTAR', shot=a.shot, clist=a.cmp_clist, trange=a.trange, norm=1)

# # Channel position correction file names
# cpc_fn = f'data/ecei_pos_37389_ECEI_{dname}0101-2408_{int(cpc_time*1000)}ms_b1.01_max_EQ_PRL.pkl'
# A.ch_pos_correction(dnum=dnum, fname=cpc_fn)

# band pass filter
A.filt(dnum=0, name='FFT_pass', fL=a.flimits[0]*1000, fH=a.flimits[1]*1000)
A.filt(dnum=1, name='FFT_pass', fL=a.flimits[0]*1000, fH=a.flimits[1]*1000)

# Calculate correlation coefficient
A.fftbins(nfft=512, window='hann', overlap=0.5, detrend=0, full=1)
A.corr_coef(done=0, dtwo=1)

# # Mark bad channels manually 
# if dname == 'GT':
#     bad_str = '0101, 0104, 0106, 0206, 0306, 0406, 0504, 0505, 0506, 0507, 0606, 0704, 0706, 0806, 0901, 0902, 0903, 0904, 0905, 0906, 1002, 1006, 1101, 1102, 1103, 1104, 1106, 1107, 1108, 1206, 1301, 1302, 1305, 1306, 1307, 1405, 1406, 1501, 1505, 1506, 1601, 1602, 1606, 1701, 1801, 1802, 1806, 1902, 1906, 1907, 1908, 2001, 2002, 2003, 2004, 2005, 2006, 2007, 2008, 2106, 2202, 2205, 2206, 2301, 2302, 2304, 2306, 2307, 2402, 2406'
# elif dname == 'GR':
#     bad_str = '0102, 0103, 0104, 0105, 0106, 0107, 0108, 0201, 0301, 0401, 0402, 0501, 0502, 0601, 0602, 0701, 0702, 0703, 0704, 0705, 0707, 0801, 0802, 0803, 0804, 0805, 0806, 0901, 0902, 0903, 1001, 1002, 1003, 1004, 1006, 1101, 1102, 1201, 1202, 1203, 1301, 1302, 1401, 1402, 1403, 1404, 1501, 1502, 1601, 1602, 1603, 1606, 1701, 1702, 1801, 1802, 1803, 1804, 1901, 1902, 2001, 2002, 2101, 2102, 2201, 2202, 2301, 2302, 2303, 2304, 2305, 2306, 2307, 2308, 2401, 2402, 2403, 2404, 2405, 2406, 2407, 2408'
# bad_cnames = [s.strip() for s in bad_str.split(',')]
# for bad_cname in bad_cnames:
#     if f'ECEI_{dname}' + bad_cname in E.Dlist[dnum].clist:
#         ch_num = E.Dlist[dnum].clist.index(f'ECEI_{dname}' + bad_cname)
#         E.Dlist[dnum].good_channels[ch_num] = 0
#     # print(f'{bad_cname}, {E.Dlist[dnum].clist[ch_num]}')

# ## EFIT (this requires geadks_dk_MDS.py; Please contact trhee@kfe.re.kr for this)
# if a.efit is not None:
#     efit_time = np.mean(a.trange)
#     geddq = geqdsk_dk_MDS(a.shot, efit_time, treename=a.efit)
#     geq.clevels = np.arange(0.1, 1.2, 0.1)
# else:
#     geq = None
geq = None

## Plot (the calculated result is saved in dnum=dtwo)
# clevels = [0.025] # values for contour lines 
clevels = None 

if a.save:
    # A.iplot(dnum=1,snum=10,type='val',tstep=500,vlimits=a.vlimits,istep=0.005,aspect_ratio=0.8,imethod='linear',bcut=0.0345,msize=3,pmethod='image',movtag=f'{a.pulse}ms')
    A.iplot(dnum=1,snum=10,type='val',tstep=a.tstep,vlimits=a.vlimits,istep=0.01,aspect_ratio=0.8,ylimits=[-0.12,0.04],geq=geq,
            imethod='linear',bcut=0.015,msize=0,pmethod='contour',clevels=clevels,movtag=f'movtag')
else:
    A.iplot(dnum=1,snum=10,type='val',tstep=a.tstep,vlimits=a.vlimits,istep=0.01,aspect_ratio=0.8,ylimits=[-0.12,0.04],geq=geq,
            imethod='linear',bcut=0.015,msize=0,pmethod='contour',clevels=clevels)
    # A.iplot(dnum=1,snum=10,type='val',tstep=a.tstep,vlimits=a.vlimits,istep=0.01,aspect_ratio=0.8,ylimits=[-0.05,0],geq=geq,
    #         imethod='linear',bcut=0.015,msize=0,pmethod='image',clevels=[0.027])

