import sys, os
sys.path.insert(0, os.pardir)
import bisect
import numpy as np
import matplotlib.pyplot as plt
from fluctana import *
import argparse
import matplotlib.gridspec as gridspec
import zipfile
import io
import functools

from geqdsk_dk_MDS import geqdsk_dk_MDS
from geqdsk_dk3 import geqdsk_dk


parser = argparse.ArgumentParser(description="Check KSTAR plasma control")
parser.add_argument("-slist", nargs='+', type=int, default=[], help="shot number list")
parser.add_argument("-trange", nargs=2, type=float, default=[0, 10],help="time range")
parser.add_argument("-dtype", type=str, default='profile', help="data type (ipf, profile)")
parser.add_argument("-efit", type=str, default='EFITRT1', help="EFIT tree name (EFIT01 or EFITRT1)")
parser.add_argument("-tv", type=str, default='TV01', help="TV camera name (TV01 or TV02)")

args = parser.parse_args()

MDSPLUS_SERVER = os.environ.get('MDSPLUS_SERVER', 'mdsr.kstar.kfe.re.kr:8005')

def shot_to_year(shot):
    thresholds = [6393, 8225, 9427, 11723, 14942, 17356, 19392, 21779, 24100, 27400, 30540, 32810, 34822, 37700, 40984]
    years = [2011, 2012, 2013, 2014, 2015, 2016, 2017, 2018, 2019, 2020, 2021, 2022, 2023, 2024, 2025, 2026]
    return years[bisect.bisect_right(thresholds, shot)]

def get_tv_path(shot):
    # User can implement the campaign folder logic here based on shot number
    thresholds = [1090, 2277, 4448, 6399, 8217, 9427, 11723, 14942, 17356, 19392, 21779, 24100, 27400, 30540, 32810, 33375, 34752, 37742, 40464, 41376]
    campaigns = ['2008C01', '2009C02', '2010C03', '2011C04', '2012C05', '2013C06', '2014C07', '2015C08', '2016C09', '2017C10', '2018C11', '2019C12', '2020C13', '2021C14', '2022C15', '2023C16', '2024C16', '2025C17', '2025C18', '2026C18', '2026C19']
    campaign = campaigns[bisect.bisect_right(thresholds, shot)]
    return f"/Diag_TV/{campaign}/"

@functools.lru_cache(maxsize=150)
def get_cached_geq_local(efit_fn):
    return geqdsk_dk(filename=efit_fn)

@functools.lru_cache(maxsize=150)
def get_cached_geq_mds(shot, time_sec, treename):
    return geqdsk_dk_MDS(shot, time_sec, treename=treename)

@functools.lru_cache(maxsize=10)
def cached_efit_dir_list(efit_dir):
    if os.path.exists(efit_dir):
        return sorted(os.listdir(efit_dir))
    return []

def find_efit_fn(shot=None, tree='EFIT01', selected_time=None):
    efit_dir = '/EFIT/EFIT_RUN/EFITDATA_{:d}/{:s}/EXP{:06d}/'.format(shot_to_year(shot), tree, shot)

    files = cached_efit_dir_list(efit_dir)
    efit_fn = None
    for i, fn in enumerate(files):
        if fn.split('.')[0] == 'g{:06d}'.format(shot):
            efit_time = int(fn.split('.')[-1]) # [ms]
            if efit_time <= selected_time:
                efit_fn = os.path.join(efit_dir, fn)

    return efit_fn

def ece_channel_selection(shot, Rrange):
    if shot <= 14386: 
        clist_temp = ['ECE{:02d}'.format(i) for i in range(1,49)]
    else:        
        clist_temp = ['ECE{:02d}'.format(i) for i in range(1,77)]

    M = KstarMds(shot=shot, clist=clist_temp)
    M.rpos[np.isnan(M.rpos)] = 0.0 # zero for nan channels
    idx = np.where((M.rpos >= Rrange[0]) * (M.rpos <= Rrange[-1]))[0]
    clist = [clist_temp[i] for i in idx]
    rpos = M.rpos[idx]

    return clist, rpos

colors = ['b','g','r','c','m','y','k']


# Data dictionary
data = {
    # Plasma current
    'ip_ref': {'clist': ['SYTIPREFA'], 'ax': (0,0), 'label': 'Ip [MA]', 'scale': -1e-6, 'linestyle': ':'},
    'ip_val': {'clist': ['PCRC03'], 'ax': (0,0), 'label': 'Ip [MA]', 'scale': -1e-6},
    # Rp
    'rp_ref': {'clist': ['LMTRREF'], 'ax': (0,1), 'label': 'Rp [cm]', 'scale': 100, 'linestyle': ':'},
    'rp_val': {'clist': ['LMSR'], 'res': 1e-2, 'ax': (0,1), 'label': 'Rp [cm]', 'scale': 100},

    # Zp
    'zp_ref': {'clist': ['LMTZREF'], 'ax': (0,2), 'label': 'Zp [cm]', 'scale': 100, 'linestyle': ':'},
    'zp_val': {'clist': ['LMSZ'], 'res': 1e-2, 'ax': (0,2), 'label': 'Zp [cm]', 'scale': 100},
    # NBI
    'nbi': {'clist': ['NB11_PNB', 'NB12_PNB', 'NB13_PNB', 'NB2A_PNB', 'NB2B_PNB', 'NB2C_PNB'], 'res': 1e-3, 'ax': (0,3), 'label': 'NBI [MW]', 'scale': 1.0, 'sum': True},
    # ECH
    'ech': {'clist': ['EC2_PWR', 'EC3_PWR', 'EC4_PWR', 'EC5_PWR', 'EC6_PWR'], 'res': 1e-3, 'ax': (0,4), 'label': 'ECH [MW]', 'scale': 1e-3, 'sum': 1},
    # RMP
    'rmp': {'clist': ['PCRMPMJULI'], 'res': 1e-2, 'ax': (0,5), 'label': 'RMP [kA]'},

    # # Bz
    # 'bz': {'clist': ['PCMP4P21Z'], 'ax': (1,0), 'label': 'Bz [T]'},
    # Total MVA
    'mva': {'clist': ['SYSTOTMVA'], 'ax': (1,0), 'label': 'MVA', 'ylim': [0, 90]},
    # Loop voltage
    'loop_v': {'clist': ['PCLV23'], 'ax': (1,1), 'label': 'LV and betap', 'scale': -1.0, 'ylim': [-2, 5], 'linestyle': '--'},
    # Betap
    'betap': {'clist': ['BETAP'], 'ax': (1,1), 'label': 'LV and betap'},
    # Density
    'density': {'clist': ['NE_TCI01'], 'res': 1e-3, 'ax': (1,2), 'label': 'ne'},
    'gas1': {'clist': ['GVTGAS1'], 'ax': (1,2), 'label': 'ne', 'linestyle': '--'},
    'gaspvd': {'clist': ['GVTPVBDD2'], 'ax': (1,2), 'label': 'ne', 'linestyle': '--'},
    # Dalpha
    'dalpha': {'clist': ['POL_HA59:FOO'], 'res': 0.5e-3, 'ax': (1,3), 'label': 'Da', 'scale': -1},    
    # IRVB (total radiation power)
    'irvb': {'clist': ['IRVB1_PRAD'], 'res': 0.01, 'ax': (1,4), 'label': 'Prad'},
    # IVC
    'ivc': {'clist': ['PCIVCURFM'], 'res': 1e-2, 'ax': (1,5), 'label': 'IVC [kA]', 'scale': 1e-3},

    # Rx Upper
    'rxu_ref': {'clist': ['EFSG2RXT'], 'ax': (3,0), 'label': 'Rx Upper [cm]', 'scale': 100, 'ylim': [133, 146], 'linestyle': ':'},
    'rxu_val': {'clist': ['EFSG2RX'], 'ax': (3,0), 'label': 'Rx Upper [cm]', 'scale': 100, 'ylim': [133, 146]},
    # Rx Lower
    'rxl_ref': {'clist': ['EFSG1RXT'], 'ax': (4,0), 'label': 'Rx Lower [cm]', 'scale': 100, 'ylim': [133, 146], 'linestyle': ':'},
    'rxl_val': {'clist': ['EFSG1RX'], 'ax': (4,0), 'label': 'Rx Lower [cm]', 'scale': 100, 'ylim': [133, 146]},
    # Zx Upper
    'zxu_ref': {'clist': ['EFSG2ZXT'], 'ax': (3,1), 'label': 'Zx Upper [cm]', 'scale': 100, 'ylim': [78, 100], 'linestyle': ':'},
    'zxu_val': {'clist': ['EFSG2ZX'], 'ax': (3,1), 'label': 'Zx Upper [cm]', 'scale': 100, 'ylim': [78, 100]},
    # Zx Lower
    'zxl_ref': {'clist': ['EFSG1ZXT'], 'ax': (4,1), 'label': 'Zx Lower [cm]', 'scale': 100, 'ylim': [-100, -78], 'linestyle': ':'},
    'zxl_val': {'clist': ['EFSG1ZX'], 'ax': (4,1), 'label': 'Zx Lower [cm]', 'scale': 100, 'ylim': [-100, -78]},

    # DrSep
    'drsep_ref': {'clist': ['IDTDRSEP'], 'ax': (4,2), 'label': 'DrSep [cm]', 'scale': 100, 'ylim': [-3, 3], 'linestyle': ':'},
    'drsep_val': {'clist': ['IDSDRSEP'], 'res': 1e-2, 'ax': (4,2), 'label': 'DrSep [cm]', 'ylim': [-3, 3], 'scale': 100},
    # # IRC
    # 'irc': {'clist': ['PCIRCURFM'], 'res': 1e-2, 'ax': (4,3), 'label': 'IRC [kA]', 'scale': 1e-3},

    # Rsurf from EFIT
    'rsurf': {'clist': ['RSURF'], 'ax': (0,1), 'label': 'Rp [cm]', 'scale': 100, 'linestyle': '--'},
    # GapIn from EFIT
    'gapin': {'clist': ['RSURF', 'AMINOR'], 'ax': (4,3), 'label': 'GapIn [cm]', 'scale': 100, 'sum': -1},
    # Triangularity Upper from EFIT
    'triu': {'clist': ['TRITOP'], 'ax': (4,4), 'label': 'Triangularity', 'linestyle': '--'},
    # Triangularity Lower from EFIT
    'tril': {'clist': ['TRIBOT'], 'ax': (4,4), 'label': 'Triangularity'}
}

if args.dtype == 'profile':
    data.update({
    # ECE channels 
    'ece_185': {'clist': ['ECE185'], 'res':1e-2, 'ax': (2,2), 'label': 'Temp (1.85 m) [keV]', 'scale': 1e-3},
    'ece_195': {'clist': ['ECE195'], 'res':1e-2, 'ax': (2,3), 'label': 'Temp (1.95 m) [keV]', 'scale': 1e-3},    
    'ece_205': {'clist': ['ECE205'], 'res':1e-2, 'ax': (2,4), 'label': 'Temp (2.05 m) [keV]', 'scale': 1e-3},    
    'ece_215': {'clist': ['ECE215'], 'res':1e-2, 'ax': (2,5), 'label': 'Temp (2.15 m) [keV]', 'scale': 1e-3},    

    # CES Ti channels
    'ces_ti03': {'clist': ['CES_TI03'], 'res':1e-2, 'ax': (2,2), 'label': 'Temp [keV]', 'scale': 1e-3, 'linestyle': '--'},
    'ces_ti07': {'clist': ['CES_TI07'], 'res':1e-2, 'ax': (2,3), 'label': 'Temp [keV]', 'scale': 1e-3, 'linestyle': '--'},
    'ces_ti10': {'clist': ['CES_TI10'], 'res':1e-2, 'ax': (2,4), 'label': 'Temp [keV]', 'scale': 1e-3, 'linestyle': '--'},
    'ces_ti14': {'clist': ['CES_TI14'], 'res':1e-2, 'ax': (2,5), 'label': 'Temp [keV]', 'scale': 1e-3, 'linestyle': '--'},

    # CES Vt channels
    'ces_vt03': {'clist': ['CES_VT03'], 'res':1e-2, 'ax': (3,2), 'label': 'Vt [km/s]'},
    'ces_vt07': {'clist': ['CES_VT07'], 'res':1e-2, 'ax': (3,3), 'label': 'Vt [km/s]'},
    'ces_vt10': {'clist': ['CES_VT10'], 'res':1e-2, 'ax': (3,4), 'label': 'Vt [km/s]'},
    'ces_vt14': {'clist': ['CES_VT14'], 'res':1e-2, 'ax': (3,5), 'label': 'Vt [km/s]'},
    })
elif args.dtype == 'ipf':
    data.update({
    # PF1 coil 
    'pf1_ref': {'clist': ['SYTPF1UL'], 'ax': (2,0), 'label': 'PF1 [kA]', 'scale': 1e-3, 'linestyle': ':'},
    'pf1_val': {'clist': ['PCPF1U'], 'ax': (2,0), 'label': 'PF1 [kA]', 'scale': 1e-3},
    # PF2 coil
    'pf2_ref': {'clist': ['SYTPF2UL'], 'ax': (2,1), 'label': 'PF2 [kA]', 'scale': 1e-3, 'linestyle': ':'},
    'pf2_val': {'clist': ['PCPF2U'], 'ax': (2,1), 'label': 'PF2 [kA]', 'scale': 1e-3},
    # PF3 Upper coil
    'pf3u_ref': {'clist': ['SYTPF3U'], 'ax': (2,2), 'label': 'PF3 Upper [kA]', 'scale': 1e-3, 'linestyle': ':'},
    'pf3u_val': {'clist': ['PCPF3U'], 'ax': (2,2), 'label': 'PF3 Upper [kA]', 'scale': 1e-3},
    # PF3 Lower coil
    'pf3l_ref': {'clist': ['SYTPF3L'], 'ax': (3,2), 'label': 'PF3 Lower [kA]', 'scale': 1e-3, 'linestyle': ':'},
    'pf3l_val': {'clist': ['PCPF3L'], 'ax': (3,2), 'label': 'PF3 Lower [kA]', 'scale': 1e-3},
    # PF4 Upper coil
    'pf4u_ref': {'clist': ['SYTPF4U'], 'ax': (2,3), 'label': 'PF4 Upper [kA]', 'scale': 1e-3, 'linestyle': ':'},
    'pf4u_val': {'clist': ['PCPF4U'], 'ax': (2,3), 'label': 'PF4 Upper [kA]', 'scale': 1e-3},
    # PF4 Lower coil
    'pf4l_ref': {'clist': ['SYTPF4L'], 'ax': (3,3), 'label': 'PF4 Lower [kA]', 'scale': 1e-3, 'linestyle': ':'},
    'pf4l_val': {'clist': ['PCPF4L'], 'ax': (3,3), 'label': 'PF4 Lower [kA]', 'scale': 1e-3},    
    # PF5 Upper coil
    'pf5u_ref': {'clist': ['SYTPF5U'], 'ax': (2,4), 'label': 'PF5 Upper [kA]', 'scale': 1e-3, 'linestyle': ':'},
    'pf5u_val': {'clist': ['PCPF5U'], 'ax': (2,4), 'label': 'PF5 Upper [kA]', 'scale': 1e-3},
    # PF5 Lower coil
    'pf5l_ref': {'clist': ['SYTPF5L'], 'ax': (3,4), 'label': 'PF5 Lower [kA]', 'scale': 1e-3, 'linestyle': ':'},
    'pf5l_val': {'clist': ['PCPF5L'], 'ax': (3,4), 'label': 'PF5 Lower [kA]', 'scale': 1e-3},    
    # PF6 Upper coil 
    'pf6u_ref': {'clist': ['SYTPF6U'], 'ax': (2,5), 'label': 'PF6 Upper [kA]', 'scale': 1e-3, 'linestyle': ':'},
    'pf6u_val': {'clist': ['PCPF6U'], 'ax': (2,5), 'label': 'PF6 Upper [kA]', 'scale': 1e-3, },
    # PF6 Lower coil 
    'pf6l_ref': {'clist': ['SYTPF6L'], 'ax': (3,5), 'label': 'PF6 Lower [kA]', 'scale': 1e-3, 'linestyle': ':'},
    'pf6l_val': {'clist': ['PCPF6L'], 'ax': (3,5), 'label': 'PF6 Lower[kA]', 'scale': 1e-3, },    
    # PF7 coil
    'pf7_ref': {'clist': ['SYTPF7UL'], 'ax': (4,5), 'label': 'PF7 [kA]', 'scale': 1e-3, 'linestyle': ':'},
    'pf7_val': {'clist': ['PCPF7U'], 'ax': (4,5), 'label': 'PF7 [kA]', 'scale': 1e-3},
    })

efit_keys = ['gapin', 'triu', 'tril', 'rsurf']


## Make figure 
fig = plt.figure(figsize=(24, 10))
gs = gridspec.GridSpec(5, 8, figure=fig)

## Create the grid of axes on the left
axs = np.array([[fig.add_subplot(gs[i, j]) for j in range(6)] for i in range(5)])
# Sync x-axes
for i in range(5):
    for j in range(6):
        axs[i, j].sharex(axs[0, 0])

## Create the large axis on the right, spanning all rows and 2 columns
image_ax = fig.add_subplot(gs[:, 6:])
image_ax.set_title('EFIT equilibrium')
image_ax.text(0.5, 0.5, 'EFIT or TV image plot (toggle key: v)', ha='center', va='center', transform=image_ax.transAxes)

## Reference EFIT plot
# efit_ref_fn = '/Users/mjchoi/Work/experiments/KSTAR_2025-2026/Zhang_SCH/g006580.01550'
# efit_ref_fn = '/home/users/mjchoi/fluctana/analysis/data/g006580.01550'
efit_ref_fn = '/abcd/efg'
efit_ref_zoff = 0.06 # [m]
if os.path.exists(efit_ref_fn) is True:
    geq_ref = geqdsk_dk(filename=efit_ref_fn)
    image_ax.plot(geq_ref.get('rbbbs'), geq_ref.get('zbbbs') + efit_ref_zoff, 'g')


## Add data, Post-process, Plot 1D data
for s, shot in enumerate(args.slist):
    A = FluctAna()

    ece_clist, ece_rpos = ece_channel_selection(shot, Rrange=[1.82, 2.21])

    for key in data.keys():
        clist = data[key]['clist'].copy()
        res = data[key].get('res', 0)
        ax = data[key]['ax']
        scale = data[key].get('scale', 1.0)
        linestyle = data[key].get('linestyle', '-')
        sum_flag = data[key].get('sum', 0)
        label = data[key]['label']

        # replace ece clist
        if key in ['ece_185', 'ece_195', 'ece_205', 'ece_215']:
            ece_idx = np.argmin(np.abs(ece_rpos - float(key.split('_')[1])/100))
            clist = [ece_clist[ece_idx]]
            print(f"{key}, clist: {clist}, R: {ece_rpos[ece_idx]:.3f} m")

        # Add data
        try:
            if key in efit_keys:
                if args.efit == 'EFIT04':
                    A.add_data(dev='KSTAR', shot=shot, tree=args.efit, clist=clist, trange=[args.trange[0]*1e3, args.trange[1]*1e3], res=10, norm=0, verbose=0)
                    A.Dlist[-1].time = A.Dlist[-1].time * 1e-3  # [ms] -> [sec]
                else:
                    A.add_data(dev='KSTAR', shot=shot, tree=args.efit, clist=clist, trange=args.trange, res=res, norm=0, verbose=0)
            else:
                A.add_data(dev='KSTAR', shot=shot, tree=None, clist=clist, trange=args.trange, res=res, norm=0, verbose=0)
        except Exception as e:
            print(f"** Error occurred in {shot}, {key}: {e}")
            continue

        if key in ['ces_ti03', 'ces_ti07', 'ces_ti10', 'ces_ti14']:
            print(f"{key}, R: {A.Dlist[-1].rpos[0]:.3f} m")

        if A.Dlist[-1].data is None:
            print(f"** Warning: No data for shot {shot}, {key}")
            continue
        else:
            print(f"Loaded data for shot {shot}, {key}")
            
        # Post-processing 
        if sum_flag == 1:
            ydata = np.sum(A.Dlist[-1].data, axis=0) * scale
        elif sum_flag == -1: 
            # for GapIn
            ydata = A.Dlist[-1].data[0,:] * scale - A.Dlist[-1].data[1,:] * scale - 126.5
        else:
            ydata = A.Dlist[-1].data[0,:] * scale

        # Plot
        axs[ax].plot(A.Dlist[-1].time, ydata, colors[s % len(colors)], linestyle=linestyle, alpha=0.7, label=f'{shot}')

        axs[ax].set_title(data[key]['label'])
        if 'ylim' in data[key]:
            axs[ax].set_ylim(data[key]['ylim'])


## Viewer Class Setup
class PlasmaControlViewer:
    def __init__(self, fig, axs, image_ax, slist, tv_cam, efit_tree):
        self.fig = fig
        self.axs = axs
        self.image_ax = image_ax
        self.slist = slist
        self.tv = tv_cam
        self.efit = efit_tree
        
        self.selected_time = 1000  # Initialize selected_time [ms]
        self.selected_time_step = 100
        self.show_tv = False
        
        self.vlines = [ax.axvline(x=self.selected_time*1e-3, color='r', alpha=0.5) for ax in self.axs.flatten()]
        print(f'Initial jump step [ms]: {self.selected_time_step}')
        print('Controls: [a/d] Prev/Next, [w/x] Step Size, [v] Toggle EFIT/TV, [mouse click] Jump to time, [q/Esc] Quit')

        self.tv_zip_cache = None
        self.tv_zip_path_cache = None
        self.current_tv_image = None
        
        self.fig.canvas.mpl_connect('key_press_event', self.on_key)
        self.fig.canvas.mpl_connect('button_press_event', self.on_click)

    ## Plot EFIT equilibrium at the time of interest
    def plot_EFIT_eq(self):
        self.image_ax.clear()

        title_str = ''
        efit_fn = None
        geq = None
        
        for s, shot in enumerate(self.slist):
            if MDSPLUS_SERVER == 'mdsr.kstar.kfe.re.kr:8005' and self.efit != 'EFITRT1':         ## For running on nKSTAR server
                efit_fn_tmp = find_efit_fn(shot=shot, tree=self.efit, selected_time=self.selected_time)
                if efit_fn_tmp is None:
                    continue
                geq = get_cached_geq_local(efit_fn_tmp)
                efit_fn = efit_fn_tmp
            elif MDSPLUS_SERVER == 'localhost:8005' or self.efit == 'EFITRT1':             ## For running on local PC
                geq = get_cached_geq_mds(shot, self.selected_time*1e-3, self.efit)
                if geq is None or not hasattr(geq, 'file_name'):
                    continue
                efit_fn = geq.file_name

            if efit_fn:
                print(efit_fn)

            # Reference EFIT
            if os.path.exists(efit_ref_fn) is True and 'geq_ref' in globals():
                self.image_ax.plot(geq_ref.get('rbbbs'), geq_ref.get('zbbbs') + efit_ref_zoff, 'g')

            if geq is not None:
                color = colors[s % len(colors)]
                self.image_ax.contour(geq.data['r'][0], geq.data['z'][0], geq._psi_spline_normal(geq.data['r'][0], geq.data['z'][0]).T, levels=np.arange(0,1.2,0.03), linewidths=0.5, colors=color, linestyles='dashed')
                self.image_ax.plot(geq.get('rbbbs'), geq.get('zbbbs'), color)

                ## Set titles
                title_str += f'{shot} ({color}) '
                
        if efit_fn:
            efit_time = int(efit_fn.split("/")[-1].split(".")[-1][-6:]) # [ms]
            title_str += f'at {efit_time*1e-3:.3f} sec'
        else:
            title_str += f'No EFIT available at {self.selected_time*1e-3:.3f} sec'

        if geq is not None:
            self.image_ax.plot(geq.get('rlim'), geq.get('zlim'), 'r')
            
        self.image_ax.set_aspect('equal', adjustable='box')
        self.image_ax.set_xlabel('R [m]')
        self.image_ax.set_ylabel('Z [m]')
        self.image_ax.set_title(title_str)

    # Note: Lru cache is handled differently to avoid global state cleanly
    # but we will just pass it inside the class without global variables
    def load_tv_frame(self, tv_zip_path, img_filename):
        if self.tv_zip_cache is None or self.tv_zip_path_cache != tv_zip_path:
            if self.tv_zip_cache is not None:
                self.tv_zip_cache.close()
            self.tv_zip_cache = zipfile.ZipFile(tv_zip_path, 'r')
            self.tv_zip_path_cache = tv_zip_path
            
        try:
            self.tv_zip_cache.getinfo(img_filename)
            with self.tv_zip_cache.open(img_filename) as f:
                return plt.imread(io.BytesIO(f.read()), format='bmp')
        except KeyError:
            return None

    ## TV image plot function
    def plot_TV_image(self):
        t_sec = self.selected_time * 1e-3
        frame = int(round((t_sec + 0.1) * 210)) + 1
        
        # We only show TV image for the first shot in the list
        if not self.slist:
            self.image_ax.set_title("No shot selected")
            return
            
        shot = self.slist[0]
        tv_zip_path = get_tv_path(shot) + f"{self.tv}/{shot:06d}_{self.tv.lower()}.zip"
        img_filename = f"{shot:06d}-{frame:05d}.bmp"
        
        title_str = f"TV: {shot} at {t_sec:.3f} s (Frame {frame})"
        
        try:
            img_data = self.load_tv_frame(tv_zip_path, img_filename)
            
            if img_data is not None:
                if self.current_tv_image is None or self.current_tv_image not in self.image_ax.images:
                    self.image_ax.clear()
                    self.current_tv_image = self.image_ax.imshow(img_data, extent=[1, 2.5, -1, 1], origin='upper')
                    self.image_ax.set_aspect('equal', adjustable='box')
                    self.image_ax.axis('off')
                else:
                    self.current_tv_image.set_data(img_data)
            else:
                self.image_ax.clear()
                self.current_tv_image = None
                self.image_ax.text(0.5, 0.5, f"Frame {img_filename} not found", ha='center', va='center', transform=self.image_ax.transAxes)
        except Exception as e:
            self.image_ax.clear()
            self.current_tv_image = None
            self.image_ax.text(0.5, 0.5, f"Cannot open TV zip:\n{tv_zip_path}\n{e}", ha='center', va='center', wrap=True, transform=self.image_ax.transAxes)
            
        self.image_ax.set_title(title_str)

    def update_right_plot(self):
        if self.show_tv:
            self.plot_TV_image()
        else:
            self.plot_EFIT_eq()

    def on_key(self, event):
        if event.key == 'd':
            self.selected_time = self.selected_time + self.selected_time_step
        elif event.key == 'a':
            self.selected_time = self.selected_time - self.selected_time_step
        elif event.key == 'w':
            self.selected_time_step = min(2000, int(self.selected_time_step * 1.5))
            print(f'Jump step [ms] = {self.selected_time_step}')
        elif event.key == 'x':
            self.selected_time_step = max(1, int(self.selected_time_step / 1.5))
            print(f'Jump step [ms] = {self.selected_time_step}')
        elif event.key == 'v':
            self.show_tv = not self.show_tv
            print(f'Toggle TV mode: {self.show_tv}')
        elif event.key == 'escape' or event.key == 'q':
            plt.close(self.fig)
            return

        self.update_right_plot()
        for vline in self.vlines:
            vline.set_xdata([self.selected_time * 1e-3, self.selected_time * 1e-3])
        self.fig.canvas.draw_idle()        

    def on_click(self, event):
        if event.inaxes in self.axs.flatten():
            self.selected_time = event.xdata * 1000 # sec -> ms

        self.update_right_plot()
        for vline in self.vlines:
            vline.set_xdata([self.selected_time * 1e-3, self.selected_time * 1e-3])
        self.fig.canvas.draw_idle()

viewer = PlasmaControlViewer(fig, axs, image_ax, args.slist, args.tv, args.efit)


plt.tight_layout()
plt.show()



