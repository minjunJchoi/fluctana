import sys, os
sys.path.insert(0, os.pardir)
from fluctana import *
import argparse
import matplotlib.gridspec as gridspec

from geqdsk_dk_MDS import geqdsk_dk_MDS
from geqdsk_dk3 import geqdsk_dk


parser = argparse.ArgumentParser(description="Check KSTAR plasma control")
parser.add_argument("-slist", nargs='+', type=int, default=[], help="shot number list")
parser.add_argument("-trange", nargs=2, type=float, help="time range")
parser.add_argument("-efit", type=str, default='EFITRT1', help="EFIT tree name (EFIT01 or EFITRT1)")

args = parser.parse_args()

def shot_to_year(shot):
    if 5073 < shot and shot < 6393:
        year = 2011
    elif 7065 < shot and shot < 8225:
        year = 2012
    elif 8639 < shot and shot < 9427:
        year = 2013
    elif 9741 < shot and shot < 11723:
        year = 2014
    elif 12272 < shot and shot < 14942:
        year = 2015
    elif 14941 < shot and shot < 17356:
        year = 2016
    elif 17963 < shot and shot < 19392:
        year = 2017
    elif 19391 < shot and shot < 21779:
        year = 2018
    elif 21778 < shot and shot < 24100:
        year = 2019
    elif 24100 < shot and shot < 27400:
        year = 2020
    elif 27400 < shot and shot < 30540:
        year = 2021
    elif 30540 < shot and shot < 32810:
        year = 2022
    elif 32810 < shot and shot < 34822:
        year = 2023
    elif 34822 < shot and shot < 37700:
        year = 2024
    else:
        year = 2025

    return year

def find_efit_fn(shot=None, tree='EFIT01', selected_time=None):
    efit_dir = '/EFIT/EFIT_RUN/EFITDATA_{:d}/{:s}/EXP{:06d}/'.format(shot_to_year(shot), tree, shot)

    files = sorted(os.listdir(efit_dir))
    for i, fn in enumerate(files):
        if fn.split('.')[0] == 'g{:06d}'.format(shot):
            efit_time = int(fn.split('.')[-1]) # [ms]
            if efit_time <= selected_time:
                efit_fn = os.path.join(efit_dir, fn)

    return efit_fn

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

    # # Te 
    # 'ece_te': {'clist': ['ECE60'], 'res':1e-4, 'ax': (4,0), 'label': 'ECE [keV]', 'scale': 1e-3}, # 1.8 T: 60 (at 2 m), 2.7 T: 25 3.0 T: 39

    # # Vt
    # 'ces_vt03': {'clist': ['CES_VT03'], 'res':1e-2, 'ax': (2,0), 'label': 'CES [km/s]', 'scale': 1e-3},
    # 'ces_vt06': {'clist': ['CES_VT06'], 'res':1e-2, 'ax': (2,1), 'label': 'CES [km/s]', 'scale': 1e-3},
    # 'ces_vt09': {'clist': ['CES_VT09'], 'res':1e-2, 'ax': (2,2), 'label': 'CES [km/s]', 'scale': 1e-3},
    # 'ces_vt12': {'clist': ['CES_VT12'], 'res':1e-2, 'ax': (2,3), 'label': 'CES [km/s]', 'scale': 1e-3},
    # 'ces_vt15': {'clist': ['CES_VT15'], 'res':1e-2, 'ax': (2,4), 'label': 'CES [km/s]', 'scale': 1e-3},

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
efit_ax = fig.add_subplot(gs[:, 6:])
efit_ax.set_title('EFIT equilibrium')
efit_ax.text(0.5, 0.5, 'Right side plot for EFIT', ha='center', va='center', transform=efit_ax.transAxes)

## Reference EFIT plot
# efit_ref_fn = '/Users/mjchoi/Work/experiments/KSTAR_2025-2026/Zhang_SCH/g006580.01550'
# efit_ref_fn = '/home/users/mjchoi/fluctana/analysis/data/g006580.01550'
efit_ref_fn = '/abcd/efg'
efit_ref_zoff = 0.06 # [m]
if os.path.exists(efit_ref_fn) is True:
    geq_ref = geqdsk_dk(filename=efit_ref_fn)
    efit_ax.plot(geq_ref.get('rbbbs'), geq_ref.get('zbbbs') + efit_ref_zoff, 'g')

## Add data, Post-process, Plot 
for s, shot in enumerate(args.slist):
    A = FluctAna()

    for key in data.keys():
        clist = data[key]['clist'].copy()
        res = data[key].get('res', 0)
        ax = data[key]['ax']
        scale = data[key].get('scale', 1.0)
        linestyle = data[key].get('linestyle', '-')
        sum_flag = data[key].get('sum', 0)
        label = data[key]['label']

        # Add data
        try:
            if key in efit_keys:
                A.add_data(dev='KSTAR', shot=shot, tree=args.efit, clist=clist, trange=args.trange, res=res, norm=0, verbose=0)
                if args.efit == 'EFIT04':
                    A.add_data(dev='KSTAR', shot=shot, tree=args.efit, clist=clist, trange=[args.trange[0]*1e3, args.trange[1]*1e3], res=10, norm=0, verbose=0)
                    A.Dlist[-1].time = A.Dlist[-1].time * 1e-3  # [ms] -> [sec]
            else:
                A.add_data(dev='KSTAR', shot=shot, tree=None, clist=clist, trange=args.trange, res=res, norm=0, verbose=0)
        except:
            print(f"** Error occurred in {shot}, {key}")
            continue

        if A.Dlist[-1].time is None:
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
        axs[ax].plot(A.Dlist[-1].time, ydata, colors[s], linestyle=linestyle, label=f'{shot}')

        axs[ax].set_title(data[key]['label'])
        if 'ylim' in data[key]:
            axs[ax].set_ylim(data[key]['ylim'])

## Plot EFIT equilibrium at the time of interest
selected_time = 1000  # Initialize selected_time [ms]
selected_time_step = 100
print(f'Initial jump step [idx]: {selected_time_step}')
print('Controls: [a/d] Prev/Next, [w/x] Step Size, [mouse click] Jump to time, [q] Quit')

MDSPLUS_SERVER = os.environ.get('MDSPLUS_SERVER', 'mdsr.kstar.kfe.re.kr:8005')

# EFIT plot function
def plot_EFIT_eq(efit_ax, selected_time):
    efit_ax.clear()

    title_str = ''
    for s, shot in enumerate(args.slist):
        if MDSPLUS_SERVER == 'mdsr.kstar.kfe.re.kr:8005' and args.efit != 'EFITRT1':         ## For running on nKSTAR server
            efit_fn = find_efit_fn(shot=shot, tree=args.efit, selected_time=selected_time)
            geq = geqdsk_dk(filename=efit_fn)
        elif MDSPLUS_SERVER == 'localhost:8005' or args.efit == 'EFITRT1':             ## For running on local PC
            geq = geqdsk_dk_MDS(shot, selected_time*1e-3, treename=args.efit)
            efit_fn = geq.file_name
        print(efit_fn)

        # Reference EFIT
        if os.path.exists(efit_ref_fn) is True:
            efit_ax.plot(geq_ref.get('rbbbs'), geq_ref.get('zbbbs') + efit_ref_zoff, 'g')

        efit_ax.contour(geq.data['r'][0], geq.data['z'][0], geq._psi_spline_normal(geq.data['r'][0], geq.data['z'][0]).T, levels=np.arange(0,1.2,0.1), linewidths=0.5, colors=colors[s], linestyles='dashed')
        efit_ax.plot(geq.get('rbbbs'), geq.get('zbbbs'), colors[s])

        ## Set titles
        title_str += f'{shot} ({colors[s]}) '
    title_str += f'at {efit_fn.split("/")[-1].split(".")[-1][-6:]} ms'

    efit_ax.plot(geq.get('rlim'), geq.get('zlim'), 'r')
    efit_ax.set_aspect('equal', adjustable='box')
    efit_ax.set_xlabel('R [m]')
    efit_ax.set_ylabel('Z [m]')
    efit_ax.set_title(title_str)

def on_key(event):
    global selected_time, selected_time_step
    if event.key == 'd':
        selected_time = selected_time + selected_time_step
    elif event.key == 'a':
        selected_time = selected_time - selected_time_step
    elif event.key == 'w':
        selected_time_step = min(2, int(selected_time_step * 1.5))
        print(f'Jump step [ms] = {selected_time_step}')
    elif event.key == 'x':
        selected_time_step = max(1, int(selected_time_step / 1.5))
        print(f'Jump step [ms] = {selected_time_step}')
    elif event.key == 'escape':
        plt.close(fig)
        return

    plot_EFIT_eq(efit_ax, selected_time)                
    fig.canvas.draw()        

def on_click(event):
    global selected_time
    if event.inaxes in axs.flatten():
        selected_time = event.xdata * 1000 # sec -> ms

    plot_EFIT_eq(efit_ax, selected_time)
    fig.canvas.draw()        

fig.canvas.mpl_connect('key_press_event', on_key)
fig.canvas.mpl_connect('button_press_event', on_click)

plt.tight_layout()
plt.show()



