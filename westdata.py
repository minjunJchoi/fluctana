"""
Author: Minjun J. Choi (mjchoi@kfe.re.kr), Claude Code Opus

Description: This code reads the WEST data from the hdf5 file if it exists,
             or using imas_west (on the WEST altair server)

HDF5 structure (WEST_PATH/{shot:06d}/WEST.{shot:06d}.h5, saved by imas_to_hdf5()) :
    /.attrs         SHOT, T_OFFSET
    /{cname}/TIME   time [s] (T_OFFSET subtracted), hard-linked between channels read together
    /{cname}/DATA   data
    /{cname}.attrs  RPOS [m], ZPOS [m], APOS [rad], IDS_NAME, OCCURRENCE, TRANGE,
                    POS_TRANGE (trange of time-averaged positions, e.g. ECE)
    /EQ_{occ}       equilibrium saved by add_eq_to_hdf5() : TIME, R, Z, PSI, PSI_BND, RHO_TOR, Q, R_IN, R_OUT, WALL_R, WALL_Z
    /EQ_{occ}.attrs IDS_NAME, OCCURRENCE (occ = 0 : NICE without, 1 : with polarimetry; each existing occurrence is saved)
    /EQ_{occ}/{VAR} channels of EQ_CHANNELS (e.g. EQ_1/W_MHD, EQ_1/beta_pol) saved by add_eq_to_hdf5(), read with tree = occ,
                    TIME hard-linked to /EQ_{occ}/TIME
"""

import os
import re
import traceback

import numpy as np
import h5py
try:
    import imas_west as iw
except ImportError:
    iw = None

# WEST hdf5 data path
WEST_PATH = os.path.expanduser('~/data/WEST/') # on altair
if not os.path.exists(WEST_PATH):
    WEST_PATH = '/Users/mjchoi/Work/data/WEST/' # on local machine

# clists saved in hdf5 file; each clist is read by one WestData instance (one kind of data)
HDF5_CLIST = [['Ip'], ['Bt'], ['P_LH1-2'], ['P_ICH1-3'], ['P_Ohm'], ['ne'], ['P_rad', 'P_rad_core'],
              ['SECE01-25'], ['FECE01-25'], ['MC06', 'MC07', 'MC30'],
              ['DI_6561_LODIVOU18', 'DI_4341_LODIVOU18', 'WI_4009_LODIVOU18']]

# cname patterns
CH_CNAME = r'^(P_LH|P_ICH|SECE|FECE|MC)(\d+)$'      # numbered channel (e.g. P_LH1, SECE01, FECE01, MC06)
VIS_CNAME = r'^([A-Za-z]+_\d{4})_([A-Za-z0-9]+)$'   # visible spectroscopy {LINE}_{LOS} (e.g. DI_4341_LODIVOU18)
# equilibrium channels {VAR : path in equilibrium IDS}, saved by add_eq_to_hdf5() as EQ_{occ}/{VAR} channels
EQ_CHANNELS = {'q_axis': 'global_quantities/q_axis', 'W_MHD': 'global_quantities/w_mhd',
               'q95': 'global_quantities/q_95', 'q_min': 'global_quantities/q_min/value', 'rho_q_min': 'global_quantities/q_min/rho_tor_norm',
               'beta_pol': 'global_quantities/beta_pol', 'beta_tor': 'global_quantities/beta_tor', 'beta_n': 'global_quantities/beta_normal',
               'li': 'global_quantities/li_3',
               'R_axis': 'global_quantities/magnetic_axis/r', 'Z_axis': 'global_quantities/magnetic_axis/z',
               'R_geo': 'boundary/geometric_axis/r', 'Z_geo': 'boundary/geometric_axis/z', 'a_minor': 'boundary/minor_radius',
               'kappa': 'boundary/elongation', 'tri_u': 'boundary/triangularity_upper', 'tri_l': 'boundary/triangularity_lower',
               'R_x': 'boundary/x_point/r', 'Z_x': 'boundary/x_point/z', 'elong_u': 'boundary/elongation_upper', 'elong_l': 'boundary/elongation_lower'}
EQ_CNAME = r'^({:s})$'.format('|'.join(EQ_CHANNELS.keys())) # equilibrium channel (occurrence given by tree; e.g. tree = 0 : NICE without, 1 : with polarimetry)

#### VAR to IDS
# Ip              : plasma current [A]                    magnetics.ip[0].data
# Bt              : vacuum toroidal field at R0 [T]       tf.b_field_tor_vacuum_r.data / tf.r0
# q_axis, W_MHD, beta_pol, li, ... : equilibrium quantities  equilibrium (occurrence 1, NICE with polarimetry).{path in EQ_CHANNELS}
#                   (e.g. q_axis : global_quantities.q_axis, W_MHD [J] : global_quantities.w_mhd, R_x : boundary.x_point.r)
#                   occurrence by tree (tree = 0 : NICE without, 1 : with polarimetry, None : 1)
#                   saved in hdf5 by add_eq_to_hdf5() as EQ_{occ}/{VAR} (each existing occurrence)
# P_LH1, P_LH2    : LH power per antenna [W]              lh_antennas.antenna[ch].power_launched.data
# P_ICH1--P_ICH3  : ICRH power per antenna [W]            ic_antennas.antenna[ch].power_launched.data
# P_ECH           : ECRH launched power [W]               ec_launchers.beam[:].power_launched.data (sum)
# P_Ohm           : ohmic power [W]                       summary.global_quantities.power_ohm.value
# P_rad           : total radiated power [W]              bolometer.power_radiated_total
# P_rad_core      : radiated power inside LCFS [W]        bolometer.power_radiated_inside_lcfs
# gas             : total gas injection rate              gas_injection.pipe[:].flow_rate.data (sum)
# ne              : line averaged density [m^-3]          summary.line_average.n_e.value
# Ti0             : central ion temperature [eV]          summary.local.magnetic_axis.t_i_average.value
# SECE01--        : slow 1D ECE electron temperature [eV] ece (occurrence 0).channel[ch].t_e.data
# FECE01--        : fast 1D ECE signal                    ece (occurrence 1).channel[ch].t_e_voltage.data, if exists
#                   positions of SECE and FECE            ece (occurrence 0).channel[ch].position.r, z (same size as ece.time)
# MC01--MC48      : MHD probe (Mirnov coil) signal [V]    magnetics (occurrence 1).b_field_pol_probe[ch].voltage.data
#                   48 fast channels, <= 1 MHz, triggered acquisition (2 s windows, 5-10 triggers per shot)
# {LINE}_{LOS}    : visible spectroscopy line radiance    spectrometer_visible.channel[name == LOS].grating_spectrometer.processed_line[name == LINE].radiance.data
#                   (e.g. DI_4341_LODIVOU18) LINE : DI_6561, DI_4341, WI_4009, ... (VISTA integrated lines)
#                   LOS : IC2L, IC2R, LH6aL, LH6aR, LH6bL, APL, LODIVIN, LODIVOU, UPDIV, INBUM + number

class WestData():
    def __init__(self, shot, clist, tree=None, trange=[0,25], t_offset=32.2, savedata=False, read_hdf5=True):
        self.shot = shot
        self.tree = tree # IMAS occurrence of equilibrium channels (0 : NICE without, 1 : with polarimetry, None : 1)
        self.t_offset = t_offset

        self.clist = self.expand_clist(clist)

        # get ids_name, imas_occurrence for given diagnostic
        self.ids_name, self.imas_occ = cname_to_imas_name_occ(self.shot, self.clist[0], tree=self.tree)

        # hdf5 file name
        self.fname = "{:s}{:06d}/WEST.{:06d}.h5".format(WEST_PATH, shot, shot)

        # save IMAS data in hdf5 format if necessary
        if os.path.exists(self.fname) == False and savedata == True:
            print('Reformat WEST data to hdf5 file')
            imas_to_hdf5(self.shot, trange=trange, t_offset=self.t_offset)

        # read from hdf5 if all channels are in the file
        self.read_hdf5 = False
        if read_hdf5 and os.path.exists(self.fname):
            with h5py.File(self.fname, 'r') as fin:
                self.read_hdf5 = all(self.hdf5_path(cname) in fin for cname in self.clist)

                # without imas_west (e.g. local machine), only channels in hdf5 are read
                if not self.read_hdf5 and iw is None:
                    print('No imas_west : {} not in hdf5, removed'.format([self.hdf5_path(cname) for cname in self.clist if self.hdf5_path(cname) not in fin]))
                    self.clist = [cname for cname in self.clist if self.hdf5_path(cname) in fin]
                    self.read_hdf5 = len(self.clist) > 0

        # without imas_west and hdf5, no channel to read
        if not self.read_hdf5 and iw is None:
            self.clist = []

        # ids read once (by channel_position or read_imas_signal) and reused; released at the end of get_data
        self.ids = None

        # get channel positions
        self.channel_position(trange)

        # data quality
        self.good_channels = np.ones(len(self.clist))
        self.offlev = np.zeros(len(self.clist))
        self.offstd = np.zeros(len(self.clist))
        self.siglev = np.zeros(len(self.clist))
        self.sigstd = np.zeros(len(self.clist))

        self.time = None
        self.data = None

    def get_data(self, trange=[0,25], norm=0, atrange=[1.0, 1.1], res=0, verbose=1):
        if verbose == 1:
            if norm == 0:
                print('Data is not normalized')
            elif norm == 1:
                print('Data is normalized by trange average')
            elif norm == 2:
                print('Data is normalized by atrange average')

        self.trange = trange

        self.time = None
        self.data = None

        data = [] # data of each channel

        if self.read_hdf5:
            # read data from hdf5
            print('Read WEST data from hdf5 file')
            with h5py.File(self.fname, 'r') as fin:
                # time base (time of the file is converted to self.t_offset)
                t = np.array(fin[f'{self.hdf5_path(self.clist[0])}/TIME']) + fin.attrs['T_OFFSET'] - self.t_offset
                idx1, idx2, aidx = time_index(t, trange, atrange)
                self.fs = round(1/(t[1] - t[0])/1000)*1000
                self.time = t[idx1:idx2]

                for cname in self.clist:
                    dset = fin[self.hdf5_path(cname)]
                    self.ids_name = dset.attrs['IDS_NAME']
                    self.imas_occ = dset.attrs['OCCURRENCE']

                    # load data
                    v = np.array(dset['DATA'][idx1:idx2])
                    av = np.array(dset['DATA'])[aidx] if norm == 2 else None

                    if verbose == 1: print("Read {:d} - {:s} : {:s} (number of data points = {:d}), t_offset = {:.1f}".format(self.shot, self.ids_name, cname, len(v), self.t_offset))

                    data.append(normalize(v, av, norm))
        else:
            # read data from IMAS
            print('Load WEST data from IMAS')

            # --- loop starts --- #
            for cname in self.clist.copy():
                try:
                    t, v, self.ids = self.read_imas_signal(cname, ids=self.ids)

                    # the first successfully read channel sets time base
                    if len(data) == 0:
                        idx1, idx2, aidx = time_index(t, trange, atrange)
                        self.fs = round(1/(t[1] - t[0])/1000)*1000
                        self.time = t[idx1:idx2]

                    av = v[aidx]
                    v = v[idx1:idx2]
                    if len(v) != len(self.time):
                        raise ValueError('Time base of {:s} is different from the first channel'.format(cname))

                    if verbose == 1: print("Read {:d} - {:s} : {:s} (number of data points = {:d}), t_offset = {:.1f}".format(self.shot, self.ids_name, cname, len(v), self.t_offset))

                    data.append(normalize(v, av, norm))
                except Exception as e:
                    self.clist.remove(cname)
                    if verbose == 1:
                        print('Failed {:d} : {:s}. {:s} is removed'.format(self.shot, cname, cname))
                        print(f"Error: {e}")
                        traceback.print_exc()
            # --- loop ends --- #

            # update channel positions with trange (failed channels are already removed from clist)
            self.channel_position(trange)

            # release ids (can be large, e.g. fast ECE)
            self.ids = None

        if len(data) > 0:
            self.data = np.array(data)

        # # check data quality
        # self.find_bad_channel()

        return self.time, self.data

    def read_imas_signal(self, cname, ids=None):
        # IN : node name, ids (reused if already read)
        # OUT : time [s], data, ids
        if ids is None and not cname.startswith('MC'): # MC : partialGet the probe only
            ids = iw.get(self.shot, self.ids_name, imas_occurrence=self.imas_occ)

        ch, k = cname_to_imas_ch_info(cname, ids=ids)

        if cname == 'Ip':
            t, v = ids.ip[0].time, ids.ip[0].data
        elif cname == 'Bt':
            t, v = ids.time, ids.b_field_tor_vacuum_r.data/ids.r0
        elif re.match(EQ_CNAME, cname): # equilibrium channels (e.g. q_axis, W_MHD, beta_pol), path in EQ_CHANNELS
            t, v = ids.time, ids
            for node in EQ_CHANNELS[cname].split('/'):
                v = getattr(v, node)
        elif cname.startswith('P_LH') or cname.startswith('P_ICH'):
            t, v = ids.antenna[ch].power_launched.time, ids.antenna[ch].power_launched.data
        elif cname == 'P_ECH':
            t = ids.time
            v = np.sum([beam.power_launched.data for beam in ids.beam], axis=0)
        elif cname == 'P_rad':
            t, v = ids.time, ids.power_radiated_total
        elif cname == 'P_rad_core':
            t, v = ids.time, ids.power_radiated_inside_lcfs
        elif cname == 'P_Ohm':
            t, v = ids.time, ids.global_quantities.power_ohm.value
        elif cname == 'gas':
            t = ids.pipe[0].flow_rate.time
            v = np.sum([pipe.flow_rate.data for pipe in ids.pipe], axis=0)
        elif cname == 'ne':
            t, v = ids.time, ids.line_average.n_e.value
        elif cname == 'Ti0':
            t, v = ids.time, ids.local.magnetic_axis.t_i_average.value
        elif cname.startswith('SECE'):
            t, v = ids.time, ids.channel[ch].t_e.data
        elif cname.startswith('FECE'):
            t, v = ids.time, ids.channel[ch].t_e_voltage.data
        elif cname.startswith('MC'):
            probe = iw.partialGet(self.shot, '{:s}/b_field_pol_probe({:d})'.format(self.ids_name, ch), imas_occurrence=self.imas_occ)
            t, v = probe.voltage.time, probe.voltage.data
            if len(t) == 0: # homogeneous_time = 1 : common time of magnetics (occurrence 1)
                t = iw.partialGet(self.shot, '{:s}/time'.format(self.ids_name), imas_occurrence=self.imas_occ)
        elif self.ids_name == 'spectrometer_visible':
            if k is None:
                raise ValueError('No line of {:s} in spectrometer_visible'.format(cname))
            line = ids.channel[ch].grating_spectrometer.processed_line[k]
            t, v = line.radiance.time, line.radiance.data # calibrated [Ph/m^2/sr/s]
            if len(v) == 0: # no absolute calibration
                t, v = line.intensity.time, line.intensity.data # standardized [Phe/s]

        # homogeneous_time = 1 : signal time is empty, use ids.time
        if len(t) == 0: t = ids.time

        return np.asarray(t) - self.t_offset, np.asarray(v), ids

    def find_bad_channel(self):
        # auto-find bad
        for c in range(len(self.clist)):
            # check signal level
            if self.siglev[c] > 0.01:
                ref = 100*self.offstd[c]/self.siglev[c]
            else:
                ref = 100
            if ref > 30:
                self.good_channels[c] = 0
                print('LOW signal level channel {:s}, ref = {:g}%, siglevel = {:g} V'.format(self.clist[c], ref, self.siglev[c]))

            # check bottom saturation
            if self.offstd[c] < 0.001:
                self.good_channels[c] = 0
                print('SAT offset data  channel {:s}, offstd = {:g}%, offlevel = {:g} V'.format(self.clist[c], self.offstd[c], self.offlev[c]))

            # check top saturation.
            if self.sigstd[c] < 0.001:
                self.good_channels[c] = 0
                print('SAT signal data  channel {:s}, offstd = {:g}%, siglevel = {:g} V'.format(self.clist[c], self.sigstd[c], self.siglev[c]))

    def channel_position(self, trange):
        cnum = len(self.clist)
        self.rpos = np.zeros(cnum)  # R [m]
        self.zpos = np.zeros(cnum)  # z [m]
        self.apos = np.zeros(cnum)  # angle [rad]

        if cnum == 0: return

        if self.read_hdf5:
            # positions from hdf5
            with h5py.File(self.fname, 'r') as fin:
                for c, cname in enumerate(self.clist):
                    dset = fin[self.hdf5_path(cname)]
                    self.rpos[c] = dset.attrs['RPOS'] # [m]
                    self.zpos[c] = dset.attrs['ZPOS'] # [m]
                    self.apos[c] = dset.attrs['APOS'] # [rad]

        elif self.ids_name == 'ece':
            # SECE, FECE : time-dependent positions in occurrence 0, averaged over trange
            # (reused as self.ids only for SECE whose data is also in occurrence 0)
            if self.imas_occ == 0:
                if self.ids is None:
                    self.ids = iw.get(self.shot, self.ids_name, imas_occurrence=0)
                ids = self.ids
            else:
                ids = iw.get(self.shot, self.ids_name, imas_occurrence=0)

            time = ids.time - self.t_offset
            idx = np.where((time >= trange[0])*(time <= trange[1]))
            for c, cname in enumerate(self.clist):
                ch, _ = cname_to_imas_ch_info(cname)
                self.rpos[c] = np.mean(ids.channel[ch].position.r[idx])
                self.zpos[c] = np.mean(ids.channel[ch].position.z[idx])
            print(f'ECE positions are averaged for trange {trange}')

        elif self.clist[0].startswith('MC'):
            # MC : probe positions (partialGet position only; iw.get loads all fast probe data)
            for c, cname in enumerate(self.clist):
                ch, _ = cname_to_imas_ch_info(cname)
                pos = iw.partialGet(self.shot, '{:s}/b_field_pol_probe({:d})/position'.format(self.ids_name, ch), imas_occurrence=self.imas_occ)
                self.rpos[c] = pos.r
                self.zpos[c] = pos.z
                self.apos[c] = pos.phi

        elif self.ids_name == 'spectrometer_visible':
            # visible spectroscopy : measurement position (end point of LOS)
            if self.ids is None:
                self.ids = iw.get(self.shot, self.ids_name, imas_occurrence=self.imas_occ)
            ids = self.ids
            for c, cname in enumerate(self.clist):
                ch, _ = cname_to_imas_ch_info(cname, ids=ids)
                self.rpos[c] = ids.channel[ch].line_of_sight.second_point.r
                self.zpos[c] = ids.channel[ch].line_of_sight.second_point.z
                self.apos[c] = ids.channel[ch].line_of_sight.second_point.phi

    def hdf5_path(self, cname):
        # hdf5 path of channel (equilibrium channels in group EQ_{occ}, e.g. EQ_1/W_MHD)
        if re.match(EQ_CNAME, cname):
            return 'EQ_{:d}/{:s}'.format(self.imas_occ, cname)
        return cname

    def expand_clist(self, clist):
        # IN : List of channel names (e.g. 'SECE01-08').
        # OUT : Expanded list (e.g. 'SECE01', ..., 'SECE08')
        exp_clist = []
        for cname in clist:
            m = re.match(r'^([A-Za-z_]+?)(\d+)-(\d+)$', cname)
            if m is None:
                exp_clist.append(cname)
            else:
                prefix, n1, n2 = m.group(1), m.group(2), m.group(3)
                for n in range(int(n1), int(n2) + 1):
                    exp_clist.append('{:s}{:0{:d}d}'.format(prefix, n, len(n1)))

        return exp_clist


#### cname to IMAS
def cname_to_imas_name_occ(shot, cname, tree=None):
    # IN : shot, node name (e.g. 'Ip', 'P_LH1', 'SECE01'), tree (IMAS occurrence of equilibrium channels)
    # OUT : ids_name, imas_occurrence
    imas_occ = 0

    if cname == 'Ip':
        ids_name = 'magnetics'
    elif cname == 'Bt':
        ids_name = 'tf'
    elif re.match(EQ_CNAME, cname):
        ids_name = 'equilibrium'
        imas_occ = int(tree) if tree is not None else 1 # 0 : NICE without, 1 : with polarimetry constraints (default)
    elif cname.startswith('P_LH'):
        ids_name = 'lh_antennas'
    elif cname.startswith('P_ICH'):
        ids_name = 'ic_antennas'
    elif cname == 'P_ECH':
        ids_name = 'ec_launchers'
    elif cname in ['P_rad', 'P_rad_core']:
        ids_name = 'bolometer'
    elif cname in ['P_Ohm', 'ne', 'Ti0']:
        ids_name = 'summary'
    elif cname == 'gas':
        ids_name = 'gas_injection'
    elif cname.startswith('SECE'):
        ids_name = 'ece' # slow 1D ECE
    elif cname.startswith('FECE'):
        ids_name = 'ece'
        imas_occ = 1 # fast 1D ECE
    elif cname.startswith('MC'):
        ids_name = 'magnetics'
        imas_occ = 1 # fast MHD acquisition
    elif re.match(VIS_CNAME, cname):
        ids_name = 'spectrometer_visible'
    else:
        raise ValueError('Unknown node {:s}'.format(cname))

    return ids_name, imas_occ


def check_imas_data(shot_number, path, occurrence=0):
    # True if IMAS path (e.g. 'ece/time', 'equilibrium/time') of occurrence has data (size > 1)
    # e.g. ece occurrence 1 : fast ECE, equilibrium occurrence 0 : NICE without polarimetry, 1 : with polarimetry
    # (partialGet prints a traceback and returns no value for an empty occurrence, e.g. no fast ECE)
    try:
        out = iw.partialGet(shot_number, '{:s}/size'.format(path), imas_occurrence=occurrence)
        return out is not None and int(out) > 1
    except Exception:
        return False


def cname_to_imas_ch_info(cname, ids=None):
    # IN : cname (e.g. 'Ip', 'P_LH1', 'SECE01', 'DI_4341_LODIVOU18'), ids (needed for visible spectroscopy)
    # OUT : channel index (0-based), line index (visible spectroscopy only; None if no line in the channel)
    ch, k = 0, None

    m = re.match(CH_CNAME, cname)
    if m is not None:
        ch = int(m.group(2)) - 1

    m = re.match(VIS_CNAME, cname)
    if m is not None:
        line_name, los_name = m.group(1), m.group(2)

        # channel index of LOS name
        los_list = [str(chan.name) for chan in ids.channel]
        if los_name not in los_list:
            raise ValueError('No LOS {:s} in spectrometer_visible'.format(los_name))
        ch = los_list.index(los_name)

        # line index of wavelength line name
        line_list = [str(line.name) for line in ids.channel[ch].grating_spectrometer.processed_line]
        if line_name in line_list:
            k = line_list.index(line_name)

    return ch, k


#### time index and normalization
def time_index(t, trange, atrange):
    # IN : time, trange, atrange
    # OUT : index range of trange (idx1:idx2; time may have gaps, e.g. MC triggered acquisition), index of atrange
    idx = np.where((t >= trange[0])*(t <= trange[1]))[0]
    aidx = np.where((t >= atrange[0])*(t <= atrange[1]))

    return int(idx[0]), int(idx[-1]+1), aidx


def normalize(v, av, norm):
    # IN : data in trange, data in atrange, norm (0 : none, 1 : trange average, 2 : atrange average)
    if norm == 1:
        v = v/np.mean(v) - 1
    elif norm == 2:
        v = v/np.mean(av) - 1

    return v


#### plasma time
def get_plasma_time(shot, frac=0.05, t_offset=32.2, verbose=1):
    # IN : shot, frac (threshold of W_MHD relative to its maximum), t_offset
    # OUT : [tstart, tend] [s] (t_offset subtracted), first and last time of W_MHD > frac*max(W_MHD)
    for occ in [1, 0]: # occurrence 0 if 1 does not exist
        w = WestData(shot, ['W_MHD'], tree=occ, t_offset=t_offset)
        w.get_data(trange=[-np.inf, np.inf], verbose=0)
        if w.data is not None: break
    if w.data is None:
        raise ValueError('No W_MHD for {:d}'.format(shot))

    t, v = w.time, w.data[0]
    idx = np.where(v > frac*np.nanmax(v))[0] # nan (e.g. no equilibrium) is excluded
    if len(idx) == 0:
        raise ValueError('No plasma for {:d}'.format(shot))

    plasma_trange = [t[idx[0]], t[idx[-1]]]
    if verbose == 1: print('Plasma time of {:d} : [{:.3f}, {:.3f}] s (W_MHD > {:g} x max), t_offset = {:.1f}'.format(shot, plasma_trange[0], plasma_trange[1], frac, t_offset))

    return plasma_trange


#### save to hdf5
def imas_to_hdf5(shot, trange=[0,25], t_offset=32.2, pos_trange=[5,6], clists=None):
    # save HDF5_CLIST data of shot from IMAS to hdf5 file
    # clists (e.g. [['P_rad', 'P_rad_core']]) : add (overwrite) only these to the existing file, without equilibrium
    mode = 'w' if clists is None else 'a'
    if clists is None: clists = HDF5_CLIST

    fname = "{:s}{:06d}/WEST.{:06d}.h5".format(WEST_PATH, shot, shot)

    # make directory if necessary
    path = '{:s}{:06d}'.format(WEST_PATH, shot)
    if os.path.exists(path) == False:
        os.makedirs(path)

    with h5py.File(fname, mode) as fout:
        if mode == 'w' or 'T_OFFSET' not in fout.attrs:
            fout.attrs['SHOT'] = shot
            fout.attrs['T_OFFSET'] = t_offset

        for clist in clists:
            # read data and positions averaged over pos_trange (flat-top) from IMAS
            try:
                g = WestData(shot, clist, trange=trange, t_offset=t_offset, read_hdf5=False)
                g.get_data(trange=trange)
                if g.data is None:
                    print('No data for {:d} : {}'.format(shot, clist))
                    continue
                g.channel_position(trange=pos_trange)
            except Exception as e:
                print('Failed {:d} : {}'.format(shot, clist))
                print(f"Error: {e}")
                traceback.print_exc()
                continue

            # add data
            for c, cname in enumerate(g.clist):
                if cname in fout:
                    del fout[cname] # overwrite
                grp = fout.create_group(cname)

                if c == 0:
                    tset = grp.create_dataset('TIME', data=g.time, compression='gzip')
                else:
                    grp['TIME'] = tset # hard link (no copy)
                grp.create_dataset('DATA', data=g.data[c,:], compression='gzip')

                # add channel positions and IMAS info as attributes
                grp.attrs['RPOS'] = g.rpos[c] # [m]
                grp.attrs['ZPOS'] = g.zpos[c] # [m]
                grp.attrs['APOS'] = g.apos[c] # [rad]
                grp.attrs['IDS_NAME'] = g.ids_name
                grp.attrs['OCCURRENCE'] = g.imas_occ
                grp.attrs['TRANGE'] = [g.time[0], g.time[-1]]
                grp.attrs['POS_TRANGE'] = pos_trange

            print('Added', g.clist)

    # add equilibrium (2D psi, q profile) of existing occurrences, only when the whole file is made
    for occ in ([0, 1] if mode == 'w' else []):
        if check_imas_data(shot, 'equilibrium/time', occurrence=occ) == False:
            print('No equilibrium occurrence {:d} for {:d}'.format(occ, shot))
            continue
        try:
            add_eq_to_hdf5(shot, t_offset=t_offset, occ=occ)
        except Exception as e:
            print('Failed {:d} : EQ_{:d}'.format(shot, occ))
            print(f"Error: {e}")
            traceback.print_exc()

    print('Saved', fname)


def add_eq_to_hdf5(shot, t_offset=32.2, occ=1):
    # add equilibrium (2D psi, q profile) of shot from IMAS to the existing hdf5 file, in group 'EQ_{occ}'
    # occ = 0 : NICE without, 1 : with polarimetry constraints
    # (existence of occ is not checked here; check with check_imas_data(shot, 'equilibrium/time', occurrence=occ) before)
    fname = "{:s}{:06d}/WEST.{:06d}.h5".format(WEST_PATH, shot, shot)
    if os.path.exists(fname) == False:
        print('No hdf5 file', fname)
        return

    equi = iw.get(shot, 'equilibrium', imas_occurrence=occ)
    wall = iw.get(shot, 'wall', imas_occurrence=0)

    with h5py.File(fname, 'a') as fout:
        eq_name = 'EQ_{:d}'.format(occ) # occurrence number in group name (e.g. EQ_1)
        if eq_name in fout:
            del fout[eq_name] # overwrite
        grp = fout.create_group(eq_name)

        grp.create_dataset('TIME', data=np.asarray(equi.time) - t_offset)                         # [nt], t_offset subtracted
        grp.create_dataset('R', data=np.asarray(equi.interp2D.r), compression='gzip')             # [nr, nz] [m]
        grp.create_dataset('Z', data=np.asarray(equi.interp2D.z), compression='gzip')             # [nr, nz] [m]
        grp.create_dataset('PSI', data=np.asarray(equi.interp2D.psi), compression='gzip')         # [nt, nr, nz] [Wb]
        grp.create_dataset('PSI_BND', data=np.asarray(equi.boundary.psi))                         # [nt] psi at separatrix
        grp.create_dataset('RHO_TOR', data=np.asarray(equi.profiles_1d.rho_tor), compression='gzip') # [nt, nrho] [m]
        grp.create_dataset('Q', data=np.asarray(equi.profiles_1d.q), compression='gzip')          # [nt, nrho]
        grp.create_dataset('R_IN', data=np.asarray(equi.profiles_1d.r_inboard), compression='gzip')   # [nt, nrho] [m] HFS midplane R of flux surface
        grp.create_dataset('R_OUT', data=np.asarray(equi.profiles_1d.r_outboard), compression='gzip') # [nt, nrho] [m] LFS midplane R of flux surface

        # wall (limiter) outline
        grp.create_dataset('WALL_R', data=np.asarray(wall.description_2d[0].limiter.unit[0].outline.r))
        grp.create_dataset('WALL_Z', data=np.asarray(wall.description_2d[0].limiter.unit[0].outline.z))

        grp.attrs['IDS_NAME'] = 'equilibrium'
        grp.attrs['OCCURRENCE'] = occ
        tset = grp['TIME']

        # EQ_{occ}/{VAR} channels of EQ_CHANNELS (same format as imas_to_hdf5), TIME hard-linked to EQ_{occ}/TIME
        for var, path in EQ_CHANNELS.items():
            cname = '{:s}/{:s}'.format(eq_name, var) # e.g. EQ_1/W_MHD (read with tree = 1)
            try:
                v = equi
                for node in path.split('/'):
                    v = getattr(v, node)
            except AttributeError:
                print('No equilibrium.{:s}, skip {:s}'.format(path.replace('/', '.'), cname))
                continue
            grp = fout.create_group(cname)

            grp['TIME'] = tset # hard link to EQ_{occ}/TIME (no copy)
            grp.create_dataset('DATA', data=np.asarray(v), compression='gzip')

            # add channel positions and IMAS info as attributes
            grp.attrs['RPOS'] = 0.0 # [m]
            grp.attrs['ZPOS'] = 0.0 # [m]
            grp.attrs['APOS'] = 0.0 # [rad]
            grp.attrs['IDS_NAME'] = 'equilibrium'
            grp.attrs['OCCURRENCE'] = occ
            grp.attrs['TRANGE'] = [tset[0], tset[-1]]

    print('Added EQ_{:d} with {:s} channels to {:s}'.format(occ, ', '.join(EQ_CHANNELS.keys()), fname))


if __name__ == "__main__":
    pass
    
    # import matplotlib
    # matplotlib.use('Agg')
    # import matplotlib.pyplot as plt
    # import seaborn as sns

    # #### check WEST data time series
    # shot = 65415
    # trange = get_plasma_time(shot, verbose=1)

    # #test each cname
    # # test_clist = [['Ip'], ['Bt'], ['P_LH1-2'], ['P_ICH1-3'], ['P_Ohm'], ['ne'], ['ECE01-32'], ['MC01', 'MC10', 'MC20', 'MC30', 'MC40']] # working okay
    # # test_clist = [['P_ECH'], ['Ti0'], ['gas']] # problematic
    # test_clist = HDF5_CLIST
    # if check_imas_data(shot, 'ece/time', occurrence=1) == False:
    #     test_clist = [cl for cl in test_clist if not any('FECE' in c for c in cl)]
    #     print('No fast ECE data for shot {:d}'.format(shot))

    # for clist in test_clist:
    #     print('========== {:s} =========='.format(clist[0]))
    #     g = WestData(shot=shot, clist=clist, trange=trange)
    #     g.get_data(trange=trange)
    #     if g.data is None:
    #         print('No data for {:s}'.format(clist[0]))
    #         continue

    #     print('clist = {}'.format(g.clist))
    #     print('time = [{:g}, {:g}] s, data shape = {}, fs = {:g}'.format(g.time[0], g.time[-1], g.data.shape, g.fs))
    #     print('data min = {:g}, max = {:g}'.format(np.nanmin(g.data), np.nanmax(g.data)))
    #     if np.any(g.rpos) or np.any(g.zpos):
    #         print('rpos = {}\nzpos = {}\napos = {}'.format(g.rpos, g.zpos, g.apos))

    #     plt.figure()
    #     for c, cname in enumerate(g.clist):
    #         plt.plot(g.time, g.data[c,:], label=cname)
    #     plt.xlabel('Time [s]')
    #     plt.ylabel(g.clist[0])
    #     plt.legend()
    #     plt.title('#{:d} {:s}'.format(shot, clist[0]))
    #     plt.tight_layout()
    #     plt.savefig('test_westdata_{:s}.png'.format(clist[0]), dpi=150)
    #     plt.close()

    


    #### check fast 1D ECE data existence
    # shot_list = [57928, 57929, 57931, 57932, 57934, 57935, 57971, 57972, 57973, 57977, 57979, 57980, 57982, 57983, 58072, 58073, 58074, 58075, 58079, 58080, 58081, 58082, 58083, 58084, 58086, 58087, 58088, 58090, 58091]
    # # shot_list = [65415, 65434, 65427]
    # for shot in shot_list:
    #     # if check_imas_data(shot, 'ece/time', occurrence=0):
    #     #     print('Shot {:d} has SLOW 1D ECE data'.format(shot))
    #     if check_imas_data(shot, 'ece/time', occurrence=1):
    #         print('Shot {:d} has FAST 1D ECE data'.format(shot))
    #     # if check_imas_data(shot, 'ece/time', occurrence=2):
    #     #     print('Shot {:d} has FAST 2D ECE data'.format(shot))
        



    #### check Bt and Ip values
    # shot_list = [57928, 57929, 57931, 57932, 57934, 57935, 57971, 57972, 57973, 57977, 57979, 57980, 57982, 57983, 58072, 58073, 58074, 58075, 58079, 58080, 58081, 58082, 58083, 58084, 58086, 58087, 58088, 58090, 58091]
    # Bt_list = []
    # Ip_list = []
    # for shot in shot_list:
    #     D = WestData(shot=shot, clist=['Bt'])
    #     D.get_data(trange=[5, 6])
    #     Bt_list.append(np.mean(D.data))

    #     D = WestData(shot=shot, clist=['Ip'])
    #     D.get_data(trange=[5, 6])
    #     Ip_list.append(np.mean(D.data))

    # for shot, Bt, Ip in zip(shot_list, Bt_list, Ip_list):
    #     print('Shot {:d} : Bt = {:g} T, Ip = {:g} MA'.format(shot, Bt, Ip/1e6))


        # B = FluctAna()
        # B.add_data(shot, dev='WEST', clist=['Bt'], trange=[3,4])
        # B.add_data(shot, dev='WEST', clist=['Ip'], trange=[3,4])




    # #### Plasma equilibrium and MC channel posistions check
    # shot = 57928
    # D = WestData(shot=shot, clist=['MC01-10'])
    # D.get_data(trange=[9,9.1])
    # # visible spectroscopy DI_4341 for lower divertor LOS (channels without data are removed)
    # # vis_obj = [('LODIVIN', 36), ('LODIVOU', 36)]
    # vis_obj = [('LODIVOU', 18)]
    # vis_clist = ['DI_4341_{:s}{:02d}'.format(obj, n) for obj, nmax in vis_obj for n in range(1, nmax+1)]
    # V = WestData(shot=shot, clist=vis_clist)
    # V.get_data(trange=[9,9.1], verbose=0)

    # # IMAS parameters
    # # For NICE example run:
    # run      = 0
    # occ_equi = 1 # Set occurrence 1 for equilibrium because it uses polarimetry constraints
    # occ_wall = 0
    # user     = 'imas_public'
    # machine  = 'west'

    # # Figures options
    # nbr_levels = 30  # Number isocontours for Psi (magnetic flux function)
    # Z_for_B_NORM = 49  # Position in Z for 1D plot in the cartesian grid

    # print('shot     =', shot)
    # print('run      =', run)
    # print('occ_equi =', occ_equi)
    # print('occ_wall =', occ_wall)
    # print('machine  =', machine)
    # print('user     =', user)

    # # Read equilibrium and wall IMAS data
    # equi = iw.get(shot, 'equilibrium', imas_run=run, imas_occurrence=occ_equi , \
    #                     imas_user=user, imas_machine=machine)
    # wall = iw.get(shot, 'wall', imas_run=run, imas_occurrence=occ_wall , \
    #                     imas_user=user, imas_machine=machine)

    # # Print number of time steps and fix time index to plot
    # print('Number times in equilibrium IDS =', len(equi.time))
    # ind_mid = int(0.5*len(equi.time))

    # # Add extra point to wall to close the wall contour for plot
    # wall_r = wall.description_2d[0].limiter.unit[0].outline.r
    # wall_z = wall.description_2d[0].limiter.unit[0].outline.z
    # wall_r = np.append(wall_r, wall_r[0])
    # wall_z = np.append(wall_z, wall_z[0])

    # # PLOTS
    # # Four subfigures are plotted
    # plt.ion()
    # fig = plt.figure(figsize=(20, 16))
    # sns.set_context('talk', font_scale=1.)
    # fig.subplots_adjust(left=0.1, right=0.9, bottom=0.1, top=0.9, wspace=0.6, hspace=0.6)
    # fig.suptitle('Equilibrium ' + 'Time = ' + '{:.3f}'.format(equi.time[ind_mid]) + ' - ' + \
    #             'Shot: ' + str(shot) + ', run: ' + str(run) + ', ' +  \
    #             'machine: ' + machine + ', user: ' + user, fontsize='medium')

    # # Top left plot - Safety Factor (q) as a function of rho_tor
    # axi = fig.add_subplot(2, 2, 1)
    # axi.plot(equi.profiles_1d.rho_tor[ind_mid, :], equi.profiles_1d.q[ind_mid, :], \
    #         '-b', label='q', linewidth=1)
    # axi.set_xlabel(r'$\rho_{tor}$ [m]')
    # axi.set_ylabel(r'$q$',)
    # axi.legend()
    # axi.grid(True)

    # # Top right plot - B_NORM at Z_for_B_NORM
    # B_NORM = np.sqrt(equi.interp2D.b_field_r[ind_mid, ...]**2. \
    #             + equi.interp2D.b_field_z[ind_mid, ...]**2. \
    #             + equi.interp2D.b_field_tor[ind_mid, ...]**2.)
    # axi = fig.add_subplot(2, 2, 2)
    # axi.plot(equi.interp2D.r[:, Z_for_B_NORM], B_NORM[:, Z_for_B_NORM], '-b',  \
    #         label='Z = ' \
    #         + '{:.2f}'.format(equi.interp2D.z[Z_for_B_NORM, Z_for_B_NORM])+' m', \
    #         linewidth=1)
    # axi.set_xlabel(r'$R$ [m]')
    # axi.set_ylabel(r'$B_{norm}$ [T]')
    # axi.legend()
    # axi.grid(True)

    # # Bottom left plot - Flux function (Psi) contours
    # axi = fig.add_subplot(2, 2, 3)
    # axi.set(aspect = 1)
    # levels_req = np.linspace(np.nanmin(equi.interp2D.psi[ind_mid, ...]), \
    #                         np.nanmax(equi.interp2D.psi[ind_mid, ...]), \
    #                         nbr_levels)
    # axi.contour(equi.interp2D.r, equi.interp2D.z, \
    #             np.squeeze(equi.interp2D.psi[ind_mid, ...]), \
    #             levels=levels_req, colors='blue', linestyles='-', \
    #             linewidths=0.5)
    # # Separatrix plot
    # axi.contour(equi.interp2D.r, equi.interp2D.z, \
    #             np.squeeze(equi.interp2D.psi[ind_mid, ...]), \
    #             levels=(equi.boundary.psi[ind_mid],), linestyles='-', colors='tab:red')
    # # Wall plot
    # axi.plot(wall_r, wall_z, color='k')
    # # MC probe locations
    # axi.plot(D.rpos, D.zpos, 'o', color='orange', markersize=5, label='MC probes')
    # for c, cname in enumerate(D.clist):
    #     axi.text(D.rpos[c], D.zpos[c], cname, color='orange', fontsize=8)
    # # visible spectroscopy measurement positions
    # axi.plot(V.rpos, V.zpos, 's', color='green', markersize=1, label='DI_4341 LOS')
    # for c, cname in enumerate(V.clist):
    #     axi.text(V.rpos[c], V.zpos[c], cname[8:], color='green', fontsize=3) # LOS name only
    # axi.set_title('Psi')
    # axi.set_xlabel('R [m]')
    # axi.set_ylabel('Z [m]')
    # axi.grid(True)

    # # Bottom right plot - B_NORM
    # axi = fig.add_subplot(2, 2, 4)
    # axi.set(aspect=1)
    # subfig = axi.contourf(equi.interp2D.r, equi.interp2D.z, B_NORM, \
    #                     cmap=plt.cm.viridis)
    # cb_labels = fig.colorbar(subfig, ax=axi, format='%.1f')
    # axi.set_title(r'$B_{norm}$ [T]')
    # axi.set_xlabel('R [m]')
    # axi.set_ylabel('Z [m]')
    # axi.grid(True)

    # # Save figure PNG format
    # time_fig = '{:.3f}'.format(equi.time[ind_mid])
    # plt.savefig('Python_equi_IMAS_Time' + time_fig.replace('.', 'P') \
    #             + '_Shot'+str(shot) + '_Run' + str(run) + '_' \
    #             + machine + '_' + user + '.png', dpi=200)



    # #### ECE channel posistions check
    # shot = 57980
    # D = WestData(shot=shot, clist=['SECE01-32'])
    # D.get_data(trange=[9,9.1])
    # ch_numbers = np.arange(1, 33)
    # plt.plot(D.rpos, ch_numbers, 'o', color='orange', markersize=5, label='ECE channels')
    # for c, cname in enumerate(D.clist):
    #     plt.text(D.rpos[c], ch_numbers[c], cname, color='orange', fontsize=8)
    # plt.xlabel('R [m]')
    # plt.ylabel('Channel number')
    # plt.title('ECE channel positions for shot {:d}'.format(shot))
    # plt.grid()
    # plt.savefig('ece_channel_positions_{:d}.png'.format(shot), dpi=150)
    # plt.close()



    # #### Download WEST data to hdf5 file 
    # shot_list = [57928, 57929, 57931, 57932, 57934, 57935, 57971, 57972, 57973, 57977, 57979, 57980, 57982, 57983, 58072, 58073, 58074, 58075, 58079, 58080, 58081, 58082, 58083, 58084, 58086, 58087, 58088, 58090, 58091, 65415, 65434, 65427]
    # # shot_list = [57928, 57929, 57931, 57932, 57934, 57935, 57971, 57972]
    # # shot_list = [57973, 57977, 57979, 57980, 57982, 57983, 58072, 58073]
    # # shot_list = [58074, 58075, 58079, 58080, 58081, 58082, 58083, 58084]
    # shot_list = [58086, 58087, 58088, 58090, 58091, 65415, 65434, 65427]
    # # shot_list = [57928]
    # for shot in shot_list:
    #     trange = get_plasma_time(shot, verbose=1)
    #     pos_range = get_plasma_time(shot, frac=0.7, verbose=1)
    #     imas_to_hdf5(shot, trange=trange, t_offset=32.2, pos_trange=pos_range)



    #### Add equilibrium to hdf5 file 
    # shot_list = [57928, 57929, 57931, 57932, 57934, 57935, 57971, 57972, 57973, 57977, 57979, 57980, 57982, 57983, 58072, 58073, 58074, 58075, 58079, 58080, 58081, 58082, 58083, 58084, 58086, 58087, 58088, 58090, 58091, 65415, 65434, 65427]
    # # shot_list = [57928, 57929, 57931, 57932, 57934, 57935, 57971, 57972]
    # # shot_list = [57973, 57977, 57979, 57980, 57982, 57983, 58072, 58073]
    # # shot_list = [58074, 58075, 58079, 58080, 58081, 58082, 58083, 58084]
    # shot_list = [58086, 58087, 58088, 58090, 58091, 65415, 65434, 65427]
    # for shot in shot_list:
    #     for occ in [0, 1]:
    #         if check_imas_data(shot, 'equilibrium/time', occurrence=occ) == False:
    #             print('No equilibrium occurrence {:d} for {:d}'.format(occ, shot))
    #             continue
    #         try:
    #             add_eq_to_hdf5(shot, t_offset=32.2, occ=occ)
    #         except Exception as e:
    #             print('Failed {:d} : EQ_{:d}'.format(shot, occ))
    #             print(f"Error: {e}")
    #             traceback.print_exc()



    ### Add selected channels to existing hdf5 file
    shot_list = [57928, 57929, 57931, 57932, 57934, 57935, 57971, 57972, 57973, 57977, 57979, 57980, 57982, 57983, 58072, 58073, 58074, 58075, 58079, 58080, 58081, 58082, 58083, 58084, 58086, 58087, 58088, 58090, 58091, 65415, 65434, 65427]
    for shot in shot_list:
        imas_to_hdf5(shot, trange=[0,25], t_offset=32.2, clists=[['P_rad', 'P_rad_core']])
