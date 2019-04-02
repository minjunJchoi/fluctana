import numpy as np

tor_ang = [1.6, 20.35, 35.35, 49.3, 70.5, 91.6, 110.35, 132.5, 142.7, 160.5, 181.6, 200.35, 215.35, 229.3, 257.3, 271.6, 290.35, 312.5, 319.3, 343.9]
pol_ang = [6.3465, 21.05855, 55.79, 65.32, 83.2, 93.71941, 99.02009, 113.0833, 120.05747, 135.00238, 153.48876, 206.60051, 225.05208, 239.96408, 247.00965, 261.02, 266.23091, 276.8, 294.68, 304.21, 338.88057, 353.6342]

def get_mc_pos():
    mc1t_apos = {}
    mc1p_apos = {}

    for i, a in enumerate(tor_ang):
        chname = 'MC1T{:02d}'.format(i+1)
        mc1t_apos[chname] = a/180.0*np.pi # [rad]

    for i, a in enumerate(pol_ang):
        chname = 'MC1P{:02d}'.format(i+1)
        mc1p_apos[chname] = a/180.0*np.pi # [rad]

    return mc1t_apos, mc1p_apos
