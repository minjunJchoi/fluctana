import sys, os
sys.path.append(os.pardir)
import kstarmds
import numpy as np

def get_tf_current(shot):
    print('Read I_TF from MDSplus')
    B = kstarmds.KstarMds(shot, ['PCITFMSRD'])
    _, data = B.get_data(trange=[3,6], norm=0)
    itf = np.mean(data)

    return itf

def get_ece_freq(shot):
    if shot < 12273:
        freqECE = [110,111,112,113,114,115,116,117,118,119,120,121,123,124,125,126,127,128,129,130,131,132,133,134,138,139,140,141,142,143,144,145,146,147,148,149,151,152,153,154,155,156,157,158,159,160,161,162,164,165,167,168,169,170,171,172,173,174,175,176,177,178,182,183,184,185,186,187,188,189,190,191,192,193,195,196]
        # print('YEAR 2014 ECE')
    elif (shot >= 12273) and (shot <= 14386):
        freqECE = [110,111,112,113,114,115,116,117,118,119,120,121,123,124,125,126,127,128,129,130,131,132,133,134,138,139,140,141,142,143,144,145,146,147,148,149,151,152,153,154,155,156,157,158,159,160,161,162]
        # print('YEAR 2015 ECE')
    elif (shot > 14386) and (shot <= 17356):
        freqECE = [110,111,112,113,114,115,116,117,118,119,120,121,123,124,125,126,127,128,129,130,131,132,133,134,138,139,140,141,142,143,144,145,146,147,148,149,151,152,153,154,155,156,157,158,159,160,161,162,78,79,81,82,83,84,85,86,87,88,89,90,91,92,96,97,98,99,100,101,102,103,104,105,106,107,109,110]
        # print('YEAR 2016 ECE')
    elif (shot > 17356) and (shot <= 19399):
        freqECE = [110,111,112,113,114,115,116,117,118,119,120,121,123,124,125,126,127,128,129,130,131,132,133,134,138,139,140,141,142,143,144,145,146,147,148,149,151,152,153,154,155,156,157,158,159,160,161,162,78,79,81,82,83,84,85,86,87,88,89,90,91,92,96,97,98,99,100,101,102,103,104,105,106,107,109,110]
        # print('YEAR 2017 ECE')
    elif shot >= 19400:
        freqECE = [110,111,112,113,114,115,116,117,118,119,120,121,123,124,125,126,127,128,129,130,131,132,133,134,138,139,140,141,142,143,144,145,146,147,148,149,151,152,153,154,155,156,157,158,159,160,161,162,78,79,81,82,83,84,85,86,87,88,89,90,91,92,96,97,98,99,100,101,102,103,104,105,106,107,109,110]
        # print('YEAR 2018 ECE')

    return freqECE

def get_ece_pos(shot):

    me = 9.1e-31        # electron mass
    e = 1.602e-19       # charge
    mu0 = 4*np.pi*1e-7  # permeability
    ttn = 56*16         # total TF coil turns
    harm = 2            # harmonic number

    itf = get_tf_current(shot)
    freqECE = get_ece_freq(shot)

    ece_rpos = {}
    for i, f in enumerate(freqECE):
        chname = 'ECE{:02d}'.format(i+1)
        
        # ece_rpos[chname] = 1.80*27.99*harm*bt/f # [m]
        ece_rpos[chname] = harm*e*mu0*ttn*itf/((2*np.pi)**2*me*f*1e9) # [m]

    return ece_rpos