import numpy as np
from scipy import signal
import math
import itertools

import pickle 

import matplotlib.pyplot as plt


def skewness(t, x, detrend=1):
    # normalize
    x = x / x[0]
        
    if detrend == 1:
        x = signal.detrend(x, type='linear')

    nx = (x - np.mean(x)) / np.std(x - np.mean(x))
    skew = np.mean(nx**3) / np.mean(nx**2)**(3.0/2.0)

    return skew


def kurtosis(t, x, detrend=1):
    # normalize
    x = x / x[0]

    if detrend == 1:
        x = signal.detrend(x, type='linear')

    nx = (x - np.mean(x)) / np.std(x - np.mean(x))
    kurt = np.mean(nx**4) / np.mean(nx**2)**2 

    return kurt


def hurst(t, x, bins=30, detrend=1, fitrange=[10,1000], **kwargs):
    # R/S method for fGm
    # (generalized hurst exponent for fBm)
    # axis
    bsize = int(1.0*len(t)/bins)
    ax = np.floor( 10**(np.arange(1.0, np.log10(bsize), 0.01)) )

    ers = np.zeros((bins, len(ax)))

    for b in range(bins):
        idx1 = b*bsize
        idx2 = idx1 + bsize

        sx = x[idx1:idx2]

        if detrend == 1:
            sx = signal.detrend(sx, type='linear')

        for i in range(len(ax)):
            ls = int( ax[i] ) # length of each sub-region
            ns = int( 1.0*ax[-1]/ls ) # number of sub-region

            delta = np.zeros((ls + 1, 1))
            for j in range(ns):
                jdx1 = j*ls
                jdx2 = jdx1 + ls

                ssx = sx[jdx1:jdx2]

                delta[1:,0] = np.cumsum(ssx) - np.cumsum(np.ones(ls))*sum(ssx)/ls

                r = np.max(delta) - np.min(delta)
                s = np.sqrt(np.sum(ssx**2)/ls - (np.sum(ssx)/ls)**2)

                ers[b,i] = ers[b,i] + r/s/ns

    # time lag axis
    dt = t[1] - t[0]
    tax = ax*dt*1e6 # [us]
    # ERS
    mean_ers = np.mean(ers, 0)
    std_ers = np.std(ers, axis=0)

    ptime = tax # time lag [us]
    pdata = mean_ers
    fidx = (fitrange[0] <= ptime) * (ptime <= fitrange[1])
    fit = np.polyfit(np.log10(ptime[fidx]), np.log10(pdata[fidx]), 1)
    fit_data = 10**(fit[1])*ptime**(fit[0])

    # Hurst exponent
    hurst_exp = fit[0]

    return tax, mean_ers, std_ers, hurst_exp, fit_data


def bp_prob(x, d=6, bins=1):
    # BP_probability
    nst = math.factorial(d) # number of possible states
    ax = np.arange(nst) + 1 # state number

    bsize = int(1.0*len(x)/bins)
    # print('For an accurate estimation of the probability, bsize {:g} should be considerably larger than nst {:g}'.format(bsize, nst))

    # possible orders
    orders = np.empty((0,d))
    for p in itertools.permutations(np.arange(d)):
        orders = np.append(orders,np.atleast_2d(p),axis=0)

    # calculate permutation probability
    val = np.zeros((nst, bins))

    for b in range(bins):
        idx1 = b*bsize
        idx2 = idx1 + bsize

        sx = x[idx1:idx2]

        jnum = len(sx) - d + 1
        for j in range(jnum):
            ssx = sx[j:(j+d)]

            sso = np.argsort(ssx)
            bingo = np.sum(np.abs(orders - np.tile(sso, (nst, 1))), 1) == 0
            val[bingo, b] = val[bingo, b] + 1.0/jnum

    pi = np.mean(val, 1) # bin averaged pi
    pierr = np.std(val, 1)

    # sort
    pio = np.argsort(-pi)
    val = pi[pio]     # bin averaged sorted pi
    std = pierr[pio]

    return ax, val, std


def ns_entropy(pi):
    nst = len(pi)
    pinz = pi[pi != 0] # to avoid blow up in entropy calculation
    spi = np.sum(-pinz * np.log(pinz)) # Shannon entropy
    nsent = spi/np.log(nst) # normalized Shannon entropy

    return nsent


def js_complexity(pi):
    # Jensen Shannon complexity with a given probability [Rosso PRL 2007]
    nst = len(pi)

    nsent = ns_entropy(pi)
    spi = nsent * np.log(nst) # Shannon entropy

    pe = 1.0*np.ones(nst)/nst
    spe = np.sum(-pe * np.log(pe))

    pieh = (pi + pe)/2.0
    spieh = np.sum(-pieh * np.log(pieh))

    # Jensen Shannon complexity
    jscom = -2.0*(spieh - spi/2.0 - spe/2.0)/((nst + 1.0)/nst*np.log(nst+1.0) - 2.0*np.log(2.0*nst) + np.log(nst))*nsent

    return jscom


def ch_measure(pi):
    # Jensen Shannon complexity, normalized Shannon entropy measure with a given BP probability [Rosso PRL 2007]
    # chaotic : moderate C and H, above fBm
    # stochastic : low C and high H, below fBm

    # normalized Shannon entropy
    nsent = ns_entropy(pi)

    # Jensen Shannon complexity
    jscom = js_complexity(pi)

    return jscom, nsent


def lmc_complexity(pi, nst):
    pe = np.ones(nst)/nst

    pinz = pi[pi != 0] # to avoid blow up in log
    nent = -1.0/np.log(nst)*np.sum(pinz * np.log(pinz))

    diseq = np.sum((pi - pe)**2)

    clmc = diseq*nent

    return clmc, nent


def complexity_limits(d):
    nst = math.factorial(d)

    pval = np.arange(1.0/nst,1,0.001)
    Hone = -1.0/np.log(nst)*(pval * np.log(pval) + (1.0-pval)*np.log((1.0-pval)/(nst-1.0)))
    Cone = np.zeros(len(Hone))
    for i in range(len(Hone)):
        pi = np.zeros(nst)
        pi[0] = pval[i]
        pi[1:] = (1.0 - pval[i])/(nst - 1.0)
        Cone[i] = js_complexity(pi)
    # plt.plot(Hone, Cone, 'k')

    Htwo = np.array([1])
    Ctwo = np.array([0])
    for n in range(nst-1):
        pmin = np.arange(0.001,1.0/(nst-n),0.001)
        # pmin = np.arange(0.001,0.1,0.001)
        Hext = -1.0/np.log(nst)*(pmin * np.log(pmin) + (1.0-pmin)*np.log((1.0-pmin)/(nst-n-1.0)))
        Cext = np.zeros(len(Hext))
        for i in range(len(Hext)):
            pi = np.zeros(nst)
            pi[0:n] = 0
            pi[n:(n+1)] = pmin[i]
            pi[(n+1):] = (1.0 - pmin[i])/(nst - n - 1.0)
            Cext[i] = js_complexity(pi)
        # plt.plot(Hext, Cext, 'k')
        Htwo = np.concatenate((Htwo, Hext), axis=0)
        Ctwo = np.concatenate((Ctwo, Cext), axis=0)
    idx = np.argsort(Htwo)
    Htwo = Htwo[idx]
    Ctwo = Ctwo[idx]

    return Hone, Cone, Htwo, Ctwo


def fbm_fgn_locus(d, data_dir='..'):
    try:
        with open('{:s}/chdata/ch_fbm_fgn_d{:d}.pkl'.format(data_dir, d), 'rb') as f:
            [c_fbm, h_fbm, c_fgn, h_fgn] = pickle.load(f)
    except:
        [c_fbm, h_fbm, c_fgn, h_fgn] = [0,0,0,0]

    return c_fbm, h_fbm, c_fgn, h_fgn


def fisher_measure(pi):
    # fisher information measure
    if ns_entropy(pi) == 0:
        f0 = 1.0
    else:
        f0 = 1.0/2.0

    fim = f0*np.sum( ( np.sqrt(pi[1:]) - np.sqrt(pi[:-1]) )**2 )

    return fim


def intermittency(t, x, bins=20, overlap=0.2, qstep=0.3, fitrange=[20.0,100.0], verbose=1, **kwargs):
    # intermittency parameter from multi-fractal analysis [Carreras PoP 2000]
    # this ranges from 0 (mono-fractal) to 1
    # add D fitting later

    # axis
    qax = np.arange(-2,8,qstep) # order axis
    N = len(x)
    Tmax = int( N/(bins - overlap*(bins - 1.0)) ) # minimum bin -> maximum data length
    Tax = np.floor( 10**(np.arange(1, np.log10(Tmax), 0.1)) ) # sub-data length axis
    nTax = Tax/N # normalized axis

    # data dimension
    eTq = np.zeros((len(Tax), len(qax)))
    K = np.zeros(len(qax))
    C = np.zeros(len(qax))
    D = np.zeros(len(qax))

    # first axes
    # x = signal.detrend(x, type='linear')

    if verbose == 1:
        fig = plt.figure(facecolor='w', figsize=(8,12))
        plt.subplots_adjust(bottom = 0.05, top = 0.95, hspace = 0.5, wspace = 0.3)
        axes1 = plt.subplot(5,1,1)
        plt.plot(t, x)
        plt.xlabel('Time [s]')

    ndxe = (x - np.mean(x))**2 / np.mean((x - np.mean(x))**2) # Eq.(7)

    for t, T in enumerate(Tax): # loop over different length T
        bins = int( N/(T - overlap*(T-1)) ) # number of bins with length T

        eT = np.zeros(bins)
        bstep = int(T*(1 - overlap))
        for j in range(bins):
            idx1 = j*bstep
            idx2 = int(idx1 + T)

            eT[j] = np.mean(ndxe[idx1:idx2]) # Eq.(9)

        # calculate moments
        for k, q in enumerate(qax):
            eTq[t, k] = np.mean(eT**(q)) # Eq.(10)

    # second axes
    if verbose == 1: plt.subplot(5,1,2)
    # calculate K
    for k, q in enumerate(qax):
        if verbose == 1: plt.plot(nTax, eTq[:,k], 'o')

        # fit range
        nT1 = fitrange[0]/N
        nT2 = fitrange[1]/N
        idx = (nT1 < nTax) * (nTax < nT2)

        lx = np.log(nTax[idx])
        ly = np.log(eTq[idx,k])

        fit = np.polyfit(lx, ly, 1)
        fit_func = np.poly1d(fit)
        K[k] = -fit[0]

        fx = np.arange(nTax.min(), nTax.max(), 1.0/N)
        fy = np.exp(fit_func(np.log(fx)))
        if verbose == 1:
            plt.plot(fx, fy)

            plt.axvline(x=nT1, color='r')
            plt.axvline(x=nT2, color='r')

    if verbose == 1:
        plt.title('Linear fit of loglog plot is -K(q)')
        plt.xlabel('T/N')
        plt.ylabel('eTq moments')
        plt.xscale('log')
        plt.yscale('log')

        # third axes
        plt.subplot(5,1,3)
        plt.plot(qax, K, '-o')
        plt.xlabel('q')
        plt.ylabel('K(q)')

    # calculate C and D
    for k, q in enumerate(qax):
        if (0.9 <= q) and (q <= 1.1):
            Kgrad = np.gradient(K, qax[1] - qax[0])
            C[k] = Kgrad[k]

            intmit = C[k]
            print('C({:g}) intermittency parameter is {:g}'.format(q, intmit))
        else:
            C[k] = K[k] / (q - 1)

        D[k] = 1 - C[k]

    if verbose == 1:
        # fourth axes
        plt.subplot(5,1,4)
        plt.plot(qax, C, '-o')
        plt.xlabel('q')
        plt.ylabel('C(q)')

        # fifth axes
        plt.subplot(5,1,5)
        plt.plot(qax, D, '-o')
        plt.xlabel('q')
        plt.ylabel('D(q)')

        plt.show()

    return intmit
