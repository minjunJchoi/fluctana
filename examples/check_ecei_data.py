import sys, os
sys.path.append(os.pardir)
from fluctana import *

## HOW TO RUN
# For data before 2018
# ./python3 check_ecei_data.py 10186 [15,16] L,H,G time

# For data since 2018
# ./python3 check_ecei_data.py 22568 [8,9] GT,GR,HT spec

shot = int(sys.argv[1]) 
trange = eval(sys.argv[2])
dlist = sys.argv[3].split(',') 
fflag = sys.argv[4]

# call fluctana
A = FluctAna()
# add data
for i, dname in enumerate(dlist):
    clist = ['ECEI_{:s}1201-1208'.format(dname)]
    A.add_data(KstarEcei(shot, clist), trange, norm=1)
    D = A.Dlist[i]

    fig = plt.figure(figsize=(10,18))
    for j in range(8):
        if j+1 == 1:
            ax1 = plt.subplot(8,1,j+1)
            plt.title(dname)
            if fflag == 'spec':
                axprops = dict(sharex = ax1, sharey = ax1)
            else:
                axprops = dict(sharex = ax1)
        else:
            plt.subplot(8,1,j+1, **axprops)
        
        x = D.time
        y = D.data[j,:]

        # do plot
        if fflag == 'spec':
            fs = round(1/(x[1] - x[0])/1000.0)*1000
            y = y - np.mean(y)
            Pxx, freqs, bins, im = plt.specgram(y, NFFT=2048, Fs=fs, noverlap=2048*0.9, xextent=[x[0],x[-1]], cmap=CM)
            plt.ylim([0, 250000])
            maxP = math.log(np.amax(Pxx+1e-14),10)*10
            minP = math.log(np.amin(Pxx+1e-14),10)*10
            dP = maxP - minP
            plt.clim([(minP+dP*0.15), maxP])
            chpos = '{:.1f}:{:.1f}'.format(D.rpos[j]*100, D.zpos[j]*100)
            plt.ylabel(chpos)
        else:
            plt.plot(x, y, color='k')
            chpos = '{:.1f}:{:.1f}'.format(D.rpos[j]*100, D.zpos[j]*100)
            plt.ylabel(chpos)

    plt.xlim([x[0], x[-1]])
    plt.xlabel('Time [s]')
    plt.show()
