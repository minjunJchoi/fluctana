import sys, os
sys.path.append(os.pardir)
from fluctana import *

import pickle
import math

shot = int(sys.argv[1]) # 20896
trange = eval(sys.argv[2]) # [1,10]
clist1 = sys.argv[3].split(',') # MC1T06

# add data
A = FluctAna()
if clist1[0][0:4] == 'ECEI':
    A.add_data(KstarEcei(shot=shot, clist=clist1), trange=trange, norm=0)
else:
    A.add_data(KstarMds(shot=shot, clist=clist1), trange=trange, norm=0)

# list data
A.list_data()

## spec
A.spec(dnum=0,cnl=range(len(A.Dlist[0].clist)), nfft=2048)
