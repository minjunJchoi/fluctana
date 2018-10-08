import sys, os
sys.path.append(os.pardir)
from fluctana import *

A = FluctAna()
A.add_data(KstarMir(shot=20119, clist=['MIR_0101']), trange=[9, 9.001])
A.add_data(KstarMds(shot=20119, clist=['CSS_I01:FOO']), trange=[9, 9.001], norm=0, res=0)

#A.mplot(dnum=0,cnum=[0],type='time')
plt.plot(A.Dlist[0].time, A.Dlist[0].data[0,:])
plt.plot(A.Dlist[1].time, A.Dlist[1].data[0,:])
plt.show()
