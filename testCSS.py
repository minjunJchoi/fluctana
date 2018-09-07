from fluctana import *

A = FluctAna()
A.add_data(KstarMir(shot=20119, clist=['MIR_0101']), trange=[1, 1.01])
A.add_data(KstarMds(shot=20119, clist=['CSS_I01']), trange=[1, 1.01], norm=0, res=0)

plt.plot(A.Dlist[0].time, A.Dlist[0].data)
plt.show()
