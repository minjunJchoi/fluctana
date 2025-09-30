import sys, os
sys.path.insert(0, os.pardir)
from fluctana import *

## HOW TO RUN
# For data before 2018
# ./python3 check_ecei_pos.py 10186 L,H,G

# For data since 2018
# ./python3 check_ecei_pos.py 22568 GT,GR,HT

# For data since 2024
# ./python3 check_ecei_pos.py 37389 GT,GR,HT save(if necessary)

shot = int(sys.argv[1])
if len(sys.argv) >= 3:
    dlist = sys.argv[2].split(',') 
elif len(sys.argv) == 2:
    if shot < 19391:
        dlist = ['L','H','G']
    else:
        dlist = ['GT','GR','HT']
savedata = True if sys.argv[-1] == 'save' else False

# plot channels
fig, (a1) = plt.subplots(1,1, figsize=(6,6))
for d, dname in enumerate(dlist):
    clist = ['ECEI_{:s}0101-2408'.format(dname)]
    if shot < 35000:
        E = KstarEcei(shot, clist)
    else:
        E = KstarEceiRemote(shot, clist, savedata=savedata)
         
    a1.plot(E.rpos, E.zpos, 'o')

    for c, cname in enumerate(E.clist):
            a1.annotate(cname[5:], (E.rpos[c], E.zpos[c]))

a1.set_title('ABCD positions (need corrections using syndia)')
a1.set_xlabel('R [m]')
a1.set_ylabel('z [m]')
plt.show()

