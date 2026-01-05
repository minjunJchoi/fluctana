import sys, os
sys.path.insert(0, os.pardir)
from fluctana import *
import argparse


## HOW TO RUN
# For data before 2018
# python3 check_ecei_pos.py -shot 10186 -dlist L H G

# For data since 2018
# python3 check_ecei_pos.py -shot 22568 -dlist GT GR 

# For data since 2024 (and also to save data in hdf5 format)
# python3 check_ecei_pos.py -shot 37389 -dlist GT GR HT --save 

parser = argparse.ArgumentParser(description="ECEI position")
parser.add_argument("-shot", type=int, default=22289, help="Shot number")
parser.add_argument("-dlist", nargs='+', type=str, default=None, help="ECEI device list")
parser.add_argument("--save", action="store_true", help="Save images")
a = parser.parse_args()

if a.dlist is not None:
    if a.shot < 19391:
        a.dlist = ['L','H','G']
    else:
        a.dlist = ['GT','GR','HT']

# plot channels
fig, (a1) = plt.subplots(1,1, figsize=(6,6))
for d, dname in enumerate(a.dlist):
    clist = ['ECEI_{:s}0101-2408'.format(dname)]
    if a.shot < 35000:
        E = KstarEcei(a.shot, clist)
    else:
        E = KstarEceiRemote(a.shot, clist, savedata=a.save)
         
    a1.plot(E.rpos, E.zpos, 'o')

    for c, cname in enumerate(E.clist):
            a1.annotate(cname[5:], (E.rpos[c], E.zpos[c]))

a1.set_title('ABCD positions (need corrections using syndia)')
a1.set_xlabel('R [m]')
a1.set_ylabel('z [m]')
plt.show()

