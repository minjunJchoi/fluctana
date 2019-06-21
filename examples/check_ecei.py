import sys, os
sys.path.append(os.pardir)
from fluctana import *
import math

# For data before 2018
# python 10186 ECEI_L0101-2408 / ECEI_H0101-2408 / ECEI_G0101-2408

# For data since 2018
# python 21328 ECEI_GT0101-2408 / ECEI_GR0101-2408 / ECEI_HT0101-2408

shot = int(sys.argv[1]) # 19158
clist = sys.argv[2].split(',') # ECEI_G0101-2408

# select channels
ecei = KstarEcei(shot=shot, clist=clist)

# check channel positions 
ecei.show_ch_position()

