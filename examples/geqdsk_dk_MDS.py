#!/usr/bin/env python
import numpy as np;
from geqdsk_dk3 import geqdsk_dk;
import imp, os, sys;
"""
Class for equilibrium magnetic quantities from MDSplus equilibrium  
Written by Tongnyeol Rhee (trhee@kfe.re.kr)
KFE(NFRI), Korea
13 May 2020.

======License===================
Copyright 2020 Tongnyeol Rhee

geqdsk_dk_MDS is free software: you can redistribute it and/or modify
it under the terms of the GNU Lesser General Public License as published by
the Free Software Foundation, either version 3 of the License, or
(at your option) any later version.

geqdsk_dk_MDS is distributed in the hope that it will be useful,
but WITHOUT ANY WARRANTY; without even the implied warranty of
MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
GNU Lesser General Public License for more details.

You should have received a copy of the GNU Lesser General Public License
along with geqdsk_dk_MDS.  If not, see <http://www.gnu.org/licenses/>.

A copy of the LGPL license is in COPYING.LESSER. Since this is based
on (and refers to) the GPL, this is included in COPYING.

=============== Initialization ==============================================
Initialization
       shot = 18597        # Shot number to be tested
       time_i = 5.5        # Time to be considered
       treename = 'EFIT01' # or       treename = 'EFITRT1'
       geq =geqdsk_dk_MDS(shot,time_i,treename=treename)
       geq.init_operators();
After using this initialization you can use all functions described in geqdsk_dk and Geqdsk class
 """

# MDSplus server address
MDSPLUS_SERVER = os.environ.get('MDSPLUS_SERVER', 'mdsr.kstar.kfe.re.kr:8005')

class geqdsk_dk_MDS(geqdsk_dk):
    def __init__(self,shot, time_i, treename='EFITRT1'): 
           self.file_name = self.mds_geqdsk(shot,time_i,treename=treename);
           geqdsk_dk.__init__(self,filename = self.file_name, gR0B0 = False, BDS = True);
    def __del__(self):
        pass

    def write_profile(self,fp,data2):
       i=0;
       for data in data2 :
              fp.write(str("%16.9e"%data));
              i=i+1;
              if(i==5):
                     fp.write(str("\n"));
                     i=0;
    
       if(i>0): fp.write(str("\n"));
       return
    
    def mds_geqdsk(self,shot,time_i,treename='EFITRT1'):
       try:
           imp.find_module('MDSplus');
           found = True;
           import MDSplus as MDS;
       except ImportError:
           found = False;
           print("MDSplus module cannot be found\n")
           return 0;
       
        
       PARAMETERS=['\\bcentr','\\bdry','\\cpasma','\\epoten','\\ffprim',
                   '\\fpol','\\gtime','\\lim','\\limitr','\\mh','\\mw','\\nbdry',
                   '\\pprime','\\pres','\\psin','\\psirz','\\qpsi','\\r','\\rgrid1',
                   '\\rhovn','\\rmaxis','\\rzero','\\ssibry','\\ssimag','\\xdim','\\z',
                   '\\zdim','\\zmaxis','\\zmid'];
       
       geqdsk=[];
    
    #    mds= MDS.Connection("172.17.250.21:8005");
       mds= MDS.Connection(MDSPLUS_SERVER);       
       try:
           eq=mds.openTree(treename,shot);
       except: 
           print("Error #1")
       else:
           try:
               for signame in PARAMETERS:
                   temp = mds.get(signame).data();
                   geqdsk.append(temp); 
                   #print geqdsk[PARAMETERS.index(signame)];
               
           except Exception as e:
               print("Can not reading the signal\n Quit the program");          
               print(e);
               sys.exit(0);
           else:
               print("END of reading")
               #plt.show();
    
    
       mds.closeTree(treename,shot);
       
       index_time = PARAMETERS.index('\\gtime');
    
       if treename == 'EFITRT1':
           pass;
       else:
           geqdsk[index_time]/=1.0e3;
       len_time = len(geqdsk[index_time])
       DTmin = 1000.;
       for i in range(len_time):
           DT = np.abs(time_i - geqdsk[index_time][i])
           if  DT <=DTmin:
               DTmin = DT;
               t_index = i;
    
       print("Time which we write %10.5f\n"%geqdsk[index_time][t_index]);
       
    
       ###### Writing to file #######
       for i in [t_index]:
           time_fileout = geqdsk[index_time][i]*1000;
           os.makedirs('./data/EFIT', exist_ok=True)
           file_name='./data/EFIT/KSTAR_%s_g%06d.%06d'%(treename,shot,time_fileout);
           print('writing..',file_name)
           
           f=open(file_name,"w");
    
           nw=int(geqdsk[PARAMETERS.index('\\mw')][i]);
           nh=int(geqdsk[PARAMETERS.index('\\mh')][i]);
    
           rleft  = geqdsk[PARAMETERS.index('\\rgrid1')][i]; 
           rdim   = geqdsk[PARAMETERS.index('\\xdim')][i]; 
           rright = rdim+rleft;
           rcentr = geqdsk[PARAMETERS.index('\\rzero')][i]; 
           zdim   = geqdsk[PARAMETERS.index('\\zdim')][i]; 
           zmid   = geqdsk[PARAMETERS.index('\\zmid')][i];
    
           rmaxis = geqdsk[PARAMETERS.index('\\rmaxis')][i];
           zmaxis = geqdsk[PARAMETERS.index('\\zmaxis')][i];
    
           simag  = geqdsk[PARAMETERS.index('\\ssimag')][i]; 
           sibry  = geqdsk[PARAMETERS.index('\\ssibry')][i];
    
           bcentr = geqdsk[PARAMETERS.index('\\bcentr')][i];
           current= geqdsk[PARAMETERS.index('\\cpasma')][i];
    
           header = "%s  #%08d%08dms                       0"%(treename,shot,time_fileout)+str("%4i"%nw)+str("%4i"%nh)+"\n";
           f.write(header);
    
           f.write(str("%16.9e"%rdim)+str("%16.9e"%zdim)+str("%16.9e"%rcentr)+str("%16.9e"%rleft)+str("%16.9e"%zmid)+"\n");
           f.write(str("%16.9e"%rmaxis)+str("%16.9e"%zmaxis)+str("%16.9e"%simag)+str("%16.9e"%sibry)+str("%16.9e"%bcentr)+"\n");
           f.write(str("%16.9e"%current)+str("%16.9e"%simag)+str("%16.9e"%0)+str("%16.9e"%rmaxis)+str("%16.9e"%0)+"\n");
           f.write(str("%16.9e"%zmaxis)+str("%16.9e"%0)+str("%16.9e"%sibry)+str("%16.9e"%0)+str("%16.9e"%0)+"\n");
    
           # profile 
    
           self.write_profile(f,geqdsk[PARAMETERS.index('\\fpol')][i]);
           self.write_profile(f,geqdsk[PARAMETERS.index('\\pres')][i]);
           self.write_profile(f,geqdsk[PARAMETERS.index('\\ffprim')][i]);
           self.write_profile(f,geqdsk[PARAMETERS.index('\\pprime')][i]);
    
           #2D psi ...
    
           l=0;
           for w in range(nw):
              for h in range(nh):
                 f.write(str("%16.9e"%geqdsk[PARAMETERS.index('\\psirz')][i][w][h]));
                 l=l+1;
                 if(l==5):
                    f.write(str("\n"));
                    l=0;
                
           if(l>0): f.write(str("\n"))
    
           # qprofile
    
           self.write_profile(f,geqdsk[PARAMETERS.index('\\qpsi')][i]);
    
           # bdry
           
           nbdry = int(geqdsk[PARAMETERS.index('\\nbdry')][i]);
           nlimt = int(geqdsk[PARAMETERS.index('\\limitr')][i]);
           
           f.write(str("%4i"%nbdry)+str("%4i"%nlimt)+str("\n"));
    
           l=0         
           for w in range(nbdry):             
             f.write(str("%16.9e"%geqdsk[PARAMETERS.index('\\bdry')][i][w][0]));
             l=l+1;
             if(l==5):
                    f.write(str("\n"));
                    l=0;
             f.write(str("%16.9e"%geqdsk[PARAMETERS.index('\\bdry')][i][w][1]));
             l=l+1;
             if(l==5):
                    f.write(str("\n"));
                    l=0;
           if(l>0): f.write(str("\n"))
    
           l=0         
           for w in range(nlimt):
             f.write(str("%16.9e"%geqdsk[PARAMETERS.index('\\lim')][w][0]));
             l=l+1;
             if(l==5):
                    f.write(str("\n"));
                    l=0;
             f.write(str("%16.9e"%geqdsk[PARAMETERS.index('\\lim')][w][1]));
             l=l+1;
             if(l==5):
                    f.write(str("\n"));
                    l=0;
           if(l>0): f.write(str("\n"))
    
           
           f.close();
           return file_name;
        
if __name__=="__main__":
       shot = 18597
       time_i = 5.5
       treename = 'EFIT01'
#       treename = 'EFITRT1'
       geq =geqdsk_dk_MDS(shot,time_i,treename=treename)
       geq.init_operators();
       

