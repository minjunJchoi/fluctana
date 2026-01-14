#!/usr/bin/env python
import numpy as np;
from geqdsk import Geqdsk;
from scipy import interpolate;
from scipy.integrate import ode;
from scipy.optimize import brentq;
from mpl_toolkits.mplot3d import Axes3D;
# import plasma_basic as pb;
import matplotlib.pyplot as plt;


"""
Class for equilibrium magnetic quantities from geqdsk file
Written by Tongnyeol Rhee (trhee@kfe.re.kr)
KFE(NFRI), Korea


======License===================
Copyright 2018 Tongnyeol Rhee

geqdsk_dk is free software: you can redistribute it and/or modify
it under the terms of the GNU Lesser General Public License as published by
the Free Software Foundation, either version 3 of the License, or
(at your option) any later version.

geqdsk_dk is distributed in the hope that it will be useful,
but WITHOUT ANY WARRANTY; without even the implied warranty of
MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
GNU Lesser General Public License for more details.

You should have received a copy of the GNU Lesser General Public License
along with geqdsk_dk.  If not, see <http://www.gnu.org/licenses/>.

A copy of the LGPL license is in COPYING.LESSER. Since this is based
on (and refers to) the GPL, this is included in COPYING.


"""

class geqdsk_dk(Geqdsk):
    def __init__(self,filename = None,gR0B0=False,BDS=True, IsReverse=False):
        """
        Contructor for drift kinetic equation equilibrium
        """
        if filename == None:
            self.data = {}
            Geqdsk.__init__(self);
        else:
            self.data = {}
            Geqdsk.__init__(self,filename=filename,gR0B0=gR0B0,BDS=BDS,IsReverse=IsReverse); 
            self.init_rbx()
            self.IsReverse=IsReverse;
            self.pol2tor_exist = False;

    def __del__(self):
        pass

    def inside_BDs(self,Rt,Zt):
           """
           Return True if inside boundary
           """
           if( self.BDs[0]<=Rt<=self.BDs[1] and self.BDs[2]<=Zt<=self.BDs[3]):
                  return True;
           else:
                  return False;
    def get_lcfs_psi(self):    #return last closed flux surface
      """
      Return last closed flux surface calculated by using interpolation method
      """
      from scipy import interpolate
      rbs = self.data['rbbbs'][0]
      zbs = self.data['zbbbs'][0]
      r   = self.data['r'][0]
      z   = self.data['z'][0]
      if not self.psi_inter:
          self.get_psi();
      nbbbs = self.data['nbbbs'][0]
      psis = np.zeros((nbbbs,),np.float64)
      for i in range(nbbbs): 
        psis[i] = self.f(rbs[i],zbs[i])
      return np.average(psis)
    
    def get_psi(self):        
      """
      Return psi interpolation on 2D
      psi_p = f(R, Z)
      """
      if not self.psi_inter:
          from scipy import interpolate
          r   = self.data['r'][0]
          z   = self.data['z'][0]
          self._psi_spline = interpolate.RectBivariateSpline(r,z, self.get('psirz').T, kx=5, ky=5, s=0);
          self.f= lambda Rq, Zq, *, dx=0, dy=0, grid=True: \
                  self._psi_spline(Rq, Zq, dx=dx, dy=dy, grid=grid)[0];
          self.psi_inter = True
      else:
          pass;
      return self.f
    def get_psi_normal(self):
        """
        Return psi_normal interpolation on 2D
        psi_p = f_normal(R,Z)
        """
        if not self.psi_inter_normal:
          from scipy import interpolate
          r   = self.data['r'][0]
          z   = self.data['z'][0]
          psi_mag = self.get('simag')
          psiw    = self.get('psiw')
          #self.f_normal=interpolate.interp2d(r,z,(self.get('psirz')-psi_mag)/psiw,kind='quintic')
          self._psi_spline_normal = interpolate.RectBivariateSpline(r,z, (self.get('psirz').T-psi_mag)/psiw, kx=5, ky=5, s=0);
          self.f_normal=lambda Rq, Zq, *, dx=0, dy=0, grid=True: \
                  self._psi_spline_normal(Rq, Zq, dx=dx, dy=dy, grid=grid)[0];
          self.psi_inter_normal = True
        else:
            pass;
        return self.f_normal
    
    
    
    def get_eps(self):
      """
      Return ellipticity
      """
      minr=np.min(self.data['rbbbs'][0])
      maxr=np.max(self.data['rbbbs'][0])
      minz=np.min(self.data['zbbbs'][0])
      maxz=np.max(self.data['zbbbs'][0])
      return (maxz-minz)/(maxr-minr)
    
    def get_delta(self):
      """
      Return dshape parameter
      """
      minr=np.min(self.data['rbbbs'][0])
      maxr=np.max(self.data['rbbbs'][0])
      acenter = (minr+maxr)*0.5
      a       = (maxr-minr)/2.
      
      minz=np.min(self.data['zbbbs'][0])
      c_ind = np.where(self.data['zbbbs'][0] == minz)
      cr = self.data['rbbbs'][0][c_ind]
      c =np.abs( acenter - cr)
      
      maxz=np.max(self.data['zbbbs'][0])
      c_ind = np.where(self.data['zbbbs'][0] == maxz)
      dr = self.data['rbbbs'][0][c_ind]
      d  = np.abs(acenter - dr)
    
      return (c+d)/2./a
    
    
    def get_a(self):
      """
      Return minor radius of last closed flux surface along R direction
      """
      minr=np.min(self.data['rbbbs'][0])
      maxr=np.max(self.data['rbbbs'][0])
      return (maxr-minr)/2
    
    def get_q95(self):       
      """
      Return q value of of 95% psi value
      """
      from scipy import interpolate
      psi = self.data['psi'][0]
      q   = self.data['qpsi'][0]
      f   = interpolate.interp1d(psi,q,kind='cubic')
      psi95 = np.min(psi)+(np.max(psi)-np.min(psi))*0.95
      return f(psi95)
    
    def q_inter(self, psi,der=0):
      """
      Return q value at a given normalized psi value or its derivative.
      Its derivative is dq/dpsi not dq/dpsi_normal
      optional keyword der = 0 : q value
                             1 : 1st derivative
                             2 : 2nd derivative  
      """
      return interpolate.splev(psi,self.data['q_inter'],der=der)/self.data['delpsi'][0]**der
    
    def q_RZ(self,R,Z):
        """
        Return q value at a given R and Z position
        """
        psi_N = self.f_normal(R,Z)
        return self.q_inter(psi_N);
    
    def q_root_inter(self, qval):
      """
      Return psi value at a given q value.
      """
      return interpolate.splev(qval,self.data['q_root_inter'])
    
    
    def g_inter(self, psi,der=0):
      """
      Return g value at a given normalized psi value
      Its derivative is dg/dpsi not dg/dpsi_normal
      optional keyword der = 0 : q value
                             1 : 1st derivative
                             2 : 2nd derivative  
      """
      return interpolate.splev(psi,self.data['g_inter'],der=der)/self.data['delpsi'][0]**der
    
    def B_field(self,R,Z,cyclic=False):
      """
      Return BR, BZ value at a given R, Z position.
      B   = grad psi * grad phi - g grad phi 
      """
      if not self.psi_inter:
         f = self.get_psi();
         del f;
      else:
         pass
      psi_RZ = (self.f(R,Z)[0]-self.data['simag'][0])/self.data['psiw'][0]
      if psi_RZ <= 1.00:
         BT = -self.g_inter(psi_RZ)
      else:
         BT = -self.data['bcentr'][0]*self.data['rcentr'][0]
    
      BR = -self.f(R,Z,dy=1)[0]
      BZ =  self.f(R,Z,dx=1)[0]
    
      if cyclic:
          return np.array([BR, BT, BZ])/R
      else: 
          return np.array([BR, BZ, BT])/R
    
    
    def B_theta(self,R,Z):
        """
        Return Btheta value at a given R,Z position
        """
        return 1./R * np.sqrt( self.f(R,Z,dy=1)[0]**2 + self.f(R,Z,dx=1)[0]**2)
        
    def Bxyz2(self,x,y,z):
        """
        Return bx, by, and bz at the x,y,z coordinate
        """
        r = np.sqrt(x**2+y**2);
        z = z;
        Bvec  = self.B_field(r,z);
        phi   = np.arctan2(y,x);
        bx    = Bvec[0]*np.cos(phi) - Bvec[2]*np.sin(phi);
        by    = Bvec[0]*np.sin(phi) + Bvec[2]*np.cos(phi);
        bz    = Bvec[1];
        #print 'br,bphi,cos,sin',Bvec[0],Bvec[2],np.cos(phi),np.sin(phi)
        return np.array([bx,by,bz])

    def Bxyz(self,x,y,z):
        """
        Return bx, by, and bz at the x,y,z coordinate
        """
        r = np.sqrt(x**2+y**2);
        z = z;
        Bvec  = self.B_field(r,z);
        phi   = np.arctan2(y,x);
        cosp  = np.cos(phi); sinp = np.sin(phi);
        [bx,by] = np.array( [[cosp,-sinp],[sinp,cosp]]).dot([Bvec[0],Bvec[2]])
        return np.array([bx,by,Bvec[1]])
    
    def local_pitch(self,R,Z):
        """
        Return local pitch defined as Bt/Bp
        """
        B = self.B_field(R,Z);
        Bp    = np.sqrt(B[0]**2+B[1]**2)
        Bt    = B[2]
        return Bt/Bp
    
    def get_ext_lim(self,rat):
        """
        Return limiter shape
        """
        R0    = self.get('rcentr')
        Z0    = 0.
        rlim  = self.get('rlim')
        zlim  = self.get('zlim')
        nlim  = len(rlim)
        rlim_new  = np.zeros(nlim,dtype='float')
        zlim_new  = np.zeros(nlim,dtype='float')
        for i in range(nlim):
            l = np.sqrt((rlim[i]-R0)**2+(zlim[i]-Z0)**2);
            phi   = np.arctan2(zlim[i]-Z0,rlim[i]-R0)
            rlim_new[i]   = l*(1.+rat)*np.cos(phi)+R0;
            zlim_new[i]   = l*(1.+rat)*np.sin(phi)+Z0;
        return rlim_new,zlim_new
    def fs(self,Rs,Zs):
        """
        Return psis from interpolation
        Input: Rs ,Zs
        Output: psis
        """
        psis=[]
        if Rs.shape != Zs.shape :
            print("Rs and Zs has different dimension")
            return 0;
    
        for i in range(Rs.shape[0]): 
            psis += [self.f(Rs[i],Zs[i])[0]];
        return np.array(psis);
    def B_fields(self,Rs,Zs):
        """
        Return B field from interpolation
        input: Rs, Zs
        output: Brs, Bzs,Bphis
        """
        Brs=[]; Bzs=[];Bphis=[];
        if Rs.shape != Zs.shape :
            print("Rs and Zs has different dimension")
            return 0;
    
        for i in range(Rs.shape[0]): 
            Br, Bz, Bphi  = self.B_field(Rs[i],Zs[i]);
            Brs += [Br]; Bzs += [Bz] ; Bphis += [Bphi]
    
        return np.array(Brs), np.array(Bzs), np.array(Bphis);

    def b_field(self,R,Z):
        """
        b field
        input: R, Z
        output: b vector (R, phi, Z)
        """
        B = self.B_field(R,Z)
        return np.array([B[0],B[2],B[1]])/np.sqrt(B.dot(B))
    
    def B_abs(self,R,Z):
        """
        B value
        input: R, Z
        output: B
        """
        return np.sqrt(self.B2(R,Z))

    def B2(self,R,Z):
        """
        B**2 value
        input: R, Z
        output: B**2
        """
        B=self.B_field(R,Z)
        return B.dot(B)

    def psin_RZ(self,psi_n, sigma = 1.):
        """
        Calculateor R, Z value having psi_n on the line from (Rmaxis, Zmaxis) 
            to (maximum R_lcfs, Zmaxis)
        input 
            psi_n: normalized psi value from 0 to 1
            ns  : number of grids of the line from Axis to boundary
        return 
            the value of [R, Z] having psi_n
        """
        if psi_n <=0.:
            print("Please input 0<psi_n<1")
            return 0.;
        """ OLD version
        rbbbs = self.get('rbbbs')   #lcfs R
        zbbbs = self.get('zbbbs')   #lcfs Z
        f = self.get_psi_normal();  #normalized poloidal flux
        R0 = self.get('rmaxis');    #magnetic axis R
        Z0 = self.get('zmaxis');    #magnetic axis Z
        Rmax = self.get('rbbbs').max();  #maximum R

        r = np.linspace(R0,Rmax, ns); 
        r_psi = np.zeros(ns);
        for i in range(ns):
            r_psi[i] = f(r[i],Z0);
        r_inter_psi = interpolate.interp1d(r, r_psi - psi_n, kind = 'cubic')
        try: 
            rv = brentq(r_inter_psi, R0, Rmax);
        except:
            print( "Error finding flux surface")
            print( "If you give close to 0 or 1 then please use diff. psi values")
            return 0.;
        else:
            return [rv,Z0]
        """
        #### New method """
        if sigma >= 0:
               return self.f_RZ_Nr(psi_n).item(), self.get('zmaxis');
        else:
               return self.f_RZ_Nl(psi_n).item(),self.get('zmaxis');

    def init_rbx(self): 
       ix1 = 0;
       nbbbs = self.data['nbbbs'][0];
       zmaxis = self.data['zmaxis'][0];
       zbbbs = self.data['zbbbs'][0]
       rbbbs = self.data['rbbbs'][0]
       izblr = np.zeros(2,dtype='int');
       for i in range(nbbbs-1): 
              if(zmaxis - zbbbs[i] == 0):
                     izblr[ix1] = i;
                     ix1+=1;
              elif((zmaxis - zbbbs[i])*(zmaxis- zbbbs[i+1])<0.): 
                     #print(" zbbbs[%d]-zmaxis =%f %f\n"%(i,zmaxis-zbbbs[i],zmaxis-zbbbs[i+1]));
                     izblr[ix1] = i; 
                     ix1+=1; 
       ####### get rbxl and rblr #######
       #double R2, R1, Z2, Z1, Rb1, Rb2, Zx;
       #print("ix1 is %d"%ix1);
       if(ix1==2):
              Zx  = zmaxis;
              ix2 = izblr[0];
              ix3 = izblr[1];
              R2 = rbbbs[ix2+1]; R1 = rbbbs[ix2];
              Z2 = zbbbs[ix2+1]; Z1 = zbbbs[ix2];
              if not R2 == R1:
                     Rb1 = (R2 - R1)/(Z2-Z1)*(Zx-(R2*Z1-R1*Z2)/(R2-R1));
              else:
                     Rb1 = R1;
              R2 = rbbbs[ix3+1]; R1 = rbbbs[ix3];
              Z2 = zbbbs[ix3+1]; Z1 = zbbbs[ix3];
              if not R2==R1: 
                     Rb2 = (R2 - R1)/(Z2-Z1)*(Zx-(R2*Z1-R1*Z2)/(R2-R1));
              else:
                     Rb2 = R1;
              #print("Rb1 is %f and Rb2 is %f\n"%(Rb1, Rb2));
              if(Rb1 >= Rb2):
              	self.rbxl = Rb2;
              	self.rbxr = Rb1;
              else:
              	self.rbxl = Rb1;
              	self.rbxr = Rb2;
       else:
       	print("plasma boundary has problems\n");
       	self.rbxl = rbbbs.min(); 
       	self.rbxr = rbbbs.max();
       
       	##high-field side initialization psi_R_l_spl;
       ddR = 0.02
       nR = 100
       
       rmaxis	= self.data['rmaxis'][0];
       simag	= self.data['simag'][0];
       psiw	= self.data['psiw'][0];
       
       dR = -(rmaxis-self.rbxl+ddR)/float(nR-1);
       Rs = np.zeros(nR,dtype='float');
       psi_Rs = np.zeros(nR,dtype='float');
       for i in range(nR):
       	Rs[i] = rmaxis+dR*float(i);
       	psi_Rs[i] = (self.f(Rs[i],zmaxis)-simag)/psiw;
       
       psi_Rs[0]	= 0.;
       self.f_RZ_Nl   = interpolate.interp1d(psi_Rs,Rs,kind='cubic')
       
       ###low-field side initialization psi_R_l_spl;
       dR = -(rmaxis-self.rbxr-ddR)/float(nR-1);
       for i in range(nR):
       	Rs[i] = rmaxis+dR*float(i);
       	psi_Rs[i] = (self.f(Rs[i],zmaxis)-simag)/psiw;
       psi_Rs[0] 	= 0.; 
       self.f_RZ_Nr   = interpolate.interp1d(psi_Rs,Rs,kind='cubic')



