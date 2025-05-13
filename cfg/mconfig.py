import sys
import os

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '../')))

import numpy as np
import math
import matplotlib.pyplot as plt
from typing import Optional
from figure.C2Boundary.C2Boundary import C2Bound
from figure.Geom_figures import Banana

class mconfig:
    _src : list[np.ndarray]
    _rcv : list[np.ndarray]
    _Ng : int
    _center : np.ndarray

    def __init__(self, Ng, center, src = None, rcv = None):
        self._Ng = Ng
        self._center = center
    @property
    def _Ns(self):
        return self._src[0].shape[1]
    @property
    def _Nr(self):
        return self._rcv[0].shape[1]
    @property
    def _Ns_total(self):
        return self._Ns* self._Ng
    @property
    def _Nr_total(self):
        return self._Nr*self._Ng
    @property
    def _data_dim(self):
        return self._Ns * self._Nr * self._Ng
    
    #Getter method
    def get_src(self):
        #Returns the full list of source coordinates
        return self._src
    #Utility functions
    def group(self, g):
        #Coordinates of sources and receivers of the g-th group
        src = self._src[g]
        rcv = self._rcv[g]
        return src, rcv
    
    def src_query(self,s):
        #Return the group index the s-th source (Input from 0 to Ns_total - 1)
        gid = math.floor(s / self._Ns)
        sid = s % self._Ns
        return gid, sid

    def src(self, sidx):
        #Gives the coordinates of sources with source index in sidx (each element of sidx must be between 0 and Ns_total-1)
        if not isinstance(sidx, list):
            sidx = [sidx]
        r = np.zeros((2, len(sidx)))
        for i, s in enumerate(sidx):
            gid, sid = self.src_query(s)
            toto = self._src[gid] # toto becomes the collection of sources of group gid
            r[:,i] = toto[:, sid] #We extract the s-th source
        return r
  
    def rcv(self,s):
        #Coordinates of receivers corresponding to s-th source (s between 0 and Ns_total-1)
        gid, _ = self.src_query(s)
        
        return self._rcv[gid]
    
    def all_src(self):
        return np.concatenate([self._src[i] for i in range(self._Ng)], axis=1)
    
    def all_rcv(self):
        return np.concatenate([self._rcv[i] for i in range(self._Ng)], axis=1)
    
    def plot(self, *args, **kwargs):
        for g in range(self._Ng):
            src, rcv = self.group(g)
            plt.plot(src[0, :], src[1, :], 'x')
            plt.plot(rcv[0, :], rcv[1, :], 'o')
        plt.legend()
        plt.axis('equal')
        plt.show()
    
    @staticmethod
    def src_rcv_circle(Naa, N00, R0, Z, theta, aov = 2 * np.pi):
        """
        Generate sources/receivers placed on concentric arcs of radius
         R0 centered at a point Z. There are in total Na arcs which are equally
         distributed between the angular range [0, aov), and each has the same angular
         coverage which is theta.
        
        Parameters:
        -------------
         Na: number of arcs
         N0: number of sources/receivers per arc
         R0: radius of measurement circle
         Z: center of measurement circle
         theta: angle of each arc
         aov: (optional) total angle of view, 2*pi by default
        Returns:
        -------------
        Xs: the positions of sources/receivers, dimension 2 X (N0*Na)
        Thetas: angle of each source/receiver, dimension (N0*Na)
        Xslist: Xs in list. Xslist[n] is the coordinates Xs of the n+1-th arc
        """
        Na = int(Naa)
        N0 = int(N00)
        Xs = np.zeros((2,N0*Na))
        Z = Z.reshape((2,1))
        Thetas = np.zeros((N0*Na))
        Xslist = [np.empty((0,0)) for _ in range(Na)]

        for n in range(Na):
            tt0 = (n / Na) * aov
            tt = tt0 + (np.arange(N0) / N0) * theta

            rr = R0 * np.vstack((np.cos(tt), np.sin(tt)))

            Xs[:,n*N0:(n+1)*N0] = rr

            Thetas[n*N0:(n+1)*N0] = tt

            v = rr + Z 
            
            Xslist[n] = v
        Xs = Xs + Z

        return Xs, Thetas, Xslist

class Concentric(mconfig):
    neutCoeff : np.ndarray
    neutRad : float

    radius_src : float
    radius_rcv : float
    equispaced = 0 

    def __init__(self, Z, Rs, Ns, Rr, Nr, viewmode = np.array([2, 2*np.pi, 2*np.pi]), grouped = 0, neutCoeff : Optional[np.ndarray] = None, neutRad=0.01):
        self.neutRad = neutRad
        if neutCoeff is None or neutCoeff.shape[0] <= 1:
            neutCoeff = np.ones(1)
        else:
            if neutCoeff.sum() != 0 or abs(neutCoeff).max() == 0:
                raise ValueError("Coefficients of Diracs must be non-zero and satisfy the zero sum condition")
        self.neutCoeff = neutCoeff
        Na = viewmode[0]
        theta = viewmode[1]
        aov = viewmode[2]

        _, _, Xslist = mconfig.src_rcv_circle(Na, Ns, Rs, Z, theta, aov)
        _, _, Xrlist = mconfig.src_rcv_circle(Na, Nr, Rr, Z, theta, aov)

        self._center = Z
        self.radius_src = Rs
        self.radius_rcv = Rr

        if grouped:
            self._Ng = Na
            self._src = Xslist
            self._rcv = Xrlist
        else:
            self._Ng = 1
            self._src = Xslist
            self._rcv = Xrlist

        if self._Ng == 1 and theta == 2*np.pi:
            self.equispaced = 1
    
    def plot(self, *args, **kwargs):
        plt.plot(self._center[0], self._center[1], 'r*')
        super().plot(*args, **kwargs)
        
    @property
    def nbDirac(self):
        return self.neutCoeff.shape[0]
    
    def neutSrc(self, s):
        """
        Get the positions (Diracs) of the s-th source fulfilling the neutrality
		condition. The Diracs are distributed on a segment of length
		obj.neutRad * obj.radius_src in the tangent direction to the
		source circle.
        
        Parameters:
        -------------
        s: source index
            int
        Returns
        -------------
        neutSrc: Coordinates of the point sources
            ndarray
        """
        psrc = self.src(s)
        if self.nbDirac == 1:
            r = psrc
        else:
            r = np.zeros((2,self.nbDirac))
            L = self.radius_src * self.neutRad
            toto = psrc - self._center
            q = np.array([toto[1], -toto[0]])
            q = L * (q / np.linalg.norm(q))
            for n in range(self.nbDirac):
                r[:, n] = (psrc + n * (q / self.nbDirac)).T
        return r

class Planewave(mconfig):
    radius_src = 1
    def __init__(self, Z, Rr, Nr, Ns, viewmode=np.array([1, 2 * np.pi, 2 * np.pi]), grouped=0):
        Na = viewmode[0]
        theta = viewmode[1]
        aov = viewmode[2]
        Xs,_, Xsblock = mconfig.src_rcv_circle(Na, Ns, 1, np.ones(2), theta, aov)
        Xr,_, Xrblock = mconfig.src_rcv_circle(Na, Nr, Rr, Z, theta, aov)

        self._center = Z
        self.radius_rcv = Rr

        if grouped:
            self._Ng = Na
            self._src = Xsblock
            self._rcv = Xrblock
        else:
            self._Ng = 1
            self._src = Xsblock
            self._rcv = Xrblock
        if self._Ng == 1 and theta == 2*np.pi:
            self.equispaced = 1
    
    def plot(self, *args, **kwargs):
        for g in range(self._Ng):
            src, rcv = self.group(g)
            src = self.radius_rcv * src + self._center
            plt.plot(src[0, :], src[1, :], 'x')

            rcv = self.radius_src*rcv
            plt.plot(rcv[0, :], rcv[1, :], 'o')
        plt.plot(self._center[0], self._center[1], 'r*')
        plt.axis('equal')
        plt.show()

#TO DO: Interior ROI, Interior rcv, mconfig classes,
#Need to define wavelet classes in order to properly build these...
class Fish_circle(mconfig):

    dipole_prv : list # directions of dipole source, list of 2D vectors

    Omega0 : C2Bound


    def __init__(self, Omega, idxRcv, Z, Rs, Ns, aov, eorgan0=None, dipole0=None, d0 = np.zeros(2), impd = None):
        """
			Omega: body of the fish, a C2boundary object
			idxRcv: index of active receivers. If idxRcv is empty, then all boundary points will be receivers.
			Z: center of the measurement circle
			Rs: radius of the measurement circle
			Ns: number of sources
			aov: angle of view covered by all sources
			eorgan0: position of reference of the electric organ (optional)
			dipole0: direction of reference of the dipole source (optional)
			d0: offset of the dipole source wrt the center of mass (optional)
	        impd: impedance of the skin (optional)
        """
        if not isinstance(Omega, C2Bound):
            raise ValueError("Type Error, the domain must be a C2Boundary")
        
        self.Omega0 = Omega
        if impd is None:
            self.impd = 0.0
        else:
            self.impd = impd
        
        if dipole0 is None:
            self.dipole0 = Omega.get_pdirection()
        else:
            self.dipole0 = dipole0 / np.linalg.norm(dipole0)
        if eorgan0 is None:
            self.eorgan0 = Omega._center_of_mass + ((np.diag(d0) @ self.dipole0) * Omega.diameter / 2).reshape(-1,1)
        else:
            self.eorgan0 = eorgan0
        
        if isinstance(Omega, Banana):
            R = np.linalg.norm(Omega.center-Omega.curvature)
            alpha = 2*d0[0]*np.arctan(max(Omega.axis_a, Omega.axis_b) / R)
            self.dipole0 = np.array([-np.sin(alpha), np.cos(alpha)])
            self.eorgan0 = np.array([np.cos(alpha), np.sin(alpha)])*R
        
        self.aov = aov
        self._center = Z.reshape(2,1)

        if isinstance(Omega, Banana):
            self.radius = 0
        else:
            self.radius = Rs

        self.fpos , self.angl, _ = mconfig.src_rcv_circle(1, Ns, self.radius, Z.reshape(2,1), aov, 2*np.pi)
        
        if idxRcv.shape[0] == 0:
            self.idxRcv = np.arange(Omega.nb_points)
        else:
            self.idxRcv = idxRcv

        self._Ng = Ns
        self.dipole_prv = []
        self._rcv = []
        self._src = []
        for i in range(Ns):
            theta = self.angl[i]
            Body = (self.Omega0 < theta)
            Body = Body + self.fpos[:,i]
            self._rcv.append(Body._points[:, self.idxRcv])
            
            rot = np.array([[np.cos(theta) , - np.sin(theta)], [np.sin(theta), np.cos(theta)]])
            Ob = rot @ self.dipole0
            Sr = self.fpos[:,i].reshape(-1,1) + rot @self.eorgan0
            self.dipole_prv.append(Ob) #Direction of the s-th dipole
            self._src.append(Sr) # Position of the s-th electric organ         

    def all_dipole(self):
        r = np.array([])
        for g in range(self._Ng):
            r = np.hstack((r, self.dipole_prv[g]))
        return r
    
    def dipole(self, n): #Watch out as some of the parent class attributes are not initialized...
        if n > self._Ns_total-1 or n < 0:
            raise ValueError('Source index out of range')
        return self.dipole_prv[n]
    def Bodies(self, n):
        if n > self._Ns_total-1 or n < 0:
            raise ValueError('Source index out of range')
        return ((self.Omega0 < self.angl[n]) + self.fpos[:,n]) 
    
    def plot(self,ax = None, idx = None, *args, **kwargs):
        if ax is None:
            fig, ax = plt.subplots()

        if idx is None:
            idx = np.arange(self._Ns_total)

        for s in idx:
            rcv = self.rcv(s)  # (2, N) array
            ax.plot(rcv[0, :], rcv[1, :], '.', **kwargs)

            Omega = self.Bodies(s)
            Omega.plot(ax=ax)  # IMPORTANT: Omega needs an ax passed

            src = self.src(s)  # (2,) vector
            dp = self.dipole(s) * Omega.diameter * 0.25

            ax.plot(src[0], src[1], 'go', **kwargs)
            ax.quiver(src[0], src[1], dp[0], dp[1], angles='xy', scale_units='xy', scale=1, width=0.003, **kwargs)

        # Plot the overall center
        ax.plot(self._center[0], self._center[1], 'r*', **kwargs)

        ax.set_aspect('equal')

class Coincided(Concentric):
    def __init__(self, Z, Rs, Ns, viewmode=np.array([1, 2 * np.pi, 2 * np.pi]), grouped=0, neutCoeff=None, neutRad=0.01):
        super().__init__(Z, Rs, Ns, Rs, Ns, viewmode, grouped, neutCoeff, neutRad)

 