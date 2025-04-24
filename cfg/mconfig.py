import numpy as np
import math
import matplotlib.pyplot as plt
from typing import Optional

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
        Thetas: angle of each source/receiver
        Xslist: Xs in list. Xslist[n] is the coordinates Xs of the n+1-th arc
        """
        Na = int(Naa)
        N0 = int(N00)
        Xs = np.zeros((2,N0*Na))
        Thetas = np.zeros((1, N0*Na))
        Xslist = [np.empty((0,0)) for _ in range(Na)]

        for n in range(Na):
            tt0 = (n / Na) * aov
            tt = tt0 + (np.arange(N0) / N0) * theta

            rr = R0 * np.vstack((np.cos(tt), np.sin(tt)))

            Xs[:,n*N0:(n+1)*N0] = rr

            Thetas[0,n*N0:(n+1)*N0] = tt

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
        super().plot(*args, **kwargs)
        plt.plot(self._center[0], self._center[1], 'r*')
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

#TO DO: Interior ROI, Interior rcv, Fish circle mconfig classes,
#Need to define wavelet classes in order to properly build these...

class Coincided(Concentric):
    def __init__(self, Z, Rs, Ns, viewmode=np.array([1, 2 * np.pi, 2 * np.pi]), grouped=0, neutCoeff=None, neutRad=0.01):
        super().__init__(Z, Rs, Ns, Rs, Ns, viewmode, grouped, neutCoeff, neutRad)

 