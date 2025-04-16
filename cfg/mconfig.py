import numpy as np
import math
import matplotlib.pyplot as plt
from numpy import pi


class mconfig:
    __src : np.ndarray
    __rcv : np.ndarray

    _Ng : int
    _center : np.ndarray

    def __init__(self, Ng, center):
        self._Ng = Ng
        self._center = center
        
    _Ns : int
    _Nr : int
    _Ns_total : int
    _Nr_total : int
    data_dim : int

    #Getter methods

    def get_Ns(self):
        return self.__src.shape[1]
    def get_Nr(self):
        return self.__rcv.shape[1]
    def get_Ns_total(self):
        return self._Ns* self._Ng
    def get_Nr_total(self):
        return self._Nr*self._Ng
    def get_data_dim(self):
        return self._Ns * self._Nr * self._Ng
    #Utility functions
    def group(self, g):
        #Coordinates of sources and receivers of the g-th group
        src = self.__src[:,g]
        rcv = self.__rcv[:,g]
        return src, rcv
    def src_query(self,s):
        #Return the group index of a source s
        if not isinstance(s, int) or s > self._Ns_total or s < 1:
            raise ValueError("Non-scalar source index or source index out of range")
        gid = math.floor((s-1) / self._Ns) +1
        sid = s- (gid-1) * self._Ns
        return gid, sid

    def src(self, sidx):
        #Coordinates of sources in sidx
        r = np.zeros((2, len(sidx)))
        for s in range(len(sidx)):
            gid, sid = self.src_query(sidx[s])
            toto = self.__src[:, gid]
            r[:,s] = toto[:, sid]
        return r   
    def rcv(self,s):
        #Coordinates of receivers corresponding to s-th source
        gid, _ = self.src_query(s)
        return self.__rcv[:,gid]
    def all_src(self):
        sr = np.array([])
        for i in range(self._Ng):
            sr = np.hstack((sr,self.__src[:,i]))
        return sr
    def all_rcv(self):
        rc = np.array([])
        for i in range(self._Ng):
            rc = np.hstack((rc, self.__rcv[:,i]))
        return rc
    
    def plot(self, *args, **kwargs):
        for g in range(self._Ng):
            src, rcv = self.group(g)
            plt.plot(src[0, :], src[1, :], 'x')
            plt.plot(rcv[0, :], rcv[1, :], 'o')
        plt.legend()
        plt.axis('equal')
        plt.show()
    
    @staticmethod
    def src_rcv_circle(Na,N0, R0, Z, theta, aov = 2 * np.pi):
        Xs = np.zeros((2, N0*Na))
        Thetas = np.zeros((1, N0*Na))
        Xsblock = np.empty(Na, dtype=object)

        for n in range(Na):
            tt0 = (n / Na) * aov
            tt = tt0 + (np.arange(N0) / N0 ) * theta

            rr = R0 * np.vstack((np.cos(tt), np.sin(tt)))
            Xs[:,(n*N0):(n+1)*N0] = rr
            Thetas[(n*N0):((n+1)*N0)] = tt

            Xsblock = rr + Z.reshape(2,1)

        Xs += Z.reshape(2, 1)

        return Xs, Thetas, Xsblock

class Concentric(mconfig):
    nbDirac : int
    neutCoeff : np.ndarray
    neutRad : float

    radius_src : float
    radius_rcv : float
    equispaced = 0 

    def __init__(self, Z, Rs, Ns, Rr, Nr, viewmode = np.array([1,2*np.pi, 2*np.pi]), grouped = 0, neutCoeff = np.ones(1), neutRad=0.01):
        self.neutRad = neutRad
        if neutCoeff.shape[0] <= 1:
            neutCoeff = np.ones(1)
        else:
            if neutCoeff.sum() != 0 or abs(neutCoeff).max() ==0:
                raise ValueError("Coefficients of Diracs must be non zero and satisfy the zero sum condition")
            self.neutCoeff = neutCoeff
        Na = viewmode[0]
        theta = viewmode[1]
        aov = viewmode[2]

        Xs, _, Xsblock = mconfig.src_rcv_circle(Na, Ns, Rs, Z, theta, aov)
        Xr, _, Xrblock = mconfig.src_rcv_circle(Na, Nr, Rr, Z, theta, aov)

        self._center = Z
        self.radius_src = Rs
        self.radius_rcv = Rr

        if grouped:
            self.Ng = Na
            self.__src = Xsblock
            self.__rcv = Xrblock
        else:
            self.Ng = 1
            self.__src = Xs
            self.__rcv = Xr

        if self.Ng == 1 and theta == 2*np.pi:
            self.equispaced = 1
    
    def plot(self, *args, **kwargs):
        super().plot(*args, **kwargs)
        plt.plot(self._center[0], self._center[1], 'r*')
    
    def get_nbDirac(self):
        return self.neutCoeff.shape[0]
    
    @property
    def neutSrc(self, s=0):
        psrc = self.__src[s]
        if self.nbDirac == 1:
            return psrc
        else:
            r = np.zeros((2,self.nbDirac))
            L = self.radius_src * self.neutRad
            toto = psrc - self._center
            q = np.array([toto[1], -toto[0]])
            q = L * (q / np.linalg.norm(q))
            for n in range(self.nbDirac):
                r[:,n] = psrc + n * (q / self.nbDirac)
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
            self.__src = Xsblock
            self.__rcv = Xrblock
        else:
            self._Ng = 1
            self.__src = Xs
            self.__rcv = Xr
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
#Need to define wavelet classes in order to properly buil these...

class Coincided(Concentric):
    def __init__(self, Z, Rs, Ns, viewmode=np.array([1, 2 * np.pi, 2 * np.pi]), grouped=0, neutCoeff=np.empty(1), neutRad=0.01):
        super().__init__(Z, Rs, Ns, Rs, Ns, viewmode, grouped, neutCoeff, neutRad)

 