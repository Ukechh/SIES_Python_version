import numpy as np
import math
import matplotlib.pyplot as plt


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
    _data_dim : int

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

class Concentric(mconfig):
    nbDirac : int
    neutSrc : np.ndarray
    neutCoeff : np.ndarray
    pass