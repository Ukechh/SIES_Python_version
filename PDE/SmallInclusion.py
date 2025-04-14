import numpy as np
from figure.C2Boundary.C2Boundary import C2Bound
from cfg import mconfig
from abc import ABC, abstractmethod

class SmallInclusion(ABC):
    _D : list[C2Bound]
    _nbIncl : int
    _cfg : str

    def __init__(self, D, cfg):
        if D.shape[0] == 1:
            self.addInclusion(D)
        else:
            for i in range(D.shape[0]):
                self.addInclusion(D[i])
        if not isinstance(cfg, mconfig):
            raise TypeError("must be an object of acq.mconfig")
        self.cfg = cfg

    def addInclusion(self,D):
        if not isinstance(D,C2Bound):
            raise TypeError('Type error: the inclusion must be an object of C2boundary')
        if self._nbIncl >= 1:
            if self._D[self._nbIncl-1]._nb_points != D._nb_points:
                raise ValueError('All inclusions must have the same number of boundary discretization points')
            
            if not self.check_inclusions(D):
                raise ValueError("Inclusions must be separated from each other")
        self._D.append(D)
        self._nbIncl += 1
    
    def check_inclusions(self, D):
        v = 1
        for i in range(self._nbIncl):
            v = v * D.isdisjoint(self._D [i] )
        return v
    
    def plot(self, *args, **kwargs):
        # Plot the inclusions first
        for n in range(self._nbIncl):
            self._D[n].plot(*args,**kwargs)
        # PLot the acquisition system
        self.cfg.plot(*args, **kwargs)
    @abstractmethod
    def data_simulation(self):
        pass