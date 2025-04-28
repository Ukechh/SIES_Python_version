import numpy as np
from typing import Any
from figure.C2Boundary.C2Boundary import C2Bound
from cfg.mconfig import mconfig
from abc import ABC, abstractmethod

class SmallInclusion(ABC):
    _nbIncl : int
    _cfg : mconfig
    _D : list[C2Bound]

    def __init__(self, D, cfg):
        """
        Initialises a conductivity problem with D a list of inclusions and cfg a configuration of sources and receivers
        Parameters:
        -------------
        D:  list of inclusions in the problem
            list[C2Bound]
        cfg: Configuration of sources and receivers
            mconfig object
        """
        self._nbIncl = 0
        self._D = []
        if len(D) == 1:
            self.addInclusion(D[0])
        else:
            for i in range(len(D)):
                self.addInclusion(D[i])
        if not isinstance(cfg, mconfig):
            raise TypeError("must be an object of cfg.mconfig")
        self._cfg = cfg

    def addInclusion(self, D):
        if not isinstance(D, C2Bound):
            raise TypeError('Type error: the inclusion must be an object of C2boundary')
        
        if self._nbIncl >= 1:
            last_D = self._D[self._nbIncl - 1]
            if last_D.nb_points != D.nb_points:
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
        self._cfg.plot(*args, **kwargs)
    
    @abstractmethod
    def data_simulation(self, *args, **kwargs) -> Any:
        pass