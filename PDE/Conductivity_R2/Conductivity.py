import sys
import os

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '../../')))

#We consider the points array as d x n matrix where d is the dimension and n is the number of points
import numpy as np
import matplotlib.pyplot as plt
import warnings
import math
import copy
from PDE.SmallInclusion import SmallInclusion
from cfg import mconfig
from FundamentalSols import green
from asymp.CGPT_methods import lbda

class Conductivity(SmallInclusion):
    __KsdS : np.ndarray
    __dGdn : np.ndarray
    _cnd : np.ndarray
    _pmtt : np.ndarray
    
    def compute_dGdn(self, sidx : list[int]):
        if sidx is None:
            sidx = self.cfg._Ns_total
        npts = self._D[0]._nb_points
        r = np.zeros((npts, self._nbIncl, len(sidx)))

        if isinstance(self.cfg, mconfig.Concentric) and self.cfg.nbDirac > 1:
            for i in range(self._nbIncl):
                toto = np.zeros((len(sidx), npts))
                for s in range(len(sidx)):
                    psrc = self.cfg.neutSrc[sidx[s]]
                    G = green.Green2D_Dn(psrc, self._D[i].points, self._D[i].normal)
                    neutCoeff = np.reshape(self.cfg.neutCoeff, (1, -1))
                    toto[s, :] = neutCoeff @ G
                r[:, i, :] = toto.T
        else:
            src = self.cfg.src(sidx)
            for i in range(self._nbIncl):
                toto = green.Green2D_Dn(src, self._D[i].points, self._D[i].normal)
                r[:, i, :] = toto.T

        r = r.reshape(npts * self._nbIncl, len(sidx))
        return r


    def compute_Phi(self, f, s):
        npts = self._D[0]._nb_points
        l = lbda(self._cnd, self._pmtt, f)
    #TO DO finish conductivity class and the CGPT methods
    def __init__(self, D, cfg):
        super().__init__(D, cfg)
    