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
from Operators.Operators import SingleLayer
from asymp.CGPT_methods import lbda, make_system_matrix_fast, make_system_matrix, make_block_matrix

class Conductivity(SmallInclusion):
    __KsdS : np.ndarray
    __dGdn : np.ndarray
    _cnd : np.ndarray
    _pmtt : np.ndarray
    
    def compute_dGdn(self, sidx = None ):
        if sidx is None:
            sidx = np.arange(self.cfg._Ns_total)
        npts = self._D[0]._nb_points
        r = np.zeros((npts, self._nbIncl, len(sidx) ))

        if isinstance(self.cfg, mconfig.Concentric) and self.cfg.nbDirac > 1:
            for i in range(self._nbIncl):
                toto = np.zeros((len(sidx), npts))
                for s in range(len(sidx)):
                    psrc = self.cfg.neutSrc[:,sidx[s]].reshape(2,1)
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


    def compute_Phi(self, f, s=None):
        npts = self._D[0]._nb_points
        l = lbda(self._cnd, self._pmtt, f)
        Amat = make_system_matrix_fast(self.__KsdS, l)
        
        if s is None:
            dGdn = self.__dGdn
        else:
            dGdn = self.compute_dGdn(s)
        
        phi = np.linalg.solve(Amat, dGdn)
        Phi = np.empty(self._nbIncl, dtype=object)
        
        idx = 0
        for i in range(self._nbIncl):
            Phi[i] = phi[idx : idx+npts, :]
            idx = idx + npts
        return Phi

    def __init__(self, D, cnd, pmtt, cfg):
        super().__init__(D, cfg)

        if len(cnd) < self._nbIncl or len(pmtt) < self._nbIncl:
            raise ValueError("Conductivity and permittivity must be specified for each inclusion!")
        
        for i in range(self._nbIncl):
            if cnd[i] == 1 or cnd[i] < 0:
                raise ValueError("Conductivity constants must be positive and different from 1!")
            if pmtt[i] <= 0:
                raise ValueError("Permittivity constants must be positive!")
        
        self._cnd = cnd
        self._pmtt = pmtt
        self.__KsdS = make_block_matrix(self._D)
        self.__dGdn = self.compute_dGdn()

    #Simulation Methods
    def data_simulation(self, f):
        out_MSR = np.empty(len(f), dtype=object)
        f_idx = 0
        for freq in f:
            Phi = self.compute_Phi(freq)
            MSR = np.zeros((self._cfg._Ns_total, self._cfg._Nr))

            for i in range(self._nbIncl):
                toto = np.zeros((self._cfg._Ns_total, self._cfg._Nr))
                
                for s in range(self._cfg._Ns_total):
                    rcv = self._cfg._rcv[s]
                    toto[s,:] = SingleLayer.eval(self._D[i], Phi[i][:,s], rcv)
                MSR += toto

            out_MSR[f_idx] = MSR

            f_idx += 1 
        return out_MSR, f
    
    def calculate_field(self, f, s, z0, width, N):
        epsilon = width / ((N-1)*5)
        Sx, Sy, mask = self._D[0].boundary_off_mask(z0, width, N, epsilon)
        Z = np.vstack((Sx.ravel(),Sy.ravel()))
        for i in range(1, self._nbIncl):
            _,_, toto = self._D[i].boundary_off_mask(z0, width, N, epsilon)
            mask *= toto
        Phi = self.compute_Phi(f,s)

        Hs = green.Green2D(Z, self._cfg._src[s])

        V = Hs.reshape(1, -1)  # 1D row vector
        for i in range(self._nbIncl):
            V += SingleLayer.eval(self._D[i], Phi[i], Z)
        F = V.reshape(N, N)
        F_bg = Hs.reshape(N, N)
        return F, F_bg, Sx, Sy, mask
    
    def plot_field(self, s, F, F_bg, Sx, Sy, nbLine, *args, **kwargs):
        src = self._cfg._src[s]

        fig, axs = plt.subplots(2, 2, figsize=(12, 10))

        # Subplot 1: Real part of total field
        cs1 = axs[0, 0].contourf(Sx, Sy, F.real, nbLine)
        axs[0, 0].plot(src[0], src[1], 'gx', *args, **kwargs)
        for i in range(self._nbIncl):
            self._D[i].plot(ax=axs[0, 0], *args, **kwargs)
        axs[0, 0].set_title("Potential field u, real part")
        axs[0, 0].axis('image')
        fig.colorbar(cs1, ax=axs[0, 0])

        # Subplot 2: Imaginary part of total field
        cs2 = axs[0, 1].contourf(Sx, Sy, F.imag, nbLine)
        axs[0, 1].plot(src[0], src[1], 'gx', *args, **kwargs)
        for i in range(self._nbIncl):
            self._D[i].plot(ax=axs[0, 1], *args, **kwargs)
        axs[0, 1].set_title("Potential field u (or u-U), imaginary part")
        axs[0, 1].axis('image')
        fig.colorbar(cs2, ax=axs[0, 1])

        # Subplot 3: Real part of perturbed field
        cs3 = axs[1, 0].contourf(Sx, Sy, (F - F_bg).real, nbLine)
        axs[1, 0].plot(src[0], src[1], 'gx', *args, **kwargs)
        for i in range(self._nbIncl):
            self._D[i].plot(ax=axs[1, 0], *args, **kwargs)
        axs[1, 0].set_title("Perturbed field u-U, real part")
        axs[1, 0].axis('image')
        fig.colorbar(cs3, ax=axs[1, 0])

        # Subplot 4: Background potential (real)
        cs4 = axs[1, 1].contourf(Sx, Sy, F_bg, nbLine)
        axs[1, 1].plot(src[0], src[1], 'gx', *args, **kwargs)
        # Optional: show only one inclusion
        # self._D[0].plot(ax=axs[1, 1], *args, **kwargs)
        axs[1, 1].set_title("Background potential field U")
        axs[1, 1].axis('image')
        fig.colorbar(cs4, ax=axs[1, 1])

        plt.tight_layout()
        plt.show()
    






