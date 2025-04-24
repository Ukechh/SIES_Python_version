import sys
import os

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '/../../')))

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
    __dGdn : np.ndarray
    _cnd : np.ndarray
    _pmtt : np.ndarray
    
    def compute_dGdn(self, sidx = None ):
        """
        Construct the right hand vector of the given source.
		If the source is not given, compute for all sources
        
        Parameters:
        -------------
        sidx:  list of source indices
            list[int]
        Returns:
        -------------
        dGdn: normal derivative of the fundamental solution with respect to the sources with indices in sidx
            ndarray
        """
        #Define the default value of sidx a list of source indices
        if sidx is None:
            sidx = list(range(self._cfg._Ns_total))
        elif isinstance(sidx,int):
            sidx = [sidx]
        #Number of points of the inclusion (All inclusions must have the same number of points)
        npts = self._D[0].nb_points
        #Create the return object
        r = np.zeros((npts, self._nbIncl, len(sidx)))
        #Test if the configuration is or not concentric
        if isinstance(self._cfg, mconfig.Concentric) and self._cfg.nbDirac > 1:
            
            neutCoeff = self._cfg.neutCoeff
            #Make neutCoeff into a row vector
            neutCoeff = np.reshape(neutCoeff, (1, -1))

            #Loop over each inclusion
            for i in range(self._nbIncl):
                toto = np.zeros((len(sidx), npts))
                #Loop over each source index, the 
                for s in range(len(sidx)):
                    psrc = self._cfg.neutSrc(sidx[s]) #We get the positions of the diracs ( shape (2, nbDirac) )
                    G = green.Green2D_Dn(psrc, self._D[i].points, self._D[i].normal) # Compute de normal derivative between boundary points and dirac points
                    toto[s , :] = neutCoeff @ G #neutCoeff has shape (1,nbDirac) and G has shape (nbDirac,npts) 

                r[:,i,:] = toto.T
        else:
            src = self._cfg.src(sidx)
            for i in range(self._nbIncl):
                toto = green.Green2D_Dn(src, self._D[0].points, self._D[0].normal)
                r[:, i, :] = toto.T
        
        r = r.reshape(npts*self._nbIncl, len(sidx)) #Returns a size (npts*nbIncl, indices) matrix
        return r


    def compute_Phi(self, f, s=None):
        """
        Construct the Phi vector for each given frequency and at source s if indicated
        
        Parameters:
        -------------
        f:  array of tested frequencies
            ndarray[float]
        s: source index
            int
        Returns:
        -------------
        P: List of (npts,1) vectors conaining the Phi vector for each inclusion, list of length nbIncl
            List[ndarray]
        """
        npts = self._D[0].nb_points
        l = lbda(self._cnd, self._pmtt, f)

        Amat, _ = make_system_matrix_fast(self.__KsdS, l) #Amat is the full system matrix of shape (npts*NbIncl, npts*Nbincl)
        
        if s is None:
            dGdn = self.__dGdn
        else:
            dGdn = self.compute_dGdn(s) #ndarray return type of size (npts*nbIncl, 1) as s is a single integer
        
        phi = np.linalg.solve(Amat, dGdn) #Returns X = (Amat)^-1 dGdn of shape (npts*nbIncl,1)
        P = []
        
        idx = 0
        for i in range(self._nbIncl):
            P.append(phi[idx:idx+npts])
            idx += npts
        return P

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
        self.__KsdS = make_block_matrix(self._D) #Block matrix (List of list of ndarrays) of shape (nbIncl,nbIncl) where each matrix is of shape (npts,npts)
        self.__dGdn = self.compute_dGdn() #nd array of shape (npts*NbIncl,1) 

    #Simulation Methods
    def data_simulation(self, f=None):
        """
        Simulate the data at the given frequencies
        
        Parameters:
        -------------
        f:  array of tested frequencies
            ndarray[float] 
        Returns:
        -------------
        out_MSR: list of data matrices, for each frequency
            list[ndarray]
        f: array of tested frequencies
            ndarray[float]
        """

        #Set the default value of freq
        if f is None:
            f = np.zeros(1)
        #Initialize the output and the index
        out_MSR = []
        for freq in f:
            Phi = self.compute_Phi(freq) #List of length NbIncl
            MSR = np.zeros((self._cfg._Ns_total, self._cfg._Nr), dtype=np.complex128) #Shape (Ns_total, Nr)

            for i in range(self._nbIncl):
                toto = np.zeros((self._cfg._Ns_total, self._cfg._Nr), dtype=np.complex128) #Pre initialize the sum matrix to a zeros matrix
                
                for s in range(self._cfg._Ns_total):
                    rcv = self._cfg.rcv(s) #Outputs a (2, Nr) ndarray
                    toto[s,:] = (SingleLayer.eval(self._D[i], Phi[i][:,s], rcv)).T #eval outputs a (Nr,1) ndarray so we transpose so we get the eval  
                MSR += toto

            out_MSR.append(MSR) 
        return out_MSR, f
    
    def calculate_field(self, f, s, z0, width, N):
        """
        Compute the total and background fields on a grid centered at `z0`, 
        excluding the inclusion boundaries, for a given frequency and source.

        Parameters:
        -----------
        f : float
            Frequency at which to evaluate the field.
        s : int
            Index of the source.
        z0 : ndarray of shape (2,)
            Center of the square computational domain.
        width : float
            Width of the square domain centered at `z0`.
        N : int
            Number of points for the grid.

        Returns:
        --------
        F : ndarray of shape (N, N)
            Total field (background + scattered) evaluated on the grid.
        F_bg : ndarray of shape (N, N)
            Background field only, without scatterers.
        Sx : ndarray of shape (N, N)
            X-coordinates of the evaluation grid.
        Sy : ndarray of shape (N, N)
            Y-coordinates of the evaluation grid.
        mask : ndarray of shape (N, N)
            Boolean mask excluding grid points that are inside any inclusion.
        """
        
        epsilon = width / ((N-1)*5) #Compute epsilon

        Sx, Sy, mask = self._D[0].boundary_off_mask(z0, width, N, epsilon) #Compute mask for the boundary Sx, Sy is of shape (N,N)

        Z = np.vstack((Sx.ravel(),Sy.ravel())) #Create a matrix of coordinates of shape (2, N**2)

        for i in range(1, self._nbIncl): #Create a mask to ignore all other boundaries
            _,_, toto = self._D[i].boundary_off_mask(z0, width, N, epsilon)
            mask *= toto
        Phi = self.compute_Phi(f,s) #Compute phi with the given source and frequencies (Output is a list of length NbIncl)

        Hs = green.Green2D(Z, self._cfg.src(s)) #Compute the background field, output has shape (N**2, 1)
        
        V = Hs.ravel().astype(np.complex128) #V has shape (N**2,)

        for i in range(self._nbIncl):
            ev = SingleLayer.eval(self._D[i], Phi[i], Z) #Phi[i] has shape (npts,1), Z has shape (2,N**2) output has shape (N**2,1)
            V += ev.ravel() 
        
        F = V.reshape((N,N)) #Total field
        F_bg = Hs.reshape((N,N)) #Background field

        return F, F_bg, Sx, Sy, mask
    
    def plot_field(self, s, F, F_bg, Sx, Sy, nbLine, *args, **kwargs):
        
        src = self._cfg._src[s]

        fig, axs = plt.subplots(2, 2, figsize=(12, 10))

        # Subplot 1: Real part of total field
        cs1 = axs[0, 0].contourf(Sx, Sy, F.real, nbLine)
        axs[0, 0].contour(Sx, Sy, F.real, nbLine, colors='k', linewidths=0.5)
        axs[0, 0].plot(src[0], src[1], 'gx', *args, **kwargs)
        
        for i in range(self._nbIncl):
            self._D[i].plot(ax=axs[0, 0], *args, **kwargs)
        axs[0, 0].set_title("Potential field u, real part")
        axs[0, 0].axis('image')
        fig.colorbar(cs1, ax=axs[0, 0])

        # Subplot 2: Imaginary part of total field
        cs2 = axs[0, 1].contourf(Sx, Sy, F.imag, nbLine)
        axs[0, 1].contour(Sx, Sy, F.real, nbLine, colors='k', linewidths=0.5)
        axs[0, 1].plot(src[0], src[1], 'gx', *args, **kwargs)
        for i in range(self._nbIncl):
            self._D[i].plot(ax=axs[0, 1], *args, **kwargs)
        axs[0, 1].set_title("Potential field u (or u-U), imaginary part")
        axs[0, 1].axis('image')
        fig.colorbar(cs2, ax=axs[0, 1])

        # Subplot 3: Real part of perturbed field
        cs3 = axs[1, 0].contourf(Sx, Sy, (F - F_bg).real, nbLine)
        axs[1, 0].contour(Sx, Sy, F.real, nbLine, colors='k', linewidths=0.5)
        axs[1, 0].plot(src[0], src[1], 'gx', *args, **kwargs)
        for i in range(self._nbIncl):
            self._D[i].plot(ax=axs[1, 0], *args, **kwargs)
        axs[1, 0].set_title("Perturbed field u-U, real part")
        axs[1, 0].axis('image')
        fig.colorbar(cs3, ax=axs[1, 0])

        # Subplot 4: Background potential (real part)
        cs4 = axs[1, 1].contourf(Sx, Sy, F_bg.real, nbLine)
        axs[1, 1].contour(Sx, Sy, F.real, nbLine, colors='k', linewidths=0.5)
        axs[1, 1].plot(src[0], src[1], 'gx', *args, **kwargs)
        axs[1, 1].set_title("Background potential field U, real part")
        axs[1, 1].axis('image')
        fig.colorbar(cs4, ax=axs[1, 1])

        plt.tight_layout()
        plt.show()

        






