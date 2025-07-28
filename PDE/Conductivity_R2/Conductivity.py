import sys
import os

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '/../../')))

#We consider the points array as d x n matrix where d is the dimension and n is the number of points
from scipy.sparse.linalg import lsqr, LinearOperator
import numpy as np
import matplotlib.pyplot as plt
from PDE.SmallInclusion import SmallInclusion
from cfg import mconfig
from FundamentalSols import green
from Operators.Operators import SingleLayer, LmKstarinv
from asymp.CGPT_methods import lbda, make_system_matrix_fast, make_block_matrix
from PDE.Conductivity_R2.make_CGPT import make_linop_CGPT


class Conductivity(SmallInclusion):
    __dGdn : np.ndarray
    _cnd : np.ndarray
    _pmtt : np.ndarray
    drude : bool
    
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
                #Loop over each source index 
                for s in range(len(sidx)):
                    psrc = self._cfg.neutSrc(sidx[s]) #We get the positions of the diracs ( shape (2, nbDirac) )
                    G = green.Green2D_Dn(psrc, self._D[i].points, self._D[i].normal) # Compute de normal derivative between boundary points and dirac points
                    toto[s , :] = neutCoeff @ G #neutCoeff has shape (1,nbDirac) and G has shape (nbDirac,npts) 

                r[:,i,:] = toto.T
        else:
            src = self._cfg.src(sidx)
            for i in range(self._nbIncl):
                toto = green.Green2D_Dn(src, self._D[i].points, self._D[i].normal)
                r[:, i, :] = toto.T
        
        r = r.reshape(npts*self._nbIncl, len(sidx)) #Returns a size (npts*nbIncl, indices) matrix
        return r #For p = i*nb_Incl + j, r(:,s) = dGdn of the j-th inclusion at the i-th point for the s-th source 

    def compute_Phi(self, f, s=None):
        """"
        Construct the Phi vectors for each inclusion at given frequencies by solving the linear system Amat * Phi = dGdn.
        The (i,j)th element of the k-th term in the output represents the potential at the i-th point 
        of the k-th inclusion boundary, due to the j-th source (if multiple sources are provided).
        f : ndarray
            Array of tested frequencies for inclusions.
            Shape must be either (1,) or (NbIncl,).
            Used to compute the contrast parameter lambda.
        s : int, optional
            Source index for computing specific dGdn.
            If None, uses precomputed dGdn for all sources.
            Default is None.
        P : list[ndarray]
            List of length nbIncl containing the Phi vectors for each inclusion.
            Each element is a ndarray of shape (npts, 1) if single source,
            or shape (npts, Ns_total) if all sources are used,
            where:
            - npts is the number of discretization points per inclusion
            - Ns_total is the total number of sources
        Notes:
        The system solves: Amat * Phi = dGdn where:
        - Amat is the full system matrix of shape (npts*NbIncl, npts*NbIncl)
        - dGdn is of shape (npts*NbIncl, 1) for single source or (npts*NbIncl, Ns_total) for all sources
        - The solution is split into NbIncl vectors, one for each inclusion
        
       
        """
        npts = self._D[0].nb_points
        l = lbda(self._cnd, self._pmtt, f, self.drude)

        Amat, _ = make_system_matrix_fast(self.__KsdS, l) #Amat is the full system matrix of shape (npts*NbIncl, npts*Nbincl)
        
        if s is None:
            dGdn = self.__dGdn
        else:
            dGdn = self.compute_dGdn(s) #ndarray return type of size (npts*nbIncl, 1) as s is a single integer
        
        phi = np.linalg.solve(Amat, dGdn) #Returns X = (Amat)^-1 dGdn of shape (npts*nbIncl,1) or shape (npts*nbIncl, Ns_total)
        P = []
        
        idx = 0
        for i in range(self._nbIncl):
            P.append(phi[idx:idx+npts])
            idx += npts
        return P

    def __init__(self, D, cnd, pmtt, cfg, drude=True):
        super().__init__(D, cfg)
        cnd = np.array([cnd]) if isinstance(cnd, (int, float)) else cnd
        pmtt = np.array([pmtt]) if isinstance(pmtt, (int, float)) else pmtt

        if len(cnd) < self._nbIncl or len(pmtt) < self._nbIncl:
            raise ValueError("Conductivity and permittivity must be specified for each inclusion!")
        
        for i in range(self._nbIncl):
            if cnd[i] == 1 or cnd[i] < 0:
                raise ValueError("Conductivity constants must be positive and different from 1!")
            if pmtt[i] <= 0:
                raise ValueError("Permittivity constants must be positive!")
        
        self.drude = drude
        self._cnd = cnd
        self._pmtt = pmtt
        self.__KsdS = make_block_matrix(self._D) #Block matrix (List of list of ndarrays) of shape (nbIncl,nbIncl) where each matrix is of shape (npts,npts)
        self.__dGdn = self.compute_dGdn() #nd array of shape (npts*NbIncl,1) 

    #Simulation Methods
    def data_simulation(self, f=None):
        """
        Simulate the data of the perturbed field at the given frequencies
        
        Parameters:
        -------------
        f:  array of tested frequencies
            ndarray[float]
        !!!!!! WARNING: ONLY WORKS FOR ONE INCLUSION OR INCLUSIONS WITH THE SAME WORKING FREQUENCY
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
        if not isinstance(f, np.ndarray) :
            f = np.array([f])
        #Initialize the output and the index
        out_MSR = []
        for freq in f:
            Phi = self.compute_Phi(freq) #List of length NbIncl containing (npts,1) vectors
            MSR = np.zeros((self._cfg._Ns_total, self._cfg._Nr), dtype=np.complex128) #Shape (Ns_total, Nr)

            for i in range(self._nbIncl):
                toto = np.zeros((self._cfg._Ns_total, self._cfg._Nr), dtype=np.complex128) #Pre initialize the sum matrix to a zeros matrix
                
                for s in range(self._cfg._Ns_total):
                    rcv = self._cfg.rcv(s) #Outputs a (2, Nr) ndarray
                    #Here Phi[i] denotes the values of the phi function on the i-th inclusion's boundary
                    #Phi[i] [:,s] corresponds to the values of the phi function on the i-th inclusion, the column corresponding to the problem corresponding to the s-th source
                    toto[s,:] = (SingleLayer.eval(self._D[i], Phi[i][:,s], rcv)).T #eval outputs a (Nr,1) ndarray so we transpose so we get the eval (1,Nr) 
                MSR += toto #The MSR corresponds to the total perturbation

            out_MSR.append(MSR) #One MSR matrix per working frequency
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
    
    def calculate_v_hat(self, f, width, N):
        """
        Compute the inner expansion function  v(ξ) = ξ + S_B (λ I - K^*)^{-1} [v](ξ)
        on a grid centered at `z0` in the ξ = (x - z0)/δ coordinate frame.

        Parameters:
        -----------
        f : float
            Frequency for lambda computation
        z0 : ndarray of shape (2,)
            Center of the inclusion (defines ξ = (x - z0) / δ).
        width : float
            Width of computational square in ξ-space.
        N : int
            Number of points of the grid
        delta : float
            Scaling factor

        Returns:
        --------
        Vx : ndarray of shape (N, N)
            x-component of far-field map.
        Vy : ndarray of shape (N, N)
            y-component of far-field map.
        Sx, Sy : ndarrays of shape (N, N)
            Grid coordinates in ξ-space.
        mask : ndarray of shape (N, N)
            Boolean mask for valid evaluation points (outside boundary).
        """

        epsilon = width / ((N - 1) * 5)

        z0 = self._D[0]._center_of_mass

        Sx, Sy, mask = self._D[0].boundary_off_mask(z0, width, N, epsilon)

        Z = np.vstack((Sx.ravel(), Sy.ravel()))  # shape (2, N**2)

        for i in range(1, self._nbIncl):
            _, _, toto = self._D[i].boundary_off_mask(z0, width, N, epsilon)
            mask *= toto

        v = Z.copy().astype(np.complex128)

        for i in range(self._nbIncl):
            Sphi = self.v_hat(v, f, i)
            v += Sphi  # shape (2, N**2)

        Vx = v[0, :].reshape((N, N))
        Vy = v[1, :].reshape((N, N))

        delta = self._D[0].delta

        sSx = Sx / delta #We only stretch the grid as it is already centered to z0
        sSy = Sy / delta
        return Vx, Vy, Sx, Sy, mask, sSx, sSy

    def v_hat(self, x, freq, idx):
        """
        Computes the close field function v(ξ) = ξ + S_B (λ I - K^*)^{-1} [v](ξ) at the point ξ = (x-z0) / delta 
        where z0 is the center of the inclusion of index idx, at frequency freq and for delta the scaling coefficient
        """

        lam = lbda(cnd=self._cnd, pmtt=self._pmtt, freq=freq, drude=self.drude)
        
        l = lam[idx]

        D = self._D[idx]
        
        z0 = D._center_of_mass.reshape((2,1))

        delta = D.delta
        
        B = (D + (-z0) ) * (1 / delta) #Recenter and rescale

        ev = np.vstack((LmKstarinv.eval(B, B._normal[0,:], l), LmKstarinv.eval(B, B._normal[1,:], l)), dtype=np.complex128)
        
        xi = (x-z0) / delta
        
        Sphi = np.vstack((SingleLayer.eval(B, ev[0,:], xi), SingleLayer.eval(B, ev[1,:], xi)), dtype=np.complex128)

        v_hat = xi + Sphi

        return v_hat

    def plot_field(self, s, F, F_bg, Sx, Sy, nbLine, *args, **kwargs):
        
        src = self._cfg.src(s)
        # Use log scale for the imaginary part of the field
        logPerturbed = np.log10(np.abs(F-F_bg) + 1e-8)

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

        # Subplot 2: Log scale of perturbed field
        cs2 = axs[0, 1].contourf(Sx, Sy, logPerturbed, nbLine)
        axs[0, 1].contour(Sx, Sy, logPerturbed, nbLine, colors='k', linewidths=0.5)
        axs[0, 1].plot(src[0], src[1], 'gx', *args, **kwargs)
        for i in range(self._nbIncl):
            self._D[i].plot(ax=axs[0, 1], *args, **kwargs)
        axs[0, 1].set_title("Log scale of perturbed field |u-U|")
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

        # Subplot 4: Imaginary part of perturbed field
        cs4 = axs[1, 1].contourf(Sx, Sy, np.abs((F-F_bg).imag), nbLine)
        axs[1, 1].contour(Sx, Sy, F.real, nbLine, colors='k', linewidths=0.5)
        axs[1, 1].plot(src[0], src[1], 'gx', *args, **kwargs)
        for i in range(self._nbIncl):
            self._D[i].plot(ax=axs[1, 1], *args, **kwargs)
        axs[1, 1].set_title("Perturbed field u-U, imaginary part")
        axs[1, 1].axis('image')
        fig.colorbar(cs4, ax=axs[1, 1])

        plt.tight_layout()
        plt.show()

    def reconstruct_CGPT(self, MSR, ord, maxiter=1e6, tol=1e-5, symmode=False, method= 'pinv', L = None):
        """
        Reconstruct contracted GPTs from the Multistatic response matrix using either a least squares or a penrose inverse procedure
        Parameters:
        -------------------
        MSR: Multistatic response matrix, one matrix per working frequency
            list[ndarray]
        ord: Order of the CGPTs
            int
        maxiter: Max # of iterations when using least squares iterative method
            int
        tol: Tolerance parameter for least squares
            float
        symmmode: Boolean to indicate wether or not the CGPT is computed symmetric
            Bool
        method: indicates which methd to use
            string
        L: Linear operator of the CGPT, is a handle and computes L(X) = SXR

        Returns:
        ------------------
        CGPT: Corresponding CGPT matrix for each working frequency the matrix is 
            list
        res: Residual of the approximation using the desired method
            list
        rres: Normalized residuals
            list
        """
        if L is None:
            Linop = make_linop_CGPT(self._cfg, ord, symmode)
        else:
            Linop = make_linop_CGPT(self._cfg, ord, symmode)
        As = Linop['As']
        Ar = Linop['Ar']
        L = Linop['L']
        res = []
        rres= []
        CGPT_block = []
        if method == 'pinv':
            for t in range(len(MSR)):
                iArT = np.linalg.pinv(Ar).T
                iAs = np.linalg.pinv(As)
                CGPT_block.append(iAs @ MSR[t] @ iArT)
                res.append(np.linalg.norm(MSR[t] - Ar @ CGPT_block[t] @ As.conj().T, 'fro'))
                rres.append(res[t] / np.linalg.norm(MSR[t], 'fro'))
        elif method=='lsqr':
            def mv(x):
                return L(x,False)
            def rv(x):
                return L(x, True)
            ns, _ = As.shape
            nr, _ = Ar.shape  # type: ignore
            L_op = LinearOperator(shape= (nr*ns, 4*ord*ord), dtype=np.complex128, matvec= mv, rmatvec = rv) # type: ignore
            # Above the form of L_op is (nr*ns, 4*ord**2) because we need to solve the least squares ||SxR-MSR|| for x in (2ord, 2ord)
            # This needs to be done using the form ||AX-MSR(:)|| because of how the solvers work, thus to translate the operator SXR into AX
            # We do this form

            for t in range(len(MSR)):
                toto = MSR[t].reshape(-1,1)
                if toto.shape == (1,1):
                    toto = toto.reshape(1)
                X, _, _, r1norm, _, _, _, _, _, _= lsqr(L_op, toto, atol=tol, btol=tol, iter_lim=maxiter)
                cgpt = X.reshape((2*ord, 2*ord))
                res.append(r1norm)
                rres.append(r1norm / np.linalg.norm(MSR[t], 'fro'))
                if symmode:
                    cgpt = cgpt + cgpt.T
                CGPT_block.append(cgpt)
        results = {
            'CGPT': CGPT_block,
            'residuals': res,
            'normalized_residuals': rres
        }
        
        return results