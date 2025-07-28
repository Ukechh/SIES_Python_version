import sys
import os

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '../../')))

#We consider the points array as d x n matrix where d is the dimension and n is the number of points
import numpy as np
import warnings
import math

#Plot packages
from matplotlib.ticker import MaxNLocator 
import matplotlib.pyplot as plt
from matplotlib import colormaps
#Utility classes and functions
from scipy.sparse.linalg import LinearOperator
from scipy.optimize import least_squares
from scipy.linalg import lstsq as lsqr
from Tools_fct.BEM_tools import P1_basis, interpolation 
from PDE.SmallInclusion import SmallInclusion
from cfg.mconfig import Fish_circle
from FundamentalSols import green
from Operators.Operators import SingleLayer, LmKstarinv, dSLdn, dDLdn, Ident, Kstar, DoubleLayer
from asymp.CGPT_methods import lbda, make_system_matrix_fast, make_block_matrix
from PDE.Conductivity_R2.make_CGPT import make_matrix_A
from Tools_fct.General_tools import cart2pol
from Tools_fct.linsys import SXR_op_symm_list, SXR_op_list

class Electric_Fish(SmallInclusion):

    def __init__(self, D, cnd, pmtt, cfg : Fish_circle, stepBEM, drude=True):
        """  Constructor of the Electric_Fish class.
        This class represents an electric fish with small exterior inclusions and handles
        the boundary element method (BEM) calculations for both the fish body and inclusions.
        Parameters
        ----------
        D : list
            List of exterior small inclusions around the fish.
        cnd : float
            Conductivity constant of the inclusions.
        pmtt : float
            Permittivity constant of the inclusions.
        cfg : Fish_circle
            Configuration object containing fish body geometry and impedance data.
        stepBEM : int
            Sampling step for the P1 boundary element method.
            Must be a divisor of the number of points in fish body discretization.
        Raises
        ------
        ValueError
            If cfg is not an instance of Fish_circle class.
            If stepBEM is not a valid divisor of the number of points in fish body.
        Notes
        -----
        - Initializes P1 basis functions for the fish body
        - Creates block matrices for inclusion interactions
        - Sets up boundary element types (P1 for fish body, P0 for inclusions)
        - Computes normal derivatives of harmonic functions """
        
        super().__init__(D, cfg)

        if not isinstance(cfg, Fish_circle):
            raise ValueError('The acquisition configuration must be an object of Fish_circle')
        self.drude = drude
        self.Omega = cfg.Omega0
        self.impd = cfg.impd
        self.cnd, self.pmtt = cnd, pmtt
        
        self.typeBEM1 = 'P1' #Type of boundary elements for fish body
        self.typeBEM2 = 'P0' #Type of boundary elements for inclusions

        if self.Omega.nb_points % stepBEM:
            raise ValueError('Sampling step for the fish is invalid')
        
        self.stepBEM1 = stepBEM
        self.stepBEM2 = 1
        
        #P1 elements for the body
        self.Psi = P1_basis(self.Omega.nb_points, self.stepBEM1) #Psi is of shape (nb_pts, nb_points/ stepBEM1)
        
        self.KsdS = make_block_matrix(self._D)
        if len(D) != 0:
             self.dHdn = self.compute_dHdn()
        else:
            self.dHdn = np.zeros((self.stepBEM1 + self.stepBEM2, self._cfg._Ns_total), dtype= np.complex128)

    def check_intersections(self):
        """
        Checks if any inclusions intersect with the fish body at any source position.
        
        Returns
        -------
        bool
            True if there are any intersections, False otherwise.
        """
        # Check intersections between inclusions and fish body at each source position
        if not isinstance(self._cfg, Fish_circle):
            raise ValueError('The acquisition configuration must be an object of Fish_circle')
        for s in range(self._cfg._Ns_total):
            Omega = self._cfg.Bodies(s)
            for incl in self._D:
                if not Omega.isdisjoint(incl): # Changed condition by removing the not 
                    return True
                    
        return False

    @property
    def Grammatrix(self):
        return self.Psi.T @ np.diag(self.Omega.sigma) @ self.Psi
    @property
    def nbBEM1(self):
        nb = math.floor(self.Omega.nb_points / self.stepBEM1)
        return nb
    @property
    def nbBEM2(self):
        return math.floor(self._D[0].nb_points / self.stepBEM2)

    def compute_dHdn(self, sidx=None):
        if not isinstance(self._cfg, Fish_circle):
            raise ValueError('The cfg must be an instance of Fish_circle')
        if sidx is None:
            sidx = np.arange(self._cfg._Ns_total)
        elif isinstance(sidx, int):
            sidx = np.array([sidx])
        H1 = np.zeros((self.nbBEM1, len(sidx)))
        H2 = np.zeros((self._nbIncl*self.nbBEM2, len(sidx)))
        
        for s in range(len(sidx)):
            Omega = self._cfg.Bodies(sidx[s])
            src = self._cfg.src(sidx[s])
            dipole = self._cfg.dipole(sidx[s])
            
            toto1 = Electric_Fish.source_vector(Omega, src, dipole)
            H1[:,s] = self.Psi.T @ (Omega.sigma * toto1)
            idx = 0
            for i in range(self._nbIncl):
                H2[idx:idx+self.nbBEM2, s] = Electric_Fish.source_vector(self._D[i], src, dipole)
                idx+=self.nbBEM2
        return np.vstack((H1,H2), dtype=np.complex128)

    def plot(self, idx=None, ax=None, **kwargs):
        if ax is None:
            fig, ax = plt.subplots(figsize=(8,6))

        for n in range(self._nbIncl):
            self._D[n].plot(ax=ax, **kwargs)

        self._cfg.plot(ax=ax, idx=idx,  **kwargs)

    def calculate_field(self, f, s, z0, width, N, fpsi_bg, fpsi, fphi):
        """
        Calculate the background potential field and the potential field due to the inclusion on a square region.

        Parameters:
        ----------------
        f : int
            Index of the frequency.
        s : int
            Index of the source.
        z0 : complex or tuple(float, float)
            Center of the square region (either as a complex number or (x0, y0)).
        width : float
            Width of the square region.
        N : int
            Number of points per side (grid resolution).
        fpsi_bg : list of ndarrays or ndarray
            Values of the function psi_bg returned by data_simulation.
        fpsi : list of ndarrays or ndarray
            Values of the function psi returned by data_simulation.
        fphi : list of ndarrays or ndarray
            Values of the function phi returned by data_simulation.

        Returns:
        ----------------
        F : ndarray
            Potential field u (total field).
        F_bg : ndarray
            Potential field U (background field).
        SX : ndarray
            X-coordinates of the rectangular region (meshgrid format).
        SY : ndarray
            Y-coordinates of the rectangular region (meshgrid format).
        """
        if not isinstance(self._cfg, Fish_circle):
            raise ValueError('The cfg must be an instance of Fish_circle')
        Omega = self._cfg.Bodies(s)
        src = self._cfg.src(s)
        dipole = self._cfg.dipole(s)

        epsilon = 1e-5

        Sx, Sy, mask = Omega.boundary_off_mask(z0, width, N, epsilon)
        Z = np.vstack((Sx.ravel(),Sy.ravel()))

        for i in range(1, self._nbIncl): #Create a mask to ignore all other boundaries
            _,_, toto = self._D[i].boundary_off_mask(z0, width, N, epsilon)
            mask *= toto
        
        Gx, Gy = green.Green2D_grad(Z, src)
        Hs = np.reshape(Gx * dipole[0] + Gy * dipole[1], (1, -1))

        fpsi_bg = fpsi_bg[:,s]
        fpsi = fpsi[f] [:,s]
        fphi = fphi[f] [:, s, :]
        # TOTAL FIELD
        SsT = SingleLayer.eval(Omega, fpsi, Z)
        DsT = DoubleLayer.eval(Omega,fpsi,Z)

        V = Hs + SsT - self.impd * DsT

        for i in range(self._nbIncl):
         #contribution of the i^th inclusion 
            V += SingleLayer.eval(self._D[i], fphi[:,i], Z)
        
        F = np.reshape(V, (N,N))
        
        #BACKGROUND FIELD
        Ss = SingleLayer.eval(Omega, fpsi_bg, Z)
        Ds = DoubleLayer.eval(Omega, fpsi_bg, Z)
    
        V = Hs + Ss - self.impd * Ds
        F_bg = np.reshape(V, (N,N))
        results = {
            'F' : F,
            'F_bg' : F_bg,
            'Sx' : Sx,
            'Sy' : Sy,
            'mask' : mask

        }
        return results
    
    def data_simulation(self, f):
        """
        Simulates the data of electric fish.

        Parameters
        -------------------
        freq : list or array-like
            A list of working frequencies used for the simulation.
        Returns
        -------------------
            - Variables starting with 'v' are coefficient vectors of boundary element basis.
            - Variables starting with 'f' are function values.
            - Variables starting with a capital letter are measurement matrices.

            The dictionary includes the following entries:
            - vpsi : list of arrays  # len(f) arrays of shape (nbBEM1, Ns_total)
                For each frequency n, each column is a solution vector ψ of the linear system (A.5) for a given source.
            - vphi[n][i] : list of arrays  # len(f) lists, each with nbIncls arrays of shape (nbBEM2, Ns_total)  
                For each frequency n and inclusion i, solution vectors φ.
            - fpsi[n] : list of 2D arrays  # len(f) arrays of shape (nbBEM1, Ns_total)
                Function values corresponding to vpsi.
            - fphi[n] : list of 3D arrays  # len(f) arrays of shape (nbBEM2, Ns_total, nbIncls)
                Function values corresponding to vphi.
            - vpsi_bg : ndarray  # shape (nbBEM1, Ns_total)
                Background solution vectors for equation (A.2).
            - fpsi_bg : ndarray  # shape (nbBEM1, Ns_total) 
                Function values for vpsi_bg.
            - fpp[n] : list of arrays  # len(f) arrays of shape (nbBEM1, Ns_total)
                The post-processed function defined in equation (4.7): (1/2 - K* - ξ dDdn)(dudn - dUdn).

            Measurements:
            
            - Current[n] : list of arrays  # len(f) arrays of shape (Ns_total, Nr)
                Surface current dUdn for each frequency n.
            - Current_bg : ndarray  # shape (Ns_total, Nr)
                Background surface current dUdn.
            - MSR[n] : list of arrays  # len(f) arrays of shape (Ns_total, Nr)
                Multi-static response matrix for CGPT reconstruction (see reference [2]).
            - SFR[n] : list of arrays  # len(f) arrays of shape (Ns_total, Nr)
                Space-Frequency response matrix dUdn - dUdn corresponding to the dipolar expansion (section 4.2 in reference [1]).
            - PP_SFR[n] : list of arrays  # len(f) arrays of shape (Ns_total, Nr)
                Post-processed response matrix, defined by equation (4.7).
                Note: the negative sign before ξ in eq (4.7) of [1] was a typo — it should be positive (+).

        Notes
        -------------------
        - For matrices named `fxxx`, each **column corresponds to a source**.
        - For measurement matrices (names starting with a capital letter), each **row corresponds to a source**.
        """
        if not isinstance(self._cfg, Fish_circle):
            raise ValueError('The cfg must be an instance of Fish_circle')
            
        if isinstance(f, (int, float)):
            f = np.array([f])

        # Initialize known variables
        SF = np.zeros((self.nbBEM1 + self._nbIncl*self.nbBEM2, len(f), self._cfg._Ns_total), dtype=np.complex128)
        SU = np.zeros((self.nbBEM1, self._cfg._Ns_total), dtype=np.complex128)
        PP = np.zeros((self.nbBEM1, len(f), self._cfg._Ns_total), dtype=np.complex128)
        
        # Convert to complex
        Gram = self.Grammatrix
        Psi_int = self.Psi.conj().T @ self.Omega.sigma
        
        # Optimize Phi_int calculation
        Phi_int = np.array([d.sigma for d in self._D]).T

        Psi_row = Psi_int.reshape(1, -1)
        Phi_row = Phi_int.reshape(1,-1)
        zeros_row1 = np.zeros((1, self.nbBEM2 * self._nbIncl), dtype= np.complex128)
        zeros_row2 = np.zeros((1, self.nbBEM1), dtype= np.complex128)
        # Main loop for different fish positions
        matrix_D_list = [make_system_matrix_fast(self.KsdS, lbda(self.cnd, self.pmtt, f[n], drude=self.drude))[0] for n in range(len(f))]
        
        for s in range(self._cfg._Ns_total):
            Omega = self._cfg.Bodies(s)

            # Build system matrices
            matrix_A = Electric_Fish.system_matrix_block_A(Omega, self.typeBEM1, self.stepBEM1, self.impd)
            matrix_B = Electric_Fish.system_matrix_block_B(Omega, self.typeBEM1, self.stepBEM1, self._D, self.typeBEM2, self.stepBEM2) 
            matrix_C = Electric_Fish.system_matrix_block_C(Omega, self.typeBEM1, self.stepBEM1, self._D, self.typeBEM2, self.stepBEM2, self.impd)

            # Background system resolution
            matrix_BEM = np.vstack([matrix_A, Psi_row])
            rhs = np.append(self.dHdn[:self.nbBEM1, s], 0)
            row, _, _, _= lsqr(matrix_BEM, rhs, lapack_driver='gelsy') #type: ignore
            SU[:, s] = row
            # Frequency dependent calculations
     
            # rhs is independent of frequency 
            rhs1 = np.concatenate([self.dHdn[:, s], [0, 0]])
            for n, matrix_D in enumerate(matrix_D_list):
                # Build combined system matrix efficiently
                matrix_BEM = np.vstack((
                    np.hstack((matrix_A, matrix_B)),
                    np.hstack((matrix_C, matrix_D)),
                    np.hstack((Psi_row, zeros_row1)),
                    np.hstack((zeros_row2, Phi_row))
                ))
               
                upd, _, _, _ = lsqr(matrix_BEM, rhs1, lapack_driver='gelsy') #type: ignore
                SF[:, n, s] = upd
                
                rhsn = matrix_B @ SF[self.nbBEM1:, n, s]
                PP[:, n, s] = -np.linalg.solve(Gram, rhsn)

        # Post-processing
        vpsi_bg = SU
        fpsi_bg = interpolation(self.Psi, SU)
        
        vpsi = [SF[:self.nbBEM1, m, :] for m in range(len(f))]
        fpsi = [interpolation(self.Psi, v) for v in vpsi]

        # Initialize vphi and MSR directly with list comprehensions
        vphi = [[SF[self.nbBEM1+j*self.nbBEM2:self.nbBEM1+(j+1)*self.nbBEM2, m, :] 
             for j in range(self._nbIncl)] for m in range(len(f))]
             
        fphi = [np.stack([v[j] for j in range(self._nbIncl)], axis=2) for m, v in enumerate(vphi)]
        
        fpp = [interpolation(self.Psi, PP[:,m,:]) for m in range(len(f))]
        
        # Calculate MSR more efficiently
        MSR = []
        for m in range(len(f)):
            msr = np.zeros((self._cfg._Ns_total, self._cfg._Nr), dtype=np.complex128)
            for j in range(self._nbIncl):
                for s in range(self._cfg._Ns_total):
                    rcv = self._cfg.rcv(s)
                    msr[s,:] += SingleLayer.eval(self._D[j], vphi[m][j][:,s], rcv)
            MSR.append(msr)

        # Final post-processing
        Current_bg = fpsi_bg.T[:,self._cfg.idxRcv]
        Current = [fpsi[n].T[:,self._cfg.idxRcv] for n in range(len(f))]
        SFR = [(Current[n] - Current_bg) for n in range(len(f))]
        PP_SFR = [fpp[n].T[:,self._cfg.idxRcv] for n in range(len(f))]

        return {
            'frequencies': f,
            'vpsi': vpsi,
            'vphi': vphi,
            'fpsi': fpsi, 
            'fphi': fphi,
            'vpsi_bg': vpsi_bg,
            'fpsi_bg': fpsi_bg,
            'fpp': fpp,
            'Current': Current,
            'Current_bg': Current_bg,
            'MSR': MSR,  
            'SFR': SFR,
            'PP_SFR': PP_SFR
        }

    def plot_field(self, s, F, F_bg, SX, SY, nbLine, subfig, *args, **kwargs):
        if not isinstance(self._cfg, Fish_circle):
            raise ValueError('The cfg must be an instance of Fish_circle')
        def create_plot(ax, data, title, inclusions=True, add_colorbar=True, custom_cmap=None):
            if not isinstance(self._cfg, Fish_circle):
                raise ValueError('The cfg must be an instance of Fish_circle')
            # Determine appropriate contour levels for better visualization
            vmin, vmax = np.nanmin(data).astype(np.float64), np.nanmax(data).astype(np.float64)
            # Prevent excessive zoom for near-constant fields
            if abs(vmax - vmin) < 1e-10:
                vmin, vmax = -1, 1

                if np.all(data >= 0):
                    cmap = colormaps['viridis']  # For positive data
                elif vmin < 0 and vmax > 0:
                    cmap = colormaps['RdBu_r']   # For diverging data
                else:
                    cmap = colormaps['coolwarm']  # Default diverging colormap
            else:
                cmap = custom_cmap
                
            # Create filled contours with improved styling
            contour = ax.contourf(SX, SY, data, nbLine, cmap=cmap, alpha=0.8, 
                        extend='both', vmin=vmin, vmax=vmax)
                
            # Add black contour lines
            ax.contour(SX, SY, data, nbLine, colors='black', linewidths=0.5, alpha=0.5)
                
            # Plot inclusions
            if inclusions:
                for n in range(self._nbIncl):
                    self._D[n].plot(ax=ax, **kwargs)
                
            # Plot main body (Omega)
            self._cfg.Bodies(s).plot(ax=ax, **kwargs)
            src = self._cfg.src(s)
               
            # Set appropriate zoom level - avoid excessive zoom
            grid_center_x = (np.max(SX) + np.min(SX)) / 2
            grid_center_y = (np.max(SY) + np.min(SY)) / 2
            
            # Set zoom to include furthest part from center plus Omega diameter
            zoom_radius_x = self.Omega.diameter * 2
            zoom_radius_y = self.Omega.diameter * 1.5
            ax.set_xlim(grid_center_x - zoom_radius_x, grid_center_x + zoom_radius_x)
            ax.set_ylim(grid_center_y - zoom_radius_y, grid_center_y + zoom_radius_y)
                
            # Add title and grid
            ax.set_title(title, pad=10, fontsize=12)
            ax.grid(True, linestyle='--', alpha=0.4)
                
            # Add colorbar
            if add_colorbar:
                cbar = plt.colorbar(contour, ax=ax, shrink=0.9)
                cbar.set_alpha(1)
                cbar.ax.yaxis.set_major_locator(MaxNLocator(nbins=6))
            #ax.plot(grid_center_x, grid_center_y, 'x', color ='r')
            #ax.plot(src[0], src[1], 'x', color= 'k')
            return contour
        
        if subfig:
            # Create 2x2 subplot grid with adjusted spacing
            fig, axs = plt.subplots(2, 2, figsize=(15, 12),
                                gridspec_kw={'hspace': 0.2, 'wspace': 0.2})
            fig.suptitle('Potential Field Analysis', y=0.98, fontsize=14, fontweight='bold')
            diff = F - F_bg
            # Plot 1: Real part of F
            contour1 = create_plot(axs[0, 0], np.real(F), 'Potential field u, real part')
            
            # Plot 2: Imaginary part of F
            contour2 = create_plot(axs[0, 1], np.imag(F), 'Potential field u, imaginary part')
            
            # Plot 3: Real part of perturbation
            R = np.real(diff)
            contour3 = create_plot(axs[1, 0], R, 'Perturbed field u-U, real part')  # Diverging colormap for difference
            
            # Plot 4: Background field with contours
            contour4 = create_plot(axs[1, 1], F_bg, inclusions=False, title= 'Background potential field U')
            
            # Apply consistent formatting to all subplots with proper axes labels
            for ax in axs.flat:
                ax.set_xlabel('x', fontsize=10)
                ax.set_ylabel('y', fontsize=10)
                
        else:
            # Single plot version with better layout
            fig, ax = plt.subplots(figsize=(10, 8))
            contour = create_plot(ax, np.real(F-F_bg), 'Perturbed field u-U, real part')
            ax.set_xlabel('x', fontsize=10)
            ax.set_ylabel('y', fontsize=10)    
        plt.tight_layout()
        
        return fig
    
    def reconstruct_CGPT(self, MSR, Current, ord, max_iter=1e5, tol=1e-3, symmode=0):
        CGPT_block = []
        res = []
        rres = []
        Q_list = []
        for t in range(len(MSR)):
            #Current varies with frequency so we make a new linear operator for each frequency
            op = Electric_Fish.make_linop_CGPT(self._cfg, Current[t], self.impd, ord, symmode)
            L = op['L']
            As = op['As']
            Ar = op['Ar']
            def mv(x):
                return L(x, False)
            def rv(x):
                return L(x, True)
            ns, _ = As.shape
            
            if isinstance(Ar, np.ndarray):
                nr, _ = Ar.shape
            else: 
                nr, _ = Ar[0].shape  # type: ignore
            
            L_op = LinearOperator(shape= (nr*ns, 4*ord*ord), dtype=np.complex128, matvec= mv, rmatvec = rv) # type: ignore
            Q = L_op.matmat(np.eye(L_op.shape[1]))
            
            toto = MSR[t].reshape(-1,1)
            if toto.shape == (1,1):
                toto = toto.reshape(1)
            #X, _, _, r1norm, _, _, _, _, _, _= lsqr(L_op, toto, atol=tol, btol=tol, iter_lim=maxiter)
            X, r1norm, _, _ = lsqr(Q, toto, lapack_driver='gelsy') # type: ignore
            cgpt = X.reshape((2*ord, 2*ord))
            
            res.append(r1norm)
            rres.append(r1norm / np.linalg.norm(MSR[t], 'fro'))
            
            if symmode:
                cgpt = cgpt + cgpt.T
            Q_list.append(Q)
            CGPT_block.append(cgpt)

        #Print out the results
        results= {
            'CGPT': CGPT_block, #type: ignore
            'residuals': res, #type: ignore
            'relative_residuals': rres, #type: ignore
            'Q' : Q_list
        }

        return results

    def reconstruct_PT(self, SFR, Current_bg, maxiter=100, tol= 1e-3, symmode=0):
        PT = []
        res = []
        rres = []
        op = Electric_Fish.make_linop_PT(self._cfg, Current_bg, self.impd, symmode)
        L = op['L']
        As = op['As']
        Ar = op['Ar']
        def mv(x):
            return L(x, False)
        def rv(x):
            return L(x, True)
        
        ns = As.shape[0]
        if isinstance(Ar, np.ndarray):
            nr, _ = Ar.shape
        else: 
            nr, _ = Ar[0].shape  # type: ignore

        L_op = LinearOperator(shape= (ns*nr, 4),dtype= np.complex128, matvec=mv, rmatvec = rv) #type: ignore
        Q = L_op.matmat(np.eye(L_op.shape[1]))

        for f in range(len(SFR)):
            toto = SFR[f].reshape(-1,1)
            if toto.shape == (1,1):
                toto = toto.reshape(1)
            #X, _, _, r1norm, _, _, _, _, _, _= lsqr(L_op, toto, atol=tol, btol=tol, iter_lim=maxiter)
            X, r1norm, _, _ = lsqr(Q, toto, lapack_driver='gelsy') # type: ignore
            pt = X.reshape((2, 2))
            
            res.append(r1norm)
            rres.append(r1norm / np.linalg.norm(SFR[f], 'fro'))
            PT.append(pt)
        results = {
            'PT': PT,
            'res': res,
            'rres': rres,
            'Q' : Q
        }
        return results

    @staticmethod
    def system_matrix_block_A(Omega, type1, step1, impd):
        Id = Ident(Omega, type1, step1)
        Ks = Kstar(Omega, type1, step1)
        dDldn = dDLdn(Omega, type1, step1)
        return (1/2 * Id.stiffmat + -1*Ks.stiffmat + impd * dDldn.stiffmat).toarray() # SHape (N/s1, N/s1), where N is the number of pts in Omega and s1 is the BEM1 step

    @staticmethod
    def system_matrix_block_B(Omega, type1, step1, D, type2, step2):
        nbIncl = len(D)
        toto = []
        for i in range(nbIncl):
            dSldn = dSLdn(D[i],type2, step2, Omega, type1, step1)
            toto.append(-1*dSldn.stiffmat)
        return np.hstack(toto)

    @staticmethod
    def system_matrix_block_C(Omega, type1, step1, D, type2, step2, impd):
        """
        Operator -dS_{D_l}dn restricted on Omega

        Parameters:
        ------------
        Omega : Fish body
            C2Bound
        typex : BEM type P0 or P1
        stepx : step size for BEM
        D: List of inclusions
            list[np.ndarray]

        Returns:
        --------------
        Matrix block A of the Operator
            np.ndarray
        """
        nbIncl = len(D)
        toto = [] 
        for i in range(nbIncl):
            dSldn = dSLdn(Omega, type1, step1, D[i], type2, step2)
            dDldn = dDLdn(Omega, type1, step1, D[i], type2, step2)
            toto.append(-1 * dSldn.stiffmat + impd * dDldn.stiffmat)
        
        matrix = np.vstack(toto)
        return matrix

    @staticmethod
    def source_vector(Omega, src, dipole):
        """Calculate the scalar product of the source vector dHdn with the boundary element basis.
        Args:
            Omega: The boundary (fish's body or inclusion) on which the source is evaluated.
                  Must contain points and normal attributes.
            src: array-like
                Position coordinates of the source point
            dipole: array-like
                Dipole direction vector
        Returns:
            ndarray: The source vector calculated as the product of the normal components,
                    Hessian of Green's function, and dipole direction.
        """
        _ , Hess = green.Green2D_Hessian(Omega.points, src)
        D = np.block([[np.diag(Omega.normal[0, :]), np.diag(Omega.normal[1, :])]])
        return  D @ Hess @ dipole

    @staticmethod
    def make_linop_CGPT(cfg, Current, impd, ord, symmode=0):
        As, Ar = Electric_Fish.make_matrix_SR(cfg, Current, cfg._center, impd, ord)
        
        if symmode:
            L = lambda x, tflag : SXR_op_symm_list(x, As, Ar, tflag)
        else:
            L = lambda x, tflag : SXR_op_list(x, As, Ar, tflag)
        
        result = {
            'L': L,
            'As': As,
            'Ar': Ar
        }

        return result
    
    @staticmethod
    def make_matrix_SR(cfg, Current, Z, impd, ord):
        """
        Construct matrices S and R involved in forward linear operator from CGPT to SFR data.
        
        Parameters
        ----------
        cfg : Fish_circle
            Acquisition configuration
        current : ndarray
            Surface current measurement (du/dn)
        Z : array-like
            Reference center coordinates
        impd : float
            Impedance of the skin
        ord : int
            Maximum order of the CGPT
        
        Returns
        -------
        S : ndarray
            Matrix mapping CGPT to measurements, shape (Ns_total, 2*ord)
        R : list
            List of matrices for each source position
        """
        S = np.zeros((cfg._Ns_total, 2*ord), dtype=np.complex128)
        R = []

        for s in range(cfg._Ns_total):
            Xr = cfg.rcv(s)  # receivers of s-th source
            
            # Right hand matrix (concerning only receivers)
            R.append(-1 * make_matrix_A(Xr, Z, ord))
            
            xs = cfg.src(s)  # s-th source
            dipole = cfg.dipole(s)
            
            mes = Current[s,:]  # surface measurement
            
            Omega0 = cfg.Bodies(s)
            Omega = Omega0.subset(cfg.idxRcv)
            sigma = Omega.sigma
            normal = Omega.normal
            
            for k in range(ord):
                m = k+1
                # Dipole terms
                phim, psim = Electric_Fish.phim_psim(m+1, np.vstack((Z[0]-xs[0], Z[1]-xs[1])) )
                
                A = (((-1)**m ) / (2*np.pi) * (dipole[0]*phim + dipole[1]*psim)).astype(np.complex128)
                B = (((-1)**m) / (2*np.pi) * (dipole[0]*psim - dipole[1]*phim)).astype(np.complex128)
                
                # Single layer terms
                phim, psim = Electric_Fish.phim_psim(m, np.vstack((Xr[0,:] - Z[0], Xr[1,:] - Z[1])) )
                


                A = (A - 1/(2*np.pi*m) * np.dot(sigma, phim)).astype(np.complex128)
                B = (B - 1/(2*np.pi*m) * np.dot(sigma, psim)).astype(np.complex128)
                
                # Double layer terms
                phim, psim = Electric_Fish.phim_psim(m+1, np.vstack((Xr[0,:] - Z[0], Xr[1,:] - Z[1])) ) 
                
                v1 = phim * normal[0,:] + psim * normal[1,:]
                A = A - impd/(2*np.pi) * np.dot(sigma, v1)
                
                v2 = psim * normal[0,:] - phim * normal[1,:]
                B = B - impd/(2*np.pi) * np.dot(sigma,v2)
                
                S[s, 2*k:2*m] = np.hstack((A, B))
                
        return S, R
    
    @staticmethod
    def phim_psim(m,x): 
        r, theta  = cart2pol(x)
        phim = np.cos(m*theta) / (r**m)
        psim = np.sin(m*theta) / (r**m)
        return phim, psim
    
    @staticmethod
    def make_linop_PT(cfg, Current, impd, symmode):
        M = Electric_Fish.make_matrix_gradUG(cfg, cfg._center, Current, impd)
        As = M['dU']
        Ar = M['dG']
        if symmode:
            L = lambda x, tflag : SXR_op_symm_list(x, As, Ar, tflag)
        else:
            L = lambda x, tflag : SXR_op_list(x, As, Ar, tflag)
        
        results = {
            'L': L,
            'As': As, 
            'Ar': Ar
        }
        
        return results
    
    @staticmethod
    def make_matrix_gradUG(cfg, Z, current_bg, impd):
        """
        Construct the matrices grad U and grad(dGdn) which are involved in the
        post-processed dipolar expansion eq 4.8
        Parameters:
        ---------------
         cfg: acquisition configuration
         Z: reference point of measurement
         current_bg: the function psi (Be carful, do not use the coefficient vector of
         the P1 basis) solution of the linear system A.2 measured on the skin
         impd: impedance
        Returns:
        -----------------
         dU: a cfg.Ns_total X 2 matrix. The s-th row correspond to grad U_s(z) with U_s
         the background solution with source at x_s, and z is the center of measurment
         dG: a list of cfg.Nr X 2 matrix, The r-th row correspond to
         grad(dGdn)(x_r, z), with x_r the r-th receiver
        """
        if not isinstance(cfg, Fish_circle):
            raise ValueError('cfg must be an instance of fish circle')
        _, H = green.Green2D_Hessian(cfg.all_src(), Z)
        
        grad_SL = np.zeros((cfg._Ns_total, 2), dtype=np.complex128)
        grad_DL = np.zeros((cfg._Ns_total,2), dtype=np.complex128)
        
        toto = cfg.all_dipole()
        grad_src = np.hstack((np.diag(toto[0,:]), np.diag(toto[1,:]))) 
        grad_src = grad_src @ H
        dG = []
        
        for s in range(cfg._Ns_total):
            Omega0 = cfg.Bodies(s)
            Fish = Omega0.subset(cfg.idxRcv)
            #Gradient of the single layer potential
            grad_SL[s,:] = SingleLayer.eval_grad(Fish, current_bg[s,:], Z).reshape((2,))
            #Gradient of the double layer
            grad_DL[s,:] = DoubleLayer.eval_grad(Fish, current_bg[s,:], Z).reshape((2,))
            #Hessian matrix at receivers
            _, H = green.Green2D_Hessian(cfg.rcv(s), Z)
            DN = np.hstack((np.diag(Fish.normal[0,:]), np.diag(Fish.normal[1,:])))
            dG.append(-DN @ H)
        dU = grad_src + grad_SL - impd*grad_DL
        results = {
            'dU': dU,  # Gradient of background potential
            'dG': dG   # Gradient of Green's function normal derivatives
        }
        
        return results