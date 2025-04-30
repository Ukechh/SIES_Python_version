import sys
import os

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '../../')))

#We consider the points array as d x n matrix where d is the dimension and n is the number of points
import numpy as np
import warnings
import math
import copy
#Plot packages
from matplotlib.ticker import MaxNLocator 
import matplotlib.pyplot as plt
#Utility classes and functions
from Tools_fct.BEM_tools import P1_basis, interpolation 
from PDE.SmallInclusion import SmallInclusion
from cfg.mconfig import Fish_circle
from FundamentalSols import green
from Operators.Operators import SingleLayer, LmKstarinv, dSLdn, dDLdn, Ident, Kstar, DoubleLayer
from asymp.CGPT_methods import lbda, make_system_matrix_fast, make_system_matrix, make_block_matrix

class Electric_Fish(SmallInclusion):

    def __init__(self, D, cnd, pmtt, cfg : Fish_circle, stepBEM):
        """
			Constructor of Electric_Fish class
			Inputs:
			D: exterior small inclusions
                list
			cnd, pmtt: conductivity and permittivity constants of D
			cfg: configuration acq.Fish_circle
			stepBEM: sampling step for the P1 BEM
        """
        super().__init__(D, cfg)
        if not isinstance(cfg, Fish_circle):
            raise ValueError('The acquisition configuration must be an object of Fish_circle')
        self.Omega = copy.deepcopy(cfg.Omega0)
        self.impd = copy.deepcopy(cfg.impd)
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
        self.dHdn = self.compute_dHdn()
    
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
        return np.vstack((H1,H2))

    def plot(self, idx=None, ax=None, **kwargs):
        if ax is None:
            fig, ax = plt.subplots(figsize=(8,6))

        for n in range(self._nbIncl):
            self._D[n].plot(ax=ax, **kwargs)

        if idx is None:
            idx = []
        self._cfg.plot(ax=ax, **kwargs)

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
        Ss = SingleLayer.eval(Omega, fpsi, Z)
        Ds = DoubleLayer.eval(Omega,fpsi,Z)

        V = Hs + Ss - self.impd * Ds

        for i in range(self._nbIncl):
         #contribution of the i^th inclusion 
            V += SingleLayer.eval(self._D[i], fphi[:,i], Z)
        F = np.reshape(V, (N,N))
        #BACKGROUND FIELD
        Ss = SingleLayer.eval(Omega, fpsi_bg, Z)
        Ds = DoubleLayer.eval(Omega, fpsi_bg, Z)
    
        V = Hs + Ss - self.impd * Ds
        F_bg = np.reshape(V, (N,N))
        
        return F, F_bg, Sx, Sy, mask 
     
    def data_simulation(self, f, *args, **kwargs):
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
            - vpsi : list of arrays
                For each frequency n, each column is a solution vector ψ of the linear system (A.5) for a given source.
            - vphi[n][i] : list of arrays
                For each frequency n and inclusion i, solution vectors φ.
            - fpsi[n] : list of 2D arrays
                Function values corresponding to vpsi; shape (nbBEM1, Ns_total).
            - fphi[n] : list of 3D arrays
                Function values corresponding to vphi; shape (nbBEM2, Ns_total, nbIncls).
            - vpsi_bg :ndarray
                Background solution vectors for equation (A.2).
            - fpsi_bg : ndarray
                Function values for vpsi_bg.
            - fpp[n] : list of arrays
                The post-processed function defined in equation (4.7): (1/2 - K* - ξ dDdn)(dudn - dUdn).

            Measurements:
            
            - Current[n] : list of arrays
                Surface current dUdn for each frequency n.
            - Current_bg : ndarray
                Background surface current dUdn.
            - MSR[n] : list of arrays
                Multi-static response matrix for CGPT reconstruction (see reference [2]).
            - SFR[n] : list of arrays
                Space-Frequency response matrix dUdn - dUdn corresponding to the dipolar expansion (section 4.2 in reference [1]).
            - PP_SFR[n] : list of arrays
                Post-processed response matrix, defined by equation (4.7).
                Note: the negative sign before ξ in eq (4.7) of [1] was a typo — it should be positive (+).

        Notes
        -------------------
        - For matrices named `fxxx`, each **column corresponds to a source**.
        - For measurement matrices (names starting with a capital letter), each **row corresponds to a source**.
        """
        if not isinstance(self._cfg, Fish_circle):
            raise ValueError('The cfg must be an instance of Fish_circle')
        ###Solve forward pb for all positions:
        if isinstance(f,int):
            f = np.array([f])
        #Solution matrix of problem, frequency dependent
        SF = np.zeros((self.nbBEM1 + self._nbIncl*self.nbBEM2, len(f), self._cfg._Ns_total), dtype=np.complex128)
        #Solution matrix of background problem
        SU = np.zeros((self.nbBEM1, self._cfg._Ns_total), dtype=np.complex128)
        #Post Processing
        PP = np.zeros((self.nbBEM1, len(f), self._cfg._Ns_total), dtype= np.complex128)
        #Gram matrix as complex:
        Gram = self.Grammatrix.astype(complex)
        #Psi and Phi integrals:
        Psi_int = self.Psi.T @ self.Omega.sigma
        Phi_int = np.zeros((self.nbBEM2, self._nbIncl))
        for i in range(self._nbIncl):
            Phi_int[:,i] = self._D[i].sigma #Boundary integral of P0 elements

        """
        The first for loop is on the fish's position (different Omega), because it is more expensive to
        build the block matrices depending on Omega (A, B, C). Remark that both the system matrix and the
        right hand vector are Omega-dependent.
        """
        for s in range(self._cfg._Ns_total):
            Omega = self._cfg.Bodies(s)

            matrix_A = Electric_Fish.system_matrix_block_A(Omega, self.typeBEM1, self.stepBEM1, self.impd)
            matrix_B = Electric_Fish.system_matrix_block_B(Omega, self.typeBEM1, self.stepBEM1, self._D, self.typeBEM2, self.stepBEM2)
            matrix_C = Electric_Fish.system_matrix_block_C(Omega, self.typeBEM1, self.stepBEM1, self._D, self.typeBEM2, self.stepBEM2, self.impd) 
            
            #Resolution of background system:
            matrix_BEM = np.vstack([matrix_A, Psi_int.reshape(1, -1)])
            rhs = np.concatenate([self.dHdn[:self.nbBEM1, s], [0.0]])
            row, _, _, _ = np.linalg.lstsq(matrix_BEM, rhs, rcond=0.0) 
            SU[:, s] = row #Here matrix BEM is of shape (n+1, n) and rhs has shape (n+1,)
            
            for n in range(len(f)):
                lam = lbda(self.cnd, self.pmtt, f[n])
                matrix_D, _ = make_system_matrix_fast(self.KsdS, lam)

                top = np.hstack([matrix_A, matrix_B], dtype=np.complex128)
                middle = np.hstack([matrix_C, matrix_D],dtype=np.complex128)
                row1 = np.hstack([Psi_int.reshape(1, -1), np.zeros((1, self.nbBEM2 * self._nbIncl))], dtype=np.complex128)
                row2 = np.hstack([np.zeros((1, self.nbBEM1)), Phi_int.reshape(1, -1)], dtype=np.complex128)
                matrix_BEM = np.vstack([top, middle, row1, row2], dtype=np.complex128)

                rhs = np.concatenate([self.dHdn[:, s], [0, 0]], dtype=np.complex128)
                upd, _, _, _ = np.linalg.lstsq(matrix_BEM, rhs)
                SF[:, n, s] = upd
                
                rhs = matrix_B @ SF[self.nbBEM1:, n, s]
                PP[:, n, s] = np.linalg.solve(-Gram, rhs)

        vpsi_bg = SU
        fpsi_bg = interpolation(self.Psi, SU)
        
        vpsi = []
        fpsi = []
        
        vphi = [[np.array([]) for _ in range(self._nbIncl)] for _ in range(len(f))]
        fphi = []
        
        fpp = []
        MSR = []
        
        for m in range(len(f)):
            sol = np.squeeze(SF[:, m, :])
            vpsi.append(sol[:self.nbBEM1, :])
            fpsi.append(interpolation(self.Psi, vpsi[m]))

            idx = self.nbBEM1
            fphi.append(np.zeros((self.nbBEM2, self._cfg._Ns_total, self._nbIncl), dtype=np.complex128))
            
            for j in range(self._nbIncl):
                vphi[m][j] = sol[idx:idx+self.nbBEM2 , :]
                idx += self.nbBEM2
                fphi[m][:,:,j] = vphi[m][j]
            
            vpp = np.squeeze(PP[:,m,:])
            fpp.append(interpolation(self.Psi, vpp))
            
            MSR.append(np.zeros((self._cfg._Ns_total, self._cfg._Nr), dtype= np.complex128))
            
            for j in range(self._nbIncl):
                toto = np.zeros((self._cfg._Ns_total, self._cfg._Nr), dtype=np.complex128)
                for s in range(self._cfg._Ns_total):
                    rcv = self._cfg.rcv(s)
                    toto[s,:] = SingleLayer.eval(self._D[j], vphi[m][j][:,s], rcv)
                MSR[m] += toto
        
        Current_bg = fpsi_bg.T
        Current = []
        SFR = []
        PP_SFR = []
        for n in range(len(f)):
            Curr = fpsi[n].T
            sfr = Curr - Current_bg
            pp_sfr = fpp[n].T

            Current.append(Curr[:, self._cfg.idxRcv])
            SFR.append(sfr[:,self._cfg.idxRcv])
            PP_SFR.append(pp_sfr[:,self._cfg.idxRcv])

        Current_bg = Current_bg[:,self._cfg.idxRcv]

        return f, vpsi, vphi, fpsi, fphi, vpsi_bg, fpsi_bg, fpp, Current, Current_bg, MSR, SFR, PP_SFR  

    def plot_field(self, s, F, F_bg, SX, SY, nbLine, subfig, *args, **kwargs):
        if not isinstance(self._cfg, Fish_circle):
            raise ValueError('The cfg must be an instance of Fish_circle')
        
        def create_plot(ax, data, title, add_colorbar=True, custom_cmap=None):
            if not isinstance(self._cfg, Fish_circle):
                raise ValueError('The cfg must be an instance of Fish_circle')
            # Determine appropriate contour levels for better visualization
            vmin, vmax = np.nanmin(data).astype(np.float64), np.nanmax(data).astype(np.float64)
            # Prevent excessive zoom for near-constant fields
            if abs(vmax - vmin) < 1e-10:
                vmin, vmax = -1, 1
            
            # Create filled contours with improved styling
            contour = ax.contourf(SX, SY, data, nbLine, alpha=0.8, extend='both', vmin=vmin, vmax=vmax)
            
            # Plot inclusions
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
            ax.plot(grid_center_x, grid_center_y, 'x', color ='r')
            ax.plot(src[0], src[1], 'x', color= 'k')
            return contour
        
        if subfig:
            # Create 2x2 subplot grid with adjusted spacing
            fig, axs = plt.subplots(2, 2, figsize=(15, 12),
                                gridspec_kw={'hspace': 0.2, 'wspace': 0.2})
            fig.suptitle('Potential Field Analysis', y=0.98, fontsize=14, fontweight='bold')
            
            # Plot 1: Real part of F
            contour1 = create_plot(axs[0, 0], np.real(F), 'Potential field u, real part')
            
            # Plot 2: Imaginary part of F
            contour2 = create_plot(axs[0, 1], np.imag(F), 'Potential field u, imaginary part')
            
            # Plot 3: Real part of perturbation
            diff = np.real(F - F_bg)
            contour3 = create_plot(axs[1, 0], diff, 'Perturbed field u-U, real part')  # Diverging colormap for difference
            
            # Plot 4: Background field with enhanced contours
            contour4 = create_plot(axs[1, 1], F_bg, 'Background potential field U')
            
            # Apply consistent formatting to all subplots with proper axes labels
            for ax in axs.flat:
                ax.set_xlabel('x', fontsize=10)
                ax.set_ylabel('y', fontsize=10)
                
        else:
            # Single plot version with better layout
            fig, ax = plt.subplots(figsize=(10, 8))
            contour = create_plot(ax, np.real(F), 'Potential field u, real part')
            ax.set_xlabel('x', fontsize=10)
            ax.set_ylabel('y', fontsize=10)
            
        plt.tight_layout()
        
        return fig
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
        _ , Hess = green.Green2D_Hessian(Omega.points, src)
        D = np.block([[np.diag(Omega.normal[0, :]), np.diag(Omega.normal[1, :])]])
        return  D @ Hess @ dipole

