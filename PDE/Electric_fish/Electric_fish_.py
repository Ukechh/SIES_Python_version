import sys
import os

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '/../../')))

#We consider the points array as d x n matrix where d is the dimension and n is the number of points
import numpy as np
import matplotlib.pyplot as plt
import warnings
import math
import copy
from Tools_fct.BEM_tools import P1_basis, interpolation
from PDE.SmallInclusion import SmallInclusion
from cfg.mconfig import Fish_circle
from FundamentalSols import green
from Operators.Operators import SingleLayer, LmKstarinv, dSLdn, dDLdn, Ident, Kstar
from asymp.CGPT_methods import lbda, make_system_matrix_fast, make_system_matrix, make_block_matrix

class Electric_Fish(SmallInclusion):

    def __init__(self, D, cnd, pmtt, cfg : Fish_circle, stepBEM):
        """
			Constructor of Electric_Fish class
			Inputs:
			D: exterior small inclusions
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
        self.Psi = P1_basis(self.Omega.nb_points, self.stepBEM1)
        
        self.KsdS = make_block_matrix(self._D) 
        self.dHdn = self.compute_dHdn()
    
    @property
    def Grammatrix(self):
        return self.Psi.T @ np.diag(self.Omega.sigma) @ self.Psi
    @property
    def nbBEM1(self):
        return math.floor(self._D[0].nb_points / self.stepBEM1)
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
            H1[:,s] = self.Psi.T @ (Omega.sigma() * toto1)
            idx = 0
            for i in range(self._nbIncl):
                H2[idx:idx+self.nbBEM2, s] = Electric_Fish.source_vector(self._D[i], src, dipole)
                idx+=self.nbBEM2
        return np.vstack((H1,H2))

    def plot(self, idx=None, ax=None, **kwargs):
        if ax is None:
            fig, ax = plt.subplots()

        for n in range(self._nbIncl):
            self._D[n].plot(ax=ax, **kwargs)

        if idx is None:
            idx = []
        
        self._cfg.plot(idx=idx, ax=ax, **kwargs)


    def data_simulation(self, f, *args, **kwargs):
        """
        Simulates the data of electric fish.

        Parameters
        -------------------
        freq : list or array-like
            A list of working frequencies used for the simulation.
        Returns
        -------------------
        output : 
            A dictionary containing various terms related to the simulation:
            
            - Variables starting with 'v' are coefficient vectors of boundary element basis.
            - Variables starting with 'f' are function values.
            - Variables starting with a capital letter are measurement matrices.

            The dictionary includes the following entries:
            - vpsi[n] : list of arrays
                For each frequency n, each column is a solution vector ψ of the linear system (A.5) for a given source.
            - vphi[n, i] : list of arrays
                For each frequency n and inclusion i, solution vectors φ.
            - fpsi[n] : list of 2D arrays
                Function values corresponding to vpsi; shape (nbBEM1, Ns_total).
            - fphi[n] : list of 3D arrays
                Function values corresponding to vphi; shape (nbBEM2, Ns_total, nbIncls).
            - vpsi_bg : list of arrays
                Background solution vectors for equation (A.2).
            - fpsi_bg : list of arrays
                Function values for vpsi_bg.
            - fpp[n] : list of arrays
                The post-processed function defined in equation (4.7): (1/2 - K* - ξ dDdn)(dudn - dUdn).

            Measurements:
            
            - Current[n] : list of arrays
                Surface current dUdn for each frequency n.
            - Current_bg : list of arrays
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
        SF = np.zeros((self.nbBEM1 + self._nbIncl*self.nbBEM2, len(f), self._cfg._Ns_total))
        #Solution matrix of background problem
        SU = np.zeros((self.nbBEM1, self._cfg._Ns_total))
        #Post Processing
        PP = np.zeros((self.nbBEM1, len(f), self._cfg._Ns_total))

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
            rhs = np.concatenate([self.dHdn[:self.nbBEM1, s], [0]])
            SU[:, s] = np.linalg.solve(matrix_BEM, rhs)
            
            for n in range(len(f)):
                lam = lbda(self.cnd, self.pmtt, f[n])
                matrix_D = make_system_matrix_fast(self.KsdS, lam)

                top = np.hstack([matrix_A, matrix_B])
                middle = np.hstack([matrix_C, matrix_D])
                row1 = np.hstack([Psi_int.reshape(1, -1), np.zeros((1, self.nbBEM2 * self._nbIncl))])
                row2 = np.hstack([np.zeros((1, self.nbBEM1)), Phi_int.reshape(1, -1)])
                matrix_BEM = np.vstack([top, middle, row1, row2])

                rhs = np.concatenate([self.dHdn[:, s], [0, 0]])

                SF[:, n, s] = np.linalg.solve(matrix_BEM, rhs)
                
                PP[:, n, s] = -np.linalg.solve(self.Grammatrix, matrix_B @ SF[self.nbBEM1:, n, s])

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
            fphi.append(np.zeros((self.nbBEM2, self._cfg._Ns_total, self._nbIncl)))
            for j in range(self._nbIncl):
                vphi[m][j] = sol[idx:idx+self.nbBEM2 , :]
                idx += self.nbBEM2
                fphi[m][:,:,j] = vphi[m][j]
            vpp = np.squeeze(PP[:,m,:])
            fpp.append(interpolation(self.Psi, vpp))
            
            MSR.append(np.zeros((self._cfg._Ns_total, self._cfg._Nr)))
            for j in range(self.nbBEM2):
                toto = np.zeros((self._cfg._Ns_total, self._cfg._Nr))
                for s in range(self._cfg._Ns_total):
                    rcv = self._cfg.rcv(s)
                    toto[s,:] = SingleLayer.eval(self._D[j], vphi[m][j][:,s], rcv)
                MSR[m] += toto
        

        ##ALMOST DONE JUST NEED TO TRANSLATE LAST PART AND IDENTIFY THE INTENDED OUTPUTS!!



        return f, vpsi_bg, fpsi_bg, 


    @staticmethod
    def system_matrix_block_A(Omega, type1, step1, impd):
        Id = Ident(Omega, type1, step1)
        Ks = Kstar(Omega, type1, step1)
        dDldn = dDLdn(Omega, type1, step1)
        return (1/2 * Id.stiffmat + -1*Ks.stiffmat + impd * dDldn.stiffmat).toarray()

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
        return np.hstack(toto)

    @staticmethod
    def source_vector(Omega, src, dipole):
        _ , Hess = green.Green2D_Hessian(Omega.points, src)
        D = np.block([[np.diag(Omega.normal[0, :]), np.diag(Omega.normal[1, :])]])
        return  D @ Hess @ dipole

