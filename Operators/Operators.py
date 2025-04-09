import numpy as np
import math
from Tools_fct import BEM_tools as BEM
from figure.C2Boundary.C2Boundary import C2Boundary
from FundamentalSols.green import Green2D
from Tools_fct import General_tools

class Operator:
    D1: C2Boundary
    D2: C2Boundary
    Kmat : np.ndarray

    def __init__(self, D1, type1, step1, D2=None,type2=None,step2=None):
        self.D1 = D1
        self.typeBEM1 = type1
        self.D2 = D2 if D2 else D1
        self.typeBEM2 = type2 if type2 else type1
        npts1 = self.D1.get_nbpts()
        npts2 = self.D2.get_nbpts()
        #1st Boundary case distinction
        if self.typeBEM1 == 'P0':
        # If P0 basis then the dimension of the approximation space equals to the number of boundary points
            self.Psi = np.eye(npts1)
            self.stepBEM1 = 1
        elif self.typeBEM1 == 'P1':
            if npts1 % step1 != 0:
                raise ValueError('Invalid sampling step for D1')
            self.stepBEM1 = step1
            self.Psi = BEM.P1_basis(npts1, step1)

        else:
            raise ValueError('Type Error: only P0 and P1 elements are supported in the current version.')
        #2nd boundary case distinction
        if self.typeBEM2 == 'P0':
            self.Phi = np.eye(npts2)
            self.stepBEM2 = 1
        elif self.typeBEM2 == 'P1':
            if step2 is None:
                step2 = step1
            if npts2 % step2 != 0:
                raise ValueError('Invalid sampling step for D1')
            self.stepBEM2 = step2
            self.Phi = BEM.P1_basis(npts2, step2)
        else:
            raise ValueError('Type Error: only P0 and P1 elements are supported in the current version.')
        self.Psi_t = self.Psi.T
        self.Phi_t = self.Phi.T

    #Utility methods

    def get_nbBEM1(self):
        return math.floor(self.D1.get_nbpts() / self.stepBEM1)
    def get_nbBEM2(self):
        return math.floor(self.D2.get_nbpts() / self.stepBEM2)
    
    def fwd(self, f):
        return self.Kmat @ self.Psi @ f
    
    @property
    def stiffmat(self):
        sigma2 = np.diag(self.D2.sigma)
        
        if self.typeBEM1 == 'P0' and self.typeBEM2 == 'P0':
            return self.Kmat
        elif self.typeBEM1 == 'P0' and self.typeBEM2 != 'P0':
            return np.dot(self.Phi_t, np.dot(sigma2, self.Kmat))
        elif self.typeBEM1 != 'P0' and self.typeBEM2 == 'P0':
            return np.dot(self.Kmat, self.Psi)
        elif self.typeBEM1 != 'P0' and self.typeBEM2 != 'P0':
            return np.dot(self.Phi_t, np.dot(sigma2, np.dot(self.Kmat, self.Psi)))
        else:
            raise ValueError('Not implemented')


class SingleLayer(Operator):
    
    def __init__(self,D1,type1,step1,D2=None,type2=None,step2=None):
        super().__init__(D1, type1, step1, D2, type2, step2)
        #TO DO: Finish writing the Single and Double Layer operators class with their make_kernel_matrix functions!





