import numpy as np
import math
from FundamentalSols import green
from Tools_fct import BEM_tools as BEM
from figure.C2Boundary.C2Boundary import C2Boundary
from Tools_fct import General_tools
from abc import ABC, abstractmethod

class Operator(ABC):
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
    
    
    @staticmethod
    def green(X,Y):
        g = 1/ (4*np.pi) * np.log(X**2 + Y**2)
        return g

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
    
    @abstractmethod
    def make_kernel_matrix(self, *args, **kwargs) -> np.ndarray:
        pass
        

class SingleLayer(Operator):
    
    def __init__(self,D1,type1,step1,D2=None,type2=None,step2=None):
        super().__init__(D1, type1, step1, D2, type2, step2)
        if self.D1 == self.D2:
            self.Kmat = SingleLayer.make_kernel_matrix(D1.points, D1.sigma)
        else:
            self.Kmat = SingleLayer.make_kernel_matrix(D1.points, D1.sigma, self.D2.points)


    @staticmethod
    def make_kernel_matrix(D,sigma,E=None):
        if not E is None:
            X1 = General_tools.tensorplus(E[0,:],-D[0,:])
            X2 = General_tools.tensorplus(E[1,:],-D[1,:])
            K = Operator.green(X1,X2) @ np.diag(sigma)
        else:
            N = D.shape()[1]
            K = np.zeros(N,N)
            for i in range(N):
                K[i, 0:i] = sigma[0:i] * Operator.green(D[0, i] - D[0, 0:i], D[1, i] - D[1, 0:i])
                K[i, i+1:N] = sigma[i+1:N] * Operator.green(D[0, i] - D[0, i+1:N], D[1, i] - D[1, i+1:N])
                K[i,i] = sigma[i] * (np.log(sigma[i]/2) - 1) / (2*np.pi)
        return K
    
    @staticmethod
    def eval(D,F,X):
        G = green.Green2D(D.points, X)
        ev = (F.reshape(1,-1)*D.sigma) @ G
        return ev
    @staticmethod
    def eval_grad(D,F,X):
        Gx, Gy = green.Green2D_grad(D.points,X)
        v1 = Gx @ (F.ravel() * D.sigma.ravel())
        v2 = Gy @ (F.ravel() * D.sigma.ravel())
        r = np.vstack((v1, v2))
        return r

class DoubleLayer(Operator):
    
    def __init__(self,D1,type1,step1,D2,type2,step2):
        super().__init__(D1, type1, step1, D2, type2, step2)
        if self.D1 == self.D2:
            raise TypeError("This operator is not defined for identical boundaries because of the jump")
        elif self.D2 is None:
            raise ValueError("D2 needs to be specified")
        else:
            self.Kmat = DoubleLayer.make_kernel_matrix(self.D1.points, self.D1.normal, self.D1.sigma, self.D2.normal)

    #Utility methods
    def fwd(self,f):
        if self.D1 == self.D2:
            raise TypeError("This operator is not defined for identical boundaries")
        return Operator.fwd(self,f)

    @staticmethod
    def eval(D,F,X):
        dGn = green.Green2D_Dn(X,D.points,D.normal)
        r = dGn @ (F.ravel()*D.sigma.ravel())
        return r.reshape(1,-1)
    
    @staticmethod
    def make_kernel_matrix(D,normal, sigma, E):
        Gx, Gy = green.Green2D_grad(E,D)
        K = -(Gx @ np.diag(normal[0,:]) ) + Gy @ np.diag(normal[1,:]) @ np.diag(sigma)
        return K
    
    @staticmethod
    def eval_grad(D,F,X):
        H, _ = green.Green2D_Hessian(D.points,X)
        vv = -(D.sigma*F.ravel()) @ (General_tools.bdiag(D.normal.T,1) @ H)
        r = np.vstack((vv[::2],vv[1::2]))
        return r
    
class Kstar(Operator):
    def __init__(self,D1,type1,step1,D2=None,type2=None,step2=None):
        super().__init__(D1, type1, step1, D2, type2, step2)