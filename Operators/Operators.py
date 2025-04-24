import sys
import os

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '../')))

import numpy as np
import math
from typing import cast
from scipy import sparse
from scipy.sparse import linalg as splinalg
from FundamentalSols import green
from Tools_fct import BEM_tools as BEM
from figure.C2Boundary.C2Boundary import C2Bound
from Tools_fct import General_tools
from abc import ABC, abstractmethod

class Operator(ABC):
    D1: C2Bound #Domain Boundary
    D2: C2Bound #Image Boundary
    Kmat : sparse.spmatrix

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
            self.Psi = sparse.eye(npts1, format='csr')
            self.stepBEM1 = 1
        elif self.typeBEM1 == 'P1':
            if npts1 % step1 != 0:
                raise ValueError('Invalid sampling step for D1')
            self.stepBEM1 = step1
            self.Psi = sparse.csr_matrix(BEM.P1_basis(npts1, step1))

        else:
            raise ValueError('Type Error: only P0 and P1 elements are supported in the current version.')
        #2nd boundary case distinction
        if self.typeBEM2 == 'P0':
            self.Phi = sparse.eye(npts2, format='csr')
            self.stepBEM2 = 1
        elif self.typeBEM2 == 'P1':
            if step2 is None:
                step2 = step1
            if npts2 % step2 != 0:
                raise ValueError('Invalid sampling step for D1')
            self.stepBEM2 = step2
            self.Phi = sparse.csr_matrix(BEM.P1_basis(npts2, step2))
        else:
            raise ValueError('Type Error: only P0 and P1 elements are supported in the current version.')
        self.Psi_t = self.Psi.transpose()
        self.Phi_t = self.Phi.transpose()

    #Utility methods

    def get_nbBEM1(self):
        return math.floor(self.D1.get_nbpts() / self.stepBEM1)
    def get_nbBEM2(self):
        return math.floor(self.D2.get_nbpts() / self.stepBEM2)
    def get_stiffmat(self):
        return self.stiffmat
    def fwd(self, f):
        return self.Kmat @ (self.Psi @ f)
    
    
    @staticmethod
    def green(X,Y):
        g = 1/ (4*np.pi) * np.log(X**2 + Y**2)
        return g

    @property
    def stiffmat(self):
        """Compute stiffness matrix based on basis function types"""
        # Sparse diagonal matrix from sigma values
        sigma2 = sparse.diags(self.D2.sigma, format='csr')
        
        if self.typeBEM1 == 'P0' and self.typeBEM2 == 'P0':
            return self.Kmat
        elif self.typeBEM1 == 'P0' and self.typeBEM2 != 'P0':
            # Matrix multiplication with sparse matrices
            return self.Phi_t @ (sigma2 @ self.Kmat)
        elif self.typeBEM1 != 'P0' and self.typeBEM2 == 'P0':
            return self.Kmat @ self.Psi
        elif self.typeBEM1 != 'P0' and self.typeBEM2 != 'P0':
            # Chain of sparse matrix multiplications
            return self.Phi_t @ (sigma2 @ (self.Kmat @ self.Psi))
        else:
            raise ValueError('Not implemented')
    
    @abstractmethod
    def make_kernel_matrix(self, *args, **kwargs)-> sparse.spmatrix:
        pass
    

class SingleLayer(Operator):
    """
    Single layer potential operator implementation using sparse matrices.
    
    This class implements the single layer potential integral operator:
    (Sf)(x) = ∫_Γ G(x,y) f(y) dσ(y)
    
    where G is the fundamental solution of the Laplace equation.
    """
    
    def __init__(self, D1, type1, step1, D2=None, type2=None, step2=None):
        # Initialize base operator class
        super().__init__(D1, type1, step1, D2, type2, step2)
        
        # Build kernel matrix based on whether boundaries are the same
        if self.D1 == self.D2:
            # Self-interaction case (single boundary)
            self.Kmat = SingleLayer.make_kernel_matrix(D1.points, D1.sigma)
        else:
            # Interaction between different boundaries
            self.Kmat = SingleLayer.make_kernel_matrix(D1.points, D1.sigma, self.D2.points)

    @staticmethod
    def make_kernel_matrix(D, sigma, E=None):
        """        
        Parameters:
        -----------
        D : numpy.ndarray
            Domain boundary points (2 x N)
        sigma : numpy.ndarray
            Weights
        E : numpy.ndarray, optional
            Image boundary points if different from source
            
        Returns:
        --------
        K : scipy.sparse.spmatrix
            Sparse matrix representation of the single layer operator
        """
        if E is not None:
            # Case: Different boundaries (D ≠ E)
            X1 = General_tools.tensorplus(E[0,:], -D[0,:])
            X2 = General_tools.tensorplus(E[1,:], -D[1,:])
            
            # Get dimensions
            N1, N2 = len(E[0,:]), len(D[0,:])
            
            # Initialize arrays for COO format
            row_indices = []
            col_indices = []
            values = []
            
            # Compute Green's function values
            for i in range(N1):
                for j in range(N2):
                    val = Operator.green(X1[i,j], X2[i,j]) * sigma[j]
                    # Only store significant values
                    if abs(val) > 1e-12:
                        row_indices.append(i)
                        col_indices.append(j)
                        values.append(val)
            
            # Create sparse matrix in COO format
            K = sparse.coo_matrix((values, (row_indices, col_indices)), 
                                 shape=(N1, N2))
        else:
            # Case: Same boundary (D = E)
            # Single boundary case can exploit symmetry and has diagonal singularities
            N = D.shape[1]
            
            # Use LIL format for efficient construction with changing sparsity pattern
            K = sparse.lil_matrix((N, N))
            
            # Fill the matrix efficiently, exploiting symmetry where possible
            for i in range(N):
                # Off-diagonal elements (upper triangle)
                if i+1 < N:
                    upper_vals = sigma[i+1:N] * Operator.green(
                        D[0, i] - D[0, i+1:N], 
                        D[1, i] - D[1, i+1:N]
                    )
                    # Only store significant values
                    significant_indices = np.where(np.abs(upper_vals) > 1e-12)[0] + i + 1
                    K[i, significant_indices] = upper_vals[significant_indices - (i+1)]
                
                # Off-diagonal elements (lower triangle)
                if i > 0:
                    lower_vals = sigma[0:i] * Operator.green(
                        D[0, i] - D[0, 0:i], 
                        D[1, i] - D[1, 0:i]
                    )
                    # Only store significant values
                    significant_indices = np.where(np.abs(lower_vals) > 1e-12)[0]
                    K[i, significant_indices] = lower_vals[significant_indices]
                
                # Diagonal elements (require special treatment due to singularity)
                K[i, i] = sigma[i] * (np.log(sigma[i]/2) - 1) / (2*np.pi)
        
        return cast(sparse.spmatrix, K.tocsr())
    
    @staticmethod
    def eval(D,F,X):
        """
        Parameters:
        -----------
        D : C2Bound
            Boundary object
        F : numpy.ndarray
            Density function values on boundary (Shape (npts,) or (npts,1))
        X : numpy.ndarray
            Points where to evaluate the potential (Shape (2,m))
            
        Returns:
        --------
        ev : numpy.ndarray
            Potential values at X
        """
        G = green.Green2D(D.points, X) #Outputs a (npts, m) matrix
        ev = (D.sigma * F.T) @ G # Product of (1,npts) (npts,m), output is (1,m)
        return ev.T #Output shape is (m,1)
    @staticmethod
    def eval_grad(D,F,X):
        """
        Parameters:
        -----------
        D : C2Bound
            Boundary object
        F : numpy.ndarray
            Density function values on boundary
        X : numpy.ndarray
            Points where we evaluate the gradient
        Returns:
        --------
        r : numpy.ndarray
            Gradient values at X (2 x len(X))
        """
        Gx, Gy = green.Green2D_grad(D.points, X)
        v1 = Gx @ (F.ravel() * D.sigma.ravel())
        v2 = Gy @ (F.ravel() * D.sigma.ravel())
        r = np.vstack((v1, v2))
        return r

class DoubleLayer(Operator):
    """
    Double layer potential operator implementation using sparse matrices.
    
    This class implements the double layer potential integral operator:
    (Kf)(x) = ∫_Γ ∂G(x,y)/∂n_y f(y) dσ(y)
    
    where G is the fundamental solution of the Laplace equation, and
    n_y is the unit normal at point y on the boundary.
    
    This operator is not defined for identical boundaries due to
    the jump condition across the boundary.
    """

    def __init__(self, D1, type1, step1, D2, type2, step2):
        # Initialize base operator class
        super().__init__(D1, type1, step1, D2, type2, step2)
        
        # Check boundary validity
        if self.D1 == self.D2:
            raise TypeError("This operator is not defined for identical boundaries because of the jump")
        elif self.D2 is None:
            raise ValueError("D2 needs to be specified")
        else:
            # Build kernel matrix for different boundaries
            self.Kmat = DoubleLayer.make_kernel_matrix(
                self.D1.points, 
                self.D1.normal, 
                self.D1.sigma, 
                self.D2.normal
            )

    # Utility methods
    def fwd(self, f):
        """
        Parameters:
        -----------
        f : numpy.ndarray
            Function values at boundary points
            
        Returns:
        --------
        result : numpy.ndarray
            Result of applying the operator to f
        """
        if self.D1 == self.D2:
            raise TypeError("This operator is not defined for identical boundaries")
        return Operator.fwd(self, f)

    @staticmethod
    def eval(D, F, X):
        """
        Parameters:
        -----------
        D : C2Bound
            Boundary object
        F : numpy.ndarray
            Density function values on boundary
        X : numpy.ndarray
            Points where to evaluate the potential
            
        Returns:
        --------
        r : numpy.ndarray
            Potential values at X (1 x len(X))
        """
        dGn = green.Green2D_Dn(X, D.points, D.normal)
        r = dGn @ (F.ravel() * D.sigma.ravel())
        return r.reshape(1, -1)

    @staticmethod
    def make_kernel_matrix(D, normal, sigma, E):
        """
        Parameters:
        -----------
        D : numpy.ndarray
            Source boundary points (2 x N)
        normal : numpy.ndarray
            Normal vectors at source points (2 x N)
        sigma : numpy.ndarray
            Weights at source points
        E : numpy.ndarray
            Target boundary points
            
        Returns:
        --------
        K : scipy.sparse.spmatrix
            Sparse matrix representation of the double layer operator
        """
        # Get gradient of Green function
        Gx, Gy = green.Green2D_grad(E, D)
        
        # Create diagonal matrices for normal components and integration weights
        nx_diag = sparse.diags(normal[0, :])
        ny_diag = sparse.diags(normal[1, :])
        sigma_diag = sparse.diags(sigma)
        
        # Both Gx and Gy might have many small values that could be truncated

        threshold = 1e-12  # Adjust based on problem precision requirements
        
        # Create sparse versions of gradient matrices
        row_Gx, col_Gx = Gx.shape
        row_Gy, col_Gy = Gy.shape
        
        # Convert Gx and Gy to sparse with thresholding
        Gx_values = Gx.flatten()
        Gx_rows, Gx_cols = np.indices((row_Gx, col_Gx))
        mask_Gx = np.abs(Gx_values) > threshold
        Gx_sparse = sparse.coo_matrix(
            (Gx_values[mask_Gx], (Gx_rows.flatten()[mask_Gx], Gx_cols.flatten()[mask_Gx])), 
            shape=(row_Gx, col_Gx)
        )
        
        Gy_values = Gy.flatten()
        Gy_rows, Gy_cols = np.indices((row_Gy, col_Gy))
        mask_Gy = np.abs(Gy_values) > threshold
        Gy_sparse = sparse.coo_matrix(
            (Gy_values[mask_Gy], (Gy_rows.flatten()[mask_Gy], Gy_cols.flatten()[mask_Gy])), 
            shape=(row_Gy, col_Gy)
        )
        
        # Compute the double layer kernel matrix
        K = -1 * (Gx_sparse @ nx_diag) + (Gy_sparse @ ny_diag @ sigma_diag)
        
        return K
    
    @staticmethod
    def eval_grad(D,F,X):
        """
        Parameters:
        -----------
        D : C2Bound
            Boundary object
        F : numpy.ndarray
            Density function on boundary
        X : numpy.ndarray
            Points where we evaluate the gradient
        
        Returns:
        --------
        r : numpy.ndarray
            Gradient values at X (2 x len(X))
        """
        H, _ = green.Green2D_Hessian(D.points,X)
        vv = -(D.sigma*F.ravel()) @ (General_tools.bdiag(D.normal.T,1) @ H)
        r = np.vstack((vv[::2],vv[1::2]))
        return r
    
class Kstar(Operator):
    def __init__(self, D1, type1, step1, D2=None, type2=None, step2=None):
        """Initialize Kstar operator
        
        Parameters:
        -----------
        D1: C2Bound
            Domain boundary
        type1: str 
            Type of basis functions ('P0' or 'P1')
        step1: int
            Step size for P1 basis
        D2: C2Bound 
            Image boundary (optional)
        type2: str 
            Type of basis functions for image boundary
        step2: int 
            Step size for P1 basis on image boundary
        """
        super().__init__(D1, type1, step1, D2, type2, step2)
        self.Kmat = self.make_kernel_matrix(D1.points, D1.tvec, D1.avec, D1.normal, D1.sigma)
    
    @staticmethod
    def make_kernel_matrix(D, tvec, avec, normal, sigma):
        """
        Parameters:
        -----------
        D:  ndarray
            Boundary points (2 x M array)
        tvec:   ndarray
            Tangent vectors
        avec: ndarray 
            Acceleration vectors
        normal: ndarray
            Normal vectors
        sigma: ndarray 
            Weights
            
        Returns:
        -----------
        r: sparse.spmatrix
            Sparse kernel matrix in CSR format
        """
        M = D.shape[1]
        Ks = sparse.lil_matrix((M, M))  # Use LIL for efficient construction
        
        # Precompute frequent values
        tvec_norm_sq = np.linalg.norm(tvec, axis=0)**2
        
        for j in range(M):
            # Difference vectors and norm of the vectors
            xy_diff = D[:, j, np.newaxis] - D
            x_dot_n = xy_diff[0] * normal[0, j] + xy_diff[1] * normal[1, j]
            norm_sq = np.sum(xy_diff**2, axis=0)
            
            # Elements before j (0 to j-1)
            if j > 0:
                values = (1 / (2 * np.pi)) * x_dot_n[:j] * sigma[:j] / norm_sq[:j]
                Ks[j, :j] = values
            
            # Elements after j (j+1 to M)
            if j < M - 1:
                values = (1 / (2 * np.pi)) * x_dot_n[j+1:] * sigma[j+1:] / norm_sq[j+1:]
                Ks[j, j+1:] = values
            
            # Diagonal element
            diag_val = (1 / (2 * np.pi)) * (-0.5) * np.dot(avec[:, j], normal[:, j]) / tvec_norm_sq[j] * sigma[j]
            Ks[j, j] = diag_val
        return cast(sparse.spmatrix,Ks.tocsr()) #Shape of output is (npts,npts) where npts is the number of points of the input C2Bound
    
    @staticmethod
    def eval(D, F):
        """Evaluate the operator on given boundary with function values
        
        Parameters
        -----------
        D: C2Bound 
            Boundary object
        F: ndarray 
            Function values on boundary
        Returns
        -----------
        r: ndarray
            Result of applying K* operator
        """
        temp_op = Kstar(D, 'P0', 1)  # Use P0 basis for evaluation
        return temp_op.Kmat @ F
    
class Ident(Operator):
     
    def __init__(self,D,type1,step1,type2=None,step2=None):
        super().__init__(D, type1, step1, type2=type2, step2=step2)
        self.Kmat = cast(sparse.spmatrix, sparse.eye(D._nb_points, format='csr'))
    
    @staticmethod
    def make_kernel_matrix(m):
        return cast(sparse.spmatrix, sparse.eye(m, format='csr'))
    @staticmethod
    def eval(F):
        return F
    
class LmKstarinv(Operator):
    def __init__(self, l, D, type, step):
        super().__init__(D,type,step,D,type,step)
        self.Kmat = LmKstarinv.make_kernel_matrix(l, D.points, D.tvec, D.normal, D.avec, D.sigma)
    
    @staticmethod
    def make_kernel_matrix(l, D, tvec, avec, normal, sigma):
        if abs(l) < 1/2:
            raise ValueError("The operator is not defined for this value of lambda!")
        Ks = Kstar.make_kernel_matrix(D, tvec, avec, normal, sigma)
        LambdaI = l * sparse.eye(D.shape[1], format='csr')
        A = LambdaI - Ks
        Kmat = splinalg.inv(A)
        return cast(sparse.spmatrix, Kmat)
    @staticmethod
    def eval():
        raise SyntaxError("Method not implemented!")

class dSLdn(Operator):
    def __init__(self, D1, type1, step1, D2, type2=None, step2=None):
        super().__init__(D1, type1, step1, D2, type2, step2)
        if D1 == D2:
            raise ValueError("This operator is not defined for identical boundaries!")
        self.Kmat = dSLdn.make_kernel_matrix(D1.points, D1.sigma, D2.points, D2.normal)
    
    @staticmethod
    def make_kernel_matrix(D, sigma_D, E, normal_E):
        """
        Parameters:
        -----------
        D : ndarray
            Source boundary points (2 x npts array)
        sigma_D : ndarray
            Weights/density values on source boundary (npts array)
        E : ndarray
            Target boundary points (2 x N array)
        normal_E : ndarray
            Normal vectors on target boundary (2 x N array) 
        Returns:
        --------
        Kmat: spmatrix
            Sparse kernel matrix in CSR format (npts x npts)
        """
        # Compute gradient of Green's function
        Gx, Gy = green.Green2D_grad(E, D)  # Shapes (N x npts)
        
        # Normal dot product with gradient
        Kx = normal_E[0,:] @ Gx  # (N,) @ (N x npts) -> (npts,)
        Ky = normal_E[1,:] @ Gy 
        
        # Diagonal matrices
        K_diag = sparse.diags(Kx + Ky, format='csr')  # (npts x npts)
        sigma_diag = sparse.diags(sigma_D, format='csr')  # (npts x npts)
        
        return K_diag @ sigma_diag #Shape (npts,npts)
    @staticmethod
    def eval():
        raise SyntaxError("Method not implemented because of the jump!")

class dDLdn(Operator):    
    hypersing: int = 0  # Hypersingular case flag
    
    def __init__(self, D1, type1, step1, D2, type2, step2):
        """
        Parameters:
        -----------
        D1 : C2Bound
            Source boundary
        type1 : str
            Basis type for D1 ('P1' required for self-interaction)
        step1 : int
            Discretization step
        D2 : Optional[C2Bound]
            Target boundary (defaults to D1)
        type2 : Optional[str]
            Basis type for D2 (defaults to type1)
        step2 : Optional[int]
            Discretization step for D2 (defaults to step1)
        """
        super().__init__(D1, type1, step1, D2, type2, step2)
        
        if self.D1 != self.D2:
            self.Kmat = self.make_kernel_matrix(self.D1.points, self.D1.normal, self.D1.sigma, self.D2.points, self.D2.normal)
        else:
            self.hypersing = 1
            self._setup_hypersingular_case()
    
    def _setup_hypersingular_case(self) -> None:
        """Configure operator for hypersingular case (D1 == D2)."""
        if self.typeBEM1 == 'P0' or self.typeBEM2 == 'P0':
            raise TypeError("P0 elements not supported for hypersingular operator")
        
        if self.typeBEM1 != 'P1' or self.typeBEM2 != 'P1':
            raise TypeError("Only P1 elements supported for hypersingular operator")
        
        npts = self.D1.get_nbpts()
        self.Psi = sparse.csr_matrix(BEM.P1_derivative(npts, self.stepBEM1, 2*np.pi))
        self.Phi = sparse.csr_matrix(BEM.P1_derivative(npts, self.stepBEM2, 2*np.pi))
        self.Psi_t = self.Psi.transpose()
        self.Phi_t = self.Phi.transpose()
        self.Kmat = sparse.csr_matrix((npts, npts))  # Empty matrix for hypersingular case
    
    def fwd(self, f: np.ndarray) -> np.ndarray:
        """
        Parameters:
        -----------
        f : np.ndarray
            Input function values 
        Returns:
        --------
        np.ndarray
            Operator applied to f
        """
        if self.D1 == self.D2:
            raise TypeError("Direct evaluation not supported for identical boundaries")
        return super().fwd(f)
    
    def apply_stiffmat(self, f: np.ndarray) -> np.ndarray:
        """
        Parameters:
        -----------
        f : np.ndarray
            Input function values
            
        Returns:
        --------
        np.ndarray
            Stiffness matrix applied to f
        """
        K = self.get_stiffmat()
        return K @ f
    
    def get_stiffmat(self) -> sparse.spmatrix:
        """Compute stiffness matrix.
        
        Returns:
        --------
        spmatrix
            Stiffness matrix in CSR format
        """
        if not self.hypersing:
            return super().get_stiffmat()
        
        sigma_diag = sparse.diags(self.D1.sigma, format='csr')
        Smat = SingleLayer.make_kernel_matrix(self.D1.points, self.D1.sigma)
        return self.Phi_t @ sigma_diag @ (Smat @ self.Psi)
    
    @staticmethod
    def make_kernel_matrix(D, normal_D, sigma_D, E, normal_E) -> sparse.spmatrix:
        """
        Parameters:
        -----------
        D : np.ndarray
            Source points (2 x M array)
        normal_D : np.ndarray
            Source normals (2 x M array)
        sigma_D : np.ndarray
            Source weights (M array)
        E : np.ndarray
            Target points (2 x N array)
        normal_E : np.ndarray
            Target normals (2 x N array)
            
        Returns:
        --------
        spmatrix
            Sparse kernel matrix (N x M) in CSR format
        """
        _, H = green.Green2D_Hessian(E, D)
        
        # Create block diagonal normal matrices
        normal_E_block = sparse.block_diag([sparse.diags(normal_E[0, :]), sparse.diags(normal_E[1, :])], format='csr')
        normal_D_block = sparse.block_diag([sparse.diags(normal_D[0, :]), sparse.diags(normal_D[1, :])], format='csr')
        
        Hn = -normal_E_block @ H @ normal_D_block
        sigma_diag = sparse.diags(sigma_D, format='csr')
        return Hn @ sigma_diag
    
    @staticmethod
    def eval() -> None:
        raise NotImplementedError("Direct evaluation not implemented for hypersingular operator")