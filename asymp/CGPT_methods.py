import sys
import os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '../')))

import numpy as np
import math
from itertools import combinations
from Operators import Operators
from scipy.sparse import csr_matrix

def lbda(cnd, pmtt = np.array([]), freq=np.array([]), drude= False, tau = 1.0):
    """
    Compute the parameter lambda(λ) used in boundary integral operators,
    based on conductivity, permittivity, and frequency.

    Parameters:
    -----------
    cnd : ndarray of shape (NbIncl,)
        Array of conductivities for each inclusion. Values must be > 0 and ≠ 1.

    pmtt : ndarray of shape (NbIncl,), optional
        Array of permittivities. Defaults to zero if not provided.

    freq : float or ndarray of shape (1,), optional
        Frequency at which to evaluate λ. Must be a positive float or array-like with one positive value.

    Returns:
    --------
    lam : ndarray of shape (NbIncl,)
        Computed complex λ values corresponding to each inclusion.
    """
    freq = np.atleast_1d(freq) #transform the float into an array if it isnt already
    if pmtt.shape[0] == 0:
        pmtt = np.zeros_like(cnd)
    if not np.issubdtype(freq.dtype, np.floating) or np.any(freq < 0):
        raise ValueError("Frequency must be a positive scalar")  
    if np.any((cnd==1) | (cnd< 0)):
        raise ValueError('Invalid value of conductivity')
    
    if drude:
        cnd = cnd * (1 / (1 + 1j * freq * tau) )
    
    toto = cnd + 1j * pmtt * freq
    
    l = ((toto + 1) / (2*(toto - 1))).ravel()

    if abs(l).any() < 1/2:
        raise Warning('Module of lambda is less than 1/2!')
    
    return l

def make_block_matrix(D, V=None):
    """
    Constructs a block matrix for multiple inclusions.
    
    Parameters
    ----------
    D : list
        List of C2Boundary inclusion objects defining the geometry.
    V : ndarray, optional
        Scaling factors for normal vectors, defaults to ones.
        
    Returns
    -------
    KsdS : list of lists
        Block matrix where diagonal blocks are Kstar kernels and 
        off-diagonal blocks are normal derivatives of Single Layer operator.
        
    Raises
    ------
    ValueError
        If inclusions are not mutually disjoint.
    """
    nbIncl = len(D)
    
    if V is None:
        V = np.ones((nbIncl,1))
    for m, n in combinations(range(nbIncl), 2):
        if not D[m].isdisjoint(D[n]):
            raise ValueError('Inclusions must be mutually disjoint.')
    
    KsdS = [[None for _ in range(nbIncl)] for _ in range(nbIncl)]

    for n in range(nbIncl):
        DiagMat = Operators.Kstar.make_kernel_matrix(D[n]._points, D[n]._tvec, D[n]._normal, D[n]._avec, D[n].sigma) # Shape (npts, npts)
        KsdS[n][n] = -csr_matrix(DiagMat).toarray()
        for m in range(nbIncl):
            if m != n:
                KsdS[m][n] = -Operators.dSLdn.make_kernel_matrix(D[n].points, D[n].sigma, D[m].points, (D[m].normal)*V[m] ) # shape (npts,npts)
    return KsdS #Block matrix of shape (NbIncl, NbIncl) where every block is of shape (npts,npts)

def make_system_matrix(D, l):
    KsdS = make_block_matrix(D)
    Amat, Acell = make_system_matrix_fast(KsdS,l)
    return Amat, Acell

def make_system_matrix_fast(KsdS, l):
    '''
    Construct the matrix A in the system A[phi] = b by reusing the block matrices.
    Parameters:
    -----------
    KsdS: List 
        list of lists of 2D numpy arrays (block matrix)
    l: ndarray 
        contrast constants for each inclusion
    Returns:
    -----------
    Amat: ndarray
        Full system matrix
    Acell: List 
        Updated block matrix
    '''
    nb_incls = len(KsdS)
    Acell = [[np.empty((0,0)) for _ in range(nb_incls)] for _ in range(nb_incls)]
    
    for i in range(nb_incls):
        for j in range(nb_incls):
            block = KsdS[i][j]
            if not isinstance(block, np.ndarray):
                raise TypeError(f"KsdS[{i}][{j}] is not a numpy array, got {type(block)}")
            Acell[i][j] = block

    for n in range(nb_incls):
        size = KsdS[n][n].shape[0]
        dtype = KsdS[n][n].dtype
        Acell[n][n] = l[n] * np.eye(size, dtype=dtype) + KsdS[n][n]

    Amat = np.block(Acell)
    return Amat, Acell
    
def theoretical_CGPT(D, lam, ord):
    KsdS = make_block_matrix(D)
    M = theoretical_CGPT_fast(D, KsdS, lam, ord)
    return M

def theoretical_CGPT_fast(D, KsdS, lam, ord): 
    epsilon = 1e-8
    npts = D[0].nb_points
    nbIncl = len(D)

    if len(lam) < nbIncl:
        raise ValueError('Value of lambda must be specified for each inclusion.')
    
    Amat0, _  = make_system_matrix_fast(KsdS, lam)
    if min(abs(lam-1/2)) < epsilon:
        r = math.floor(Amat0.shape[1] / nbIncl)
        o = np. ones((1,r))
        k = np.kron(np.eye(nbIncl), o)
        Amat = np.vstack((Amat0,  k))
    else:
        Amat = Amat0
    CC = np.zeros((ord, ord), dtype=np.complex128)
    CS = np.zeros((ord, ord), dtype=np.complex128)
    SC = np.zeros((ord, ord), dtype=np.complex128)
    SS = np.zeros((ord, ord), dtype=np.complex128)
    for m in range(ord):
        B = np.zeros((npts, nbIncl), dtype=np.complex128)
        for i in range(nbIncl):
            dm = m * (D[i].cpoints)**(m-1)
            toto = D[i].normal[0,:] * dm + D[i].normal[1,:] * dm * 1j;
            B[:, i] = toto.ravel()
            
            if min(abs(lam-1/2)) < epsilon:
                z = np.zeros((nbIncl,1), dtype=np.complex128)
                b = np.vstack( (B.ravel(), z))
            else:
                b = B.ravel()
            toto = np.linalg.solve(Amat, b.real)
            rphi = toto.reshape(npts, nbIncl)
            
            toto = np.linalg.solve(Amat, b.imag)
            iphi = toto.reshape(npts, nbIncl)
            for n in range(ord):
                for i in range(nbIncl):
                    zn = D[i].cpoints ** n * D[i].sigma
                    CC[m,n] = CC[m,n] + zn.real @ rphi[:,i]
                    CS[m,n] = CS[m,n] + zn.imag @rphi[:,i]
                    SC[m,n] = SC[m,n] + zn.real @ iphi[:,i]
                    SS[m,n] = SS[m,n] + zn.imag @ iphi[:,i]
    M = np.block([[CC, CS],[SC, SS]])
    return M

