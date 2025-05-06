import sys
import os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '../')))

import numpy as np
from itertools import combinations
from Operators import Operators
from scipy.sparse import csr_matrix

def lbda(cnd, pmtt = np.array([]), freq=np.array([])):
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
    
    toto = cnd + 1j * pmtt * freq
    return ((toto + 1) / (2*(toto - 1))).ravel()

def make_block_matrix(D, V=None):
    """
    Parameters:
    -----------
    D : list
        A list of C2Boundary inclusion objects
    V : ndarray, optional
        A vector of scaling factors, where `V[m]` scales the normal vector for the off-diagonal 
        blocks.

    Returns:
    --------
    KsdS : list of lists of numpy.ndarray
        A 2D list representing the block matrix. Each element is a dense 
        numpy array (2D) corresponding to a block in the matrix. The diagonal 
        blocks are the Kstar kernel matrix and the off-diagonal
        blocks are the kernel matrix of the normal derivative of the Single Layer operator
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
    




