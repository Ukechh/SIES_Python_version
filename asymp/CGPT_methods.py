import sys
import os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '../')))

import numpy as np
from itertools import combinations
from Operators import Operators

def lbda(cnd, pmtt = np.array([]), freq=0.0):
    freq = np.atleast_1d(freq) #transform the float into an array if it isnt already
    if pmtt.shape[0] == 0:
        pmtt = np.zeros_like(cnd)
    if not np.issubdtype(freq.dtype, np.floating) or np.any(freq < 0):
        raise ValueError("Frequency must be a positive scalar")  
    if np.any((cnd==1) | (cnd< 0)):
        raise ValueError('Invalid value of conductivity')
    
    toto = cnd + 1j * pmtt * freq
    return (toto + 1) / (2*(toto - 1))

def make_block_matrix(D,V=None):
    #
    nbIncl = len(D)
    if V is None:
        V = np.ones((nbIncl,1))
    for m, n in combinations(range(nbIncl), 2):
        if not D[m].isdisjoint(D[n]):
            raise ValueError('Inclusions must be mutually disjoint.')
    KsdS = np.empty((nbIncl, nbIncl), dtype=object)
    for n in range(nbIncl):
        KsdS[n][n] = -Operators.Kstar.make_kernel_matrix(D[n]._points, D[n]._tvec, D[n]._normal, D[n]._avec, D[n].sigma)
        for m in range(nbIncl):
            if m != n:
                KsdS[m][n] = -Operators.dSLdn.make_kernel_matrix(D[n].points, D[n].sigma, D[m].points, (D[m].normal)*V[m] )
    return KsdS

def make_system_matrix(D, l):
    KsdS = make_block_matrix(D)
    Amat, Acell = make_system_matrix_fast(KsdS,l)
    return Amat, Acell

def make_system_matrix_fast(KsdS, l):
    
    #Construct the matrix A in the system A[phi] = b by reusing the block matrices.
    
    #Parameters:
    #- KsdS: list of lists of 2D numpy arrays (block matrix)
    #- lam: array-like, contrast constants for each inclusion
    
    #Returns:
    #- Amat: the full system matrix
    #- Acell: the updated block matrix
    
    nb_incls = len(KsdS)

    if len(l) < nb_incls:
        raise ValueError('Value of lambda must be specified for each inclusion.')

    Acell = [row[:] for row in KsdS]

    for n in range(nb_incls):
        size = KsdS[n][n].shape[0]
        Acell[n][n] = l[n] * np.eye(size) + KsdS[n][n]

    # Convert block matrix to full matrix
    Amat = np.block(Acell)

    return Amat, Acell
    




