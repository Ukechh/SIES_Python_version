import sys
import os

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '../../')))

import numpy as np
import math
from Tools_fct.General_tools import cart2pol
from cfg.mconfig import mconfig, Coincided
from Tools_fct.linsys import SXR_op, SXR_op_symm, SXR_op_list, SXR_op_symm_list

def make_linop_CGPT(cfg, ord, symmode : bool = False):
    """
    Constructs linear operator for Generalized Polarization Tensors (GPT) computation.
    This function creates a linear operator for computing CGPTs (Contracted Generalized 
    Polarization Tensors) based on the given configuration and order. It handles both single 
    and multiple Dirac sources, and supports symmetric and non-symmetric modes.
    Parameters
    ----------
    cfg : Configuration object
        Contains all configuration parameters including source and receiver positions,
        center point, and Dirac source related parameters.
    ord : int
        Order of the GPT computation.
    symmode : bool, optional
        Whether to use symmetric mode operators. Defaults to False.
    Returns
    -------
    tuple
        - L : callable
            Linear operator function that takes array x and tflag as arguments.
            The tflag determines whether to apply forward or adjoint operation.
        - As : ndarray
            Source coefficient matrix of shape (Ns_total, 2*ord) for single Dirac case,
            or (Ns_total, 2*ord) for multiple Dirac case after contraction.
        - Ar : ndarray or list
            Receiver coefficient matrix. For single group (Ng=1), it's an array of shape 
            (Nr_total, 2*ord). For multiple groups, it's a list of arrays, each of shape
            (Nr, 2*ord).
    Notes
    -----
    The function handles three main cases:
    1. Single Dirac source configuration
    2. Multiple Dirac sources configuration
    3. Different receiver group configurations (single vs multiple groups)
    The returned linear operator L is constructed differently based on these cases and
    the symmode parameter, using either SXR_op or SXR_op_symm for single groups,
    or SXR_op_list or SXR_op_symm_list for multiple groups.
    """

    #As is of shape (Ns_total, 2*ord)
    if cfg.nbDirac == 1:
        As = make_matrix_A(cfg.all_src(), cfg._center, ord)
    else:
        src = np.zeros((2, cfg.nbDirac*cfg._Ns_total)) 
        for s in range(cfg._Ns_total):
            src[:,s*cfg.nbDirac: (s+1)*cfg.nbDirac] = cfg.neutSrc(s)
        As0 = make_matrix_A(src, cfg._center, ord) #As0 has shape (nbDirac*Ns_total, 2*ord)
        K = np.kron(np.eye(cfg._Ns_total), cfg.neutCoeff.reshape(1, -1)) #K has shape (Ns_total*1, Ns_total*nbDirac)
        As = K @ As0 #As is of shape (Ns_total, 2*ord)

    if cfg._Ng == 1:
        if isinstance(cfg, Coincided) and cfg.nbDirac == 1:
            Ar = As.copy() #Ar has shape (Ns_total, 2*ord)
        else:
            Ar = make_matrix_A(cfg.all_rcv, cfg._center, ord) #Shape (Nr_total, 2*ord)
        if symmode:
            L = lambda x, tflag: SXR_op_symm(x, As, Ar, tflag)
        else:
            L = lambda x, tflag: SXR_op(x, As, Ar, tflag)
    else:
        Ar = []
        for n in range(cfg._Ns_total):
            Ar.append(make_matrix_A(cfg.rcv(n), cfg._center, ord)) #Each Ar is of shape (Nr, 2*ord)
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

def make_matrix_A(Xs, z, ord):
    """
    Creates the A matrix involved in reconstruction of CGPTs
    Parameters:
    ---------------
    Xs: Coordinates of sources / receivers
        np.ndarray (2, N)
    z: Reference center
    ord: Highest order of CGPT
    
    Returns:
    -------------
    matrix A of shape (N, 2*ord)
    """
    N = Xs.shape[1]
    A = np.zeros((N, 2*ord), dtype=np.complex128)
    z = z.reshape((2,1))

    toto = Xs - z
    r, thetas = cart2pol(toto)
    for m in range(ord):
        A[:, 2*m:2*(m+1)] = np.array([np.cos((m+1)*thetas) / (2*np.pi*(m+1)*(r**(m+1)) ), np.sin((m+1)*thetas) / (2*np.pi*(m+1)*(r**(m+1)) ) ]).T    
    return A

