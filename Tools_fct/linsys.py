import numpy as np

def SXR_op(X, S : np.ndarray, R : np.ndarray,transp_flag : bool):
    """
    This function implements the linear operator:
    L(M) := S M R^H, (^H is the hermitian of a matrix)
    S and R are matrices of source and receiver
    and the adjoint of L is:
        L^*(D) := S^H D R 

    """
    Ns, Ms = S.shape
    Nr, Mr = R.shape

    if transp_flag:
        Y = S.conj().T @ X.reshape((Ns, Nr)) @ R
    else:
        Y = S @ X.reshape((Ms, Mr)) @ R.conj().T
    return Y.ravel(order='F')

def SXR_op_symm(X, S, R, transp_flag : bool):
    """
    This function implements the linear operator:
    L(M) := S (M+M^T) R^H
    and the adjoint of L is:
    L^H(D) := S^H D R + R^T D^T conj(S)
    """
    Ns, Ms = S.shape
    Nr, Mr = R.shape
    
    if transp_flag:
        Xm = X.reshape(Ns,Nr)
        Y = S.conj().T @ Xm @ R + R.T @ Xm.T @ S.conj()
    else:
        Xm = X.reshape((Ms,Mr))
        Y = S @ (Xm + Xm.T) @ R.conj().T
    return Y.ravel(order='F')

def SXR_op_list(X,S,R, transp_flag):
    Ns, Ms = S.shape
    Nr, Mr = R[0].shape

    if transp_flag:
        Z = X.reshape(Ns, Nr)
        W = np.zeros((Ns, Mr), dtype=X.dtype)
        for s in range(Ns):
            W[s, :] = Z[s, :] @ R[s]
        Y = S.conj().T @ W 
    else:
        Xm = X.reshape(Ms, Mr)
        Y = np.zeros((Ns, Nr), dtype=X.dtype)
        for s in range(Ns):
            Y[s, :] = S[s, :] @ Xm @ R[s].conj().T
    return Y.ravel(order='F')

def SXR_op_symm_list(X,S,R,transp_flag):
    """
    This function evaluates S X R for S = (Ns,Ms) R = (Nr, Mr) a
    """
    Ns, Ms = S.shape
    Nr, Mr = R[0].shape  # R is a list of Ns matrices

    if transp_flag:
        Z = X.reshape(Ns, Nr)
        Zt = Z.T
        W1 = np.zeros((Ns, Mr), dtype=X.dtype)
        W2 = np.zeros((Mr, Ns), dtype=X.dtype)
        for s in range(Ns):
            W1[s, :] = Z[s, :] @ R[s]
            W2[:, s] = R[s].T @ Zt[:, s]
        Y = S.T @ W1 + W2 @ np.conj(S)
    else:
        Xm = X.reshape(Ms, Mr)
        Xm = Xm + Xm.T  # symmetric part
        Y = np.zeros((Ns, Nr), dtype=X.dtype)
        for s in range(Ns):
            Y[s, :] = S[s, :] @ Xm @ R[s].T  
    return Y.ravel(order='F')
