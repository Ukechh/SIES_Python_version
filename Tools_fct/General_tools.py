import numpy as np

def tensorplus(X,Y):
    #Computes the tensor sum of two vectors
    X = np.ravel(X)[:, None]  #Makes X into a row vector
    Y = np.ravel(Y)[None, :] #Make Y into a column vector
    return X + Y 

def bdiag(A,idx):
    M,N = A.shape
    if idx == 0:
        D = np.zeros((M * N, N))
        for n in range(N):
            D[n*M:(n+1)*M, n] = A[:, n]
    else:
        D = np.zeros((M, M * N))
        for m in range(M):
            D[m, m*N:(m+1)*N] = A[m, :]
    return D

def cart2pol(x):
    """
    Transforms a (2,) shape array from cartesian to polar coordinates
    """
    r = np.hypot(x[0], x[1])
    theta = np.arctan2(x[1], x[0])
    return r, theta