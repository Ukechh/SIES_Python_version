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

def add_white_noise_mat(X, nlvl):
    W = np.random.normal(size=X.shape)
    m,n = X.shape
    Y = X + W * nlvl* np.linalg.norm(X,'fro') / np.sqrt(m*n)
    sigma = np.linalg.norm(X,'fro') * nlvl / np.sqrt(m*n)
    
    return Y, sigma

def add_white_noise_mat_complex(X, nlvl):
    W = np.random.normal(size=X.shape)
    V = np.random.normal(size=X.shape)
    
    epsilon = np.mean(abs(X))*nlvl
    
    Y = X + (W + 1j*V) * epsilon

    return Y, epsilon

def add_white_noise_list(data, nlvl):
    noisy_data = []
    variance = []
    for f in range(len(data)):
        X = data[f]
        if data[f].dtype() == np.complex128:
            Y , sigma = add_white_noise_mat_complex(X, nlvl)
        else:
            Y, sigma = add_white_noise_mat(X, nlvl)
        noisy_data.append(Y)
        variance.append(sigma)
    return noisy_data, variance