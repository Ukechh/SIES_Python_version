import numpy as np


def P1_basis(M,m):
    if M % m:
        raise ValueError("The number of boundary points must be a multiplier of the step length.")
    N= M/m
    u = np.zeros(N,M)
    #Hat function centered at the first boundary point, which is periodic
    hat = np.zeros(1,M)
    hat[0:2*m+1] = np.concatenate([np.arange(m,dtype=float) / m, [1], np.arange(m-1,-1,-1,dtype=float) / m])
    u[0,:] = np.roll(hat, shift=-m, axis = 1)
    for i in range(1, N):
        u[i,:] = np.roll(u[i-1,:], shift = m, axis= 1)
    u = u.T
    return u

def P1_derivative(M,m,L):
    if M % m != 0:
        raise ValueError("The number of boundary points must be a multiplier of the step length")
    N = M /m
    du = np.zeros((N,M))
    
    dhat = np.zeros(M)
    dhat[0:2*m+1] = np.concatenate((np.ones(m),[0],-np.ones(m)))

    du[0,:] = np.roll(dhat,shift=-m)
    for i in range(1,N):
        du[i,:] = np.roll(du[i-1,:], shift=m)

    du = M / (L*m) * du.T
    return du  

def interpolation(Psi, Y, idx=None):
    if idx is None:
        idx = []
    V = Psi @ Y.reshape(Psi.shape[1], -1)
    if len(idx) != 0:
        return V[idx,:]
    else:
        return V


