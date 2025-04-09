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

