import numpy as np
from Tools_fct import General_tools
#Functions that compute the fundamental solution of the Laplacian in 2D and 3D

def Green2D(x, y):
    #Inputs: X,Y arrays (2,n), (2,m) of points
    #Output. Matrix of size (n,m) where the (i,j)-th entry is the 2D Green function evaluated at (X[:,i]-Y[:,j])
   if x.shape()[0] != 2 or y.shape()[0] != 2:
       raise ValueError("The inputs must have 2 rows!")
   X1 = General_tools.tensorplus(x[0,:],-y[0,:])
   X2 = General_tools.tensorplus(x[1,:],-y[1,:])
   G = 1/(4*np.pi)*np.log(X1**2 + X2**2)
   return G

def Green2D_grad(x,y):
    #Inputs: X,Y arrays (2,n), (2,m) of points
    #Output: matrix Gx (Gy) size (n, m) where the (i,j)-th entry is the Dx derivative (resp. Dy) of the 2D Green function
    #evaluated at X[:,i]-Y[:,j]
    X1 = General_tools.tensorplus(x[0,:],-y[0,:])
    X2 = General_tools.tensorplus(x[1,:],-y[1,:])
    S = X1**2 + X2**2
    Gx = (1 / (2*np.pi)) * X1 / S
    Gy = (1 / (2*np.pi)) * X2 / S
    return Gx, Gy

def Green3D(x,y):
    z = x-y;
    r = (-1/(4*np.pi))*np.linalg.norm(z,2,axis=0)**(-1);
    return r
