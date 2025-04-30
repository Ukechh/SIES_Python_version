import numpy as np
from Tools_fct import General_tools
#Functions that compute the fundamental solution of the Laplacian in 2D

def Green2D(x, y):
    #Inputs: X,Y arrays (2,n), (2,m) of points
    #Output. Matrix of size (n,m) where the (i,j)-th entry is the 2D Green function evaluated at (X[:,i]-Y[:,j])
   if x.shape[0] != 2 or y.shape[0] != 2:
       raise ValueError("The inputs must have 2 rows!")
   X1 = General_tools.tensorplus(x[0,:],-y[0,:])
   X2 = General_tools.tensorplus(x[1,:],-y[1,:])

   G = 1/(4*np.pi)*np.log(X1**2 + X2**2) #The fundamental solution to the Laplacian is given by G(x,y) = 1/(2*pi) log(|x-y|), here we first do the element-wise substraction
   #And then use that log(|X|) = log(|X|**2) / 2 to define G 
   return G

def Green2D_grad(x,y):
    #Inputs: X,Y arrays (2,n), (2,m) of points
    #Output: matrix Gx (Gy) size (n, m) where the (i,j)-th entry is the Dx derivative (resp. Dy) of the 2D Green function
    #evaluated at X[:,i]-Y[:,j]
    if x.shape[0] != 2 or y.shape[0] != 2 :
        raise ValueError("Value error, given arrays are not valid!")
    X1 = General_tools.tensorplus(x[0,:], -y[0,:])
    X2 = General_tools.tensorplus(x[1,:], -y[1,:])
    S = X1**2 + X2**2
    Gx = (1 / (2*np.pi)) * X1 / S
    Gy = (1 / (2*np.pi)) * X2 / S
    return Gx, Gy

def Green2D_Dn(x,y, normal):
    # Normal derivative of the 2D Green function F(y)=G(y,x) = 1/2/pi * log|y-x| with respect to y on a
    # boundary. We compute <DF(y), n_y>. Remark that although mathematically speaking G(x,y)=G(y,x),
    # however the order of the arguments must be "X, Y, normal" here.
    #
    # Inputs:
    # X: 2 X M points
    # Y, normal: 2 X N boundary points and the normal vector
    # Output:
    # G is a matrix of shape (M,N) whose (m,n)-th term is the normal derivative evlauated at (X(:,m), Y(:,n)).
    Gx, Gy = Green2D_grad(y,x)
    Gn = np.diag(normal[0, :]) @ Gx + np.diag(normal[1, :]) @ Gy 
    return Gn.T


def Green2D_Hessian(X,Y):
    n = X.shape[1]
    m = Y.shape[1]
    H = np.zeros((2*n,2*m))

    XY1 = General_tools.tensorplus(X[0,:],-Y[0,:])
    XY2 = General_tools.tensorplus(X[1,:],-Y[1,:])

    DN = 2*np.pi*(XY1**2+XY2**2)**2
    M1 = (XY2**2-XY1**2)/ DN
    M2 = -(2*XY1*XY2) / DN
    M3 = (XY1**2-XY2**2)/ DN
    
    H[0::2, 0::2] = M1
    H[0::2, 1::2] = M2
    H[1::2, 0::2] = M2
    H[1::2, 1::2] = M3

    H1 = np.block([[M1, M2],
               [M2, M3]])
    return H, H1 #Shape of both returns is (2n, 2m)
