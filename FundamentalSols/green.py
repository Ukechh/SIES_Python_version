import numpy as np

#Functions that compute the fundamental solution of the Laplacian in 2D and 3D

#Inputs: x-> the center (2x1 matrix)
#   y-> Array of points, (2xn matrix)

def Green2D(x, y):
    z = x-y
    r = (1/(2*np.pi))*np.log(np.linalg.norm(z,2,axis=0));
    return r

def Green3D(x,y):
    z = x-y;
    r = (-1/(4*np.pi))*np.linalg.norm(z,2,axis=0)**(-1);
    return r
