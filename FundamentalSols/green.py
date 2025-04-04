import numpy as np

#Functions that compute the fundamental solution of the Laplacian in 2D and 3D

#Inputs: x-> the center (1x2 matrix)
#   y-> Array of points, (nx2 matrix)
def Green2D(c, y):
    z = c-y
    r = (1/(2*np.pi))*np.log(np.linalg.norm(z,2,axis=1));
    return r

def Green3D(x,y=np.zeros(3)):
    z = x-y;
    r = (-1/(4*np.pi))*np.linalg.norm(z,2,axis=0)**(-1);
    return r
