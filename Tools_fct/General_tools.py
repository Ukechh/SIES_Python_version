import numpy as np

def tensorplus(X,Y):
    #Computes the tensor sum of two vectors
    X = np.ravel(X)[:, None]  #Makes X into a row vector
    Y = np.ravel(Y)[None, :] #Make Y into a column vector
    return X + Y 
