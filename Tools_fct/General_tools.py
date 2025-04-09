import numpy as np

def tensorplus(X,Y):
    X = np.ravel(X)[:, None]  
    Y = np.ravel(Y)[None, :]
    return X + Y
