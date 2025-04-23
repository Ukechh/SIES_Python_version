import numpy as np

def convfix(X, width):
    Y = X;
    if width <= 0:
        return Y
    else:
        X = X.reshape(1,-1)
        if len(X) == 0:
            raise ValueError("Input vector must not be empty!")
        h = np.ones(width) / width
        toto = np.ones(width-1)
        X1 = np.array([toto * X[0], X, toto * X[-1]])
        Y1 = np.convolve(X1,h, 'full')
        Y = Y1[width - 1 : -width + 1]
    return Y
