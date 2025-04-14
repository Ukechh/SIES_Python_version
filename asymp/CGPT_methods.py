import numpy as np

def lbda(cnd, pmtt = np.array([]), freq=0.0):
    if pmtt.shape[0] == 0:
        pmtt = np.zeros_like(cnd)
    if (not isinstance(freq, float)) or freq < 0:
        raise ValueError("Frequency must be a positive scalar")  
    if np.any((cnd==1) | (cnd< 0)):
        raise ValueError('Invalid value of conductivity')
    
    toto = cnd + 1j * pmtt * freq
    return (toto + 1) / (2*(toto - 1))

def make_system_matrix():
    pass
    




