import numpy as np
from figure.C2Boundary.C2Boundary import C2Bound


class SmallInclusion:
    _D : list[C2Bound]
    _nbIncl : int
    _cfg : str

    def __init__(self, D, cfg):
        if D.shape[0] == 1:
            self.addInclusion(D)
        else:
            for i in range(D.shape[0]):
                self.addInclusion(D[i])
        