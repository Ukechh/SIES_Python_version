import sys
import os

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '../../')))

import numpy as np
from figure.Geom_figures import Ellipse
from figure.C2Boundary.C2Boundary import C2Bound
from cfg import mconfig
from PDE.Conductivity_R2 import Conductivity

#Make an inclusion
B = Ellipse(1, 1/2, 2**10)
#Plot the inclusion
B = B<(-np.pi/4)
#B.plot()

cnd = 10*np.array([1,1]) 
pmtt = 1*np.array([1,1])

cfg = mconfig.Coincided(np.zeros((2,1)), 10.0, 50)
B = np.array([B])
#cfg.plot()
#P = Conductivity.Conductivity(B, cnd, pmtt, cfg)

