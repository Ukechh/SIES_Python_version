import sys
import os

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '../../')))

import numpy as np
from figure.Geom_figures import Ellipse
from figure.C2Boundary.C2Boundary import C2Bound
from cfg import mconfig
from PDE.Conductivity_R2 import Conductivity
from FundamentalSols.green import Green2D_Dn

#Make an inclusion
B = Ellipse(1, 1/2, 2**10)
#Plot the inclusion
B = B<(-np.pi/4)
#B.plot()
npts = B.get_nbpts()

#Set conductivity and permitivitty

cnd = 10*np.array([1,1]) 
pmtt = 1*np.array([1,1])

cfg = mconfig.Coincided(np.zeros((2,1)), 10, 50, np.array([1.0, 1/16 *np.pi, 2*np.pi]), False, np.array([1, -1]), 0.01)
Gn = Green2D_Dn(cfg.src(0), B.points, B.normal)
e = cfg.neutSrc

B = np.array([B])
#cfg.plot()
#P = Conductivity.Conductivity(B, cnd, pmtt, cfg)
#freq = np.linspace(0.1,2*np.pi, 10)
#print(freq)
#P.compute_Phi(freq)

