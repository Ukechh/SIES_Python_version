import sys
import os

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '../../')))

import numpy as np
import matplotlib.pyplot as plt
from figure.Geom_figures import Ellipse, Banana, Triangle, Rectangle
from figure.C2Boundary.C2Boundary import C2Bound
from cfg import mconfig
from PDE.Conductivity_R2.Invariant_testing import compare_invariants_over_transformations

B = Ellipse(1, 1 / 2, phi=0, NbPts=2 ** 9)
# B = Rectangle(1,1/2, 200)
# B = Triangle(1, np.pi/3, npts= 2**10)
# B = Banana(np.zeros(2), 1, 1/10, np.array([1/2,1/2]).reshape(2,), 2**10)
D = [B]
# Plot the inclusion
# D = [B*1.5]
# D = [(B < np.pi / 3)]


#Set conductivity and permitivitty

cnd = 10*np.array([1]) 
pmtt = 5*np.array([1])

#Configuration of sources on a circle
cfg = mconfig.Coincided(np.zeros((2,1)), 2, 50, np.array([1.0, 2*np.pi, np.pi]), 0)

freq = np.linspace(0,5*np.pi, 10)

compare_invariants_over_transformations(freq, B, 10, cnd, pmtt, cfg, ord=2, drude=True)