import sys
import os

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '../../')))

import numpy as np
import matplotlib.pyplot as plt
from figure.Geom_figures import Ellipse, Banana, Triangle, Rectangle, Flower
from PDE.Conductivity_R2.Conductivity import Conductivity
from cfg import mconfig
from PDE.Conductivity_R2.Invariant_testing import compare_invariants_over_rotations, compare_invariants_over_translations

B = Ellipse(1, 1 / 2, phi=np.pi / 4, NbPts=2 ** 9)
E = Rectangle(1,1/2, 200)
C = Triangle(1, np.pi/3, npts= 2**10)
A = Flower(0.5, 1, 2**10, 5,0.3, tau=0.1)
# B = Banana(np.zeros(2), 1, 1/10, np.array([1/2,1/2]).reshape(2,), 2**10)
E = E*0.3
A = (A < np.pi*1.8)*0.3
#E = E + np.array([0.5, -0.5]).reshape((2, 1))
D = [A]
# Plot the inclusion
# D = [B*1.5]
# D = [(B < np.pi / 3)]


#Set conductivity and permitivitty

cnd = 10*np.array([1]) 
pmtt = 5 * np.array([1])

#Configuration of sources on a circle
cfg = mconfig.Coincided(np.zeros((2,1)), 1, 50, np.array([1.0, 2*np.pi, np.pi]), 0)
#Set up working frequency
freq = np.linspace(0,5*np.pi, 10)
#Generate a conductivity instance
P = Conductivity(D, cnd, pmtt, cfg, drude=True)
#Set up center
z0 = P._cfg._center
#Set up the source
s = 12
#Set up size (number of points and width) of the grid
N = 100
width = 3
#Set up the grid
for f in freq:
    F, F_bg, Sx, Sy, mask = P.calculate_field(f, s, z0, width, N)
    P.plot_field(s, F, F_bg, Sx, Sy, nbLine=30)

compare_invariants_over_rotations(freq, A, 3, cnd, pmtt, cfg, ord=2, drude=True, normalize='sum')
""" compare_invariants_over_rotations(freq, B, 15, cnd, pmtt, cfg, ord=2, drude=True, normalize='max')
compare_invariants_over_translations(freq, B, 15, cnd, pmtt, cfg, ord=2, drude=True, normalize='sum')
compare_invariants_over_translations(freq, B, 15, cnd, pmtt, cfg, ord=2, drude=True, normalize='max') """