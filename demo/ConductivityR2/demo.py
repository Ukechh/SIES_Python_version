import sys
import os

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '../../')))

import numpy as np
from figure.Geom_figures import Ellipse
from figure.C2Boundary.C2Boundary import C2Bound
from cfg import mconfig
from Operators.Operators import SingleLayer, LmKstarinv
from PDE.Conductivity_R2 import Conductivity
from FundamentalSols.green import Green2D_Dn, Green2D
from asymp.CGPT_methods import make_system_matrix_fast, make_block_matrix, lbda

#Make an inclusion
B = Ellipse(1, 1/2, phi= 0, NbPts=2**10)
#Plot the inclusion
D = [(B<np.pi /3)]

#B.plot()

#npts = B.get_nbpts()
#print(npts)

#Set conductivity and permitivitty

cnd = 10*np.array([1]) 
pmtt = 1*np.array([1])

#Configuration of sources on a circle
cfg = mconfig.Coincided(np.zeros((2,1)), 10, 50, np.array([1.0, 2*np.pi, 2*np.pi]), 0)

#Single Dirac point source
cfg2 = mconfig.Coincided(np.zeros((2,1)), 1, 1)
#cfg.plot()
#cfg2.plot()
#Gn = Green2D_Dn(cfg2.neutSrc(0), B.points, B.normal)

#P = Conductivity.Conductivity([B], cnd, pmtt, cfg)

P = Conductivity.Conductivity(D, cnd, pmtt, cfg)

P2 = Conductivity.Conductivity(D, cnd, pmtt, cfg2)

freq = np.linspace(0,100*np.pi, 5)
#print(freq)
#print(R[0][:,].shape)
data, f = P.data_simulation(freq)

#Calculate and plot field
sidx = 0
z0 = np.zeros((2,1))
N = 100
width=6
epsilon = width / ((N-1)*5)

#Sx, Sy, mask = P._D[0].boundary_off_mask(np.zeros((2,1)), width, N, epsilon)
#Z = np.vstack((Sx.ravel(),Sy.ravel()))
#Hs = Green2D(Z, P._cfg._src[0])

#Phi = P.compute_Phi(0.01, s=sidx)
#print(Phi[0].shape)

F, F_bg, SX, SY, mask = P.calculate_field(np.array([0.5]), sidx, z0, width, N)
#F, F_bg, SX, SY, mask = P2.calculate_field(np.array([0.01]), sidx, z0, width, N)

P.plot_field(sidx, F, F_bg, SX, SY, 100)
#P2.plot_field(sidx, F, F_bg, SX, SY, 100)

M = LmKstarinv.make_kernel_matrix(1, B._points, B._tvec, B._avec, B._normal, B.sigma)

Vx, Vy, Sx, Sy, mask = P.calculate_FFv(np.array([0.5]), z0, width, N)
#Vx, Vy, Sx, Sy, mask = P2.calculate_FFv(np.array([0.01]), z0, width, N)

P.plot_far_field(Vx, Vy, Sx, Sy, mask, 0.5)
#P2.plot_far_field(Vx, Vy, Sx, Sy, mask, 0.01)
