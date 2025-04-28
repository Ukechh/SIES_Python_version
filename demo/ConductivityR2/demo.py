import sys
import os

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '../../')))

import numpy as np
import matplotlib.pyplot as plt
from figure.Geom_figures import Ellipse, Banana, Triangle, Rectangle
from figure.C2Boundary.C2Boundary import C2Bound
from cfg import mconfig
from Operators.Operators import SingleLayer, LmKstarinv
from PDE.Conductivity_R2 import Conductivity
from FundamentalSols.green import Green2D_Dn, Green2D
from asymp.CGPT_methods import make_system_matrix_fast, make_block_matrix, lbda

#Make an inclusion
#B = Ellipse(1, 1/2, phi= 0, NbPts=2**10)
B = Rectangle(1,1/2, 2**10)
#B = Triangle(1, np.pi/3, npts= 2**10)
#B = Banana(np.zeros(2), 1, 1/10, np.array([1/2,1/2]).reshape(2,), 2**10)
#Plot the inclusion

#D = [B*1.5]
#D = [(B < np.pi / 3)]
D = [B]

fig, ax = plt.subplots(figsize=(6,6))  # size in inches

B.plot(ax=ax)
plt.show()

#npts = B.get_nbpts()
#print(npts)

#Set conductivity and permitivitty

cnd = 10*np.array([1]) 
pmtt = 1*np.array([1])

#Configuration of sources on a circle
cfg = mconfig.Coincided(np.zeros((2,1)), 2, 50, np.array([1.0, 2*np.pi, np.pi]), 0)

#Single Dirac point source
#cfg = mconfig.Coincided( np.array([-1,1]), 1, 1, np.array([1.0, np.pi, 2*np.pi]))


cfg.plot()
#cfg2.plot()
#Gn = Green2D_Dn(cfg2.neutSrc(0), B.points, B.normal)

P = Conductivity.Conductivity(D, cnd, pmtt, cfg)

freq = np.linspace(0,100*np.pi, 20)
#print(freq)
#print(R[0][:,].shape)
data, f = P.data_simulation(freq)

#Calculate and plot field
sidx = 20
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

freq = np.linspace(1,100*np.pi, endpoint=False, num= 1)
#Vx, Vy, Sx, Sy, mask = P2.calculate_FFv(np.array([0.01]), z0, width, N)

for f in freq:
    Vx, Vy, Sx, Sy, mask = P.calculate_FFv(np.array([f]), z0, width, N)
    P.plot_far_field(Vx, Vy, Sx, Sy, mask, f)
    P.plot_far_field_streamlines(Vx, Vy, Sx, Sy, mask)

#P2.plot_far_field(Vx, Vy, Sx, Sy, mask, 0.01)


