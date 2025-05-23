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
from PDE.Conductivity_R2.make_CGPT import make_matrix_A
from FundamentalSols.green import Green2D_Dn, Green2D
from asymp.CGPT_methods import make_system_matrix_fast, make_block_matrix, lbda
from GPT.Expansions import inner_expansion_conductivityR2_0, plot_inner_expansion

#Make an inclusion
#B = Ellipse(1, 1/2, phi= 0, NbPts=2**9)
#B = Rectangle(1,1/2, 200)
B = Triangle(1, np.pi/3, npts= 2**10)
#B = Banana(np.zeros(2), 1, 1/10, np.array([1/2,1/2]).reshape(2,), 2**10)
#Plot the inclusion
B = B.global_perturbation(0.01,12,1)
print('B has nb of points:', B.nb_points)
ax = plt.subplot()
B.plot(ax=ax, color='blue')

z = np.array([0.5,0.5]).reshape((2,1))
B = (B + z)*0.2 < np.pi / 3
B.plot(ax=ax, color='red')
plt.show()
#D = [B*1.5]
#D = [(B < np.pi / 3)]
D = [B]
print(f'B has {B.points.shape} points')
#fig, ax = plt.subplots(figsize=(6,6))  # size in inches

#B.plot(ax=ax)
#plt.show()

#Set conductivity and permitivitty
cnd = 10*np.array([1])  
pmtt = 5*np.array([1])

#Configuration of sources on a circle
cfg = mconfig.Coincided(np.zeros((2,1)), 2, 20, np.array([1.0, 2*np.pi, np.pi]), 0)
#cfg.plot()

#Create the center of the array
z = np.zeros((2,1))

#Single Dirac point source
#cfg = mconfig.Coincided( np.array([-1,1]), 1, 1, np.array([1.0, np.pi, 2*np.pi]))




P = Conductivity.Conductivity(D, cnd, pmtt, cfg)

#Calculate and plot field
sidx = 0
N = 100
width=4
epsilon = width / ((N-1)*5)

F, F_bg, SX, SY, mask = P.calculate_field(np.array([0.5]), sidx, z, width, N)

P.plot_field(sidx, F, F_bg, SX, SY, 100)

