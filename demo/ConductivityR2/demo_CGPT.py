import sys
import os

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '../../')))

import numpy as np
import matplotlib.pyplot as plt
from figure.Geom_figures import Ellipse, Banana, Triangle, Rectangle
from figure.C2Boundary.C2Boundary import C2Bound
from cfg import mconfig
from asymp.CGPT_methods import theoretical_CGPT
from PDE.Conductivity_R2 import Conductivity
from PDE.Conductivity_R2.make_CGPT import make_matrix_A
from FundamentalSols.green import Green2D_Dn, Green2D
from asymp.CGPT_methods import make_system_matrix_fast, make_block_matrix, lbda
from GPT.Expansions import inner_expansion_conductivityR2_0, plot_inner_expansion



#Make an inclusion
B1 = Ellipse(1, 1/2, phi= np.pi, NbPts=2**8)
B2 = Triangle(1, np.pi/3, npts= 2**8) 
#Define the center of the inclusion
z1 = np.array([0.5,0.5]).reshape((2,1))
z2 = np.array([-0.8,-0.8]).reshape((2,1))

#Define a figure to plot inclusions
axe = plt.subplot()
#Translate and scale inclusions
B1 = (B1 + z1) * 0.1
B2 = (B2 + z2) * 0.1

#Add the inclusion to the list of inclusions
#D = [B1]
#Multiple inclusions
D = [B1, B2]

#Set conductivity and permitivitty (Single inclusion)
cnd = 10*np.array([1]) 
pmtt = 5*np.array([1])

#Set conductivity and permitivitty (Multiple inclusions)
cnd = 10*np.array([1,0.5])  
pmtt = 5*np.array([1,0.5])

#Set up a list of working frequencies
freq = np.linspace(0.01, 150*np.pi, endpoint=False, num=100)

#Single Dirac point source
cfg = mconfig.Coincided( np.array([-1,1]), 1, 50, np.array([1.0, 2*np.pi, 2*np.pi]))

#Define a COnductivity instance
P = Conductivity.Conductivity(D, cnd, pmtt, cfg)
P.plot(ax=axe)
plt.show()
#Choose the order
order = 2

#Initialize lists to hold the differences
lsqr_diff = []
pinv_diff = []
data, _ = P.data_simulation(freq)

#Reconstruct the CGPT matrix from lsqr
ls, ls_res, ls_rres = P.reconstruct_CGPT(data, ord = order, method='lsqr')
#print(f'The reconstructed CGPT matrix of order 2 for frequency {f}, using lsqr is: \n {ls}')

#Reconstruct the CGPT matrix from penrose inverse
pinv, pinv_res, pinv_rres = P.reconstruct_CGPT(data, ord=order, method='pinv')
#print(f'The reconstructed CGPT matrix of order 2 for frequency {f}, using pinv is: \n {pinv}')

for i, f in enumerate(freq):
    #Compute the lambdas corresponding to the permittivity
    lam = lbda(cnd, pmtt, f, True)
    
    #Compute theoretical CGPT for the inclusion D
    M = theoretical_CGPT(D, lam, order)
    #print(f'The theoretical CGPT matrix of order 2 for frequency {f} is: \n {M}')

    #Store the frobenius norm of the difference of theoretical and reconstructed CGPT's
    lsqr_diff.append(np.linalg.norm(ls[i]-M, 'fro'))
    pinv_diff.append(np.linalg.norm(pinv[i]-M, 'fro'))

plt.figure(figsize=(8, 5))
plt.plot(freq, lsqr_diff, marker='o', linestyle='-', color='crimson', label='Least squares')
plt.plot(freq, pinv_diff, marker='o', linestyle='--', color='magenta', label='Penrose Inverse')
plt.xlabel('Working frequency')
plt.ylabel('Frobenius norm')
plt.title('Difference between theoretical and estimated CGPTs')
plt.grid(True)
plt.tight_layout()
plt.legend()
plt.show()