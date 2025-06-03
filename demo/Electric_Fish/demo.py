import sys
import os

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '../../')))

import numpy as np
import matplotlib.pyplot as plt
from figure.Geom_figures import Ellipse, Banana, Triangle, Rectangle
from figure.ImageShape.Image_shape import ImgShape
from figure.C2Boundary.C2Boundary import C2Bound
from cfg.mconfig import Fish_circle
from Operators.Operators import SingleLayer, LmKstarinv
from PDE.Electric_fish.Electric_fish_ import Electric_Fish
from FundamentalSols.green import Green2D_Dn, Green2D
from asymp.CGPT_methods import lbda, theoretical_CGPT



delta = 1

npts = 2**10
c = np.array([-0.2,1])
c2 = np.array([1,-1])

#Initialize inclusions:
D = []
B1 = ImgShape('Test_images/A.png', npts=npts)
#D.append((B1*0.4+ 0.2*np.array([-1,1]).T) < -np.pi /34 )

#B1 = Triangle(delta, np.pi / 7, npts=npts)
#D.append((B2+np.array([1.0,-1.0]).T)*0.2)
#B3 = Banana(c, 1/3, 1/4, c2, npts)
D.append(B1*0.3)

#Conductivity and permittivity values for
#One inclusion
cnd = np.array([100])
pmtt = np.array([40])

#Two inclusions
#cnd = np.array([10, 4])
#pmtt = np.array([0.1, 0.2])

#Three inclusions
#cnd = np.array([10, 4,100])
#pmtt = np.array([0.1, 0.2,10])


mcenter = np.zeros(2)
mradius = D[0].diameter * 3

#Initialize fish body
Omega = Ellipse(delta/2, delta/4, 0, 2**9)
Omega = Omega< np.pi/2

#Set skin impedance
impd = 1

#Give indices of active receptors
idxRcv = np.arange(0, Omega.nb_points-1,2**5)
Ns = 8
cfg = Fish_circle(Omega, idxRcv, mcenter, mradius, Ns, 2*np.pi, impd=impd)

stepBEM = 4

P = Electric_Fish(D, cnd, pmtt, cfg, stepBEM)
ax = plt.subplot()

#PLot the fish positions and inclusions
P.plot(ax=ax)
plt.show()

#We start a list of working frequencies
freq = np.linspace(0.1,100*np.pi,1)

#Simulate fish data:
data = P.data_simulation(freq)

fpsi_bg = data['fpsi_bg']
fpsi = data['fpsi']
fphi = data['fphi']


print('Verify that the solution of the forward system (functions phi and psi) have zero-mean:')
print(f"psi norm: {np.abs(np.sum(Omega.sigma * fpsi[0][:,0]))}")  
print(f"phi norm: {np.abs(np.sum(D[0].sigma * fphi[0][:,0,0]))}")

#Plot field
sidx = 1  # source index to be calculated
fidx = 0  # frequency index to be calculated
F, F_bg, SX, SY, mask = P.calculate_field(fidx, sidx, np.array([0,0]), 5, 100, fpsi_bg, fpsi, fphi)
P.plot_field(sidx, F, F_bg, SX, SY, 100, False, color='k', linewidth=1.4)
plt.show()

##CGPT RECONSTRUCTION

# Compute theoretical CGPT
ord = 2  # maximum order of reconstruction
symmode = 0  # force solution to be symmetric
fidx = 0  # frequency index to be reconstructed

CGPTD = []
for f in range(len(freq)):
    lambda_val = lbda(cnd, pmtt, freq[f])
    CGPTD.append(theoretical_CGPT(D, lambda_val, ord))

#We work with noiseless data:
MSR = data['MSR']
Current = data['Current']

#Reconstruct CGPTs
rCGPT = P.reconstruct_CGPT(MSR, Current, ord, 10000, 1e-10, symmode)

CGPT = rCGPT['CGPT']

print('\n************ Exp. 1 **************')
print(f'Theoretical CGPT matrix at the frequency {freq[fidx]}:')
print(CGPTD[fidx])  # theoretical value

print(f'\nReconstructed CGPT matrix at the frequency {freq[fidx]}:')
print(CGPT[fidx])  # reconstruction

print('\nRelative error:')
print(np.linalg.norm(CGPT[fidx] - CGPTD[fidx], 'fro') / np.linalg.norm(CGPTD[fidx], 'fro'))  # relative error