import sys
import os

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '../../')))

import numpy as np
import matplotlib.pyplot as plt
from figure.Geom_figures import Ellipse, Banana, Triangle, Rectangle
from figure.C2Boundary.C2Boundary import C2Bound
from cfg.mconfig import Fish_circle
from Operators.Operators import SingleLayer, LmKstarinv
from PDE.Electric_fish.Electric_fish_ import Electric_Fish
from FundamentalSols.green import Green2D_Dn, Green2D
from asymp.CGPT_methods import make_system_matrix_fast, make_block_matrix, lbda



delta = 1

npts = 600
c = np.array([-0.6,0.5])
c2 = np.array([1,-1])

#Initialize inclusions:
D = []
B1 = Ellipse(delta,delta/3, NbPts=npts)
D.append((B1*0.2+ 0.2*np.array([-1,1]).T) < np.pi /2 )

B2 = Triangle(delta, np.pi / 7, npts=npts)
D.append((B2+np.array([1.0,-1.0]).T)*0.2)
B3 = Banana(c, 1/2, 1/4, c2, npts)
D.append(B3*0.8)
#Conductivity and permittivity values for

#One inclusion
#cnd = np.array([10])
#pmtt = np.array([0.1])

#Two inclusions
#cnd = np.array([10, 4])
#pmtt = np.array([0.1, 0.2])

#Three inclusions
cnd = np.array([10, 4,100])
pmtt = np.array([0.1, 0.2,10])


mcenter = np.zeros(2)
mradius = D[0].diameter * 2.5

#Initialize fish body
Omega = Ellipse(delta/2, delta/4, 0, 200)
Omega = Omega< np.pi/2

#Set skin impedance
impd = 0.01

#Give indices of active receptors
idxRcv = np.arange(0, Omega.nb_points-1,2)
Ns = 3
cfg = Fish_circle(Omega, idxRcv, mcenter, mradius, Ns, 2*np.pi, impd=impd)
#cfg.plot()
#plt.show()

stepBEM = 2

P = Electric_Fish(D, cnd, pmtt, cfg, stepBEM)
#PLot the fish positions and inclusions
#P.plot()
#plt.show()

#We start a list of working frequencies
freq = np.linspace(0.1,100*np.pi,5)
#Simulate fish data:
_, vpsi, vphi, fpsi, fphi, vpsi_bg, fpsi_bg, fpp, Current, Current_bg, MSR, SFR, PP_SFR = P.data_simulation(freq)

f = 1 #Index of the frequency
s = 0
z0 = np.zeros(2)
width = 10
N = 100
#Calculate field for fish data


#for s in range(Ns):
    #for f in range(len(freq)):
F, F_bg, Sx, Sy, mask = P.calculate_field(f, s, z0, width, N, fpsi_bg, fpsi, fphi)
fig = P.plot_field(s, F, F_bg, Sx, Sy, 100, True)
plt.show()