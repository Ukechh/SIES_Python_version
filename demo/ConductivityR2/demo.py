import sys
import os

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '../../')))

import numpy as np
from figure import Geom_figures as shape
from figure.C2Boundary.C2Boundary import C2Bound

#Make an inclusion
B = shape.Ellipse(1,1/2, 2**10)
#Create a C2 boundary object for the inclusion
Dr = C2Bound(B.boundary_points, B.tvec, B.avec, B.normal, B.nb_points)
#Plot the inclusion
D = Dr<(np.pi/4)
D.plot()

cnd = 10*np.array([1,1]) 
pmtt = 1*np.array([1,1])

