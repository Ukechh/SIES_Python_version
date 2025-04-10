from figure.Geom_figures import Ellipse
from figure.C2Boundary import C2Boundary
import numpy as np
import matplotlib.pyplot as plt

E = Ellipse()
x = E.get_points()
plt.plot(x[0,:],x[1,:])
plt.show()
