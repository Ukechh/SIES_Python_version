from FundamentalSols import green
import numpy as np
import matplotlib.pyplot as plt

c= np.array([0,0])
x= np.linspace(-1,1,20);
y = np.linspace(-1,1,20);
X, Y = np.meshgrid(x,y);
YY  = np.vstack([ X.ravel(),Y.ravel()]).T;

grid_shape = (len(y),len(x));

Z = green.Green2D(c,YY);
Z = Z.reshape(grid_shape)

plt.contourf(X,Y,Z,70);
plt.show()