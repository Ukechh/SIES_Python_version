from FundamentalSols import green
import numpy as np
import matplotlib.pyplot as plt


c= np.zeros((2,1))

x= np.linspace(-1,1,30);
y = np.linspace(-1,1,30);
X, Y = np.meshgrid(x,y);
YY  = np.vstack([ X.ravel(),Y.ravel()]);

grid_shape = (len(y),len(x));

Z = green.Green2D(c,YY);
Z = Z.reshape(grid_shape)

plt.contourf(X,Y,Z,100);
plt.contour(X,Y, Z, 20, colors= 'black');
plt.show()