import sys
import os

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '../../')))

#We consider the points array as d x n matrix where d is the dimension and n is the number of points
import numpy as np
import matplotlib.pyplot as plt
import warnings
import math
import copy
import numbers
from figure.C2Boundary import Boundary_methods as bm
from Tools_fct import convfix
from scipy.interpolate import CubicSpline

class C2Bound:
    def __init__(self, points, tvec, avec, normal,  com= None, nstr= None, npts=1):
        self._points = points
        self._tvec = tvec
        self._avec = avec
        self._normal = normal
        self.nb_points = npts
        self.delta = 1
        if npts is None:
            self.nb_points = points.shape[1]
        flag = self.check_sampling(points)
        if not flag:
            warnings.warn("Curve may contain singluarities", RuntimeWarning)
        self._center_of_mass = com if com is not None else  self.get_com(points, tvec, normal)
        self._center_of_mass = self._center_of_mass.reshape((2,1))
        self._name_str = nstr if nstr is not None else ''
        self._tvec_norm = np.linalg.norm(self._tvec, axis = 0)
        


    #Usual methods:

    def get_box(self):
        #This method returns the width and height of the smallest box containing the boundary
        dd = self._points - self._center_of_mass.reshape(-1, 1)
        w = np.max(dd[0, :]) - np.min(dd[0, :]);
        h = np.max(dd[1, :]) - np.min(dd[1, :]);
        return np.array([w, h])
    
    def get_pdirection(self):
        #This method returns the principal direction of the boundary points
        D = self._points - self._center_of_mass.reshape(-1, 1)
        M = D @ D.T ;
        U, _, _ = np.linalg.svd(M);
        X = U[:, 0];
        t = np.arctan2(X[1], X[0]);
        return np.array([np.cos(t), np.sin(t)])
    
    def get_nbpts(self):
        #This method returns the number of points in the boundary
        return self._points.shape[1]

    def get_theta(self):
        #This method gives a non-tied off parametrization between [0,2pi) of the boundary with the number of points
        return 2 * np.pi * np.arange(self.nb_points) / self.nb_points
    
    def get_center_of_mass(self):
        return self._center_of_mass

    @property
    def cpoints(self):
        #This method returns the points as a complex numbers with x- axis being the real part and y-axis being the imaginary part
        return self._points[0,:] + 1j * self._points[1,:]
    
    @property
    def points(self):
        return self._points
    @property
    def sigma(self):
        return 2*np.pi / (self.nb_points) * self._tvec_norm; 

    @property
    def normal(self):
        return self._normal

    @property
    def diameter(self):
        D = self._points - self._center_of_mass.reshape(-1, 1)
        N = np.linalg.norm(D, axis=0);
        return 2*max(N)

    def tvec_norm(self):
        return self._tvec_norm

    #Operator overloading

    def __add__(self, z0):
        #This method overloads the plus operation and allows for translation by a 2D vector
        if not isinstance(z0, np.ndarray) or not np.issubdtype(z0.dtype, np.floating):
            raise TypeError("Type error: only a numpy array of floats can be used for translation.")

        # Check shape (2, 1) or (2,)
        if z0.shape not in [(2, 1), (2,)]:
            raise ValueError("Size error: translation vector must have shape (2,) or (2,1)")

        z0 = z0.reshape(2, 1)  # Ensure it's column-shaped
        new_boundary = copy.deepcopy(self)
        new_boundary._points = new_boundary._points + z0
        new_boundary._center_of_mass = new_boundary._center_of_mass + z0
        return new_boundary
    
    def __sub__(self,z0):
        #Overload of the minus operator
        return self + (-z0)

    def __mul__(self,m):
        #Scalng the boundary by a scalar m
        new_boundary = copy.deepcopy(self)
        new_boundary._points = new_boundary._points * m;
        new_boundary._center_of_mass = new_boundary._center_of_mass*m;
        new_boundary._tvec = new_boundary._tvec*m;
        new_boundary._avec = new_boundary._avec*m;
        new_boundary.delta = new_boundary.delta * m;
        return new_boundary
    
    def __lt__(self, phi):
        if not isinstance(phi, (int, float, np.number)):
            raise TypeError("Type error: only a scalar float or int can be used for rotation. Got type {}".format(type(phi)))
        #Rotation matrix
        phi = float(phi)
        rot = np.array([
            [np.cos(phi), -np.sin(phi)],
            [np.sin(phi),  np.cos(phi)]
        ])

        new_boundary = copy.deepcopy(self)

        new_boundary._points = rot @ self._points
        new_boundary._center_of_mass = rot @ self._center_of_mass
        new_boundary._tvec = rot @ self._tvec
        new_boundary._normal = rot @ self._normal
        new_boundary._avec = rot @ self._avec
        return new_boundary
    
    # Utility methods:

    def subset(self, idx):
        if max(idx.shape) > self.nb_points:
            raise ValueError("Value Error: the index of the subset is wrong!");
        points = self._points[:,idx]
        tvec = self._tvec[:,idx]
        avec = self._avec[:,idx]
        normal = self._normal[:,idx]
        sub = C2Bound(points, tvec, avec, normal)
        return sub

    def isinside(self, x):
        if x.shape not in [(2,), (2, 1)]:
            raise ValueError("Size error: x must have shape (2,) or (2,1)")
        flag = 1;
        if np.linalg.norm(x-self._center_of_mass) >= self.diameter / 2:
            flag = 0;
        return flag

    def isdisjoint(self, B):
        if not isinstance(B, C2Bound):
            raise TypeError("The argument must be an instance of C2Boundary.")
        flag = 1;
        d = np.linalg.norm(self._center_of_mass-B._center_of_mass);
        if d <= (self.diameter + B.diameter) / 2 :
            flag = 0;
        return flag
    
    def plot(self, *args, **kwargs):
        ax = kwargs.pop("ax", None)
        if ax is None:
            raise ValueError("An Axes object must be passed using the 'ax' keyword argument.")
        ax.plot(self._points[0, :], self._points[1, :], *args, **kwargs)
    
    def interior_mesh(self, w, N):
        # This method generates a square mesh centered at the center and compute the mask of
	    # the mesh points included in the interior of the domain.
        N = 2 * (N // 2)
        z0 = self._center_of_mass
        Sx, Sy, mask = self.boundary_off_mask(z0, w, N, w / N * 2)
        return Sx, Sy, mask
    
    def boundary_off_mask(self, z0, w, N, epsilon):
        """
        This method computes a binary mask of size N X N inside the square region (z0, width) so that the
		mesh points on the boundary are turned off (0).
		Parameters:
         ------------
		z0: center of the mesh
            ndarray
        w: width of the mesh
            float
        N: number of points by side
            int
        Returns:
         ------------
	    Sx, Sy: mesh coordinates of boundary points, each of shape (N,N)
            ndarray
		mask: binary mask of shape (N,N)
            ndarray 
        
        """
        sx = np.linspace(z0[0] - w / 2, z0[0] + w / 2, N)
        sy = np.linspace(z0[1] - w / 2, z0[1] + w / 2, N)
        Sx, Sy = np.meshgrid(sx,sy)

        Z = np.vstack((Sx.ravel(), Sy.ravel()))

        mask = np.ones(Z.shape[1])

        for n in range(Z.shape[1]):
            dd = np.linalg.norm(self._points- Z[:,n].reshape(2,1), axis=0)
            if dd.any() < epsilon:
                mask[n] = 0

        mask = mask.reshape(Sx.shape)

        return Sx, Sy, mask

    def smooth(self, hw, pos=None, w=None):
        # This method smooths a segment of the boundary by convolution using a constant window.
			# Inputs:
			# hwidth: the length (integer) of the constant convolution window
			# pos, width: the boundary to be smoothed is on [pos-width/2,
			# pos+width/2], and 0<pos<=1 is the normalized curve
			# parameteration and 0<width<=1 is the normalized width of the
			# segment of boundary
			# Output:
			# new_boundary: a new object representing the smoothed shape
        hw = math.floor(hw)
        if hw > 0:
            if pos is None or w is None:
                p1 = convfix.convfix(self._points[0,:], hw)
                p2 = convfix.convfix(self._points[1,:], hw)
            else:
                pos = pos % 1
                w = w % 1
                idx = max(int(np.floor(pos * self.nb_points)), 1)
                Lt = max(1, int(np.floor(self.nb_points * w / 2)))

                s1, s2 = 0, self.nb_points

                if idx - Lt >= 1:
                    s1 = idx - Lt
                if idx + Lt <= self.nb_points:
                    s2 = idx + Lt

                q1 = convfix.convfix(self._points[0, s1:s2], hw)
                q2 = convfix.convfix(self._points[1, s1:s2], hw)

                p1 = np.concatenate([self._points[0, :s1], q1, self._points[0, s2:]])
                p2 = np.concatenate([self._points[1, :s1], q2, self._points[1, s2:]])

            D = np.vstack([p1, p2])
            N = max(p1.shape)
            theta = 2 * np.pi * np.arange(N) / N

            D1, tvec1, avec1, normal1 = bm.rescale(D, theta, self.nb_points, self.get_box(), dspl=None)
            new_boundary = C2Bound(D1, tvec1, avec1, normal1, self.nb_points)
        else:
            new_boundary = copy.deepcopy(self)

        return new_boundary

    def global_perturbation(self, epsilon, p, n):
        """ 
        Perturb the boundary by a global perturbation
        Parameters:
        -------------
        epsilon: strength of the perturbation
		p: periodicity of the perturbation
		n: strength of the smooth filter, integer
        
        Returns:
        --------------
            new_boundary: perturbed boundary
                C2Bound
        """
        #Define theta and the box
        theta0 = self.get_theta()
        box = self.get_box()

        if abs(epsilon) > 0:
            # Create periodic perturbation along normal direction
            perturbation = np.cos(p*theta0)
            D = self._points + epsilon * perturbation * self._normal
            
            # Smooth the boundary if n > 0
            if n > 0:
                k = np.ones((1,n)) / n
                mode = 'same'
            
                M = D.shape[1]  # length of D along axis 1
                d1 = np.concatenate([D[0,:], D[0,:], D[0,:]])
                d2 = np.concatenate([D[1,:], D[1,:], D[1,:]])

                d1 = np.convolve(d1.flatten(), k.flatten(), mode)
                d2 = np.convolve(d2.flatten(), k.flatten(), mode)
            
                D = np.vstack([d1[M:2*M], d2[M:2*M]])
            
            # Rescale the boundary
            D1, tvec, avec, normal = bm.rescale(D, theta0, self.nb_points, box)
            new_boundary = C2Bound(D1, tvec, avec, normal, nstr=self._name_str, npts=D1.shape[1])
        else:
            new_boundary = copy.deepcopy(self)
        return new_boundary
    
    def local_perturbation(self, epsilon, pos, width, theta=None):
        theta0 = self.get_theta()
        if abs(epsilon)>0:
            if width > 0.5:
                raise ValueError("Wrong value of width, must be smaller than 0.5!")
            pos = pos % 1
            width = width % 1
            idx = math.floor(pos*self.nb_points)

            h = lambda t : np.exp(-10*(t**2))
            Lt = max(1,int(math.floor(self.nb_points*width)))

            toto = np.concatenate([h(np.linspace(-1,1,Lt)), np.zeros(self.nb_points-Lt)])
            
            H = np.roll(toto, shift = idx-math.floor(Lt/2),axis=1)
            
            if theta is None:
                R = np.eye(2)
            else:
                R = np.array([[np.cos(theta), -np.sin(theta)], [np.sin(theta), np.cos(theta)]])
            D = self._points + epsilon*(R@self._normal)*H
            
            D1, tvec, avec,normal = bm.rescale(D, theta0, self.nb_points)
            
            new_boundary = C2Bound(D1,tvec,avec,normal,com=[], nstr=self._name_str)
        else:
            new_boundary = copy.deepcopy(self)
        return new_boundary
    
    #Static methods
    @staticmethod
    def check_sampling(points):
        #This method checks wether the boundary has any singularities
        npts = points.shape[1];
        val = 1;
        for p in range(npts):
            x = points[:,p];
            if p == 0:
                y = points[:,npts-1];
                z = points[:, p+1];
            elif p == npts:
                y = points[:,p-2];
                z = points[:,0];
            else:
                y = points[:,p-2];
                z = points[:,p];
            denom = np.linalg.norm(y - x) * np.linalg.norm(z - x)
            if denom < 1e-14:
                toto = 0.0
            else:
                toto = np.dot(z - x, x - y) / denom

            if toto <= 0:
                val = 0;
        return val
    @staticmethod
    def get_com(points, tvec, normal):
        npts = points.shape[1];
        tvec_norm = np.linalg.norm(tvec, axis = 0)
        sigma = 2*np.pi / npts * tvec_norm;
        mass = (np.sum(points[0,:] * normal[0,:] * sigma) + np.sum(points[1,:]*normal[1,:]*sigma)) / 2;
        Cx = np.sum(0.5 * (points[0, :] ** 2) * normal[0, :] * sigma);
        Cy = np.sum(0.5 * (points[1, :] ** 2) * normal[1, :] * sigma);
        return np.array([Cx, Cy]) / mass
    @staticmethod
    def rescale(D0, theta0, nbPoints, nsize=None, dspl=1):
        """
        Parameters:
        -------------
            D0: array of shape (2, N) — original boundary points
            theta0: array of shape (N,) — original parameterization angles
            nbPoints: int — number of points after reparametrization
            nsize: tuple or list (width, height), optional — resize box
            dspl: int — down-sampling factor (>=1)

        Returns:
        --------------
            D, tvec, avec, normal — resampled boundary, tangent, acceleration, normal vectors
        """
        dspl = int(np.ceil(dspl))
        if dspl < 1:
            raise ValueError('Down-sampling factor must be positive!')

        D0 = D0.copy()  # work on a copy

        # Resize the boundary to fit in box of size nsize
        if not nsize is None:
            minx, maxx = D0[0, :].min(), D0[0, :].max()
            miny, maxy = D0[1, :].min(), D0[1, :].max()

            z0 = np.array([(minx + maxx) / 2, (miny + maxy) / 2])
            D0[0, :] = (D0[0, :] - z0[0]) * (nsize[0] / (maxx - minx))
            D0[1, :] = (D0[1, :] - z0[1]) * (nsize[1] / (maxy - miny))

        # Down-sampling
        nbPoints0 = theta0.size
        idx = np.concatenate([np.arange(0, nbPoints0-1, dspl), [nbPoints0-1]])
        theta0_dspl = theta0[idx]
        D0_dspl = D0[:, idx]

        # Target theta grid
        theta = np.linspace(theta0[0], theta0[-1], nbPoints)

        # Spline interpolation
        D, tvec, avec, normal = C2Bound.boundary_vec_interpl(D0_dspl, theta0_dspl, theta)

        return D, tvec, avec, normal
    @staticmethod
    def rescale_diff(D0, theta0, nbPoints, nsize=None):
        
        if not nsize is None:
            minx = np.min(D0[0,:]) 
            maxx = np.max(D0[0,:]) 
            miny = np.min(D0[1,:]) 
            maxy = np.max(D0[1,:])
            
            z0 = np.array([(minx+maxx)/2, (miny+maxy)/2]).reshape((2,-1))
            D0 = np.array([(D0[0,:]-z0[0])*(nsize[0]/(maxx-minx)), (D0[1,:]-z0[1])*(nsize[1]/(maxy-miny))]).reshape(2,-1)

        theta = np.arange(nbPoints) / nbPoints* 2*np.pi
        # Use scipy's CubicSpline for interpolation
        cs_x = CubicSpline(theta0, D0[0, :])
        cs_y = CubicSpline(theta0, D0[1, :])
        
        # Interpolate points
        px = cs_x(theta)
        py = cs_y(theta)
        D = np.vstack((px, py))
        
        # Get vectors using boundary_vec_interpl method
        D, tvec, avec, normal = C2Bound.boundary_vec_interpl(D, theta)
        
        return D, tvec, avec, normal
    @staticmethod
    def boundary_vec_interpl(points0, theta0,theta=None):
        """
        Given a curve and its parameterization, compute:
        - tangent (first derivative),
        - normal (rotated tangent),
        - acceleration (second derivative).
        """
        if theta is None:
            theta = theta0
        cs_x = CubicSpline(theta0, points0[0, :])
        cs_y = CubicSpline(theta0, points0[1, :])

        px = cs_x(theta)
        py = cs_y(theta)
        points = np.vstack((px, py))

        tx = cs_x(theta, 1)
        ty = cs_y(theta, 1)
        tvec = np.vstack((tx, ty))

        rotation = np.array([[0, 1], [-1, 0]])
        normal = rotation @ tvec
        normal = normal / np.linalg.norm(normal, axis=0, keepdims=True)

        ax = cs_x(theta, 2)
        ay = cs_y(theta, 2)
        avec = np.vstack((ax, ay))

        return points, tvec, avec, normal
    @staticmethod
    def smooth_out_singularity(points, com, hw, box=None):
        npts = points.shape[1]
        if box is None or box == []:
            w = max(points[0,:])-min(points[0,:])
            h = max(points[1,:])-min(points[1,:])
            box = [w, h]
        
        p1 = convfix.convfix(points[0,:], hw)
        
        p2 = convfix.convfix(points[1,:], hw)
        
        D = np.vstack([p1, p2])
        
        N = max(p1.shape)
        
        theta = np.linspace(0, 2 * np.pi, N, endpoint=False)
        
        D1, tvec1,avec1,normal1 = bm.rescale(D,theta,npts,box)
        
        com1 = C2Bound.get_com(D1, tvec1,normal1)
        
        D1 = D1 - (com1 - com)[:, np.newaxis]

        return D1, tvec1, avec1, normal1




