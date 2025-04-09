#We consider the points array as d x n matrix where d is the dimension and n is the number of points
import numpy as np
import matplotlib.pyplot as plt
import warnings
import math
import copy
import Tools_fct.convfix
import C2Boundary.Boundary_methods as bm


class C2Boundary:
    def __init__(self, points, tvec, avec, normal, npts, nstr= None, com= None):
        self._points = points;
        self._tvec = tvec;
        self._avec = avec;
        self.__normal = normal;
        self._nb_points = npts;
        
        flag = self.check_sampling(points)
        if not flag:
            warnings.warn("Curve may contain singluarities", RuntimeWarning);
        self._center_of_mass = com if com is not None else  self.get_com(points, tvec, normal);
        self._name_str = nstr if nstr is not None else '';
        self._tvec_norm = np.linalg.norm(self._tvec, axis = 0);


    #Usual methods:

    def get_box(self):
        #This method returns the width and height of the smallest box containing the boundary
        dd = self._points - np.tile(self._center_of_mass.reshape(-1, 1), (1, self._nb_points));
        w = np.max(dd[0, :]) - np.min(dd[0, :]);
        h = np.max(dd[1, :]) - np.min(dd[1, :]);
        return np.array([w, h])
    
    def get_pdirection(self):
        #This method returns the principal direction of the boundary points
        D = self._points - np.tile(self._center_of_mass.reshape(-1, 1), (1, self._nb_points));
        M = D @ D.T ;
        U, _, _ = np.linalg.svd(M);
        X = U[:, 0];
        t = np.arctan2(X[1], X[0]);
        return np.array([np.cos(t), np.sin(t)])
    
    def get_nbpts(self):
        #This method returns the number of points in the boundary
        return self._points.shape()[1]

    def get_diam(self):
        #This method gives the diameter of the smallest ball with center COM and containing the boundary
        D = self._points - np.tile(self._center_of_mass.reshape(-1, 1), (1, self._nb_points));
        N = np.linalg.norm(D, axis=0);
        return 2*max(N)

    def get_theta(self):
        #This method gives a non-tied off parametrization between [0,2pi) of the boundary with the number of points
        return 2 * np.pi * np.arange(self._nb_points) / self._nb_points
    
    def cpoints(self):
        #This method returns the points as a complex numbers with x- axis being the real part and y-axis being the imaginary part
        return self._points[0,:] + 1j * self._points[1,:];
    @property
    def sigma(self):
        return 2*np.pi / self._nb_points * self._tvec_norm; 

    def tvec_norm(self):
        return self._tvec_norm

    #Operator overloading

    def __add__(self, z0):
        #This method overloads the plus operation and allows for translation by a 2D vector
        if not isinstance(z0, np.ndarray) or not np.issubdtype(z0.dtype, np.floating):
            raise TypeError("Type error: only a numpy array of floats can be used for translation.")

        # Check shape (2, 1) or (2,)
        if z0.shape not in [(2,), (2, 1)]:
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
        if not np.issubdtype(m, np.floating) or max(m.shape) != 1 or m <=0 :
            raise TypeError("Type error: only positive floats can be used for scaling")
        new_boundary = copy.deepcopy(self)
        new_boundary._points = new_boundary._points * m;
        new_boundary._center_of_mass = new_boundary._center_of_mass*m;
        new_boundary._tvec = new_boundary._tvec*m;
        new_boundary._avec = new_boundary._avec*m;
        return new_boundary
    def __lt__(self, phi):
        if not isinstance(phi, (int, float)):
            raise TypeError("Type error: only a scalar float or int can be used for rotation.")
        #Rotation matrix
        rot = np.array([
            [np.cos(phi), -np.sin(phi)],
            [np.sin(phi),  np.cos(phi)]
        ])

        new_boundary = copy.deepcopy(self)

        new_boundary._points = rot @ self._points
        new_boundary._center_of_mass = rot @ self._center_of_mass
        new_boundary._tvec = rot @ self._tvec
        new_boundary.__normal = rot @ self.__normal
        new_boundary._avec = rot @ self._avec
        return new_boundary
    
    # Utility methods:

    def subset(self, idx):
        if max(idx.shape) > self._nb_points:
            raise ValueError("Value Error: the index of the subset is wrong!");
        sub = copy.deepcopy(self);
        sub._points = self._points[:,idx];
        sub._nb_points = max(idx.shape);
        sub._tvec = self._tvec[:,idx];
        sub._avec = self._avec[:,idx];
        sub.__normal = self.__normal[:,idx];
        return sub

    def isinside(self, x):
        if x.shape not in [(2,), (2, 1)]:
            raise ValueError("Size error: x must have shape (2,) or (2,1)")
        flag = 1;
        if np.linalg.norm(x-self._center_of_mass) >= self.get_diam() / 2:
            flag = 0;
        return flag

    def isdisjoint(self, B):
        if not isinstance(B, C2Boundary):
            raise TypeError("The argument must be an instance of C2Boundary.")
        flag = 1;
        d = np.linalg.norm(self._center_of_mass-B.get_center_of_mass);
        if d <= (self.get_diam()+ B.get_diam()) / 2 :
            flag = 0;
        return flag
    
    def plot(self, *args, **kwargs):
       # Plots the boundary points
       plt.plot(self._points[0, :], self._points[1, :], *args, **kwargs)
       plt.show()
    
    def interior_mesh(self, w, N):
        # This method generates a square mesh centered at the center and compute the mask of
	    # the mesh points included in the interior of the domain.
        N = 2 * (N // 2)
        z0 = self._center_of_mass
        Sx, Sy, mask = self.boundary_off_mask(z0, w, N, w / N * 2)
        return Sx, Sy, mask
    
    def boundary_off_mask(self, z0, w, N, epsilon):
        #This method computes a binary mask of size N X N inside the square region (z0, width) so that the
		#	 mesh points on the boundary are turned off (0).
		#	 Inputs:
		#	 z0: center of the mesh
		#	 w: width of the mesh
		#	 N: number of points by side
		#	 Outputs:
		#    Sx, Sy: mesh coordinates of boundary points
		#	 mask: binary mask
        sx = np.linspace(z0[0] - w / 2, z0[0] + w / 2, N)
        sy = np.linspace(z0[1] - w / 2, z0[1] + w / 2, N)
        Sx, Sy = np.meshgrid(sx,sy)

        Z = np.vstack((Sx.ravel(), Sy.ravel()))

        mask = np.ones(Z.shape[1]);

        for n in range(Z.shape[1]):
            dd = np.linalg.norm(self._points- Z[:,n])
            if dd < epsilon:
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
                p1 = Tools_fct.convfix.convfix(self._points[0,:], hw)
                p2 = Tools_fct.convfix.convfix(self._points[1,:], hw)
            else:
                pos = pos % 1
                w = w % 1
                idx = max(int(np.floor(pos * self._nb_points)), 1)
                Lt = max(1, int(np.floor(self._nb_points * w / 2)))

                s1, s2 = 0, self._nb_points

                if idx - Lt >= 1:
                    s1 = idx - Lt
                if idx + Lt <= self._nb_points:
                    s2 = idx + Lt

                q1 = Tools_fct.convfix.convfix(self._points[0, s1:s2], hw)
                q2 = Tools_fct.convfix.convfix(self._points[1, s1:s2], hw)

                p1 = np.concatenate([self._points[0, :s1], q1, self._points[0, s2:]])
                p2 = np.concatenate([self._points[1, :s1], q2, self._points[1, s2:]])

            D = np.vstack([p1, p2])
            N = max(p1.shape)
            theta = 2 * np.pi * np.arange(N) / N

            D1, tvec1, avec1, normal1 = bm.rescale(D, theta, self._nb_points, self.get_box(), dspl=None)
            new_boundary = C2Boundary(D1, tvec1, avec1, normal1, self._nb_points)
        else:
            new_boundary = copy.deepcopy(self)

        return new_boundary

    def global_perturbation(self, epsilon, p, n):
        theta0 = self.get_theta()
        box = self.get_box()
        if abs(epsilon) > 0:
            D = self._points + epsilon*np.cos(p*theta0)*self.__normal
            if n > 0:
                k = np.ones(1,n) / n
                mode = 'same'
                M = max(D.shape)
                d1 = np.array([D[0,:], D[0,:], D[0,:]])
                d2 = np.array([D[1,:], D[1,:], D[1,:]])

                d1 = np.convolve(d1,k,mode)
                d2 = np.convolve(d2,k,mode)
                D = np.array([d1[M:2*M], d2[M:2*M]])
            D1, tvec, avec, normal = bm.rescale(D, theta0, self._nb_points, box)
            new_boundary = C2Boundary(D1,tvec,avec,normal,[],self._name_str)
        else:
            new_boundary= copy.deepcopy(self)
        return new_boundary
    
    def local_perturbation(self, epsilon, pos, width, theta=None):
        theta0 = self.get_theta
        if abs(epsilon)>0:
            if width > 0.5:
                raise ValueError("Wrong value of width, must be smaller than 0.5!")
            pos = pos % 1
            width = width % 1
            idx = math.floor(pos*self._nb_points)

            h = lambda t : np.exp(-10*(t**2))
            Lt = max(1,int(math.floor(self._nb_points*width)))
            toto = np.concatenate([h(np.linspace(-1,1,Lt)), np.zeros(1,self._nb_points-Lt)])
            H = np.roll(toto, shift = idx-math.floor(Lt/2),axis=1)
            if theta is None:
                R = np.eye(2)
            else:
                R = np.array([[np.cos(theta), -np.sin(theta)], [np.sin(theta), np.cos(theta)]])
            D = self._points + epsilon*(R@self.__normal)*H
            D1, tvec, avec,normal = bm.rescale(D, theta0, self._nb_points)
            new_boundary = C2Boundary(D1,tvec,avec,normal,[], self._name_str)
        else:
            new_boundary = copy.deepcopy(self)
        return new_boundary

    def get_center_of_mass(self):
        return self._center_of_mass
    #Static methods
    @staticmethod
    def check_sampling(points):
        #This method checks wether the boundary has any singularities
        npts = points.shape()[1];
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
            toto = np.dot(z-x,x-y) / (np.linalg.norm(y-x) * np.linalg.norm(z-x))
            if toto <= 0:
                val = 0;
        return val
    @staticmethod
    def get_com(points, tvec, normal):
        npts = points.shape()[1];
        tvec_norm = np.linalg.norm(tvec, axis = 0)
        sigma = 2*np.pi / npts * tvec_norm;
        mass = (np.sum(points[0,:] * normal[0,:] * sigma) + np.sum(points[1,:]*normal[1,:]*sigma)) / 2;
        Cx = np.sum(0.5 * (points[0, :] ** 2) * normal[0, :] * sigma);
        Cy = np.sum(0.5 * (points[1, :] ** 2) * normal[1, :] * sigma);
        return np.array([Cx, Cy]) / mass
    
    @staticmethod
    def smooth_out_singularity(points, com, hw, box=None):
        npts = points.shape()[1]
        if box is None or box == []:
            w = max(points[0,:])-min(points[0,:])
            h = max(points[1,:])-min(points[1,:])
            box = [w, h]
        
        p1 = Tools_fct.convfix.convfix(points[0,:], hw)
        p2 = Tools_fct.convfix.convfix(points[1,:], hw)
        D = np.vstack([p1, p2])
        N = max(p1.shape)
        theta = np.linspace(0, 2 * np.pi, N, endpoint=False)
        D1, tvec1,avec1,normal1 = bm.rescale(D,theta,npts,box)
        com1 = C2Boundary.get_com(D1, tvec1,normal1)
        D1 = D1 - (com1 - com)[:, np.newaxis]
        return D1, tvec1, avec1, normal1




