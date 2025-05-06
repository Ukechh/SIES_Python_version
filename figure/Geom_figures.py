import sys
import os

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '../../')))

import numpy as np
import math
from figure.C2Boundary.C2Boundary import C2Bound

class Ellipse(C2Bound):
    R = np.array([[0, 1], [-1, 0]])
    def __init__(self, a=1.0, b=1.0, phi=0.0, NbPts=100):
        if a < b:
            raise ValueError("Value error: the semi-major axis must be longer than the semi-minor one.")

        self.axis_a = a
        self.axis_b = b
        self.phi = phi
        self.nb_points = NbPts
        self._center_of_mass = np.zeros((2,1))

        theta = np.linspace(0, 2 * np.pi, NbPts, endpoint=False)
        x = a * np.cos(theta)
        y = b * np.sin(theta)
        points = self._center_of_mass + np.vstack((x, y))

        tvec = np.vstack((-a * np.sin(theta), b * np.cos(theta)))
        avec = np.vstack((-a * np.cos(theta), -b * np.sin(theta)))

        normal = self.R @ tvec
        norm = np.linalg.norm(normal, axis=0, keepdims=True)
        normal = normal / norm

        name_str = "Circle" if a == b else "Ellipse"

        super().__init__(points, tvec, avec, normal, self._center_of_mass, name_str, npts=NbPts)
    
    def get_points(self):
        return self._points
    def get_normal(self):
        return self._normal
    def get_tvec(self):
        return self._tvec
    def get_avec(self):
        return self._avec

class Banana(C2Bound):

    def __init__(self,center,a, b, curvature, nbPoints):
        if a < b:
            raise ValueError('Value error: semi-major axis must be longer than the semi minor one.');
        x0 , y0 = center[0], center[1]
        xc, yc = curvature[0], curvature[1]
        R = np.linalg.norm(center-curvature)

        theta0 = np.arctan2(y0-yc, x0-xc)
    
        alpha = a / R
        theta = np.linspace(0,2*np.pi, nbPoints, endpoint=False)
        
        t = theta0+alpha * np.cos(theta)
        
        points = np.array([xc + (R+ b * np.sin(theta))*np.cos(t), yc + (R + b * np.sin(theta))*np.sin(t) ])

        tvec = np.array([b*np.cos(theta)*np.cos(t) + alpha * np.sin(theta)* (R+b*np.sin(theta)) * np.sin(t), 
                         b*np.cos(theta)*np.sin(t) - alpha* np.sin(theta) * (R+b*np.sin(theta)) * np.cos(t)])
        
        rotation = Ellipse.R
        normvec = rotation @ tvec
        normal = normvec / np.linalg.norm(normvec, axis=0)
        
        avec = np.array( [-b*np.sin(theta) * np.cos(t) + alpha* b * np.cos(theta)* np.sin(t) + alpha*np.cos(theta)* (R+b*np.sin(theta))*np.sin(t) +
                          alpha*b*((np.cos(theta))**2) * np.sin(t) - (alpha*np.sin(theta))**2 * (R+b * np.sin(theta))* np.cos(t)
                         , -b*np.sin(theta) * np.cos(t) - alpha* b * np.sin(theta)* np.cos(t) - alpha*np.cos(theta)* (R+b*np.sin(theta))*np.cos(t) - 
                         alpha*b*((np.cos(theta))**2) * np.cos(t) - (alpha*np.sin(theta))**2 * (R+b * np.sin(theta))* np.sin(t)])
        super().__init__(points, tvec, avec, normal, None, 'Banana', nbPoints)
        
        self.axis_a = a
        self.axis_b = b
        self.center = center
        self.curvature = curvature #The curvature means the center of the curvature of the ellipse
    
    ##Overloading of operators

    def __add__(self, z0):
        r = super().__add__(z0)
        r.center = z0 + self.center
        r.curvature = z0 + self.curvature
        return r   
    def __mul__(self, m):
        r = super().__mul__(m)
        r.curvature = m * self.curvature
        r.center = m * self.center
        r.axis_a = m * self.axis_a
        r.axis_b = m * self.axis_b
        return r
    def __lt__(self, phi):
        r = super().__lt__(phi)
        rot = np.array([
            [np.cos(phi), -np.sin(phi)],
            [np.sin(phi),  np.cos(phi)]
        ])
        r.center = rot @ self.center
        r.curvature = rot @ self.curvature
        return r
    
class Triangle(C2Bound):

    def __init__(self, a, angle, npts, dspl=10):

        h = a * np.cos(angle/2)
        b = a * np.sin(angle/2)
        t1, t2 = a / (2*(a+b)) , b / (a+b)
        t3 = t1
        n1, n2 = math.floor(t1*npts), math.floor(t2*npts)
        n3 = npts - (n1+n2)
        A = np.array([0, 2/3 * h])
        B = np.array([-b, -h/3])
        C = np.array([b, -h/3])

        t1 = np.linspace(0, 1, n1, endpoint=False)   
        t2 = np.linspace(0, 1, n2, endpoint=False)   
        t3 = np.linspace(0, 1, n3, endpoint=False)

        AB = A[:, None] + (B - A)[:, None] * t1
        BC = B[:, None] + (C - B)[:, None] * t2
        CA = C[:, None] + (A - C)[:, None] * t3

        points0 = np.hstack([AB, BC, CA])

        theta = np.linspace(0, 2*np.pi, npts, endpoint=False)

        if dspl >= 1:
            t0 = n3 // 2
            points = np.roll(points0, shift=t0, axis=1)

            points, tvec, avec, normal = C2Bound.rescale(points, theta, npts, None, dspl)
        else:
            tvec = np.hstack([
                (B - A)[:, None] / t1.size,
                (C - B)[:, None] / t2.size,
                (A - C)[:, None] / t3.size
            ]) / (2*np.pi)

            rotation = np.array([[0, 1], [-1, 0]])
            normal = rotation @ tvec
            normal = normal / np.linalg.norm(normal, axis=0, keepdims=True)

            avec = np.zeros((2, npts))

            t0 = n3 // 2
            points = np.roll(points0, shift=t0, axis=1)
            tvec = np.roll(tvec, shift=t0, axis=1)
            normal = np.roll(normal, shift=t0, axis=1)
        super().__init__(points, tvec, avec, normal, com=np.zeros(2), nstr='Triangle', npts=npts)
        self.lside = a
        self.angle = angle
    
    def __mul__(self, m):
        r = super().__mul__(m)
        r.lside = self.lside * m
        return r
    
class Rectangle(C2Bound):
    def __init__(self, a, b, npts, dspl=None):
        if dspl is None:
            dspl = 10
        t1, t2 = b / (2*(a+b)), a / (2*(a+b))
        t3, t4 = t1, t2
        n1, n2, n3 = math.floor(t1*npts), math.floor(t2*npts), math.floor(t3*npts)
        n4 = npts - (n1+n2+n3)

        A = np.array([-b,a]) / 2
        B = np.array([-b,-a]) / 2
        C = np.array([b, -a]) / 2
        D = np.array([b, a]) / 2

        t = np.arange(n1) / n1
        AB = A[:, None] + (B - A)[:, None] * t 

        t = np.arange(n2) / n2
        BC = B[:, None] + (C - B)[:, None] * t 

        t = np.arange(n3) / n3
        CD = C[:, None] + (D - C)[:, None] * t 

        t = np.arange(n4) / n4
        DA = D[:, None] + (A - D)[:, None] * t

        points0 = np.hstack((AB, BC, CD, DA))
        theta = np.arange(npts) * 2 * np.pi / npts

        if dspl >= 1:
            t0 = math.floor(n4/ 2)
            points = np.roll(points0, shift = t0, axis = 1)
            points, tvec, avec, normal = C2Bound.rescale(points, theta, npts, dspl=dspl)
        else:
            tvec = np.hstack([
                np.tile((B - A)[:, np.newaxis], (1, n1)) / t1,
                np.tile((C - B)[:, np.newaxis], (1, n2)) / t2,
                np.tile((D - C)[:, np.newaxis], (1, n3)) / t3,
                np.tile((A - D)[:, np.newaxis], (1, n4)) / t4
            ]) / (2 * np.pi)
            rotation = np.array([[0, 1], [-1, 0]])
            normal = rotation @ tvec
            norm = np.linalg.norm(normal, axis=0, keepdims=True)  # shape (1, N)
            normal = normal / norm

            avec = np.zeros((2, npts))
            t0 = n4 // 2
            points = np.roll(points0, shift=t0, axis=1)
            tvec = np.roll(tvec, shift=t0, axis=1)
            normal = np.roll(normal, shift=t0, axis=1)  
        if a == b:
            nstr = 'Square'
        else:
            nstr = 'Rectangle'
        self.width = b
        self.height = a
        super().__init__(points, tvec, avec, normal, np.zeros(2), nstr, npts)
    
    def __mul__(self, m):
        r = super().__mul__(m)
        r.width = self.width*m
        r.height = self.height*m
        return r
