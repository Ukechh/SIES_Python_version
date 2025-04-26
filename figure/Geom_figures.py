import sys
import os

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '../../')))

import numpy as np
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