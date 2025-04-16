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

        super().__init__(points, tvec, avec, normal, self._center_of_mass, name_str)
    
    def get_points(self):
        return self._points
    def get_normal(self):
        return self._normal
    def get_tvec(self):
        return self._tvec
    def get_avec(self):
        return self._avec


class Banana:
    nature = 'figure';
    def __init__(self,center,a, b, curvature):
        if a < b:
            raise ValueError('Value error: semi-major axis must be longer than the semi minor one.');
        self.axis_a = a
        self.axis_b = b
        self.center = center
        self.curvature = curvature