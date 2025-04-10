import numpy as np

class Ellipse:

    nature = 'figure';
    R = np.array([[0,-1],[1,0]]);
    def __init__(self,a=1.0,b=1.0,phi=0.0, NbPts=100):
        if a < b:
            raise ValueError('Value error: the semi-major axis must be longer than the semi-minor one.');
        self.axis_a = a
        self.axis_b = b
        self.phi = phi
        self.nb_points = NbPts
        self.theta = np.linspace(0, 2*np.pi,NbPts)
        x = a * np.cos(self.theta)
        y = b * np.sin(self.theta)
        self.boundary_points = np.vstack((x,y))
        tx = -a * np.sin(self.theta)
        ty = b * np.cos(self.theta)
        self.tvec = np.vstack((tx,ty))
        ax = -x
        ay = -y
        self.avec = np.vstack((ax,ay))  
        normal = self.R @ self.tvec
        self.normal = normal / np.linalg.norm(normal, axis = 0)  
    
    def get_points(self):
        return self.boundary_points
    def get_normal(self):
        return self.normal
    def get_tvec(self):
        return self.tvec
    def get_avec(self):
        return self.avec


class Banana:
    nature = 'figure';
    def __init__(self,center,a, b, curvature):
        if a < b:
            raise ValueError('Value error: semi-major axis must be longer than the semi minor one.');
        self.axis_a = a
        self.axis_b = b
        self.center = center
        self.curvature = curvature