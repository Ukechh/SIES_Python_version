class Ellipse:

    nature = 'figure';
    def __init__(self,a=1.0,b=1.0,phi=0.0):
        if a < b:
            raise ValueError('Value error: the semi-major axis must be longer than the semi-minor one.');
        self.axis_a = a
        self.axis_b = b
        self.phi = phi

class Banana:
    nature = 'figure';
    def __init__(self,center,a, b, curvature):
        if a < b:
            raise ValueError('Value error: semi-major axis must be longer than the semi minor one.');
        self.axis_a = a
        self.axis_b = b
        self.center = center
        self.curvature = curvature