import math
import numpy as np
from scipy.interpolate import CubicSpline
from scipy.differentiate import derivative

def curve_derivative(D,t):
    dD = D - np.roll(D, shift=1,axis=1)
    dt = np.mod(t-np.roll(t,shift=1,axis=1), 2*np.pi)

    tvec = dD / dt
    return tvec

def curve_derivative2(D,t):
    Dr = np.roll(D, shift = 1, axis=1)
    Dl = np.roll(D,shift = -1, axis=1)

    dD = Dr -2*D+Dl
    dt = np.mod(t-np.roll(t,shift=1,axis=1), 2*np.pi) ** 2

    avec = dD / dt
    avec = np.roll(avec, shift = -1, axis=1)
    return avec


def boundary_vec(D,t):
    tvec = curve_derivative(D,t)
    avec = curve_derivative2(D,t)

    R = np.array([[0, 1],[-1, 0]])
    normal = R@tvec
    normal = normal / np.linalg.norm(normal, axis =0)
    
    return tvec, avec, normal

def boundary_vec_interpl(pts0,theta0,theta):

    if theta is None:
        theta = theta0
    fx = CubicSpline(theta0, pts0[0, :], bc_type='natural')
    fy = CubicSpline(theta0, pts0[1,:],bc_type='natural')

    px = fx(theta)
    py = fy(theta)

    points = np.vstack((px, py))

    tx = fx.derivative(1)(theta)
    ty = fy.derivative(1)(theta)
    tvec = np.vstack((tx, ty))

    R = np.array([[0, 1],[-1, 0]])
    normal = R@tvec
    normal = normal / np.linalg.norm(normal, axis = 0)

    ax = fx.derivative(2)(theta)
    ay = fy.derivative(2)(theta)
    avec = np.vstack((ax, ay))

    return points, tvec, avec, normal

def rescale(D0, theta0, npts, nsize=[], dspl=None):
    if dspl is None:
        dspl = 1
    dspl = math.ceil(dspl)
    if dspl < 1:
        raise ValueError("Down-sampling factor must be positive")
    if nsize is None:
        nsize = np.array([])
    if not nsize.size == 0:
        minx = min(D0[0,:])
        maxx = max(D0[0,:])
        miny = min(D0[1,:])
        maxy = max(D0[1,:])

        z0 = np.array([(minx+maxx)/2, (miny+maxy)/2])
        D0 = np.array([(D0[0,:]-z0[0]*(nsize[0] / (maxx-minx))), (D0[1,:]-z0[1]*(nsize[1] / (maxy-miny)))])
    
    #Resampling
    nbPoints = len(theta0)
    idx = np.concatenate([np.arange(0, nbPoints - 1, dspl), [nbPoints - 1]])
    theta = np.linspace(theta0[0], theta0[-1], npts)

    D, tvec, avec, normal = boundary_vec_interpl( D0[:,idx], theta0[idx], theta)

    return D, tvec, avec, normal

def rescal_diff(D0, theta0, npts, nsize):
    if nsize is None:
        nsize = np.array([])
    if not nsize.size == 0:
        minx = min(D0[0,:])
        maxx = max(D0[0,:])
        miny = min(D0[1,:])
        maxy = max(D0[1,:])

        z0 = np.array([(minx+maxx)/2, (miny+maxy)/2])
        D0 = np.array([(D0[0,:]-z0[0]*(nsize[0] / (maxx-minx))), (D0[1,:]-z0[1]*(nsize[1] / (maxy-miny)))])
    theta = np.linspace(0, 2*np.pi, npts,endpoint=False)
    cx = CubicSpline(theta0,D0[0,:])
    cy = CubicSpline(theta0,D0[1,:])
    Dx = cx(theta)
    Dy = cy(theta)

    D = np.vstack([Dx,Dy])

    tvec, avec, normal = boundary_vec(D, theta)

    return D, tvec, avec, normal