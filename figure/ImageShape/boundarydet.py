import numpy as np
from scipy.interpolate import interp2d
def boundarydet(acc, IM):
    """
    Parameterizes a simply connected region from an image.
    
    The algorithm starts at a boundary point, advects along the tangent for a distance 
    equal to acc, then returns to the boundary using the normal vector. This process 
    repeats until returning to the original point.
    
    Args:
        acc (float): Approximate distance between consecutive points in parameterization (pixels)
        IM (numpy.ndarray): Binary image matrix (foreground/background)
    
    Returns:
        tuple: (D, t) where:
            - D: nx2 array of boundary points coordinates
            - t: array of parameter values from 0 to 2π
    
    Notes:
        Originally by Anton Bongio Karrman (karrman@caltech.edu)
        Modified from original to return untied boundary points
    """
    ny, nx = IM.shape
    dx = dy = 1

    # Initialize level set
    RIiter = 20
    X, Y = np.meshgrid(np.arange(1, nx+1), np.arange(1, ny+1))
    phi = 2 * (IM - 0.5)
    cfl = 0.5
    dt0 = min(dx, dy) * cfl

    # Level set iterations
    for n in range(RIiter):
        phin = shift2n('n', phi)
        phis = shift2n('s', phi)
        phie = shift2n('e', phi)
        phiw = shift2n('w', phi)
        
        phinn = shift2n('n', phin)
        phiss = shift2n('s', phis)
        phiee = shift2n('e', phie)
        phiww = shift2n('w', phiw)

        dxm = (phi - phiw) / dx
        dxp = (phie - phi) / dx
        dym = (phi - phis) / dy
        dyp = (phin - phi) / dy

        dxmxm = (phi - 2*phiw + phiww) / (dx**2)
        dxpxp = (phiee - 2*phie + phi) / (dx**2)
        dxpxm = (phie - 2*phi + phiw) / (dx**2)

        dymym = (phi - 2*phis + phiss) / (dy**2)
        dypyp = (phinn - 2*phin + phi) / (dy**2)
        dypym = (phin - 2*phi + phis) / (dy**2)

        partA = dxm + 0.5*dx*minmod(dxmxm, dxpxm)
        partB = dxp - 0.5*dx*minmod(dxpxp, dxpxm)
        partC = dym + 0.5*dy*minmod(dymym, dypym)
        partD = dyp - 0.5*dy*minmod(dypyp, dypym)

        delp2 = g(partA, partB, partC, partD)
        delm2 = g(partB, partA, partD, partC)

        nabla = 0.5*(dxm**2 + dxp**2 + dym**2 + dyp**2)
        sphi = phi / np.sqrt(phi**2 + np.sqrt(dx**2 + dy**2)*nabla/10)
        sphip = np.maximum(sphi, 0)
        sphim = np.minimum(sphi, 0)

        phi = phi - dt0*(sphip*delp2 + sphim*delm2 - sphi)

    # Compute derivatives and normals
    phix = (phie - phiw)/(2*dx) #type: ignore
    phiy = (phin - phis)/(2*dy) #type: ignore
    epscurv = min(dx, dy)/40
    mag = np.sqrt(phix**2 + phiy**2 + epscurv**2)
    normx = phix/mag
    normy = phiy/mag
    tanx = -normy
    tany = normx

    # Find starting point
    ny2 = int(ny/2)
    x0_idx = np.max(np.where(np.round(phi[ny2, :]) < 0)[0])
    x0 = np.array([x0_idx, ny2])

    # Initial boundary adjustment
    for i in range(2):
        txy = np.array([
            float(np.interp(x0[0], X[0,:], tanx[int(x0[1]),:],)),
            float(np.interp(x0[1], Y[:,0], tany[:,int(x0[0])]))
        ])
        x0 = x0 + acc * txy
        nxy = np.array([
            float(np.interp(x0[0], X[0,:], normx[int(x0[1]),:],)),
            float(np.interp(x0[1], Y[:,0], normy[:,int(x0[0])]))
        ])
        phi_val = float(np.interp(x0[0], X[0,:], phi[int(x0[1]),:]))
        x0 = x0 - phi_val*nxy

    # Main boundary tracing loop
    D = [x0]
    k = 1
    while np.linalg.norm(D[0] - D[-1]) > 2*acc or k < 4:
        x = D[-1].copy()
        txy = np.array([
            float(np.interp(x[0], X[0,:], tanx[int(x[1]),:],)),
            float(np.interp(x[1], Y[:,0], tany[:,int(x[0])]))
        ])
        x = x + acc*txy
        nxy = np.array([
            float(np.interp(x[0], X[0,:], normx[int(x[1]),:],)),
            float(np.interp(x[1], Y[:,0], normy[:,int(x[0])]))
        ])
        phi_val = float(np.interp(x[0], X[0,:], phi[int(x[1]),:]))
        x = x - phi_val*nxy
        D.append(x)
        k += 1

    D = np.array(D).T
    t = 2*np.pi*np.arange(D.shape[1])/D.shape[1]
    
    return D, t

def shift2n(direction, phi):
    """
    Space shift function using Neumann boundary conditions.
    
    Shifts matrix over one index in specified direction to help compute derivatives.
    
    Args:
        direction (str): Direction to shift ('e','w','n','s')
        phi (numpy.ndarray): Input matrix to shift
        
    Returns:
        numpy.ndarray: Shifted matrix with Neumann boundary conditions
    """
    m, n = phi.shape
    phishift = np.zeros((m, n))
    
    if direction == 'e':    # Shift West
        phishift[:,:-1] = phi[:,1:]
        phishift[:,-1] = phi[:,-1]
    elif direction == 'w':  # Shift East
        phishift[:,1:] = phi[:,:-1]
        phishift[:,0] = phi[:,0]
    elif direction == 'n':  # Shift North
        phishift[:-1,:] = phi[1:,:]
        phishift[-1,:] = phi[-1,:]
    elif direction == 's':  # Shift South
        phishift[1:,:] = phi[:-1,:]
        phishift[0,:] = phi[0,:]
        
    return phishift

def minmod(phi1, phi2):
    """
    MINMOD FUNCTION
    
    This function is essential for second order schemes used to solve
    the level set equation (as well as for reinitialization).
    
    Args:
        phi1 (float or numpy.ndarray): First input value/array
        phi2 (float or numpy.ndarray): Second input value/array
        
    Returns:
        float or numpy.ndarray: Minmod of the inputs
    """
    sphi1 = np.sign(phi1)
    sphi2 = np.sign(phi2)
    return np.maximum(0, sphi1 * sphi2) * sphi1 * np.minimum(np.abs(phi1), np.abs(phi2))

def g(u1, u2, v1, v2):
    """
    Numerical flux function for Hamilton Jacobi equations.
    
    This function computes the numerical flux based on the given parameters using
    the following formula:
    sqrt(max(u1,0)^2 + min(u2,0)^2 + max(v1,0)^2 + min(v2,0)^2)
    
    Args:
        u1 (float or numpy.ndarray): First u component
        u2 (float or numpy.ndarray): Second u component
        v1 (float or numpy.ndarray): First v component
        v2 (float or numpy.ndarray): Second v component
    
    Returns:
        float or numpy.ndarray: Computed numerical flux
        
    Notes:
        Originally created by Anton Bongio Karrman (karrman@caltech.edu)
        Based on level-set#type: ignore routines by Gregoire Allaire
   #type: ignore      Translated to Python from original MATLAB implementation (23-SEP-2011)
    """
    return np.sqrt(np.maximum(u1, 0.)**2 + 
                  np.minimum(u2, 0.)**2 +
                  np.maximum(v1, 0.)**2 + 
                  np.minimum(v2, 0.)**2)