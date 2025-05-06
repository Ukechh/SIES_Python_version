import sys
import os

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '../')))

import numpy as np
import math
import matplotlib.pyplot as plt
from FundamentalSols.green import Green2D, Green2D_grad
from figure.C2Boundary.C2Boundary import C2Bound
from PDE.Conductivity_R2.Conductivity import Conductivity

def outer_expansion_conductivityR2():
    pass

def inner_expansion_conductivityR2_0(D : C2Bound):
    '''
    Parameters:
    -----------------
    D: C2Bound
        Inclusion in the field, we use the center of the inclusion to compute the first term of the expansion
    Returns:
    ---------------
    u0: np.array
        Value of the background solution at the center of the inclusion
    '''
    if D._center_of_mass.all() == 0:
        raise ValueError("The background solution is not defined at 0")
    return Green2D(D._center_of_mass, np.zeros((2,1)) ).reshape(1).astype(np.complex128)

def inner_expansion_conductivityR2_1(C : Conductivity, freq : float, w = 2.0, N = 100):
    '''
    This function computes the value of the first term of the inner expansion for the conductivity equation for a given inclusion and frequency
    This function only supports a conductivity instance with only one inclusion.
    
    Parameters:
    ---------------
    C : Conductivity instance, we use this for determining the inclusion around which we compute the far field expansion
    freq: frequency we use to compute the expansion
    w = witdh of the grid
    N: number of points in the grid

    Returns:
    ----------------
    u1: an NxN grid with the value u1(xi) at every point xi
    Sx, Sy: streched and translated xi grids: xi = (x-z0)/delta
    '''
    
    z0 = C._D[0]._center_of_mass

    epsilon = w / ((N-1)*5) #Compute epsilon

    Sx, Sy, _ = C._D[0].boundary_off_mask(z0, w, N, epsilon)
    
    Z = np.vstack((Sx.ravel(), Sy.ravel()), dtype=np.complex128) #Shape (2, N**2)

    v = C.far_field(Z, freq, 0) #Far field computes v((Z-z0)/delta), i.e 
    
    Ux, Uy = Green2D_grad(z0, np.zeros((2,1)))
    
    U = np.vstack((Ux,Uy))
    
    u1 = np.dot(U.T, v).astype(np.complex128)
    
    u1 = u1.reshape(N,N)

    return u1, Sx, Sy

def plot_inner_expansion(C : Conductivity, freq : float, s : int,  w : float, N : int, nbLine=50, *args, **kwargs):
    #Compute the values of the approximation at the grid
    u0 = inner_expansion_conductivityR2_0(C._D[0])
    u1, Sx, Sy = inner_expansion_conductivityR2_1(C, freq, w=w, N=N)
    
    delta = C._D[0].delta
    
    z0 = C._D[0]._center_of_mass
    
    F, F_bg, _, _, _ = C.calculate_field(freq, s, z0=z0, width=w, N=N)

    approx = u0 + delta*u1
    
    u_minus = F - u0 - delta * u1
    
    u_abs = abs(u_minus)

    src = C._cfg.src(s)

    # === Create figure with extra row ===
    fig, axs = plt.subplots(2, 3, figsize=(20, 20))

    # --- Plot 1: Real part of total field ---
    cs1 = axs[0, 0].contourf(Sx, Sy, F.real, nbLine)
    axs[0, 0].contour(Sx, Sy, F.real, nbLine, colors='k', linewidths=0.5)
    #axs[0, 0].plot(src[0], src[1], 'gx', *args, **kwargs)
    for D in C._D:
        D.plot(ax=axs[0, 0], *args, **kwargs)
    axs[0, 0].set_title("Potential field u, real part")
    axs[0, 0].axis('image')
    fig.colorbar(cs1, ax=axs[0, 0])

    # --- Plot 2: Imaginary part of total field ---
    cs2 = axs[1, 0].contourf(Sx, Sy, F.imag, nbLine)
    axs[1, 0].contour(Sx, Sy, F.real, nbLine, colors='k', linewidths=0.5)
    #axs[1, 0].plot(src[0], src[1], 'gx', *args, **kwargs)
    for D in C._D:
        D.plot(ax=axs[1,0], *args, **kwargs)
    axs[1, 0].set_title("Potential field u, imaginary part")
    axs[1, 0].axis('image')
    fig.colorbar(cs2, ax=axs[1, 0])

    # --- Plot 3: Real part of perturbed field ---
    cs3 = axs[0, 1].contourf(Sx, Sy, (F - F_bg).real, nbLine)
    axs[0, 1].contour(Sx, Sy, F.real, nbLine, colors='k', linewidths=0.5)
    #axs[0, 1].plot(src[0], src[1], 'gx', *args, **kwargs)
    for D in C._D:
        D.plot(ax=axs[0, 1], *args, **kwargs)
    axs[0, 1].set_title("Perturbed field u - U, real part")
    axs[0, 1].axis('image')
    fig.colorbar(cs3, ax=axs[0, 1])

    # --- Plot 4: Imaginary part of perturbed field ---
    cs4 = axs[1, 1].contourf(Sx, Sy, (F - F_bg).imag, nbLine)
    axs[1, 1].contour(Sx, Sy, F.real, nbLine, colors='k', linewidths=0.5)
    #axs[1, 1].plot(src[0], src[1], 'gx', *args, **kwargs)
    for D in C._D:
        D.plot(ax=axs[1, 1], *args, **kwargs)
    axs[1, 1].set_title("Perturbed field u - U, imaginary part")
    axs[1, 1].axis('image')
    fig.colorbar(cs4, ax=axs[1, 1])

    # --- Plot 5: u₀ + δ·u₁ --- 
    cs5 = axs[0, 2].contourf(Sx, Sy, u_minus.real, levels=nbLine)
    axs[0, 2].contour(Sx, Sy, (approx - F_bg).real, nbLine, colors='k', linewidths=0.5)
    #axs[0, 2].plot(src[0], src[1], 'gx', *args, **kwargs)
    for D in C._D:
        D.plot(ax=axs[0, 2], *args, **kwargs)
    axs[0, 2].set_title("Perturbed field with approximation, real part (u₀ + δ·u₁-U)")
    axs[0, 2].axis('image')
    fig.colorbar(cs5, ax=axs[0, 2])

    # --- Plot 6: Calculated field minus inner expansion module ---- 
    cs6 = axs[1,2].contourf(Sx, Sy, u_abs, levels=nbLine)
    axs[1,2].contour(Sx, Sy, u_minus.imag, nbLine, colors='k', linewidths=0.5)
    #axs[1,2].plot(src[0], src[1], 'gx', *args, **kwargs)
    for D in C._D:
        D.plot(ax=axs[1,2], *args, **kwargs)
    axs[1,2].set_title("| U - (u₀ + δ·u₁) |")
    axs[1,2].axis('image')
    fig.colorbar(cs6, ax=axs[1,2])