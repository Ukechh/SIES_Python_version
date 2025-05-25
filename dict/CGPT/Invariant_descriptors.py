import sys
import os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '../')))

import numpy as np
import math
from asymp.CGPT_methods import CGPT2CCGPT, CCGPT_transform
from PDE.Conductivity_R2.Conductivity import Conductivity
from PDE.Electric_fish.Electric_fish_ import Electric_Fish
from Tools_fct.General_tools import add_white_noise_list

def ShapeDescriptors_CGPT(CGPT):
    N1, N2 = CGPT2CCGPT(CGPT)
    ord = N1.shape[0]
    u = N2[0,1] / (2*N2[0,0])
    J1, J2 = CCGPT_transform(N1, N2, -u, 1)
    
    S1, S2 = np.zeros((ord, ord), dtype=np.complex128), np.zeros((ord, ord), dtype=np.complex128)
    I1, I2 = np.zeros((ord, ord), dtype=np.complex128), np.zeros((ord, ord), dtype=np.complex128)
    
    for m in range(ord):
        for n in range(ord):
            S1[m,n] = J1[m,n] / np.sqrt(J2[m,m]*J2[n,n])
            S2[m,n] = J2[m,n] / np.sqrt(J2[m,m]*J2[n,n])
    
    I1 = np.abs(S1)
    I2 = np.abs(S2)
    
    return J1, J2, S1, S2, I1, I2

def Compute_Invariants(dict, cfg, cnd, pmtt, freq, pde, ord=2, noise_level=0.0):
    """
    Computes the rigid motion invariants of the CGPTs of the dictionary elements
    Parameters:
    ---------------
    dict: list of C2Bound
        The dictionary of shapes
    cfg: mconfig object 
        Acquisition system
    cnd: array
        Conductivity of the inclusions
    pmtt: array
        Permittivity of the inclusions
    freq: array
        Working frequencies at which the data is computed
    pde: string
        The PDE used for the simulation
    ord: int
        Maximum order of the CGPTs
    Returns:
    ---------------
    I1: list of list
        List of CGPT invariants for each shape in the dictionary and for each working frequency
        I1[n,m] is the invariant matrix of the nth shape in the dictionary at the mth working frequency
    I2: list of list
        List of CGPT invariants for each shape in the dictionary and for each working frequency
        I2[n,m] is the invariant matrix of the nth shape in the dictionary at the mth working frequency
    """

    
    I1 = np.array([[np.zeros((ord, ord)) for _ in range(len(freq))] for _ in range(len(dict))], dtype=object)
    I2 = np.array([[np.zeros((ord, ord)) for _ in range(len(freq))] for _ in range(len(dict))], dtype=object)
    mu = []
    tau = np.array([[np.zeros((2,)) for _ in range(len(freq))] for _ in range(len(dict))], dtype=object)
    # Loop over the shapes in the dictionary
    for i, shape in enumerate(dict):
        # Generate the conductivity instance
        if pde == 'conductivity':
            P = Conductivity([shape], np.array([cnd[i]]), np.array([pmtt[i]]), cfg, drude=False)
            data, _ = P.data_simulation(freq)
            # Add noise to the data
            if noise_level > 0:
                data, _ = add_white_noise_list(data, noise_level)
            # Compute CGPTs
            rCGPT = P.reconstruct_CGPT(data, ord, method='lsqr')
            CGPT = rCGPT['CGPT']
            for j in range(len(freq)):
                #Compute the invariants for the jth frequency
                _,_,_,_, I1_, I2_ = ShapeDescriptors_CGPT(CGPT[j])
                I1[i][j] = I1_
                I2[i][j] = I2_
        elif pde == 'fish':
            P = Electric_Fish([shape], cnd, pmtt, cfg, stepBEM=2)
            # Compute data for the shape inclusion
            data = P.data_simulation(freq)
            
            if ord >= 2 :
                MSR = data['MSR']
                Current = data['Current']

                # Compute CGPTs
                rCGPT = P.reconstruct_CGPT(MSR, Current, ord)
                CGPT = rCGPT['CGPT']
                for j in range(len(freq)):
                    #Compute the invariants for the jth frequency
                    _,_,_,_, I1_, I2_ = ShapeDescriptors_CGPT(CGPT[j])
                    I1[i][j] = I1_
                    I2[i][j] = I2_
                print(" Invariants computed! ")    
            else:
                SFR = data['SFR']
                Current_bg = data['Current_bg']
                #Compute PTs
                p = P.reconstruct_PT(SFR, Current_bg)
                PT = p['PT']
                ti = np.empty(1)
                for j in range(len(freq)):
                    tau[i][j] = ShapeDescriptors_PT(PT[j]) #type: ignore
                    if j == 0:
                        ti = tau[i][j].reshape(2,1)
                    else:
                        ti = np.hstack((ti, (tau[i][j]).reshape(2,1)))
                tmax = np.array(
                    (np.max(ti[0,:]),
                    np.max(ti[1,:])) ).reshape(2,1)
                mu.append(ti / tmax)
                print('Ratio computed!')
    if ord >= 2:
        return I1, I2
    else:
        print(f"The shape of elements in mu is {mu[0].shape}")
        return tau, mu

def ShapeRecognition_ShapeInvariants(I1_dico, I2_dico, I1, I2):
    """
    From the CGPT invariants, utilize the invariants to attribute the shape to one in the dictionary
    Parameters:
    ---------------
    I1_dico: list of ndarray
        List of CGPT invariants for each shape in the dictionary
    I2_dico: list of ndarray
        List of CGPT invariants for each shape in the dictionary
    I1: ndarray
        CGPT invariant
    I2: list of ndarray
        CGPT invariant
    Returns:
    ---------------
    index: int
        The index of the shape in the dictionary that is closest to the shape
    """
    ord = I1[0].shape[0]
    #Initialize the difference lists
    e = np.zeros((len(I1_dico)), dtype=np.complex128)
    #Compute the difference between the CGPT invariants of the shape and the figures in the dictionary
    for i in range(len(I1_dico)):
        e[i] = np.linalg.norm(I1_dico[i] - I1, 'fro')**2 + np.linalg.norm(I2_dico[i] - I2, 'fro')**2
    #Find the index of the shape in the dictionary that is closest to the shape
    index = np.argmin(e)
    return index, e

def ShapeRecognition_CGPT_frequency(I1_dico_freq, I2_dico_freq, I1, I2):
    """
    From the precomputed CGPT invariants o the dictionary, depending on frequency, determine the shape by majority voting
    Parameters:
    ---------------
    I1_dico_freq: list of list of ndarray
        List of CGPT invariants for each shape in the dictionary and for each frequency
    I2_dico_freq: list of list of ndarray
        List of CGPT invariants for each shape in the dictionary and for each frequency
    I1: list of ndarray
        CGPT invariant of the unknown shape
    I2: list of ndarray
        CGPT invariant of the unknown shape
    Returns:
    ---------------
    index: int
        The index of the shape in the dictionary determined by majority voting on frequency dependent invariants
    """
    e = np.zeros((len(I1_dico_freq)), dtype=np.complex128)
    #Loop over the frequencies
    for i in range(len(I1_dico_freq[0])):
        I1_dico = I1_dico_freq[:,i]
        I2_dico = I2_dico_freq[:,i]
        index,  q = ShapeRecognition_ShapeInvariants(I1_dico, I2_dico, I1[0][i], I2[0][i])
        e += q
    shape_index = np.argmin(abs(e))
    
    return shape_index, abs(e)

def ShapeRecognition_CGPT_majority_voting_frequency(I1_dico_freq, I2_dico_freq, I1, I2):
    """
    From the precomputed CGPT invariants o the dictionary, depending on frequency, determine the shape by majority voting
    Parameters:
    ---------------
    I1_dico_freq: list of list of ndarray
        List of CGPT invariants for each shape in the dictionary and for each frequency
    I2_dico_freq: list of list of ndarray
        List of CGPT invariants for each shape in the dictionary and for each frequency
    I1: list of ndarray
        CGPT invariant of the unknown shape
    I2: list of ndarray
        CGPT invariant of the unknown shape
    Returns:
    ---------------
    index: int
        The index of the shape in the dictionary determined by majority voting on frequency dependent invariants
    """
    votes = np.zeros((len(I1_dico_freq)))
    #Loop over the frequencies
    for i in range(len(I1_dico_freq[0])):
        I1_dico = I1_dico_freq[:,i]
        I2_dico = I2_dico_freq[:,i]
        index,  q = ShapeRecognition_ShapeInvariants(I1_dico, I2_dico, I1[0][i], I2[0][i])
        votes[index] += 1
    shape_index = np.argmax(votes)
    
    return shape_index, votes

def ShapeDescriptors_PT(PT):
    
    tau = np.linalg.svd(PT, compute_uv=False)
    
    tau = tau.reshape((2,))
    
    return tau

def ShapeRecognition_PT_freq(mu_dico, mu):
    d_total = np.zeros(len(mu_dico))
    for j in range(len(mu_dico)):
        d = np.abs(mu_dico[j] - mu[0])  # Compute absolute difference
        d_sum = np.sum(d, axis=0)  # Sum along columns
        d_total[j] = np.sum(d_sum)  # Sum all elements
        print(f'Error vector for shape {j} is {d_sum}')
    shape_index = np.argmin(d_total)
    print(d_total)
    return shape_index