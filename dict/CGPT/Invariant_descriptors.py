import sys
import os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '../')))

import numpy as np
import math
from asymp.CGPT_methods import CGPT2CCGPT, CCGPT_transform
from PDE.Conductivity_R2.Conductivity import Conductivity
from PDE.Electric_fish.Electric_fish_ import Electric_Fish
from Tools_fct.General_tools import add_white_noise_list
from tqdm import tqdm
from multiprocessing import Pool

def ShapeDescriptors_CGPT(CGPT):
    N1, N2 = CGPT2CCGPT(CGPT)
    ord = N1.shape[0]
    u = N2[0,1] / (2*N2[0,0])
    J1, J2 = CCGPT_transform(N1, N2, T0=-u, S0=1, Phi0=0.0)
    sqrtJ2_diag = np.sqrt(np.outer(np.diag(J2), np.diag(J2)))
    S1 = J1 / sqrtJ2_diag
    S2 = J2 / sqrtJ2_diag
    I1 = np.abs(S1)
    I2 = np.abs(S2)
    return J1, J2, S1, S2, I1, I2

def Compute_Invariants_fish(dict, cfg, cnd, pmtt, freq, ord=2, noise_level=0.0, verbose=True):
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
    verbose: bool
        If True, display progress bars
    Returns:
    ---------------
    I1: list of list
        List of CGPT invariants for each shape in the dictionary and for each working frequency
        I1[n,m] is the invariant matrix of the nth shape in the dictionary at the mth working frequency
    I2: list of list
        List of CGPT invariants for each #type:ignoreshape in the dictionary and for each working frequency
        I2[n,m] is the invariant matrix of the nth shape in the dictionary at the mth working frequency
    or
    tau: list of list
        List of frequency dependent singular values of the PT
    mu: normalized singular values. normalized according to the 
    """
    
    I1 = np.array([[np.zeros((ord, ord)) for _ in range(len(freq))] for _ in range(len(dict))], dtype=object)
    I2 = np.array([[np.zeros((ord, ord)) for _ in range(len(freq))] for _ in range(len(dict))], dtype=object)
    mu = []
    tau = np.array([[np.zeros((2,)) for _ in range(len(freq))] for _ in range(len(dict))], dtype=object)
    # Loop over the shapes in the dictionary
    shape_iter = tqdm(enumerate(dict), desc="Computing shapes", total=len(dict)) if verbose else enumerate(dict)
    for i, shape in shape_iter:
    # Generate the fish instance
        P = Electric_Fish([shape], cnd, pmtt, cfg, stepBEM=2)
        # Compute data for the shape inclusion
        data = P.data_simulation(freq)
        #Compute invariants given the order
        if ord >= 2:
            I1_, I2_ = Compute_fish_Invariants_from_data(data, ord, P, noise_level)
            I1[i] = I1_
            I2[i] = I2_    
        else:
            tau[i], muu = Compute_fish_Invariants_from_data(data, ord, P, noise_level)
            mu.append(muu)
    if ord >= 2:
        return I1, I2
    else:
        return tau, mu

def Compute_Invariants_conductivity(dict, cfg, cnd, pmtt, freq, ord=2, noise_level=0.0, verbose=True):
    I1 = np.array([[np.zeros((ord, ord)) for _ in range(len(freq))] for _ in range(len(dict))], dtype=object)
    I2 = np.array([[np.zeros((ord, ord)) for _ in range(len(freq))] for _ in range(len(dict))], dtype=object)
    mu = []
    tau = np.array([[np.zeros((2,)) for _ in range(len(freq))] for _ in range(len(dict))], dtype=object)
    # Loop over the shapes in the dictionary
    shape_iter = tqdm(enumerate(dict), desc="Computing shapes", total=len(dict)) if verbose else enumerate(dict)
    for i, shape in shape_iter:
    # Generate the conductivity instance
        P = Conductivity([shape], np.array([cnd[i]]), np.array([pmtt[i]]), cfg, drude=True)
        data, _ = P.data_simulation(freq)
        # Add noise to the data
        if noise_level > 0:
            data, _ = add_white_noise_list(data, noise_level)
        # Compute CGPTs
        rCGPT = P.reconstruct_CGPT(data, ord, method='lsqr')
        CGPT = rCGPT['CGPT']
        freq_iter = tqdm(range(len(freq)), desc=f"Computing frequencies for shape {i+1}", leave=False) if verbose else range(len(freq))
        for j in freq_iter:
            #Compute the invariants for the jth frequency
            _,_,_,_, I1_, I2_ = ShapeDescriptors_CGPT(CGPT[j])
            I1[i][j] = I1_
            I2[i][j] = I2_
    return I1, I2

def Compute_fish_Invariants_from_data(data, ord, fishP, noise_level, verbose=True):
    freq = data['frequencies']
    I1 = [np.zeros((ord, ord)) for _ in range(len(freq))]
    I2 = [np.zeros((ord, ord)) for _ in range(len(freq))]
    tau = [np.zeros((2,)) for _ in range(len(freq))]
    mu = []
    if ord >= 2:
        Current = data['Current']
        MSR, _ = add_white_noise_list(data['MSR'], noise_level)
        # Compute CGPTs
        rCGPT = fishP.reconstruct_CGPT(MSR, Current, ord)
        CGPT = rCGPT['CGPT']
        freq_iter = tqdm(range(len(freq)), desc=f"Computing frequency dependent Invariants", leave=False) if verbose else range(len(freq))
        for j in freq_iter:
            #Compute the invariants for the jth frequency
            _,_,_,_, I1_, I2_ = ShapeDescriptors_CGPT(CGPT[j])
            I1[j] = I1_    #type: ignore
            I2[j] = I2_    #type: ignore
    else:
        Current_bg = data['Current_bg']
        SFR, _ = add_white_noise_list(data['SFR'], noise_level)
        #Compute PTs
        p = fishP.reconstruct_PT(SFR, Current_bg)
        PT = p['PT']
        t = np.empty(1)
        freq_iter = tqdm(range(len(freq)), desc=f"Computing frequency dependent PTs", leave=False) if verbose else range(len(freq))
        for j in freq_iter:
            tau[j] = ShapeDescriptors_PT(PT[j]) #type: ignore
            if j == 0:
                t = tau[j].reshape(2,1)
            else:
                t = np.hstack((t, (tau[j]).reshape(2,1)))
        tmax = np.array(
            (np.max(t[0,:]),
            np.max(t[1,:])) ).reshape(2,1)
        mu = [t / tmax]
    if ord >= 2:
        return I1, I2
    else:
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
    
    if len(I1) == 1:
        I1 = [A for A in I1[0]]
        I2 = [A for A in I2[0]]
    #Loop over the frequencies
    for i in range(len(I1_dico_freq[0])):
        I1_dico = I1_dico_freq[:,i]
        I2_dico = I2_dico_freq[:,i]
        _,  q = ShapeRecognition_ShapeInvariants(I1_dico, I2_dico, I1[i], I2[i])
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
    if len(I1) == 1:
        I1 = [l for l in I1[0]]
        I2 = [l for l in I2[0]]
    for i in range(len(I1_dico_freq[0])):
        I1_dico = I1_dico_freq[:,i]
        I2_dico = I2_dico_freq[:,i]
        index,  q = ShapeRecognition_ShapeInvariants(I1_dico, I2_dico, I1[i], I2[i])
        votes[index] += 1
    shape_index = np.argmax(votes)
    
    return shape_index, votes

def ShapeDescriptors_PT(PT):
    
    tau = np.linalg.svd(PT, compute_uv=False)
    
    tau = tau.reshape((2,))
    
    return tau

def ShapeRecognition_PT_freq(mu_dico, mu, mode='mu'):
    d_total = np.zeros(len(mu_dico))
    if mode == 'mu':
        for j in range(len(mu_dico)):
            d = np.abs(mu_dico[j] - mu[0])  # Compute absolute difference
            d_sum = np.sum(d, axis=0)  # Sum along columns
            d_total[j] = np.sum(d_sum)  # Sum all elements
        shape_index = np.argmin(d_total)
    else:
        for s in range(len(mu_dico)):
            mu_dic = mu_dico[s]
            d = (mu_dic-mu)**2
            d_total[s] = np.sum(d)
        shape_index = np.argmin(d_total)
    return shape_index