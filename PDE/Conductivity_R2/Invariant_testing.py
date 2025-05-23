import sys
import os

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '/../../')))

#We consider the points array as d x n matrix where d is the dimension and n is the number of points
from scipy.sparse.linalg import lsqr, LinearOperator
import numpy as np
import matplotlib.pyplot as plt
import math
from PDE.Conductivity_R2.Conductivity import Conductivity

def compare_invariants_over_rotations(
    freq, inclusion, nrotations, cnd, pmtt, cfg, drude=True, ord=1, normalize='sum'):

    # Generate rotated shapes
    rotated_shapes = generate_rotated_shapes(inclusion, nrotations)

    eigenvalue_freq_magn = []
    for shape in rotated_shapes:
        cond = Conductivity([shape], cnd, pmtt, cfg, drude=drude)

        MSR, _ = cond.data_simulation(freq)

        reconstructed_CGPT, _, _ = cond.reconstruct_CGPT(MSR, ord, method='lsqr')

        eig_magnitudes = compute_eigenvalue_magnitudes(reconstructed_CGPT)
        eigenvalue_freq_magn.append(eig_magnitudes)
    #Normalization of eigenvalue magnitudes
    if normalize == 'sum':
        # Normalize each set of eigenvalue magnitudes by their sum for each CGPT
        eigenvalue_freq_magn = [
            [eigvals / np.sum(eigvals) if np.sum(eigvals) != 0 else eigvals for eigvals in freq_magn]
            for freq_magn in eigenvalue_freq_magn
        ]
    elif normalize == 'max':
        # Normalize each set of eigenvalue magnitudes by their maximum for each CGPT
        eigenvalue_freq_magn = [
            [eigvals / np.max(eigvals) if np.max(eigvals) != 0 else eigvals for eigvals in freq_magn]
            for freq_magn in eigenvalue_freq_magn
        ]
    # Plot the eigenvalue magnitudes
    if hasattr(freq, '__len__'):
        # freq is iterable, plot for each frequency
        num_freqs = len(freq)
        num_eigenvalues = len(eigenvalue_freq_magn[0][0])
        indices = list(range(nrotations))
        for f_idx in range(num_freqs):
            plt.figure(figsize=(8, 5))
            for eig_idx in range(num_eigenvalues):
                magnitudes = [eigenvalue_freq_magn[rot_idx][f_idx][eig_idx] for rot_idx in range(nrotations)]
                plt.plot(indices, magnitudes, marker='o', label=f'Eigenvalue {eig_idx+1} ({normalize} normalized)')
            plt.title(f'Eigenvalue Magnitudes vs Rotation Index for Frequency {freq[f_idx]}')
            plt.xlabel('Rotation Index n (rotation angle = 2π×n/{}rad)'.format(nrotations))
            plt.ylabel(f'Magnitude ({normalize} normalized)')
            plt.legend()
            plt.grid(True)
            plt.show()
    else:
        # freq is a single value, plot as before
        indices = list(range(nrotations))
        num_eigenvalues = len(eigenvalue_freq_magn[0][0])
        plt.figure(figsize=(8, 5))
        for eig_idx in range(num_eigenvalues):
            magnitudes = [eig_magnitudes[0][eig_idx] for eig_magnitudes in eigenvalue_freq_magn]
            plt.plot(indices, magnitudes, marker='o', label=f'Eigenvalue {eig_idx+1} ({normalize} normalized)')
        plt.title(f'Eigenvalue Magnitudes vs Rotation Index for Frequency {freq}')
        plt.xlabel('Rotation Index n (rotation angle = 2π×n/{}rad)'.format(nrotations))
        plt.ylabel(f'Magnitude ({normalize} normalized)')
        plt.legend()
        plt.grid(True)
        plt.show()

def compare_invariants_over_translations(
    freq, inclusion, ntranslations, cnd, pmtt, cfg, drude=True, ord=1, normalize='sum'):

    # Generate translated shapes
    translated_shapes = generate_translated_shapes(inclusion, ntranslations)

    eigenvalue_freq_magn = []
    # The right subplot (ax_eig) will be used later for eigenvalue plots per frequency
    for idx, shape in enumerate(translated_shapes):
        cond = Conductivity([shape], cnd, pmtt, cfg, drude=drude)

        MSR, _ = cond.data_simulation(freq)

        reconstructed_CGPT, _, _ = cond.reconstruct_CGPT(MSR, ord, method='lsqr')

        eig_magnitudes = compute_eigenvalue_magnitudes(reconstructed_CGPT)
        eigenvalue_freq_magn.append(eig_magnitudes)

    # Normalization of eigenvalue magnitudes
    if normalize == 'sum':
        eigenvalue_freq_magn = [
            [eigvals / np.sum(eigvals) if np.sum(eigvals) != 0 else eigvals for eigvals in freq_magn]
            for freq_magn in eigenvalue_freq_magn
        ]
    elif normalize == 'max':
        eigenvalue_freq_magn = [
            [eigvals / np.max(eigvals) if np.max(eigvals) != 0 else eigvals for eigvals in freq_magn]
            for freq_magn in eigenvalue_freq_magn
        ]
    # Plot the eigenvalue magnitudes
    if hasattr(freq, '__len__'):
        num_freqs = len(freq)
        num_eigenvalues = len(eigenvalue_freq_magn[0][0])
        indices = list(range(ntranslations))
        for f_idx in range(num_freqs):
            plt.figure(figsize=(8, 5))
            for eig_idx in range(num_eigenvalues):
                magnitudes = [eigenvalue_freq_magn[trans_idx][f_idx][eig_idx] for trans_idx in range(ntranslations)]
                plt.plot(indices, magnitudes, marker='o', label=f'Eigenvalue {eig_idx+1}')
            plt.title(f'Eigenvalue Magnitudes vs Translation Index for Frequency {freq[f_idx]}')
            plt.xlabel('Translation Index')
            plt.ylabel('Magnitude')
            plt.legend()
            plt.grid(True)
            plt.tight_layout()
            plt.show()
    else:
        indices = list(range(ntranslations))
        num_eigenvalues = len(eigenvalue_freq_magn[0][0])
        plt.figure(figsize=(8, 5))
        for eig_idx in range(num_eigenvalues):
            magnitudes = [eig_magnitudes[0][eig_idx] for eig_magnitudes in eigenvalue_freq_magn]
            plt.plot(indices, magnitudes, marker='o', label=f'Eigenvalue {eig_idx+1}')
        plt.title(f'Eigenvalue Magnitudes vs Translation Index for Frequency {freq}')
        plt.xlabel('Translation Index')
        plt.ylabel('Magnitude')
        plt.legend()
        plt.grid(True)
        plt.tight_layout()
        plt.show()

def compute_eigenvalue_magnitudes(MSR):
    """
    Compute the magnitudes of eigenvalues for a list of matrices.

    Parameters
    ----------
    MSR : list of np.ndarray
        List of square matrices.

    Returns
    -------
    list of np.ndarray
        List where each element is an array of magnitudes of eigenvalues for the corresponding matrix.
    """
    magnitudes = []
    for mat in MSR:
        eigvals = np.linalg.eigvals(mat)
        magnitudes.append(np.abs(eigvals))
    return magnitudes

def generate_rotated_shapes(shape, nrotations):
    """
    Generate a list of rotated versions of a shape.

    Parameters
    ----------
    shape : C2Bound
        The original shape object.
    nrotations : int
        Number of rotations (N). The n-th shape is rotated by 2*pi*n/N.

    Returns
    -------
    rotated_shapes : list of C2Bound
        List of rotated shape objects.
    """
    rotated_shapes = []
    
    for n in range(nrotations):
        angle = 2 * np.pi * n / nrotations
        rotated = shape < angle  # Uses __lt__ operator for rotation
        fig = plt.figure(figsize=(6, 6))
        rotated.plot(ax=fig.gca())
        rotated_shapes.append(rotated)
    return rotated_shapes

def generate_translated_shapes(shape, ntranslations):
    """
    Generate a list of translated versions of a shape.
    Parameters
    ----------
    shape : C2Bound
        The original shape object.
    ntranslations : int
        Number of translations.

    Returns
    -------
    translated_shapes : list of C2Bound
        List of translated shape objects.
    """
    translated_shapes = []
    for n in range(ntranslations):
        theta = 2 * np.pi * n / ntranslations
        translation_vector = np.array([np.cos(theta), np.sin(theta)])
        translated = shape + translation_vector  # Uses __add__ operator for translation
        translated_shapes.append(translated)
    return translated_shapes

def generate_complex_harmonic_coefficients(order):
    """
    Generate complex harmonic coefficients for a given order, that is: a_{\alpha}^n = \binom{n}{k} i^{n-k} for \alpha = (k, n-k).

    Parameters
    ----------
    order : int
        The order of the harmonic polynomial.

    Returns
    -------
    coeffs : np.ndarray
        Array of complex harmonic coefficients.
    """
    coeffs = np.zeros((order,), dtype=np.complex128)
    for n in range(order):
        coeffs[n] = math.comb(order, n) * (1j)**(order-n) 
    
    return coeffs

