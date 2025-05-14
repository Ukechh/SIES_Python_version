import sys
import os

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '/../../')))

#We consider the points array as d x n matrix where d is the dimension and n is the number of points
from scipy.sparse.linalg import lsqr, LinearOperator
import numpy as np
import matplotlib.pyplot as plt
from PDE.Conductivity_R2.Conductivity import Conductivity

def compare_invariants_over_transformations(
    freq, inclusion, nrotations, cnd, pmtt, cfg, drude=True, ord=1):

    # Generate rotated shapes
    rotated_shapes = generate_rotated_shapes(inclusion, nrotations)

    eigenvalue_freq_magn = []
    for shape in rotated_shapes:
        cond = Conductivity([shape], cnd, pmtt, cfg, drude=drude)

        MSR, _ = cond.data_simulation(freq)

        reconstructed_CGPT, _, _ = cond.reconstruct_CGPT(MSR, ord, method='lsqr')

        eig_magnitudes = compute_eigenvalue_magnitudes(reconstructed_CGPT)
        eigenvalue_freq_magn.append(eig_magnitudes)
    
    # Plot the eigenvalue magnitudes
    if hasattr(freq, '__len__'):
        # freq is iterable, plot for each frequency
        num_freqs = len(freq)
        num_eigenvalues = len(eigenvalue_freq_magn[0][0])
        angles = [2 * np.pi * n / nrotations for n in range(nrotations)]
        for f_idx in range(num_freqs):
            plt.figure(figsize=(8, 5))
            for eig_idx in range(num_eigenvalues):
                magnitudes = [eigenvalue_freq_magn[rot_idx][f_idx][eig_idx] for rot_idx in range(nrotations)]
                plt.plot(angles, magnitudes, marker='o', label=f'Eigenvalue {eig_idx+1}')
            plt.title(f'Eigenvalue Magnitudes vs Rotation Angle for Frequency {freq[f_idx]}')
            plt.xlabel('Rotation Angle (radians)')
            plt.ylabel('Magnitude')
            plt.legend()
            plt.grid(True)
            plt.show()
    else:
        # freq is a single value, plot as before
        angles = [2 * np.pi * n / nrotations for n in range(nrotations)]
        num_eigenvalues = len(eigenvalue_freq_magn[0][0])
        plt.figure(figsize=(8, 5))
        for eig_idx in range(num_eigenvalues):
            magnitudes = [eig_magnitudes[0][eig_idx] for eig_magnitudes in eigenvalue_freq_magn]
            plt.plot(angles, magnitudes, marker='o', label=f'Eigenvalue {eig_idx+1}')
        plt.title(f'Eigenvalue Magnitudes vs Rotation Angle for Frequency {freq}')
        plt.xlabel('Rotation Angle (radians)')
        plt.ylabel('Magnitude')
        plt.legend()
        plt.grid(True)
        plt.show()


def compute_eigenvalue_magnitudes(MSR):
        """
        Compute the magnitudes of eigenvalues for a list of matrices.

        Parameters
        ----------
        matrices : list of np.ndarray
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
        rotated_shapes.append(rotated)
    return rotated_shapes

