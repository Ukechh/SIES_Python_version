import sys
import os

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '../../')))

import numpy as np
import matplotlib.pyplot as plt
from figure.Geom_figures import Ellipse, Triangle, Rectangle, Flower
from PDE.Conductivity_R2.Conductivity import Conductivity
from cfg import mconfig
from dict.CGPT.Invariant_descriptors import Compute_Invariants_conductivity, ShapeRecognition_CGPT_frequency, ShapeRecognition_CGPT_majority_voting_frequency


#Set up the number of points
N = 2**9
#Create different inclusions
B1 = Ellipse(1, 1/2, phi=0.0, NbPts=N)
B2 = Triangle(1, np.pi/3, npts= N)
B3 = Rectangle(1, 1/2, N)
B4 = Flower(1, 1, N, 5,0.3, tau=0) 
B5 = Ellipse(1, 1/2, phi=0.0, NbPts=N) * 0.3

#Define a dictionary of shapes
D = [B1, B2, B3, B4, B5]

#Set conductivity and permitivitty (Single inclusion)
cnd = 10*np.array([1,1,1,1, 0.5])
pmtt = 5*np.array([1,1,1,1, 0.5])

#Set up a list of working frequencies
freq = np.linspace(0.01, 3*np.pi, endpoint=False, num=10)

#Set up an acquisition system
cfg = mconfig.Coincided( np.array([-1,1]), 1, 20, np.array([1.0, 2*np.pi, 2*np.pi]))
#Compute the invariants of the Dictionary shapes
I1, I2 = Compute_Invariants_conductivity(D, cfg, cnd, pmtt, freq, ord=2) 

#Define a true index for the shape:
true_index = 1
#Bnew = Triangle(1, np.pi/4, npts=N) * 0.3
Bnew = (D[true_index] * 0.4 ) < np.pi / 4

epsilon = 0.01
p = -0.2
n = 0

Bnew = Bnew.global_perturbation(epsilon, p, n)
cnd_new = 10*np.array([1])
pmtt_new = 5*np.array([1])

#Compute the invariants of the new shape
I1_new, I2_new = Compute_Invariants_conductivity([Bnew], cfg, cnd_new, pmtt_new, freq, ord=2, noise_level=0.0)
# Recognize the shape
index, errors = ShapeRecognition_CGPT_frequency(I1, I2, I1_new, I2_new)
#Print the found results
print( "Recognized shape index:", index)
print("True index:", true_index)
print("Total error vector:", errors)
# Plot the original and new shape in subplots
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(10, 5))

# Plot shapes using the axes
D[true_index].plot(ax=ax1, color='blue')
ax1.set_title('Original Shape')
ax1.legend()

Bnew.plot(ax=ax2, color='red')
ax2.set_title('New Shape')
ax2.legend()

plt.tight_layout()
plt.show()
# Define noise levels to test and number of trials per noise level
noise_levels = np.logspace(-4, 1, 15)  # from 10^-4 to 10^0
trials_per_noise = 5
success_rates = []

# The true index for Bnew (assuming it matches one of the shapes in D)
true_index = 1  # Adjust this to the correct index

# Test each noise level
for noise in noise_levels:
    successes = 0
    
    # Perform multiple trials for each noise level
    for _ in range(trials_per_noise):
        # Compute the invariants of the new shape with current noise level
        I1_new, I2_new = Compute_Invariants_conductivity([Bnew], cfg, cnd_new, pmtt_new, freq,ord=2, noise_level=noise)
        
        # Recognize the shape
        index, error = ShapeRecognition_CGPT_frequency(I1, I2, I1_new, I2_new)
        print("Recognized shape index:", index)
        print("True index:", true_index)
        print("Total errors wrt quantities of interest:", abs(error) )
        # Check if recognition was successful
        if index == true_index:
            successes += 1
            # Calculate rate of correct votes for this successful trial

    
    # Calculate success rate for current noise level
    success_rate = successes / trials_per_noise
    success_rates.append(success_rate)
    

# Create a figure with two subplots
plt.figure(figsize=(15, 6))

# First subplot for success rates
plt.subplot(1, 2, 1)
plt.semilogx(noise_levels, success_rates, 'bo-')
plt.grid(True)
plt.xlabel('Noise Level')
plt.ylabel('Success Rate')
plt.title('Shape Recognition Success Rate vs Noise Level')

plt.tight_layout()
plt.show()
