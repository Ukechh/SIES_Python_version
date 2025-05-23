import sys
import os

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '../../')))

import numpy as np
import matplotlib.pyplot as plt
from figure.Geom_figures import Ellipse, Triangle, Rectangle, Flower
from PDE.Conductivity_R2.Conductivity import Conductivity
from cfg import mconfig
from dict.CGPT.Invariant_descriptors import Compute_Invariants, ShapeRecognition_CGPT_frequency, ShapeRecognition_CGPT_majority_voting_frequency


#Set up the number of points
N = 2**9
#Set up typical size
delta = 0.3
#Create different inclusions
B1 = Ellipse(1, 1/2, phi=0.0, NbPts=N)*0.1
B2 = Triangle(1, np.pi/3, npts= N)
B3 = Rectangle(1, 1/2, N)
B4 = Flower(1, 1, N, 5,0.3, tau=0) 
B5 = Ellipse(1, 1/2, phi=0.0, NbPts=N) * 0.3

#Define a dictionary of shapes
D = [B1*delta, B2*delta, B3*delta, B4*delta, B5*delta]


#Define a true index for the shape:
true_index = 1

Bnew = (D[true_index]) < np.pi / 4

epsilon = 0
p = 0
n = 0
#Perturb the chosen shape a bit
Bnew = Bnew.global_perturbation(epsilon, p, n)
axx = plt.subplot()
Bnew.plot(ax=axx)
plt.show()

#Set conductivity and permitivitty (Single inclusion)
cnd = 10*np.array([1,1,1,1, 0.5])
pmtt = 5*np.array([1,1,1,1, 0.5])

#Set up a list of working frequencies
freq = np.linspace(0.01, 3*np.pi, endpoint=False, num=20)

#Set up the fish body
Omega = Ellipse(1, 1/3, phi=0.0, NbPts=N) * 0.5
ax = plt.subplot()

#Set up specific parameters for acquisition method
idxRcv = np.arange(0, Omega.nb_points-1,2)
Ns = 8
impd = 1
#Set up the fish acquisition system
cfg = mconfig.Fish_circle(Omega, idxRcv, np.zeros((2,1)), 1, Ns, 2*np.pi, impd=impd)
cfg.plot(ax=ax)
for shape in D:
    shape.plot(ax=ax)
plt.show()

#Compute Invariants for the dictionary shapes
I1, I2 = Compute_Invariants(D, cfg, cnd, pmtt, freq, 'fish', ord=2)
""" I1_1, I2_1 = Compute_Invariants(D, cfg, cnd, pmtt, freq, 'fish', ord=3) """
#Define new conductivity and permitivitty values
cnd_new = 10*np.array([1])
pmtt_new = 5*np.array([1])

#Compute the invariants of the new shape
I1_new, I2_new = Compute_Invariants([Bnew], cfg, cnd_new, pmtt_new, freq, 'fish', ord=2, noise_level=0.0)

#I1_new_1, I2_new_1 =  Compute_Invariants([Bnew], cfg, cnd_new, pmtt_new, freq, 'fish', ord=3, noise_level=0.0)

# Recognize the shape
index, error = ShapeRecognition_CGPT_frequency(I1, I2, I1_new, I2_new)
index_maj, votes = ShapeRecognition_CGPT_majority_voting_frequency(I1, I2, I1_new, I2_new)
""" index1, votes1 = ShapeRecognition_CGPT_frequency(I1_1, I2_1, I1_new_1, I2_new_1) """
 
#Print the found results
print(f"Results for sum of QOI")
print( "Recognized shape index QOI:", index)
print(f'Total errors are:{error}')
print( "Recognized shape index MV:", index_maj)
print("Votes per shape:", votes)
print("True index:", true_index)


"""
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
plt.show() """
""" # Define noise levels to test and number of trials per noise level
noise_levels = np.logspace(-4, 1, 15)  # from 10^-4 to 10^0
trials_per_noise = 10
success_rates = []
frequency_vote_rates = [] 

# The true index for Bnew (assuming it matches one of the shapes in D)
true_index = 1  # Adjust this to the correct index
 """
""" # Test each noise level
for noise in noise_levels:
    successes = 0
    correct_vote_rates = []  # Store vote rates for successful trials
    
    # Perform multiple trials for each noise level
    for _ in range(trials_per_noise):
        # Compute the invariants of the new shape with current noise level
        I1_new, I2_new = Compute_Invariants([Bnew], cfg, cnd_new, pmtt_new, freq, ord=2, noise_level=noise)
        
        # Recognize the shape
        index, frequency_votes = ShapeRecognition_CGPT_majority_voting_frequency(I1, I2, I1_new, I2_new)
        print("Recognized shape index:", index)
        print("True index:", true_index)
        print("Votes per shape:", frequency_votes)
        # Check if recognition was successful
        if index == true_index:
            successes += 1
            # Calculate rate of correct votes for this successful trial
            correct_votes = frequency_votes[index]
            correct_vote_rate = correct_votes / len(freq)
            correct_vote_rates.append(correct_vote_rate)
    
    # Calculate success rate for current noise level
    success_rate = successes / trials_per_noise
    success_rates.append(success_rate)
    
    # Calculate average frequency vote rate for this noise level
    avg_vote_rate = np.mean(correct_vote_rates) if correct_vote_rates else 0
    frequency_vote_rates.append(avg_vote_rate)

# Create a figure with two subplots
plt.figure(figsize=(15, 6))

# First subplot for success rates
plt.subplot(1, 2, 1)
plt.semilogx(noise_levels, success_rates, 'bo-')
plt.grid(True)
plt.xlabel('Noise Level')
plt.ylabel('Success Rate')
plt.title('Shape Recognition Success Rate vs Noise Level')

# Second subplot for frequency vote rates
plt.subplot(1, 2, 2)
plt.semilogx(noise_levels, frequency_vote_rates, 'ro-')
plt.grid(True)
plt.xlabel('Noise Level')
plt.ylabel('Correct Vote Rate')
plt.title('Frequency Correct Vote Rate vs Noise Level')

plt.tight_layout()
plt.show()

print("Success rates and vote rates for different noise levels:")
for noise, s_rate, v_rate in zip(noise_levels, success_rates, frequency_vote_rates):
    print(f"Noise level {noise:.1e}: Success={s_rate:.2%}, Vote rate={v_rate:.2%}") """