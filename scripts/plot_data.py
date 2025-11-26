import json
import numpy as np
import matplotlib.pyplot as plt

# Load the data
with open('data/sos_6D_benchmark.json', 'r') as f:
    data = json.load(f)

# Extract data
sos_data = np.array(data['sos_log_likelihoods'])  # Shape: (15, 10)
ekf_data = np.array(data['ekf_log_likelihoods'])   # Shape: (10,)
wsasos_data = np.array(data['wsasos_log_likelihoods'])  # Shape: (10,)

# Create timesteps (0 to 9)
timesteps = np.arange(len(ekf_data))

# Calculate mean and std for SOS (across 15 trials)
sos_mean = np.mean(sos_data, axis=0)
sos_std = np.std(sos_data, axis=0)

# Create the plot
plt.figure(figsize=(10, 6))

# Plot SOS with mean and shaded variance region
plt.plot(timesteps, sos_mean, label='SOS', linewidth=2, color='blue')
plt.fill_between(timesteps, sos_mean - sos_std, sos_mean + sos_std, 
                 alpha=0.3, color='blue')

# Plot EKF as a line
plt.plot(timesteps, ekf_data, label='EKF', linewidth=2, color='red')

# Plot WSASOS as a line
plt.plot(timesteps, wsasos_data, label='WSASOS', linewidth=2, color='green')

# Add labels and title
plt.xlabel('Timestep', fontsize=12)
plt.ylabel('Log Likelihood', fontsize=12)
plt.title('Log Likelihood vs Timestep', fontsize=14)
plt.legend(fontsize=11)
plt.grid(True, alpha=0.3)

# Adjust layout and show
plt.tight_layout()
plt.show()

