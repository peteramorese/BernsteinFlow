import numpy as np
import matplotlib.pyplot as plt

def sample_modal_gaussian(n_samples: int, means, covariances, weights=None, seed=None):
    if seed is not None:
        np.random.seed(seed)

    num_components = len(means)
    if weights is None:
        weights = np.ones(num_components) / num_components
    else:
        weights = np.array(weights, dtype=np.float32)
        weights /= weights.sum()  # Normalize to sum to 1

    # Choose component indices for each sample
    component_ids = np.random.choice(num_components, size=n_samples, p=weights)

    # Generate samples
    samples = np.array([
        np.random.multivariate_normal(mean=means[i], cov=covariances[i])
        for i in component_ids
    ])

    return samples

def plot_data(ax : plt.Axes, data : np.ndarray):
    ax.scatter(data[:, 0], data[:, 1], alpha=0.5, s=1)