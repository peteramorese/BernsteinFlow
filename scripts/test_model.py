from bernstein_flow.DistributionTransform import GaussianDistTransform
from bernstein_flow.Model import BernsteinFlowModel, train

from .TestDataSets import sample_modal_gaussian, plot_data

import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import torch
from torch.utils.data import DataLoader, TensorDataset

def evaluate_density_on_grid(model, resolution=100, device=None):
    """
    Evaluate a trained model on a 2D grid over [0, 1]^2.

    Args:
        model: A trained PyTorch model that maps 2D inputs to scalar densities.
        resolution: Number of points along each axis (default: 100).
        device: Device to run the model on. If None, uses model's device.

    Returns:
        X, Y: Meshgrid coordinates (numpy arrays)
        Z: Density values on the grid (numpy array of shape [resolution, resolution])
    """
    if device is None:
        device = next(model.parameters()).device

    # Create 2D grid over [0, 1] x [0, 1]
    x = np.linspace(0, 1, resolution)
    y = np.linspace(0, 1, resolution)
    X, Y = np.meshgrid(x, y)

    # Flatten grid and convert to tensor
    grid_points = np.stack([X.ravel(), Y.ravel()], axis=-1)  # shape: (resolution^2, 2)
    grid_tensor = torch.tensor(grid_points, dtype=torch.float32, device=device)

    # Evaluate model
    with torch.no_grad():
        model.eval()
        densities = model(grid_tensor).cpu().numpy()  # shape: (resolution^2,)

    Z = densities.reshape(resolution, resolution)
    return X, Y, Z

def plot_density_surface(X, Y, Z):
    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')
    
    # Create surface plot
    surf = ax.plot_surface(X, Y, Z, cmap='viridis', linewidth=0, antialiased=True)
    # Add colorbar
    fig.colorbar(surf, ax=ax, shrink=0.5, aspect=20)
    return fig, ax

def plot_density(ax : Axes3D, X, Y, Z):
    ax.contourf(X, Y, Z, levels=50, cmap='viridis')


dim = 2

gdt = GaussianDistTransform(mean=dim*[0.0], variances=dim*[5.0])

#means = [[-1.5, -1.5], [1.5, 1.5]]
#covariances = [torch.eye(dim)*1.5, torch.eye(dim)*0.5]
#X_data = sample_modal_gaussian(5000, means=means, covariances=covariances, weights=[.3, .7])
means = [[0.0, 3.0]]
covariances = [torch.eye(dim)*1.5]
X_data = sample_modal_gaussian(5000, means=means, covariances=covariances, weights=[1.0])

fig, axes = plt.subplots(2, 2)

plot_data(axes[0, 0], X_data)

U_data = gdt.X_to_U(X_data)
#U_data = torch.tensor(
#    [[0.5, 0.5],
#    [0.6, 0.5],
#    [0.7, 0.5],
#    [0.5, 0.6],
#    [0.6, 0.6]])
#print("U data: ", U_data)

plot_data(axes[0, 1], U_data)
axes[0, 1].set_xlim((0, 1))
axes[0, 1].set_ylim((0, 1))

U_data_torch = torch.tensor(U_data, dtype=torch.float32)
dataset = TensorDataset(U_data_torch)
dataloader = DataLoader(dataset, batch_size=128, shuffle=True)

transformer_degrees = [10, 10]
conditioner_degrees = [5, 5]
model = BernsteinFlowModel(dim=dim, transformer_degrees=transformer_degrees, conditioner_degrees=conditioner_degrees)

optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)

# Train
train(model, dataloader, optimizer, epochs=100)

X, Y, Z = evaluate_density_on_grid(model, resolution=100)
plot_density(axes[1, 1], X, Y, Z)

# Show the transformers
fig2, axes = plt.subplots(dim, 2)
xi_vals = torch.linspace(0, 1, 100, requires_grad=False)
for i in range(dim):
    x_vals = 0.5 * torch.ones(100, dim, requires_grad=False)
    x_vals[:, i] = xi_vals
    tf_vals = model.transformer(x_vals, i)
    tf_deriv_vals = model.transformer_deriv(x_vals, i)
    axes[i, 0].plot(xi_vals.numpy(), tf_vals.detach().numpy())
    axes[i, 0].set_ylabel(f"TF {i}")
    axes[i, 1].plot(xi_vals.numpy(), tf_deriv_vals.detach().numpy())
    axes[i, 1].set_ylabel(f"dTF/dx {i}")

print("UNCONSTRAINED Model params:")
for i in range(dim):
    print(model.A[i])

print("Model params:")
for i in range(dim):
    print(model.get_constrained_parameters(i))

#fig3, ax = plot_density_surface(X, Y, Z)
#ax.set_zlim((0, 1))


plt.show()
