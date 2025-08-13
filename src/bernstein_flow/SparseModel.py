import torch
from torch.utils.data import DataLoader, TensorDataset
import time
import sys

from .Polynomial import Polynomial, Basis, decasteljau_composition


class SparseBetaModel(torch.nn.Module):
    def __init__(self, dim : int, n_components : int, max_degree : int, device = None, dtype = torch.float32):
        """
        Create a Sparse BFM simple density estimation model

        Args:
            dim : dimension of the support
        """
        super().__init__()

        self.dim = dim
        self.n_components = n_components
        self.max_degree = float(max_degree)
        self.device = device
        self.dtype = dtype

        n_dense_components = (max_degree + 1)**dim
        sparsity = 1.0 - n_components / n_dense_components
        if sparsity > 0.0:
            print("Creating model with sparsity: ", 100 * sparsity, "%")
        else:
            print("Warning: model has no sparsity")
            n_components = n_dense_components
        

        # Alpha/Beta params
        self.A_unconstrained = torch.nn.Parameter(torch.rand(self.n_components, self.dim, device=device, dtype=dtype))
        self.B_unconstrained = torch.nn.Parameter(torch.rand(self.n_components, self.dim, device=device, dtype=dtype))
        self.weights_unconstrained = torch.nn.Parameter(torch.rand(self.n_components, device=self.device, dtype=dtype))
    
    
#    def forward(self, x : torch.Tensor):
#        # Evaluate the logarithm of each component, then exponentiate after
#
#        log_x = torch.log(x)
#        log_1mx = torch.log(1 - x)
#
#        # Get constrained parameters
#        A_bnd, B_bnd = self.get_constrained_parameters()
#
#        log_component_density_factors = (A_bnd - 1.0) * log_x + (B_bnd - 1) * log_1mx - torch.special.gammaln(A_bnd) - torch.special.gammaln(B_bnd) - torch.special.gammaln(A_bnd + B_bnd)
#        log_component_density = torch.sum(log_component_density_factors, dim = 1)
#        component_density = torch.exp(log_component_density)
#        return torch.sum(component_density)

    def forward(self, x: torch.Tensor):
        # log(x) and log(1-x) -> shape (p, 1, d)
        log_x = torch.log(x).unsqueeze(1)       # (p, 1, d)
        log_1mx = torch.log(1 - x).unsqueeze(1) # (p, 1, d)

        # Get constrained parameters -> shape (n, d)
        A_bnd, B_bnd, norm_weights = self.get_constrained_parameters()
        A_bnd = A_bnd.unsqueeze(0)  # (1, n, d)
        B_bnd = B_bnd.unsqueeze(0)  # (1, n, d)

        # Compute log density per dimension
        log_component_density_factors = (
            (A_bnd - 1.0) * log_x +
            (B_bnd - 1.0) * log_1mx -
            torch.special.gammaln(A_bnd) -
            torch.special.gammaln(B_bnd) +
            torch.special.gammaln(A_bnd + B_bnd)
        )  # shape (p, n, d)

        # Sum over dimensions -> shape (p, n)
        log_component_density = torch.sum(log_component_density_factors, dim=2)

        # Exponentiate to get densities -> shape (p, n)
        weighted_component_density = norm_weights * torch.exp(log_component_density)

        # Here: decide if you want to return per-component densities
        # or sum over components (mixture)
        return torch.sum(weighted_component_density, dim=1)  # shape (p,)

    def get_constrained_parameters(self):
        # Map to be in (0, max_degree)
        A_bounded, B_bounded = self.max_degree * torch.nn.functional.sigmoid(self.A_unconstrained), self.max_degree * torch.nn.functional.sigmoid(self.B_unconstrained)
        normalized_weights = torch.nn.functional.softmax(self.weights_unconstrained, dim=0)
        return A_bounded, B_bounded, normalized_weights

    def nll_loss(self, data):
        density = self(data)
        log_density = torch.log(density + 1e-10)
        loss = -log_density.mean()
        return loss

    def decasteljau_torch(self, coeffs : torch.Tensor, x : torch.Tensor):
        if x.ndim != 2:
            raise ValueError(f"Input tensor x must be 2-dimensional (batch_size, d), but got {x.ndim} dimensions.")

        batch_size, d = x.shape

        assert coeffs.ndim == d, "x vector dimension does not match dimension of p"

        degrees = [s - 1 for s in coeffs.shape]

        expand_shape = (batch_size,) + coeffs.shape
        current_coeffs = coeffs.expand(*expand_shape) 

        for i in range(d):
            t = x[:, i]
            view_shape = [batch_size] + [1] * (d - i)
            t = t.reshape(*view_shape)

            degree = degrees[i]

            for _ in range(degree):
                current_coeffs = (
                    (1 - t) * current_coeffs[:, :-1, ...] +
                    t * current_coeffs[:, 1:, ...]
                )
            
            if i < d - 1:
                current_coeffs = current_coeffs.squeeze(dim=1)
        return current_coeffs.squeeze()
    

def train_step(model, x_data, optimizer):

    model.train()
    optimizer.zero_grad()
    
    loss = model.nll_loss(x_data)
    loss.backward()
    optimizer.step()
    return loss.item()

def optimize(model, data_loader : DataLoader, optimizer, epochs=100, log_buffer_size = 20):

    stdout_buffer = []

    for epoch in range(epochs):
        start_time = time.time()
        total_loss = 0.0
        for x_batch in data_loader:
            x_batch = x_batch[0].to(next(model.parameters()).device)
            loss = train_step(model, x_batch, optimizer)
            total_loss += loss
        avg_loss = total_loss / len(data_loader)
        
        line = f"Epoch {epoch+1}: Avg Loss = {avg_loss:.6f}, time: {time.time() - start_time:.3f}"
        stdout_buffer.append(line)
        if len(stdout_buffer) <= log_buffer_size:
            print(line)
        else:
            stdout_buffer.pop(0)
            sys.stdout.write("\033[F" * len(stdout_buffer))
            for l in stdout_buffer:
                sys.stdout.write("\033[K")
                print(l)
    