import torch
from torch.utils.data import DataLoader, TensorDataset
import time
import sys
import numpy as np

from .Polynomial import Polynomial, Basis, decasteljau_composition
from .SparsePolynomial import SparseBernsteinPolynomial


class SparseBetaModel(torch.nn.Module):
    def __init__(self, dim : int, n_components : int, max_degree : int):
        """
        Create a Sparse BFM simple density estimation model

        Args:
            dim : dimension of the support
        """
        super().__init__()

        self.dim = dim
        self.n_components = n_components
        self.max_degree = float(max_degree)

        n_dense_components = (max_degree + 1)**dim
        sparsity = 1.0 - n_components / n_dense_components
        if sparsity > 0.0:
            print("Creating model with sparsity: ", 100 * sparsity, "%")
        else:
            print("Warning: model has no sparsity")
            n_components = n_dense_components
        

        # Alpha/Beta params
        self.A_unconstrained = torch.nn.Parameter(torch.rand(self.n_components, self.dim))
        self.B_unconstrained = torch.nn.Parameter(torch.rand(self.n_components, self.dim))
        self.weights_unconstrained = torch.nn.Parameter(torch.rand(self.n_components))
    
    
    def forward(self, x: torch.Tensor):
        # log(x) and log(1-x)
        log_x = torch.log(x).unsqueeze(1)
        log_1mx = torch.log(1 - x).unsqueeze(1) 

        # Get constrained parameters
        A_bnd, B_bnd, norm_weights = self.get_constrained_parameters()
        A_bnd = A_bnd.unsqueeze(0) 
        B_bnd = B_bnd.unsqueeze(0) 

        # Compute log density per dimension
        log_component_density_factors = (
            (A_bnd - 1.0) * log_x +
            (B_bnd - 1.0) * log_1mx -
            torch.special.gammaln(A_bnd) -
            torch.special.gammaln(B_bnd) +
            torch.special.gammaln(A_bnd + B_bnd)
        )  

        log_component_density = torch.sum(log_component_density_factors, dim=2)

        weighted_component_density = norm_weights * torch.exp(log_component_density)

        return torch.sum(weighted_component_density, dim=1)  # shape (p,)

    def get_constrained_parameters(self):
        # Map to be in (1, max_degree)
        A_bounded, B_bounded = 1.0 + (self.max_degree + 1.0) * torch.nn.functional.sigmoid(self.A_unconstrained), self.max_degree * torch.nn.functional.sigmoid(self.B_unconstrained)
        normalized_weights = torch.nn.functional.softmax(self.weights_unconstrained, dim=0)
        return A_bounded, B_bounded, normalized_weights

    def nll_loss(self, data):
        density = self(data)
        log_density = torch.log(density + 1e-10)
        loss = -log_density.mean()
        return loss

    def create_sparse_bern_poly(self):
        """
        Create the closest sparse Bernstein polynomial with the same number of coefficients. Each term is created by
        rounding the beta distribution components to the nearset integer before converting to Bernstein form
        """
        with torch.no_grad():
            A_bnd, B_bnd, norm_weights = self.get_constrained_parameters()
            A_round, B_round = torch.round(A_bnd).to(dtype=torch.long), torch.round(B_bnd).to(dtype=torch.long)
            idx = A_round - 1.0 
            deg = B_round - 1.0 + idx
            coeffs = norm_weights * torch.prod(deg + 1.0, dim=1)
            return SparseBernsteinPolynomial(coeffs.numpy(), idx.numpy(), deg.numpy())

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
    

class ConditionalSparseBetaModel(torch.nn.Module):
    def __init__(self, dim : int, 
                 cond_dim : int,
                 nv : int, 
                 max_var_degree : int,
                 nf : int,
                 max_factor_degree : int,
                 ns : int = 1,
                 factor_type = 'full',
                 nt : int = 1):
        
        """
        Conditional sparse beta mixture model for modeling p(y | x)

        Args:
            dim : dimension of the support
            cond_dim : dimension of the conditioner variable
            nv : number of beta mixture components for the y density
            max_var_degree : maximum degree of each y density mixture component (controls max sharpness)
            nf : number of conditional factor components (usually much lower than nv for scalability)
            ns : number of sub-mixture components for each weighting function. Defaults to 1, where each conditional factor
                function is assigned to only a single beta component
            max_factor_degree : maximum degree of conditional factor components
            factor_type : {'full', 'sliding' (TODO), 'stationary' (TODO)} structural constraint on the conditional factor components
            nt : number of total components, i.e. number of convex combinations of y-density mixture/x-conditional weight functions
        """
        
        super().__init__()

        self.dim = dim
        self.cond_dim = cond_dim

        self.nv = nv # Number of random variable (y) components
        self.ns = ns # Number of random variable sub components
        self.max_var_degree = float(max_var_degree)
        self.nf = nf
        self.max_factor_degree = max_factor_degree
        self.factor_type = factor_type # TODO

        self.nt = nt # Number of total mixands

        # Alpha/Beta params for each total component
        self.A_unconstrained = torch.nn.Parameter(torch.randn(self.nv, self.ns, self.dim, self.nt)) # Alpha values for density mixands
        self.B_unconstrained = torch.nn.Parameter(torch.randn(self.nv, self.ns, self.dim, self.nt)) # Beta values for density mixands
        self.var_sub_comp_weights = torch.nn.Parameter(torch.randn(self.nv, self.ns, self.nt)) # Fixed weights for each subcomponent mixture

        # Alpha/Beta params for each factor function
        self.Gamma_unconstrained = torch.nn.Parameter(torch.randn(self.nf, self.cond_dim, self.nt)) # Alpha values for weight factors
        self.Delta_unconstrained = torch.nn.Parameter(torch.randn(self.nf, self.cond_dim, self.nt)) # Beta values for weight factors
        self.factor_weights = torch.nn.Parameter(torch.randn(self.nv, self.nf, self.nt)) # Weights for each conditional factor

        self.total_comp_weights = torch.nn.Parameter(torch.randn(self.nt))
    
    def forward(self, x : torch.Tensor, y : torch.Tensor):
        log_x = torch.log(x)
        log_1mx = torch.log(1 - x)
        log_y = torch.log(y)
        log_1my = torch.log(1 - y)

        # Unsqueeze to the right shape (p, nv, ns, d, nt)
        log_y = log_y[:, None, None, :, None]
        log_1my = log_1my[:, None, None, :, None]

        # Unsqueeze to the right shape (p, nf, dc, nt)
        log_x = log_x[:, None, :, None]
        log_1mx = log_1mx[:, None, :, None]


        A, B, Gamma, Delta, var_sub_comp_weights, factor_weights, total_comp_weights = self.get_constrained_parameters()

        # Unsqueeze to the right shape
        A = A.unsqueeze(0)
        B = B.unsqueeze(0)
        Gamma = Gamma.unsqueeze(0)
        Delta = Delta.unsqueeze(0)
        var_sub_comp_weights = var_sub_comp_weights.unsqueeze(0)
        factor_weights = factor_weights.unsqueeze(0)
        total_comp_weights.unsqueeze(0)

        # Calculate the evaluation of var basis functions at y
        log_var_component_per_dim = (
            (A - 1.0) * log_y +
            (B - 1.0) * log_1my +
            torch.special.gammaln(A) - 
            torch.special.gammaln(B) +
            torch.special.gammaln(A + B)
        )

        log_var_component_density = torch.sum(log_var_component_per_dim, dim=3) # Sum out d
        weighted_var_component_density = var_sub_comp_weights * torch.exp(log_var_component_density)
        var_density = torch.sum(weighted_var_component_density, dim=2) # Sum out ns

        # Calculate the conditional weights
        log_max_factor_vals_per_dim = Gamma * torch.log(Gamma) + Delta * torch.log(Delta) - (Gamma + Delta) * torch.log(Gamma + Delta)
        log_max_factor_vals = torch.sum(log_max_factor_vals_per_dim, dim=2)

        # Calculate the evaluation of the conditional basis functions at x
        log_factor_component_per_dim = (
            (Gamma - 1.0) * log_x +
            (Delta - 1.0) * log_1mx
        )

        log_factor_value = torch.sum(log_factor_component_per_dim, dim=2)

        # Normalize and weight the conditional factor value (divide by max value and multiply by weight)
        log_factor_value = log_factor_value - log_max_factor_vals 
        log_factor_value = log_factor_value.unsqueeze(1) # Add nv dimension
        log_factor_value += factor_weights

        # Compute the product of all the conditional factors
        cond_weight = torch.prod(1.0 - torch.exp(log_factor_value), dim=2)

        mixand_densities = torch.sum(cond_weight * var_density, dim=1)

        # Weight all of the mixands and combine
        density = torch.sum(total_comp_weights * mixand_densities, dim=1)

        return density


    def get_constrained_parameters(self):
        # Map Alpha, Beta, Gamma, Delta to be in (1, max_degree)
        A_b = 1.0 + (self.max_var_degree) * torch.nn.functional.sigmoid(self.A_unconstrained)
        B_b = 1.0 + (self.max_var_degree) * torch.nn.functional.sigmoid(self.B_unconstrained)
        Gamma_b = 1.0 + (self.max_factor_degree) * torch.nn.functional.sigmoid(self.Gamma_unconstrained)
        Delta_b = 1.0 + (self.max_factor_degree) * torch.nn.functional.sigmoid(self.Delta_unconstrained)

        norm_var_sub_comp_weights = torch.nn.functional.softmax(self.var_sub_comp_weights, dim=1)

        # Map each element to be positive, increasing over each var component, and in (0, 1)
        norm_ordered_factor_weights = torch.softmax(torch.cumsum(torch.nn.functional.softplus(self.factor_weights), dim=0), dim=0)

        norm_total_comp_weights = torch.nn.functional.softmax(self.total_comp_weights, dim=0)

        return A_b, B_b, Gamma_b, Delta_b, norm_var_sub_comp_weights, norm_ordered_factor_weights, norm_total_comp_weights
        

#if __name__ == "__main__":
#    model = SparseBetaModel(2, 60, max_degree=40)