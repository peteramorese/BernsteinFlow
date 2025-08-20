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
                 n_var_components : int, 
                 max_var_degree : int,
                 n_factor_components : int,
                 max_factor_degree : int,
                 n_var_sub_components : int = 1,
                 factor_type = 'full',
                 n_total_components : int = 1):
        
        """
        Conditional sparse beta mixture model for modeling p(y | x)

        Args:
            dim : dimension of the support
            cond_dim : dimension of the conditioner variable
            n_var_components : number of beta mixture components for the y density
            max_var_degree : maximum degree of each y density mixture component (controls max sharpness)
            n_factor_components : number of conditional factor components (usually much lower than n_var_components for scalability)
            n_var_sub_components : number of sub-mixture components for each weighting function. Defaults to 1, where each conditional factor
                function is assigned to only a single beta component
            max_factor_degree : maximum degree of conditional factor components
            factor_type : {'full', 'sliding' (TODO), 'stationary' (TODO)} structural constraint on the conditional factor components
            n_total_components : number of total components, i.e. number of convex combinations of y-density mixture/x-conditional weight functions
        """
        
        super().__init__()

        self.dim = dim
        self.cond_dim = dim

        self.n_var_components = n_var_components
        self.n_var_sub_components = n_var_sub_components
        self.max_var_degree = float(max_var_degree)
        self.n_factor_components = n_factor_components
        self.max_factor_degree = max_factor_degree
        self.factor_type = factor_type # TODO

        self.n_total_components = n_total_components

        # Alpha/Beta params for each total component
        self.A_unconstrained = torch.nn.Parameter(torch.randn(self.n_var_components, self.n_var_sub_components, self.dim, self.n_total_components)) # Alpha values for density mixands
        self.B_unconstrained = torch.nn.Parameter(torch.randn(self.n_var_components, self.n_var_sub_components, self.dim, self.n_total_components)) # Beta values for density mixands
        self.var_sub_comp_weights = torch.nn.Parameter(torch.randn(self.n_var_components, self.n_var_sub_components, self.n_total_components)) # Fixed weights for each subcomponent mixture

        # Alpha/Beta params for each factor function
        self.Gamma_unconstrained = torch.nn.Parameter(torch.randn(self.n_factor_components, self.cond_dim, self.n_total_components)) # Alpha values for weight factors
        self.Delta_unconstrained = torch.nn.Parameter(torch.randn(self.n_factor_components, self.cond_dim, self.n_total_components)) # Beta values for weight factors
        self.factor_weights = torch.nn.Parameter(torch.randn(self.n_var_components, self.n_factor_components, self.n_total_components)) # Weights for each conditional factor

        self.total_comp_weights = torch.nn.Parameter(torch.randn(self.n_total_components))
    
    def forward(self, x : torch.Tensor, y : torch.Tensor):
        log_x = torch.log(x).unsqueeze(1)
        log_1mx = torch.log(1 - x).unsqueeze(1) 
        log_y = torch.log(y).unsqueeze(1)
        log_1my = torch.log(1 - y).unsqueeze(1) 

        A, B, Gamma, Delta, var_sub_comp_weights, factor_weights, total_comp_weights = self.get_constrained_parameters()

        # Calculate the var components
        log_var_component_per_dim = (
            (A - 1.0) * log_y +
            (B - 1.0) * log_1my +
            torch.special.gammaln(A) - 
            torch.special.gammaln(B) +
            torch.special.gammaln(A + B)
        )

        log_var_component_density = torch.sum(log_var_component_per_dim, dim=2)
        weighted_var_component_density = var_sub_comp_weights * torch.exp(log_var_component_density)
        var_density = torch.sum(weighted_var_component_density, dim=1)

        # Calculate the conditional weights
        log_max_factor_vals_per_dim = Gamma * torch.log(Gamma) + Delta * torch.log(Delta) - (Gamma + Delta) * torch.log(Gamma + Delta)
        log_max_factor_vals = torch.sum(log_max_factor_vals_per_dim, dim=2)

        log_factor_component_per_dim = (
            (Gamma - 1.0) * log_x +
            (Delta - 1.0) * log_1mx
        )

        log_factor_value = torch.sum(log_factor_component_per_dim, dim=2)

        # Normalize and weight the conditional factor value (divide by max value and multiply by weight)
        log_factor_value = log_factor_value - log_max_factor_vals + factor_weights

        # Compute the product of all the conditional factors
        log_cond_value = torch.sum(torch.log(1.0 - torch.exp(log_factor_value)), dim=2)


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
        
        #log_max_factor_vals = np.zeros_like(Gamma_bounded, device=Gamma_bounded.device, dtype=Gamma_bounded.dtype)
        #log_max_factor_vals = np.zeros(self.n_factor_components, self.n_total_components, device=Gamma_b.device, dtype=Gamma_b.dtype)

        #log_max_factor_vals_per_dim = Gamma_b * torch.log(Gamma_b) + Delta_b * torch.log(Delta_b) - (Gamma_b + Delta_b) * torch.log(Gamma_b + Delta_b)
        #log_max_factor_vals = torch.sum(log_max_factor_vals_per_dim, dim=2)
        #max_factor_vals = torch.exp(log_max_factor_vals)


#if __name__ == "__main__":
#    model = SparseBetaModel(2, 60, max_degree=40)