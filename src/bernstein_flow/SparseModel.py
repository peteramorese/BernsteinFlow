import torch
from torch.utils.data import DataLoader, TensorDataset
import time
import sys
import numpy as np

from .Polynomial import Polynomial, Basis, decasteljau_composition
from .SparsePolynomial import SparseBernsteinPolynomial


class BetaMixtureModel(torch.nn.Module):
    def __init__(self, dim : int, n_components : int, max_degree : int):
        """
        Beta mixture model

        Args:
            dim : dimension of the support
        """
        super().__init__()

        self.dim = dim
        self.n_components = n_components
        self.max_degree = float(max_degree)

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


class ConditionalFactorModel(torch.nn.Module):
    def __init__(self, d : int, 
                 dc : int,
                 nv : int, 
                 nf : int,
                 ns : int,
                 max_var_degree : int,
                 max_factor_degree : int,
                 factor_type = 'full',
                 nt : int = 1):
        
        """
        Conditional beta mixture model for modeling p(y | x)

        Args:
            d : dimension of the support
            dc : dimension of the conditioner variable
            nv : number of beta mixture components for the y density
            nf : number of conditional factor components (usually MUCH smaller (<10) than nv for scalability)
            ns : number of sub-mixture components for each weighting function. Defaults to 1, where each conditional factor
                function is assigned to only a single beta component
            max_var_degree : maximum degree of each y density mixture component (controls max sharpness)
            max_factor_degree : maximum degree of conditional factor components
            factor_type : {'full', 'sliding' (TODO), 'stationary' (TODO)} structural constraint on the conditional factor components
            nt : number of total components, i.e. number of convex combinations of y-density mixture/x-conditional weight functions
        """
        
        super().__init__()

        self.d = d
        self.dc = dc

        self.nv = nv # Number of random variable (y) components
        self.ns = ns # Number of random variable sub components
        self.max_var_degree = float(max_var_degree)
        self.nf = nf
        self.max_factor_degree = max_factor_degree
        self.factor_type = factor_type # TODO

        self.nt = nt # Number of total mixands

        # Alpha/Beta params for each total component
        self.A_unconstrained = torch.nn.Parameter(torch.randn(self.nv, self.ns, self.d, self.nt)) # Alpha values for density mixands
        self.B_unconstrained = torch.nn.Parameter(torch.randn(self.nv, self.ns, self.d, self.nt)) # Beta values for density mixands
        self.var_sub_comp_weights = torch.nn.Parameter(torch.randn(self.nv, self.ns, self.nt)) # Fixed weights for each subcomponent mixture

        # Alpha/Beta params for each factor function
        self.Gamma_unconstrained = torch.nn.Parameter(torch.randn(self.nf, self.dc, self.nt)) # Alpha values for weight factors
        self.Delta_unconstrained = torch.nn.Parameter(torch.randn(self.nf, self.dc, self.nt)) # Beta values for weight factors
        self.factor_weights = torch.nn.Parameter(torch.randn(self.nv - 1, self.nf, self.nt)) # Weights for each conditional factor

        self.total_comp_weights = torch.nn.Parameter(torch.randn(self.nt))
    
    def forward(self, yx : torch.Tensor):
        """
        Inference of density given y and x

        Args:
            yx : torch Tensor of size (p, d + dc)
        """
        assert yx.shape[1] == self.d + self.dc
        y = yx[:, :self.d]
        x = yx[:, self.d:]
        #print("y bounds: ", torch.max(y), torch.min(y))

        log_x = torch.log(x)
        log_1mx = torch.log(1 - x)
        log_y = torch.log(y)
        log_1my = torch.log(1 - y)

        # Unsqueeze to the right shape (p, nv, ns, d, nt)
        log_y = log_y[:, None, None, :, None]
        log_1my = log_1my[:, None, None, :, None]
        #print("log y bounds: ", torch.max(log_y), torch.min(log_y))
        #print("log 1my bounds: ", torch.max(log_1my), torch.min(log_1my))

        # Unsqueeze to the right shape (p, nf, dc, nt)
        log_x = log_x[:, None, :, None]
        log_1mx = log_1mx[:, None, :, None]


        A, B, Gamma, Delta, var_sub_comp_weights, factor_weights, total_comp_weights = self.get_constrained_parameters()
        #print("A bounds: ", torch.max(A).item(), torch.min(A).item())

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
            (A - 1.0) * log_y
            + (B - 1.0) * log_1my
            - torch.special.gammaln(A)
            - torch.special.gammaln(B)
            + torch.special.gammaln(A + B)
        )
        #print("lvcpd: ", torch.max(log_var_component_per_dim), torch.min(log_var_component_per_dim))

        log_var_component_density = torch.sum(log_var_component_per_dim, dim=3) # Sum out d
        weighted_var_component_density = var_sub_comp_weights * torch.exp(log_var_component_density)
        var_density = torch.sum(weighted_var_component_density, dim=2) # Sum out ns

        # Calculate the conditional weights
        Gamma_term = torch.where(Gamma - 1.0 > 0, (Gamma - 1.0) * torch.log(Gamma - 1.0), torch.zeros_like(Gamma))
        Delta_term = torch.where(Delta - 1.0 > 0, (Delta - 1.0) * torch.log(Delta - 1.0), torch.zeros_like(Delta))
        Gamma_Delta_term = torch.where(Gamma + Delta - 2.0 > 0, (Gamma + Delta - 2.0) * torch.log(Gamma + Delta - 2.0), torch.zeros_like(Delta))

        log_max_factor_vals_per_dim = Gamma_term + Delta_term - Gamma_Delta_term

        log_max_factor_vals = torch.sum(log_max_factor_vals_per_dim, dim=2)

        # Calculate the evaluation of the conditional basis functions at x
        log_factor_component_per_dim = (
            (Gamma - 1.0) * log_x
            + (Delta - 1.0) * log_1mx
        )

        log_factor_value = torch.sum(log_factor_component_per_dim, dim=2)

        #print("log factor value: ", log_factor_value.shape)

        # Normalize and weight the conditional factor value (divide by max value and multiply by weight)
        log_factor_value -= log_max_factor_vals 
        log_factor_value = log_factor_value.unsqueeze(1) # Add nv dimension
        log_factor_value = log_factor_value + torch.log(factor_weights)

        # Compute the product of all the conditional factors
        cumulative_weights = torch.prod(1.0 - torch.exp(log_factor_value), dim=2)

        pad_shape = cumulative_weights[:, [0], :].shape
        separated_weights = -torch.diff(cumulative_weights, dim=1, prepend=torch.ones(pad_shape), append=torch.zeros(pad_shape))

        mixand_densities = torch.sum(separated_weights * var_density, dim=1)

        # Weight all of the mixands and combine
        density = torch.sum(total_comp_weights * mixand_densities, dim=1)

        return density

    def get_constrained_parameters(self):
        # Map Alpha, Beta, Gamma, Delta to be in (1, max_degree)
        #print("A uncst bounds: ", torch.min(self.A_unconstrained), torch.max(self.A_unconstrained))
        #print("A tanh bounds: ", torch.min(torch.tanh(self.A_unconstrained)), torch.max(torch.tanh(self.A_unconstrained)))
        #A_b = 1.0 + (self.max_var_degree) * torch.relu(torch.tanh(self.A_unconstrained))
        #B_b = 1.0 + (self.max_var_degree) * torch.relu(torch.tanh(self.B_unconstrained))
        #Gamma_b = 1.0 + (self.max_factor_degree) * torch.relu(torch.tanh(self.Gamma_unconstrained))
        #Delta_b = 1.0 + (self.max_factor_degree) * torch.relu(torch.tanh(self.Delta_unconstrained))
        A_b = 1.0 + (self.max_var_degree) * torch.nn.functional.sigmoid(self.A_unconstrained)
        B_b = 1.0 + (self.max_var_degree) * torch.nn.functional.sigmoid(self.B_unconstrained)
        Gamma_b = 1.0 + (self.max_factor_degree) * torch.nn.functional.sigmoid(self.Gamma_unconstrained)
        Delta_b = 1.0 + (self.max_factor_degree) * torch.nn.functional.sigmoid(self.Delta_unconstrained)

        norm_var_sub_comp_weights = torch.nn.functional.softmax(self.var_sub_comp_weights, dim=1)

        # Map each element to be positive, increasing over each var component, and in (0, 1)
        norm_ordered_factor_weights = torch.softmax(torch.cumsum(torch.nn.functional.softplus(self.factor_weights), dim=0), dim=0)

        norm_total_comp_weights = torch.nn.functional.softmax(self.total_comp_weights, dim=0)

        return A_b, B_b, Gamma_b, Delta_b, norm_var_sub_comp_weights, norm_ordered_factor_weights, norm_total_comp_weights
        

class ConditionalPowerFunctionModel(torch.nn.Module):
    def __init__(self, d : int, 
                 dc : int,
                 nv : int, 
                 max_var_exp : float,
                 max_pf_exp : float,
                 ns : int = 1,
                 nt : int = 1):
        
        """
        Conditional beta mixture model for modeling p(y | x)

        Args:
            d : dimension of the support
            dc : dimension of the conditioner variable
            nv : number of beta mixture components for the y density
            ns : number of sub-mixture components for each weighting function. Defaults to 1, where each conditional power function
                function is assigned to only a single beta component
            max_var_degree : maximum degree of each y density mixture component (controls max sharpness)
            max_pf_degree : maximum degree of conditional power function. Uses the reciprocal value for the lower bound
            nt : number of total components, i.e. number of convex combinations of y-density mixture/x-conditional weight functions
        """
        
        super().__init__()

        self.d = d
        self.dc = dc

        self.nv = nv # Number of random variable (y) components
        self.ns = ns # Number of random variable sub components
        self.max_var_exp = max_var_exp
        self.max_pf_exp = max_pf_exp
        self.min_pf_exp = 1.0 / max_pf_exp

        self.nt = nt # Number of total mixands

        # Alpha/Beta params for each total component
        self.A_unconstrained = torch.nn.Parameter(torch.randn(self.nv, self.ns, self.d, self.nt)) # Alpha values for density mixands
        self.B_unconstrained = torch.nn.Parameter(torch.randn(self.nv, self.ns, self.d, self.nt)) # Beta values for density mixands
        self.var_sub_comp_weights = torch.nn.Parameter(torch.randn(self.nv, self.ns, self.nt)) # Fixed weights for each subcomponent mixture

        self.power_fcn_exps_unconstrianed = torch.nn.Parameter(torch.randn(self.nv - 1, self.dc, self.nt)) # Power function exponent vectors

        self.total_comp_weights = torch.nn.Parameter(torch.randn(self.nt))
    
    def forward(self, yx : torch.Tensor):
        """
        Inference of y-density given x

        Args:
            yx : torch Tensor of size (p, d + dc)
        """
        assert yx.shape[1] == self.d + self.dc
        y = yx[:, :self.d]
        x = yx[:, self.d:]
        #print("y bounds: ", torch.max(y), torch.min(y))

        log_x = torch.log(x)
        log_y = torch.log(y)
        log_1my = torch.log(1 - y)

        # Unsqueeze to the right shape (p, nv, ns, d, nt)
        log_y = log_y[:, None, None, :, None]
        log_1my = log_1my[:, None, None, :, None]

        # Unsqueeze to the right shape (p, nv, dc, nt)
        log_x = log_x[:, None, :, None]


        A, B, var_sub_comp_weights, pf_exps, total_comp_weights = self.get_constrained_parameters()

        # Unsqueeze to the right shape
        A = A.unsqueeze(0)
        B = B.unsqueeze(0)
        var_sub_comp_weights = var_sub_comp_weights.unsqueeze(0)
        pf_exps = pf_exps.unsqueeze(0)
        total_comp_weights.unsqueeze(0)

        # Calculate the evaluation of var basis functions at y
        log_var_component_per_dim = (
            (A - 1.0) * log_y
            + (B - 1.0) * log_1my
            - torch.special.gammaln(A)
            - torch.special.gammaln(B)
            + torch.special.gammaln(A + B)
        )
        #print("lvcpd: ", torch.max(log_var_component_per_dim), torch.min(log_var_component_per_dim))

        log_var_component_density = torch.sum(log_var_component_per_dim, dim=3) # Sum out d (multiply all basis functions)
        weighted_var_component_density = var_sub_comp_weights * torch.exp(log_var_component_density)
        var_density = torch.sum(weighted_var_component_density, dim=2) # Sum out ns
        #print("var density min: ", torch.min(var_density).item())
        #print(torch.any(torch.isnan(var_density)))

        # Calculate the conditional weights
        #print("pf exps min: ", torch.min(pf_exps).item(), " max: ", torch.max(pf_exps).item())
        log_pf_vals = torch.sum(pf_exps * log_x, dim=2)
        pf_vals = torch.exp(log_pf_vals)
        #print(torch.any(torch.isnan(pf_vals)))
        #print("pf min: ", torch.min(pf_vals).item(), " max: ", torch.max(pf_vals).item())

        pad_shape = pf_vals[:, [0], :].shape
        pf_diff_vals = -torch.diff(pf_vals, dim=1, prepend=torch.ones(pad_shape), append=torch.zeros(pad_shape))
        #print("pf diff min: ", torch.min(pf_diff_vals).item())

        mixand_densities = torch.sum(pf_diff_vals * var_density, dim=1)

        # Weight all of the mixands and combine
        density = torch.sum(total_comp_weights * mixand_densities, dim=1)
        #print(torch.any(torch.isnan(density)))
        #print("Density min: ", torch.min(density).item())

        #input("...")
        return density

    def get_constrained_parameters(self):
        A_b = 1.0 + (self.max_var_exp) * torch.nn.functional.sigmoid(self.A_unconstrained)
        B_b = 1.0 + (self.max_var_exp) * torch.nn.functional.sigmoid(self.B_unconstrained)

        norm_var_sub_comp_weights = torch.nn.functional.softmax(self.var_sub_comp_weights, dim=1)

        # Map each element to be positive, increasing over each dimension, and in (0, 1)
        max_exp_diff = self.max_pf_exp - self.min_pf_exp

        norm_ordered_pf_exps = self.min_pf_exp + max_exp_diff * torch.softmax(torch.cumsum(torch.nn.functional.softplus(self.power_fcn_exps_unconstrianed), dim=0), dim=0)

        norm_total_comp_weights = torch.nn.functional.softmax(self.total_comp_weights, dim=0)

        return A_b, B_b, norm_var_sub_comp_weights, norm_ordered_pf_exps, norm_total_comp_weights
        

class ConditionalGMM(torch.nn.Module):
    def __init__(self, d : int, 
                 dc : int, 
                 nv : int, 
                 min_var_sigma : float, 
                 min_erf_sigma : float, 
                 ns : int=1, 
                 nt : int = 1):
        
        """
        Conditional Gaussian mixture model for modeling p(y | x) (x \in R^d, y \in R^dc)

        Args:
            d : dimension of the support
            dc : dimension of the conditioner variable
            nv : number of beta mixture components for the y density
            ns : number of sub-mixture components for each weighting function. Defaults to 1, where each conditional power function
                function is assigned to only a single beta component
            min_var_sigma : minimum scale of Gaussian component
            min_erf_sigma : minimum scale of conditional erf multipliers
            nt : number of total components, i.e. number of convex combinations of y-density mixture/x-conditional weight functions
        """
        
        super().__init__()

        self.d = d
        self.dc = dc

        self.nv = nv # Number of random variable (y) components
        self.ns = ns # Number of random variable sub components
        self.min_var_sigma = min_var_sigma
        self.min_erf_sigma = min_erf_sigma

        self.nt = nt # Number of total mixands

        self.var_mu = torch.nn.Parameter(torch.randn(self.nv, self.ns, self.d, self.nt))
        self.var_sigma = torch.nn.Parameter(torch.randn(self.nv, self.ns, self.d, self.nt))
        self.var_sub_comp_weights = torch.nn.Parameter(torch.randn(self.nv, self.ns, self.nt)) # Fixed weights for each subcomponent mixture

        self.erf_mu = torch.nn.Parameter(torch.randn(self.nv - 1, self.dc, self.nt))
        self.erf_sigma = torch.nn.Parameter(torch.randn(self.dc, self.nt))

        self.total_comp_weights = torch.nn.Parameter(torch.randn(self.nt))

    def forward(self, yx : torch.Tensor):
        assert yx.shape[1] == self.d + self.dc
        y = yx[:, :self.d]
        x = yx[:, self.d:]

        # Shape (p, nv, ns, d, nt)
        y = y[:, None, None, :, None]
        # Shape (p, nv, dc, nt)
        x = x[:, None, :, None]

        var_mu, var_sigma, var_sub_comp_weights, erf_mu, erf_sigma, total_comp_weights = self.get_constrained_parameters()

        var_mu = var_mu.unsqueeze(0)
        var_sigma = var_sigma.unsqueeze(0)
        var_sub_comp_weights = var_sub_comp_weights.unsqueeze(0)
        erf_mu = erf_mu.unsqueeze(0)
        erf_sigma = erf_sigma[None, None, :, :]
        total_comp_weights = total_comp_weights.unsqueeze(0)
    
        log_sqrt_2_pi = 0.5 * torch.log(2.0 * torch.tensor(torch.pi))
        log_norm_const = torch.sum(-(log_sqrt_2_pi + torch.log(var_sigma)), dim=3)
        log_exp_term = torch.sum(-(y - var_mu)**2 / (2.0 * var_sigma**2), dim=3)
        log_var_component_density = log_norm_const + log_exp_term
        weighted_var_component_density = var_sub_comp_weights * torch.exp(log_var_component_density)
        var_density = torch.sum(weighted_var_component_density, dim=2) # Shape (p, nv, nt)

        sqrt_2 = torch.sqrt(torch.tensor(2.0))
        erf_mult = torch.sum(0.5 * (1.0 + torch.erf((x - erf_mu) / (erf_sigma * sqrt_2))), dim=2)

        pad_shape = erf_mult[:, [0], :].shape
        erf_diff_vals = -torch.diff(erf_mult, dim=1, prepend=torch.ones(pad_shape), append=torch.zeros(pad_shape))

        mixand_densities = torch.sum(erf_diff_vals * var_density, dim=1)

        density = torch.sum(total_comp_weights * mixand_densities, dim=1)

        return density

    
    def get_constrained_parameters(self):
        pos_var_sigma = torch.nn.functional.softplus(self.var_sigma)
        norm_var_sub_comp_weights = torch.nn.functional.softmax(self.var_sub_comp_weights, dim=1)

        first_erf_mean_vecs = self.erf_mu[0:1, :, :]
        erf_mean_increments = torch.nn.functional.softplus(self.erf_mu[1:, :, :])
        ordered_erf_mu = torch.cumsum(torch.cat((first_erf_mean_vecs, erf_mean_increments), dim=0), dim=0)

        pos_erf_sigma = torch.nn.functional.softplus(self.erf_sigma)

        norm_total_comp_weights = torch.nn.functional.softmax(self.total_comp_weights, dim=0)
        
        return self.var_mu, pos_var_sigma, norm_var_sub_comp_weights, ordered_erf_mu, pos_erf_sigma, norm_total_comp_weights
        
def optimize(model, data_loader : DataLoader, optimizer, epochs=100, log_buffer_size = 20):
    def nll_loss(model, data):
        density = model(data)
        log_density = torch.log(density + 1e-10)
        loss = -log_density.mean()
        return loss

    def train_step(data):
        model.train()
        optimizer.zero_grad()
        loss = nll_loss(model, data)
        loss.backward()
        optimizer.step()
        return loss.item()

    stdout_buffer = []

    for epoch in range(epochs):
        start_time = time.time()
        total_loss = 0.0
        for x_batch in data_loader:
            x_batch = x_batch[0].to(next(model.parameters()).device)
            loss = train_step(x_batch)
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
    

#if __name__ == "__main__":
#    model = SparseBetaModel(2, 60, max_degree=40)