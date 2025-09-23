import torch
from torch.utils.data import DataLoader, TensorDataset
import time
import sys
import numpy as np


class SOSModel(torch.nn.Module):
    def __init__(self, dy : int, dx : int, n : int, m : int, phi_param_dim : int, psi_param_dim : int, gamma : float = 1.1, eta : float = 0.25):
        """
        SOS form conditional density model for p(y | x)
        Args:
            dy : dimension of the support
            dx : dimension of the conditioner variable
            n : number of x-basis functions (excluding 1)
            m : number of duplicates of the x-basis functions (making n * m y-basis functions)
            phi_param_dim : dimension of the phi (x-basis) parameters
            psi_param_dim : dimension of the psi (y-basis) parameters
            gamma : augmented lagrangian quadratic scaling
            eta : augmented lagrangian linear residual threshold
        """

        super().__init__()

        self.dy = dy
        self.dx = dx
        self.n = n + 1
        self.m = m
        self.phi_param_dim = phi_param_dim
        self.psi_param_dim = psi_param_dim
        self.gamma = gamma
        self.eta = eta

        # Initialize with smaller values to prevent explosion
        self.phi_params = torch.nn.Parameter(0.1 * torch.randn(self.n - 1, phi_param_dim))
        self.psi_params = torch.nn.Parameter(0.1 * torch.randn(self.n*self.m, psi_param_dim)) 

        self.L_a = torch.nn.Parameter(0.1 * torch.randn(self.n*self.m, self.n*self.m))
        self.L_b = torch.nn.Parameter(0.1 * torch.randn(self.n*self.m, self.n*self.m))

        # Augmented Lagrangian multipliers
        self.register_buffer("lagr_mult", torch.zeros(self.m, self.m)) # Linear penalty multipliers for each equality block
        self.sigma = 1.0 # Quadratic penalty multiplier
        self.v = 1.0 # Initial value of the augmented lagrangian linear residual

    def phi(self, x : torch.Tensor):
        """
        Evaluate the phi basis function vector at x using self.phi_params. Must return a tensor of size (p, n)
        """
        raise NotImplementedError()

    def psi(self, y : torch.Tensor):
        """
        Evaluate the psi basis function vector at y using self.phi_params. Must return a tensor of size (p, (n+1)*m)
        """
        raise NotImplementedError()

    def psi_inner_product_mat(self):
        """
        Compute the inner product matrix of the psi basis functions. Must return a tensor of size ((n+1)*m, (n+1)*m)
        """
        raise NotImplementedError()

    def __get_phi(self, x : torch.Tensor):
        phi_vec = self.phi(x)
        phi_vec = torch.cat([torch.ones(x.shape[0], 1, dtype=x.dtype, device=x.device), phi_vec], dim=1)
        return phi_vec

    def forward(self, yx : torch.Tensor):
        """
        Inference of density given y and x

        Args:
            yx : torch Tensor of size (p, dy + dx)
        """

        assert yx.shape[1] == self.dy + self.dx
        y = yx[:, :self.dy]
        x = yx[:, self.dy:]
        phi_vec = self.__get_phi(x) # (p, n)
        #print("phi_vec: ", phi_vec)

        phi_vec = phi_vec.repeat_interleave(self.m, dim=1)  # (p, n*m)
        #print("phi_vec repeated: ", phi_vec)
        #input("...")
        psi_vec = self.psi(y) # (p, n*m)
        basis_vals = phi_vec * psi_vec  # Shape: (p, n*m)

        A_mat = self.get_A_mat()  # Shape: ((n)*m, (n)*m)

        print("A mat:\n", A_mat)
        print("Phi: ", phi_vec[0, :])
        print("Psi: ", psi_vec[0, :])
        print("basis vals: ", basis_vals)
        # Per-sample quadratic form: for each sample i, basis_vals[i]^T A basis_vals[i]
        density = torch.einsum("pi,ij,pj->p", basis_vals, A_mat, basis_vals)
        
        return density 
        
    def get_A_mat(self):
        return self.L_a @ self.L_a.T
    
    def get_residual_mat(self):
        psi_mat = self.psi_inner_product_mat()
        Gamma = psi_mat * self.get_A_mat() # Hadamard product between psi inner product mat and A

        # Reshape Gamma to group blocks and sum over each block
        # Gamma shape: ((n+1)*m, (n+1)*m) -> (m, n, m, n) -> (m, m)
        Gamma_block_view = Gamma.view(self.m, self.n, self.m, self.n)

        # Ensure all block elements sum to zero except for the first block which sums to 1
        sum_gamma_violation = torch.sum(Gamma_block_view, dim=(1, 3))  # Sum over the block dimensions
        sum_gamma_violation[0, 0] -= 1.0
        return sum_gamma_violation

    def aug_lagrangian_loss(self):
        v_mat = self.get_residual_mat()

        linear_penalty = torch.sum(self.lagr_mult * v_mat)

        quadratic_penalty = 0.5 *self.sigma * torch.sum(torch.square(v_mat))

        return -linear_penalty + quadratic_penalty
    
    def loss(self, yx : torch.Tensor):
        density = self(yx)
        log_density = torch.log(density + 1e-10)
        nll_loss = -log_density.mean()

        aug_lagrangian_loss = self.aug_lagrangian_loss()
        print("nll loss: ", nll_loss.item(),"aug lagrangian loss: ", aug_lagrangian_loss.item())
        loss = nll_loss + aug_lagrangian_loss
        #input("...")
        return loss

    def update_lagrangians(self):
        print("Updating lagrangians...")
        with torch.no_grad():
            residuals = self.get_residual_mat()

            # Lagrange update iteration
            v = torch.sum(residuals**2)
            print("v: ", v)
            if v  < self.eta * self.v:
                self.lagr_mult -= self.sigma * residuals
            else:
                self.sigma *= self.gamma

            print("sigma: ", self.sigma)

            self.v = v
        input("...")

class BetaSOSModel(SOSModel):
    def __init__(self, dy : int, dx : int, n : int, m : int, gamma : float = 1.1, eta : float = 0.25):
        # Two parameters for each basis function (alpha and beta) for each dimension
        super().__init__(dy, dx, n, m, 2 * dy, 2 * dx, gamma, eta)
    
    def phi(self, x : torch.Tensor):
        # Make each parameter positive and unsqeeze to data shape
        alpha_beta = torch.nn.functional.softplus(self.phi_params)
        alpha = alpha_beta[:, :self.dx].unsqueeze(0) # Shape (p, n, dx)
        beta = alpha_beta[:, self.dx:].unsqueeze(0)

        log_x = torch.log(x)
        log_1mx = torch.log(1 - x)
        log_x = log_x[:, None, :]
        log_1mx = log_1mx[:, None, :]

        log_phi_per_dim = (
            (alpha - 1.0) * log_x
            + (beta - 1.0) * log_1mx
            - torch.special.gammaln(alpha)
            - torch.special.gammaln(beta)
            + torch.special.gammaln(alpha + beta)
        )

        # Sum in log space over the dimension
        log_phi = torch.sum(log_phi_per_dim, dim=2) # Shape (p, n)

        return torch.exp(log_phi)

    def psi(self, y : torch.Tensor):
        # Make each parameter positive and unsqeeze to data shape
        alpha_beta = torch.nn.functional.softplus(self.psi_params)
        alpha = alpha_beta[:, :self.dy].unsqueeze(0) # Shape (p, (n+1)*m, dy)
        beta = alpha_beta[:, self.dy:].unsqueeze(0)

        log_y = torch.log(y)
        log_1my = torch.log(1 - y)
        log_y = log_y[:, None, :]  # Shape (p, 1, dy)
        log_1my = log_1my[:, None, :]  # Shape (p, 1, dy)

        log_psi_per_dim = (
            (alpha - 1.0) * log_y
            + (beta - 1.0) * log_1my
            - torch.special.gammaln(alpha)
            - torch.special.gammaln(beta)
            + torch.special.gammaln(alpha + beta)
        )

        # Sum in log space over the dimension
        log_psi = torch.sum(log_psi_per_dim, dim=2) # Shape (p, n*m)

        return torch.exp(log_psi)

    def psi_inner_product_mat(self):
        # Make each parameter positive and unsqeeze to data shape
        alpha_beta = torch.nn.functional.softplus(self.psi_params)
        alpha = alpha_beta[:, :self.dy] # Shape ((n+1)*m, dy)
        beta = alpha_beta[:, self.dy:]
        
        # For each dimension, compute the inner product between all pairs of beta distributions
        # The inner product of Beta(α₁, β₁) and Beta(α₂, β₂) is:
        # B(α₁ + α₂ - 1, β₁ + β₂ - 1) / (B(α₁, β₁) * B(α₂, β₂))
        # where B(α, β) = Γ(α) * Γ(β) / Γ(α + β)
        
        log_inner_products = torch.zeros(self.n * self.m, self.n * self.m, dtype=alpha.dtype, device=alpha.device)
        
        for d in range(self.dy):
            # Get alpha and beta for dimension d
            alpha_d = alpha[:, d]  # Shape: ((n+1)*m,)
            beta_d = beta[:, d]    # Shape: ((n+1)*m,)
            
            # Compute inner products for this dimension
            # For each pair (i, j), compute the inner product of psi_i^d and psi_j^d
            alpha_i = alpha_d.unsqueeze(1)  # Shape: ((n+1)*m, 1)
            beta_i = beta_d.unsqueeze(1)    # Shape: ((n+1)*m, 1)
            alpha_j = alpha_d.unsqueeze(0)  # Shape: (1, (n+1)*m)
            beta_j = beta_d.unsqueeze(0)    # Shape: (1, (n+1)*m)
            
            # Compute the inner product for this dimension
            # B(αᵢ + αⱼ - 1, βᵢ + βⱼ - 1) / (B(αᵢ, βᵢ) * B(αⱼ, βⱼ))
            alpha_sum = alpha_i + alpha_j - 1
            beta_sum = beta_i + beta_j - 1
            
            # Compute log of the inner product to avoid overflow
            log_inner_d = (
                torch.special.gammaln(alpha_sum) + torch.special.gammaln(beta_sum) - torch.special.gammaln(alpha_sum + beta_sum)
                - torch.special.gammaln(alpha_i) - torch.special.gammaln(beta_i) - torch.special.gammaln(alpha_j) - torch.special.gammaln(beta_j)
                + torch.special.gammaln(alpha_i + beta_i) + torch.special.gammaln(alpha_j + beta_j)
            )
            
            # Convert back from log space and multiply with existing inner products
            log_inner_products += log_inner_d
        
        return torch.exp(log_inner_products)

def optimize(model : SOSModel, data_loader : DataLoader, optimizer, epochs=100, lagrangian_update_interval=10, log_buffer_size = 20):
    def train_step(data):
        model.train()
        optimizer.zero_grad()
        loss = model.loss(data)
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

        if epoch % lagrangian_update_interval == 0:
            model.update_lagrangians()
        
        #line = f"Epoch {epoch+1}: Avg Loss = {avg_loss:.6f}, time: {time.time() - start_time:.3f}"
        #stdout_buffer.append(line)
        #if len(stdout_buffer) <= log_buffer_size:
        #    print(line)
        #else:
        #    stdout_buffer.pop(0)
        #    sys.stdout.write("\033[F" * len(stdout_buffer))
        #    for l in stdout_buffer:
        #        sys.stdout.write("\033[K")
        #        print(l)