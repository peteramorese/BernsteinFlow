import torch
from .SOSModel import SOSModel

class BetaSOSModel(SOSModel):
    def __init__(self, dy : int, dx : int, n : int, m : int, min_alpha_beta : float = 1.00, max_alpha_beta : float = 50.0, **kwargs):
        # Two parameters for each basis function (alpha and beta) for each dimension
        super().__init__(dy, dx, n, m, 2 * dy, 2 * dx, **kwargs)

        self.min_alpha_beta = min_alpha_beta
        self.max_alpha_beta = max_alpha_beta

    def phi(self, x : torch.Tensor):
        # Make each parameter positive and unsqeeze to data shape
        alpha_beta = (self.max_alpha_beta - self.min_alpha_beta) * torch.nn.functional.sigmoid(self.phi_params) + self.min_alpha_beta
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
        alpha_beta = (self.max_alpha_beta - self.min_alpha_beta) * torch.nn.functional.sigmoid(self.psi_params) + self.min_alpha_beta
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

    def psi_gram(self):
        # Make each parameter positive and unsqeeze to data shape
        alpha_beta = (self.max_alpha_beta - self.min_alpha_beta) * torch.nn.functional.sigmoid(self.psi_params) + self.min_alpha_beta
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
    
    def get_phi_params(self):
        return (self.max_alpha_beta - self.min_alpha_beta) * torch.nn.functional.sigmoid(self.phi_params) + self.min_alpha_beta

    def get_psi_params(self):
        return (self.max_alpha_beta - self.min_alpha_beta) * torch.nn.functional.sigmoid(self.psi_params) + self.min_alpha_beta