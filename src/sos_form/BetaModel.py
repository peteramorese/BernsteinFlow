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
        alpha = alpha_beta[:, :self.dy].unsqueeze(0) # Shape (p, n*m-1, dy)
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
        log_psi = torch.sum(log_psi_per_dim, dim=2) # Shape (p, n*m-1)

        return torch.exp(log_psi)

    def psi_gram(self):
        # Make each parameter positive and unsqeeze to data shape
        alpha_beta = (self.max_alpha_beta - self.min_alpha_beta) * torch.nn.functional.sigmoid(self.psi_params) + self.min_alpha_beta
        alpha = alpha_beta[:, :self.dy] # Shape (n*m-1, dy)
        beta = alpha_beta[:, self.dy:]
        
        # Initialize the full gram matrix including the constant function
        # psi_0 = 1, psi_1, ..., psi_{n*m-1}
        gram_matrix = torch.zeros(self.n * self.m, self.n * self.m, dtype=alpha.dtype, device=alpha.device)
        
        # For each dimension, compute the inner products
        for d in range(self.dy):
            alpha_d = alpha[:, d]  # Shape: (n*m-1,)
            beta_d = beta[:, d]    # Shape: (n*m-1,)
            
            # Compute integrals of individual beta functions: ∫ psi_i(y) dy
            # For Beta(α, β), the integral over [0,1] is 1 (normalized)
            # So ∫ psi_i(y) dy = 1 for all i
            beta_integrals = torch.ones(self.n * self.m - 1, dtype=alpha.dtype, device=alpha.device)
            
            # Fill in the gram matrix:
            # G[0, 0] = ⟨1, 1⟩ = 1
            gram_matrix[0, 0] = 1.0
            
            # G[0, j] = ⟨1, psi_j⟩ = ∫ psi_j(y) dy = 1 for j > 0
            gram_matrix[0, 1:] = beta_integrals
            
            # G[i, 0] = ⟨psi_i, 1⟩ = ∫ psi_i(y) dy = 1 for i > 0  
            gram_matrix[1:, 0] = beta_integrals
            
            # G[i, j] = ⟨psi_i, psi_j⟩ for i, j > 0
            # The inner product of Beta(αᵢ, βᵢ) and Beta(αⱼ, βⱼ) is:
            # B(αᵢ + αⱼ - 1, βᵢ + βⱼ - 1) / (B(αᵢ, βᵢ) * B(αⱼ, βⱼ))
            alpha_i = alpha_d.unsqueeze(1)  # Shape: (n*m-1, 1)
            beta_i = beta_d.unsqueeze(1)    # Shape: (n*m-1, 1)
            alpha_j = alpha_d.unsqueeze(0)  # Shape: (1, n*m-1)
            beta_j = beta_d.unsqueeze(0)    # Shape: (1, n*m-1)
            
            # Compute the inner product for this dimension
            alpha_sum = alpha_i + alpha_j - 1
            beta_sum = beta_i + beta_j - 1
            
            # Compute log of the inner product to avoid overflow
            log_inner_d = (
                torch.special.gammaln(alpha_sum) + torch.special.gammaln(beta_sum) - torch.special.gammaln(alpha_sum + beta_sum)
                - torch.special.gammaln(alpha_i) - torch.special.gammaln(beta_i) - torch.special.gammaln(alpha_j) - torch.special.gammaln(beta_j)
                + torch.special.gammaln(alpha_i + beta_i) + torch.special.gammaln(alpha_j + beta_j)
            )
            
            # Add to the submatrix for i, j > 0
            gram_matrix[1:, 1:] += log_inner_d
        
        # Convert the submatrix from log space
        gram_matrix[1:, 1:] = torch.exp(gram_matrix[1:, 1:])
        
        return gram_matrix
    
    def get_phi_params(self):
        return (self.max_alpha_beta - self.min_alpha_beta) * torch.nn.functional.sigmoid(self.phi_params) + self.min_alpha_beta

    def get_psi_params(self):
        return (self.max_alpha_beta - self.min_alpha_beta) * torch.nn.functional.sigmoid(self.psi_params) + self.min_alpha_beta