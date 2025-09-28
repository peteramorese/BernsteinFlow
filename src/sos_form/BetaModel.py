import torch
from .SOSModel import SOSModel

class BetaSOSModel(SOSModel):
    def __init__(self, dy : int, dx : int, n : int, min_alpha_beta : float = 1.00, max_alpha_beta : float = 50.0, **kwargs):
        # Two parameters for each basis function (alpha and beta) for each dimension
        super().__init__(dy, dx, n, 2 * dy, 2 * dx, **kwargs)

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
        alpha = alpha_beta[:, :self.dy].unsqueeze(0) # Shape (p, n, dy)
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
        log_psi = torch.sum(log_psi_per_dim, dim=2) # Shape (p, n)

        return torch.exp(log_psi)

    def gram_tensor(self):
        # Make each parameter positive
        phi_alpha_beta = (self.max_alpha_beta - self.min_alpha_beta) * torch.nn.functional.sigmoid(self.phi_params) + self.min_alpha_beta
        psi_alpha_beta = (self.max_alpha_beta - self.min_alpha_beta) * torch.nn.functional.sigmoid(self.psi_params) + self.min_alpha_beta
        
        phi_alpha = phi_alpha_beta[:, :self.dx]  # Shape (n, dx)
        phi_beta = phi_alpha_beta[:, self.dx:]   # Shape (n, dx)
        psi_alpha = psi_alpha_beta[:, :self.dy]  # Shape (n, dy)
        psi_beta = psi_alpha_beta[:, self.dy:]   # Shape (n, dy)
        
        # Initialize the 4D gram tensor: (n, n, n, n)
        log_gram_tensor = torch.zeros(self.n, self.n, self.n, self.n, dtype=phi_alpha.dtype, device=phi_alpha.device)
        
        # For each dimension, compute the inner products
        for d in range(self.dy):
            phi_alpha_d = phi_alpha[:, d]  # Shape: (n,)
            phi_beta_d = phi_beta[:, d]    # Shape: (n,)
            psi_alpha_d = psi_alpha[:, d]  # Shape: (n,)
            psi_beta_d = psi_beta[:, d]    # Shape: (n,)
            
            # Compute the 4D tensor for this dimension
            # Each element is ∫ φᵢ(y)φⱼ(y)ψₖ(y)ψₗ(y) dy
            # This is the integral of the product of four beta PDFs
            
            # Create all combinations of indices
            i_idx = torch.arange(self.n, device=phi_alpha.device)
            j_idx = torch.arange(self.n, device=phi_alpha.device)
            k_idx = torch.arange(self.n, device=phi_alpha.device)
            l_idx = torch.arange(self.n, device=phi_alpha.device)
            
            # Broadcast to 4D tensors
            phi_alpha_i = phi_alpha_d[i_idx, None, None, None]  # (n, 1, 1, 1)
            phi_alpha_j = phi_alpha_d[None, j_idx, None, None]  # (1, n, 1, 1)
            psi_alpha_k = psi_alpha_d[None, None, k_idx, None]  # (1, 1, n, 1)
            psi_alpha_l = psi_alpha_d[None, None, None, l_idx]  # (1, 1, 1, n)
            
            phi_beta_i = phi_beta_d[i_idx, None, None, None]    # (n, 1, 1, 1)
            phi_beta_j = phi_beta_d[None, j_idx, None, None]    # (1, n, 1, 1)
            psi_beta_k = psi_beta_d[None, None, k_idx, None]    # (1, 1, n, 1)
            psi_beta_l = psi_beta_d[None, None, None, l_idx]   # (1, 1, 1, n)
            
            # The integral of the product of four beta PDFs is:
            # B(αᵢ + αⱼ + αₖ + αₗ - 3, βᵢ + βⱼ + βₖ + βₗ - 3) / 
            # (B(αᵢ, βᵢ) * B(αⱼ, βⱼ) * B(αₖ, βₖ) * B(αₗ, βₗ))
            
            # Sum all alpha and beta parameters
            total_alpha = phi_alpha_i + phi_alpha_j + psi_alpha_k + psi_alpha_l - 3
            total_beta = phi_beta_i + phi_beta_j + psi_beta_k + psi_beta_l - 3
            
            # Compute log of the integral to avoid overflow
            log_integral = (
                torch.special.gammaln(total_alpha) + torch.special.gammaln(total_beta) - torch.special.gammaln(total_alpha + total_beta)
                - torch.special.gammaln(phi_alpha_i) - torch.special.gammaln(phi_beta_i)
                - torch.special.gammaln(phi_alpha_j) - torch.special.gammaln(phi_beta_j)
                - torch.special.gammaln(psi_alpha_k) - torch.special.gammaln(psi_beta_k)
                - torch.special.gammaln(psi_alpha_l) - torch.special.gammaln(psi_beta_l)
                + torch.special.gammaln(phi_alpha_i + phi_beta_i)
                + torch.special.gammaln(phi_alpha_j + phi_beta_j)
                + torch.special.gammaln(psi_alpha_k + psi_beta_k)
                + torch.special.gammaln(psi_alpha_l + psi_beta_l)
            )
            
            ## DEBUG: Check for NaN values in log_integral
            #if torch.any(torch.isnan(log_integral)):
            #    print(f"NaN detected in log_integral for dimension {d}")
            #    print("total_alpha min/max:", torch.min(total_alpha), torch.max(total_alpha))
            #    print("total_beta min/max:", torch.min(total_beta), torch.max(total_beta))
            #    print("phi_alpha_i min/max:", torch.min(phi_alpha_i), torch.max(phi_alpha_i))
            #    print("phi_beta_i min/max:", torch.min(phi_beta_i), torch.max(phi_beta_i))
            #    print("log_integral min/max:", torch.min(log_integral), torch.max(log_integral))
            #    print("log_integral has NaN at indices:", torch.isnan(log_integral).nonzero())
            #    input("Press Enter to continue...")
            
            # Add to the gram tensor for this dimension
            log_gram_tensor += log_integral
        
        # Convert from log space
        gram_tensor = torch.exp(log_gram_tensor)

        #print("Phi alpha: ", phi_alpha)
        #print("Phi beta: ", phi_beta)
        #print("Psi alpha: ", psi_alpha)
        #print("Psi beta: ", psi_beta)
        #print("Gram tensor [0, 0, 0, 0]: ", gram_tensor[0, 0, 0, 0])
        #input("...")
        
        return gram_tensor
    
    def get_phi_params(self):
        return (self.max_alpha_beta - self.min_alpha_beta) * torch.nn.functional.sigmoid(self.phi_params) + self.min_alpha_beta

    def get_psi_params(self):
        return (self.max_alpha_beta - self.min_alpha_beta) * torch.nn.functional.sigmoid(self.psi_params) + self.min_alpha_beta