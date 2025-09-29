import torch
from .SOSModel import SOSModel

class BetaSOSModel(SOSModel):
    def __init__(self, dy : int, dx : int, n : int, min_alpha_beta : float = 1.00, max_alpha_beta : float = 50.0, **kwargs):
        # Two parameters for each basis function (alpha and beta) for each dimension
        super().__init__(dy=dy, dx=dx, n=n, phi_param_dim=2 * dx, psi_param_dim=2 * dy, **kwargs)

        self.min_alpha_beta = min_alpha_beta
        self.max_alpha_beta = max_alpha_beta

    def constrained_phi_params(self):
        return (self.max_alpha_beta - self.min_alpha_beta) * torch.nn.functional.sigmoid(self.phi_params_uc) + self.min_alpha_beta

    def constrained_psi_params(self):
        return (self.max_alpha_beta - self.min_alpha_beta) * torch.nn.functional.sigmoid(self.psi_params_uc) + self.min_alpha_beta

    def phi(self, x : torch.Tensor):
        # Make each parameter positive and unsqeeze to data shape
        alpha_beta = self.get_phi_params()
        alpha = alpha_beta[:, :self.dx].unsqueeze(0) # Shape (p, n, dx)
        beta = alpha_beta[:, self.dx:].unsqueeze(0)

        log_x = torch.log(x + 1e-8)
        log_1mx = torch.log(1 - x + 1e-8)
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
        alpha_beta = self.get_psi_params()
        alpha = alpha_beta[:, :self.dy].unsqueeze(0) # Shape (p, n, dy)
        beta = alpha_beta[:, self.dy:].unsqueeze(0)

        log_y = torch.log(y + 1e-8)
        log_1my = torch.log(1 - y + 1e-8)
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

    def gram_tensor(self, cross_gram_model = None):
        """
        Compute the gram tensor using self's phi and psi parameters. If cross_gram_model is provided, use its psi parameters.
        """
        if cross_gram_model is None:
            phi_alpha_beta = self.get_phi_params()
            psi_alpha_beta = self.get_psi_params()
        else:
            phi_alpha_beta = self.get_phi_params()  # Current model's phi parameters
            psi_alpha_beta = cross_gram_model.get_psi_params()  # Other model's psi parameters

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
            
            # Add to the gram tensor for this dimension
            log_gram_tensor += log_integral
        
        # Convert from log space
        gram_tensor = torch.exp(log_gram_tensor)
        
        return gram_tensor


    def marginalize(self, dims_to_integrate):
        """
        Compute the exact marginal over the kept dimensions, returning a TensorMarginalSOSModel.
        """
        device = self.get_phi_params().device
        dtype  = self.get_phi_params().dtype
        n = self.n

        dims_to_integrate = sorted(set(dims_to_integrate))
        kept = [d for d in range(self.dy) if d not in dims_to_integrate]
        dy_new = len(kept)
        dx_new = dy_new

        # slice params
        def slice_ab(ab, D, keep_idx):
            alpha = ab[:, :D][:, keep_idx]
            beta  = ab[:, D:][:, keep_idx]
            return torch.cat([alpha, beta], dim=1)

        phi_ab = self.get_phi_params()
        psi_ab = self.get_psi_params()
        phi_new = slice_ab(phi_ab, self.dx, kept).detach()
        psi_new = slice_ab(psi_ab, self.dy, kept).detach()

        # build coefficient tensor T
        Q, R = self.get_QR_matrices()
        phi_alpha = self.get_phi_params()[:, :self.dx]
        phi_beta  = self.get_phi_params()[:, self.dx:]
        psi_alpha = self.get_psi_params()[:, :self.dy]
        psi_beta  = self.get_psi_params()[:, self.dy:]

        logC = torch.zeros(n, n, n, n, device=device, dtype=dtype)
        idx = torch.arange(n, device=device)

        for d in dims_to_integrate:
            pa, pb = phi_alpha[:, d], phi_beta[:, d]
            qa, qb = psi_alpha[:, d], psi_beta[:, d]

            pa_i = pa[idx, None, None, None]
            pa_j = pa[None, idx, None, None]
            qa_k = qa[None, None, idx, None]
            qa_l = qa[None, None, None, idx]
            pb_i = pb[idx, None, None, None]
            pb_j = pb[None, idx, None, None]
            qb_k = qb[None, None, idx, None]
            qb_l = qb[None, None, None, idx]

            total_alpha = pa_i + pa_j + qa_k + qa_l - 3
            total_beta  = pb_i + pb_j + qb_k + qb_l - 3

            log_int = (torch.special.gammaln(total_alpha) + torch.special.gammaln(total_beta)
                    - torch.special.gammaln(total_alpha + total_beta)
                    - torch.special.gammaln(pa_i) - torch.special.gammaln(pb_i)
                    - torch.special.gammaln(pa_j) - torch.special.gammaln(pb_j)
                    - torch.special.gammaln(qa_k) - torch.special.gammaln(qb_k)
                    - torch.special.gammaln(qa_l) - torch.special.gammaln(qb_l)
                    + torch.special.gammaln(pa_i + pb_i)
                    + torch.special.gammaln(pa_j + pb_j)
                    + torch.special.gammaln(qa_k + qb_k)
                    + torch.special.gammaln(qa_l + qb_l))
            logC += log_int

        C = torch.exp(logC)
        # coefficients T[i,j,k,l] = R[i,j] * Q[k,l] * C[i,j,k,l]
        T = (R[:, :, None, None] * Q[None, None, :, :]) * C

        return TensorMarginalSOSModel(phi_new, psi_new, T,
                                    min_alpha_beta=self.min_alpha_beta,
                                    max_alpha_beta=self.max_alpha_beta)


class TensorMarginalSOSModel(torch.nn.Module):
    """
    Marginalized SOS model that stores the exact 4-tensor coefficients
    after integrating out some dimensions. 

    Density is:
        p(x_kept) = sum_{i,j,k,l} T[i,j,k,l] *
                    prod_d phi_i^{(d)}(x_d) * phi_j^{(d)}(x_d) *
                    prod_d psi_k^{(d)}(x_d) * psi_l^{(d)}(x_d)
    """
    def __init__(self, phi_params, psi_params, coeff_tensor, min_alpha_beta=1.0, max_alpha_beta=50.0):
        """
        Args:
            phi_params : (n, 2*dx_new) tensor of Beta (alpha,beta) params for kept dims
            psi_params : (n, 2*dy_new) tensor
            coeff_tensor : (n, n, n, n) tensor of coefficients T[i,j,k,l]
        """
        super().__init__()
        self.register_buffer("phi_params", phi_params)
        self.register_buffer("psi_params", psi_params)
        self.register_buffer("coeff_tensor", coeff_tensor)
        self.n = phi_params.shape[0]
        self.dx = phi_params.shape[1] // 2
        self.dy = psi_params.shape[1] // 2
        self.min_alpha_beta = min_alpha_beta
        self.max_alpha_beta = max_alpha_beta

    def phi(self, x):
        """Evaluate phi basis functions at x (shape (p, dx_new)) -> (p, n)."""
        alpha = self.phi_params[:, :self.dx].unsqueeze(0)  # (1,n,dx)
        beta  = self.phi_params[:, self.dx:].unsqueeze(0)
        log_x = torch.log(x + 1e-8)[:, None, :]            # (p,1,dx)
        log_1mx = torch.log(1 - x + 1e-8)[:, None, :]
        log_phi = ((alpha-1)*log_x + (beta-1)*log_1mx
                   - torch.special.gammaln(alpha)
                   - torch.special.gammaln(beta)
                   + torch.special.gammaln(alpha+beta))
        return torch.exp(log_phi.sum(dim=2))  # (p,n)

    def psi(self, y):
        """Evaluate psi basis functions at y (shape (p, dy_new)) -> (p, n)."""
        alpha = self.psi_params[:, :self.dy].unsqueeze(0)  # (1,n,dy)
        beta  = self.psi_params[:, self.dy:].unsqueeze(0)
        log_y = torch.log(y + 1e-8)[:, None, :]            # (p,1,dy)
        log_1my = torch.log(1 - y + 1e-8)[:, None, :]
        log_psi = ((alpha-1)*log_y + (beta-1)*log_1my
                   - torch.special.gammaln(alpha)
                   - torch.special.gammaln(beta)
                   + torch.special.gammaln(alpha+beta))
        return torch.exp(log_psi.sum(dim=2))  # (p,n)

    def forward(self, xy, return_log=False):
        """
        Evaluate marginal density at (y,x) where xy has shape (p, dy+dx).
        """
        assert xy.shape[1] == self.dy + self.dx
        y = xy[:, :self.dy]
        x = xy[:, self.dy:]

        phi_x = self.phi(x)   # (p,n)
        psi_y = self.psi(y)   # (p,n)

        # Contract over i,j,k,l
        # dens[p] = sum_{i,j,k,l} T[i,j,k,l] * phi_x[p,i]*phi_x[p,j]*psi_y[p,k]*psi_y[p,l]
        dens = torch.einsum("ijkl,pi,pj,pk,pl->p", self.coeff_tensor, phi_x, phi_x, psi_y, psi_y)

        if return_log:
            return torch.log(dens + 1e-12)
        else:
            return dens
