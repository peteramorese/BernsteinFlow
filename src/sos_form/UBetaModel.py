import torch
from .SOSModel import SOSModel


class UBetaSOSModel(SOSModel):
    def __init__(self, dy: int, dx: int, n: int, min_alpha_beta: float = 1.00, max_alpha_beta: float = 50.0, regularization_weight: float = 0, **kwargs):
        # Two parameters for each basis function (alpha and beta) for each dimension
        super().__init__(dy=dy, dx=dx, n=n, phi_param_dim=2 * dx, psi_param_dim=2 * dy, **kwargs)

        self.min_alpha_beta = min_alpha_beta
        self.max_alpha_beta = max_alpha_beta
        self.regularization_weight = regularization_weight

    def constrained_phi_params(self):
        return (self.max_alpha_beta - self.min_alpha_beta) * torch.nn.functional.sigmoid(self.phi_params_uc) + self.min_alpha_beta

    def constrained_psi_params(self):
        return (self.max_alpha_beta - self.min_alpha_beta) * torch.nn.functional.sigmoid(self.psi_params_uc) + self.min_alpha_beta

    def phi(self, x: torch.Tensor):
        # Unnormalized Beta basis: prod_d x_d^{alpha-1} (1-x_d)^{beta-1}
        alpha_beta = self.get_phi_params()
        alpha = alpha_beta[:, :self.dx].unsqueeze(0)  # (1, n, dx)
        beta = alpha_beta[:, self.dx:].unsqueeze(0)

        log_x = torch.log(x + 1e-8)[:, None, :]
        log_1mx = torch.log(1 - x + 1e-8)[:, None, :]

        log_phi_per_dim = (alpha - 1.0) * log_x + (beta - 1.0) * log_1mx
        log_phi = torch.sum(log_phi_per_dim, dim=2)  # (p, n)
        return torch.exp(log_phi)

    def psi(self, y: torch.Tensor):
        # Unnormalized Beta basis: prod_d y_d^{alpha-1} (1-y_d)^{beta-1}
        alpha_beta = self.get_psi_params()
        alpha = alpha_beta[:, :self.dy].unsqueeze(0)  # (1, n, dy)
        beta = alpha_beta[:, self.dy:].unsqueeze(0)

        log_y = torch.log(y + 1e-8)[:, None, :]
        log_1my = torch.log(1 - y + 1e-8)[:, None, :]

        log_psi_per_dim = (alpha - 1.0) * log_y + (beta - 1.0) * log_1my
        log_psi = torch.sum(log_psi_per_dim, dim=2)  # (p, n)
        return torch.exp(log_psi)

    def gram_tensor(self, cross_gram_model=None):
        """
        Compute the gram tensor using self's phi and psi parameters. If cross_gram_model is provided, use its psi parameters.
        Uses unnormalized Beta basis, so inner products omit normalization denominators.
        """
        if cross_gram_model is None:
            phi_alpha_beta = self.get_phi_params()
            psi_alpha_beta = self.get_psi_params()
        else:
            phi_alpha_beta = self.get_phi_params()
            psi_alpha_beta = cross_gram_model.get_psi_params()

        phi_alpha = phi_alpha_beta[:, :self.dx]
        phi_beta = phi_alpha_beta[:, self.dx:]
        psi_alpha = psi_alpha_beta[:, :self.dy]
        psi_beta = psi_alpha_beta[:, self.dy:]

        log_gram_tensor = torch.zeros(self.n, self.n, self.n, self.n, dtype=phi_alpha.dtype, device=phi_alpha.device)

        for d in range(self.dy):
            pa = phi_alpha[:, d]
            pb = phi_beta[:, d]
            qa = psi_alpha[:, d]
            qb = psi_beta[:, d]

            idx = torch.arange(self.n, device=phi_alpha.device)

            pa_i = pa[idx, None, None, None]
            pa_j = pa[None, idx, None, None]
            qa_k = qa[None, None, idx, None]
            qa_l = qa[None, None, None, idx]
            pb_i = pb[idx, None, None, None]
            pb_j = pb[None, idx, None, None]
            qb_k = qb[None, None, idx, None]
            qb_l = qb[None, None, None, idx]

            total_alpha = pa_i + pa_j + qa_k + qa_l - 3
            total_beta = pb_i + pb_j + qb_k + qb_l - 3

            # For unnormalized Beta basis, integral is B(total_alpha, total_beta)
            log_int = (
                torch.special.gammaln(total_alpha)
                + torch.special.gammaln(total_beta)
                - torch.special.gammaln(total_alpha + total_beta)
            )

            log_gram_tensor += log_int

        return torch.exp(log_gram_tensor)

    def regularization_loss(self):
        return self.regularization_weight * torch.sum(self.get_phi_params() ** 2 + self.get_psi_params() ** 2)

    def marginalize(self, dims_to_integrate):
        """
        Compute the exact marginal over the kept dimensions, returning a UTensorMarginalSOSModel.
        """
        device = self.get_phi_params().device
        dtype = self.get_phi_params().dtype
        n = self.n

        dims_to_integrate = sorted(set(dims_to_integrate))
        kept = [d for d in range(self.dy) if d not in dims_to_integrate]
        dy_new = len(kept)
        dx_new = dy_new

        def slice_ab(ab, D, keep_idx):
            alpha = ab[:, :D][:, keep_idx]
            beta = ab[:, D:][:, keep_idx]
            return torch.cat([alpha, beta], dim=1)

        phi_ab = self.get_phi_params()
        psi_ab = self.get_psi_params()
        phi_new = slice_ab(phi_ab, self.dx, kept).detach()
        psi_new = slice_ab(psi_ab, self.dy, kept).detach()

        Q, R = self.get_QR_matrices()
        phi_alpha = self.get_phi_params()[:, :self.dx]
        phi_beta = self.get_phi_params()[:, self.dx:]
        psi_alpha = self.get_psi_params()[:, :self.dy]
        psi_beta = self.get_psi_params()[:, self.dy:]

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
            total_beta = pb_i + pb_j + qb_k + qb_l - 3

            # Unnormalized integral: B(total_alpha, total_beta)
            log_int = (
                torch.special.gammaln(total_alpha)
                + torch.special.gammaln(total_beta)
                - torch.special.gammaln(total_alpha + total_beta)
            )
            logC += log_int

        C = torch.exp(logC)
        T = (R[:, :, None, None] * Q[None, None, :, :]) * C

        return UTensorMarginalSOSModel(
            phi_new,
            psi_new,
            T,
            min_alpha_beta=self.min_alpha_beta,
            max_alpha_beta=self.max_alpha_beta,
        )


class UTensorMarginalSOSModel(torch.nn.Module):
    """
    Marginalized unnormalized SOS model that stores the exact 4-tensor coefficients
    after integrating out some dimensions.

    Density is:
        p(z) = sum_{i,j,k,l} T[i,j,k,l] *
               phi_i(z) * phi_j(z) * psi_k(z) * psi_l(z)
    where z ∈ [0,1]^{d'} with reduced dimension.
    """

    def __init__(self, phi_params, psi_params, coeff_tensor, min_alpha_beta=1.0, max_alpha_beta=50.0):
        super().__init__()
        self.register_buffer("phi_params", phi_params)
        self.register_buffer("psi_params", psi_params)
        self.register_buffer("coeff_tensor", coeff_tensor)
        self.n = phi_params.shape[0]
        self.dz = phi_params.shape[1] // 2
        self.min_alpha_beta = min_alpha_beta
        self.max_alpha_beta = max_alpha_beta

    def phi(self, z):
        """Evaluate unnormalized phi basis functions at z (p,dz) -> (p,n)."""
        alpha = self.phi_params[:, :self.dz].unsqueeze(0)
        beta = self.phi_params[:, self.dz:].unsqueeze(0)
        log_z = torch.log(z + 1e-8)[:, None, :]
        log_1mz = torch.log(1 - z + 1e-8)[:, None, :]
        log_phi = (alpha - 1.0) * log_z + (beta - 1.0) * log_1mz
        return torch.exp(log_phi.sum(dim=2))

    def psi(self, z):
        """Evaluate unnormalized psi basis functions at z (p,dz) -> (p,n)."""
        alpha = self.psi_params[:, :self.dz].unsqueeze(0)
        beta = self.psi_params[:, self.dz:].unsqueeze(0)
        log_z = torch.log(z + 1e-8)[:, None, :]
        log_1mz = torch.log(1 - z + 1e-8)[:, None, :]
        log_psi = (alpha - 1.0) * log_z + (beta - 1.0) * log_1mz
        return torch.exp(log_psi.sum(dim=2))

    def forward(self, z, return_log=False):
        """
        Evaluate marginal density at z ∈ [0,1]^{dz}.

        Args:
            z : tensor (p, dz) of inputs
        """
        assert z.shape[1] == self.dz

        phi_z = self.phi(z)  # (p,n)
        psi_z = self.psi(z)  # (p,n)

        dens = torch.einsum("ijkl,pi,pj,pk,pl->p", self.coeff_tensor, phi_z, phi_z, psi_z, psi_z)

        if return_log:
            return torch.log(dens + 1e-12)
        else:
            return dens


