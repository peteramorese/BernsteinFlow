import torch
from .SOSModel import SOSModel

class SumBetaSOSModel(SOSModel):
    """
    Each basis is a polynomial-like sum of Beta products over the y-dimensions:

        phi_i(z) = sum_{t=1..T} c^phi_{i,t} * Π_{r=1..dy} Beta(z_r; α^phi_{i,t,r}, β^phi_{i,t,r})
        psi_k(z) = sum_{t=1..T} c^psi_{k,t} * Π_{r=1..dy} Beta(z_r; α^psi_{k,t,r}, β^psi_{k,t,r})

    where Beta(·; a,b) is the *pdf* on [0,1] with normalization 1/B(a,b).

    Notes:
      • To make E[i,j,k,l] = ∫ phi_i(y) phi_j(y) psi_k(y) psi_l(y) dy correct,
        BOTH φ and ψ inside the integral must live in dy dimensions.
      • Coefficients may be negative; α,β are constrained to [min,max].
    """

    def __init__(self, dy: int, dx: int, n: int,
                 n_terms: int = 5,
                 min_alpha_beta: float = 1.0,
                 max_alpha_beta: float = 50.0,
                 regularization_weight: float = 0,
                 **kwargs):
        self.n_terms = n_terms
        self.min_alpha_beta = min_alpha_beta
        self.max_alpha_beta = max_alpha_beta
        self.regularization_weight = regularization_weight

        # IMPORTANT: For a correct Gram over y, param dims must be based on dy
        #   (coeff + α,β per y-dimension) per term.
        phi_param_dim = n_terms * (1 + 2 * dy)
        psi_param_dim = n_terms * (1 + 2 * dy)

        super().__init__(dy=dy, dx=dx, n=n,
                         phi_param_dim=phi_param_dim,
                         psi_param_dim=psi_param_dim,
                         **kwargs)

    # ---------- constrained parameter access (override shape/behavior like Beta model) ----------
    def constrained_phi_params(self):
        """
        Returns params with α,β constrained; coefficients unchanged.
        Layout per basis i:
          [ c_1, ..., c_T,  α_{1,1..dy}, β_{1,1..dy}, α_{2,1..dy}, β_{2,1..dy}, ..., α_{T,1..dy}, β_{T,1..dy} ]
        """
        T = self.n_terms
        raw = self.phi_params_uc
        coeff = raw[:, :T]
        ab_raw = raw[:, T:]
        ab = (self.max_alpha_beta - self.min_alpha_beta) * torch.sigmoid(ab_raw) + self.min_alpha_beta
        return torch.cat([coeff, ab], dim=1)

    def constrained_psi_params(self):
        T = self.n_terms
        raw = self.psi_params_uc
        coeff = raw[:, :T]
        ab_raw = raw[:, T:]
        ab = (self.max_alpha_beta - self.min_alpha_beta) * torch.sigmoid(ab_raw) + self.min_alpha_beta
        return torch.cat([coeff, ab], dim=1)

    # ---------- helpers to split (coeff, alpha, beta) ----------
    def _split_phi(self):
        T, dy = self.n_terms, self.dy
        P = self.get_phi_params()  # uses constrained_* under the hood in your base class
        coeff = P[:, :T]                                # (n,T)
        ab = P[:, T:].reshape(self.n, T, 2 * dy)        # (n,T,2*dy)
        alpha = ab[:, :, :dy]
        beta  = ab[:, :, dy:]
        return coeff, alpha, beta

    def _split_psi(self):
        T, dy = self.n_terms, self.dy
        P = self.get_psi_params()
        coeff = P[:, :T]                                # (n,T)
        ab = P[:, T:].reshape(self.n, T, 2 * dy)        # (n,T,2*dy)
        alpha = ab[:, :, :dy]
        beta  = ab[:, :, dy:]
        return coeff, alpha, beta

    # ---------- basis evaluations (sum of Beta pdf products over dy) ----------
    def phi(self, y: torch.Tensor):
        """
        y: (p, dy)  ->  (p, n)
        φ_i(y) = sum_t c_{i,t} * Π_r Beta(y_r; α_{i,t,r}, β_{i,t,r})
        """
        n, T, dy = self.n, self.n_terms, self.dy
        coeff, alpha, beta = self._split_phi()

        # log Beta pdf per-dimension, then sum over dims
        log_y   = torch.log(y.clamp_min(1e-8))[:, None, None, :]       # (p,1,1,dy)
        log_1my = torch.log((1 - y).clamp_min(1e-8))[:, None, None, :] # (p,1,1,dy)

        a = alpha[None, :, :, :]  # (1,n,T,dy)
        b = beta [None, :, :, :]  # (1,n,T,dy)

        # log pdf: (a-1)log y + (b-1)log(1-y) - ln B(a,b)
        lnB = torch.special.gammaln(a) + torch.special.gammaln(b) - torch.special.gammaln(a + b)
        log_pdf = (a - 1.0) * log_y + (b - 1.0) * log_1my - lnB         # (p,n,T,dy)

        log_term = log_pdf.sum(dim=-1)                                  # (p,n,T)
        return (torch.exp(log_term) * coeff[None, :, :]).sum(dim=-1)    # (p,n)

    def psi(self, y: torch.Tensor):
        """
        y: (p, dy)  ->  (p, n)
        ψ_k(y) = sum_t c_{k,t} * Π_r Beta(y_r; α_{k,t,r}, β_{k,t,r})
        """
        n, T, dy = self.n, self.n_terms, self.dy
        coeff, alpha, beta = self._split_psi()

        log_y   = torch.log(y.clamp_min(1e-8))[:, None, None, :]       # (p,1,1,dy)
        log_1my = torch.log((1 - y).clamp_min(1e-8))[:, None, None, :] # (p,1,1,dy)

        a = alpha[None, :, :, :]  # (1,n,T,dy)
        b = beta [None, :, :, :]  # (1,n,T,dy)

        lnB = torch.special.gammaln(a) + torch.special.gammaln(b) - torch.special.gammaln(a + b)
        log_pdf = (a - 1.0) * log_y + (b - 1.0) * log_1my - lnB         # (p,n,T,dy)

        log_term = log_pdf.sum(dim=-1)                                  # (p,n,T)
        return (torch.exp(log_term) * coeff[None, :, :]).sum(dim=-1)    # (p,n)

    def regularization_loss(self):
        _, phi_alpha, phi_beta = self._split_phi()
        _, psi_alpha, psi_beta = self._split_psi()
        exp_sq = phi_alpha ** 2 + phi_beta ** 2 + psi_alpha ** 2 + psi_beta ** 2
        return self.regularization_weight * torch.sum(exp_sq)

    # ---------- fully vectorized, grad-capable Gram tensor over y ----------
    def gram_tensor(self, cross_gram_model=None):
        """
        Returns E[i,j,k,l] = ∫ φ_i(y) φ_j(y) ψ_k(y) ψ_l(y) dy  (order i,j,k,l).

        NOTE: This is fully vectorized and memory-heavy by design; it's the fastest form.
        Make sure both φ and ψ parameter blocks are dy-dimensional (as done here).
        """
        # φ and ψ params for dy dimensions
        phi_c, phi_a, phi_b = self._split_phi()  # (n,T), (n,T,dy), (n,T,dy)
        if cross_gram_model is None:
            psi_c, psi_a, psi_b = self._split_psi()
        else:
            # use cross model's ψ params
            psi_c, psi_a, psi_b = cross_gram_model._split_psi()

        n, T, dy = self.n, self.n_terms, self.dy

        # Precompute ln B per (basis, term, dim)
        lnB_phi = torch.special.gammaln(phi_a) + torch.special.gammaln(phi_b) - torch.special.gammaln(phi_a + phi_b)  # (n,T,dy)
        lnB_psi = torch.special.gammaln(psi_a) + torch.special.gammaln(psi_b) - torch.special.gammaln(psi_a + psi_b)  # (n,T,dy)

        # Expand coefficients to (n,n,n,n,T,T,T,T)
        ca = phi_c[:, None, None, None, :, None, None, None]  # (n,n,n,n,T,1,1,1)
        cb = phi_c[None, :, None, None, None, :, None, None]  # (n,n,n,n,1,T,1,1)
        cc = psi_c[None, None, :, None, None, None, :, None]  # (n,n,n,n,1,1,T,1)
        cd = psi_c[None, None, None, :, None, None, None, :]  # (n,n,n,n,1,1,1,T)
        coeff_prod = ca * cb * cc * cd                        # (n,n,n,n,T,T,T,T)

        # Expand α,β and lnB for each role (i,j,k,l) with dy in the last axis
        ai = phi_a[:, None, None, None, :, None, None, None, :]  # (n,n,n,n,T,1,1,1,dy)
        bi = phi_b[:, None, None, None, :, None, None, None, :]
        aj = phi_a[None, :, None, None, None, :, None, None, :]
        bj = phi_b[None, :, None, None, None, :, None, None, :]
        ak = psi_a[None, None, :, None, None, None, :, None, :]
        bk = psi_b[None, None, :, None, None, None, :, None, :]
        al = psi_a[None, None, None, :, None, None, None, :, :]
        bl = psi_b[None, None, None, :, None, None, None, :, :]

        lnB_i = lnB_phi[:, None, None, None, :, None, None, None, :]     # (n,n,n,n,T,1,1,1,dy)
        lnB_j = lnB_phi[None, :, None, None, None, :, None, None, :]
        lnB_k = lnB_psi[None, None, :, None, None, None, :, None, :]
        lnB_l = lnB_psi[None, None, None, :, None, None, None, :, :]

        # Per-dimension log integral:
        # log I_r = ln B( (ai+aj+ak+al - 3), (bi+bj+bk+bl - 3) ) - (lnB_i + lnB_j + lnB_k + lnB_l)
        A_sum = ai + aj + ak + al - 3.0
        B_sum = bi + bj + bk + bl - 3.0

        log_num = (torch.special.gammaln(A_sum)
                   + torch.special.gammaln(B_sum)
                   - torch.special.gammaln(A_sum + B_sum))                 # (n,n,n,n,T,T,T,T,dy)
        log_den = lnB_i + lnB_j + lnB_k + lnB_l
        log_I_dim = log_num - log_den                                      # (n,n,n,n,T,T,T,T,dy)

        # Product over dimensions => sum of logs over dy
        log_I = log_I_dim.sum(dim=-1)                                      # (n,n,n,n,T,T,T,T)
        I = torch.exp(log_I)

        # Contract over term indices (a,b,c,d) with the coefficient product
        E = torch.einsum('ijklabcd,ijklabcd->ijkl', coeff_prod, I)         # (n,n,n,n)
        return E

    def marginalize(self, dims_to_integrate):
        """
        Compute the exact marginal over the kept dimensions for a *non-conditional* model p(z).
        Returns a SumTensorMarginalSOSModel that stores:
            - reduced φ, ψ params over kept dims
            - a coefficient tensor T[i,j,k,l] that includes integration over dims_to_integrate.
        """
        device = self.get_phi_params().device
        dtype  = self.get_phi_params().dtype
        n, T_full, dy = self.n, self.n_terms, self.dy

        dims_to_integrate = sorted(set(int(d) for d in dims_to_integrate))
        kept = [d for d in range(dy) if d not in dims_to_integrate]
        dz = len(kept)

        # ---- helper to slice SumBeta params to kept dims ----
        def slice_sum_beta_params(params_flat, n_terms, dy_full, kept_idx):
            # params layout per basis: [ c_1..c_T,  α_{1,1..dy},β_{1,1..dy}, α_{2,1..dy},β_{2,1..dy}, ... ]
            n = params_flat.shape[0]
            coeff = params_flat[:, :n_terms]                              # (n,T)
            ab = params_flat[:, n_terms:].reshape(n, n_terms, 2*dy_full)  # (n,T,2*dy)
            alpha = ab[:, :, :dy_full][:, :, kept_idx]                    # (n,T,dz)
            beta  = ab[:, :, dy_full:][:, :, kept_idx]                    # (n,T,dz)
            ab_kept = torch.cat([alpha, beta], dim=-1)                    # (n,T,2*dz)
            return torch.cat([coeff, ab_kept.reshape(n, n_terms*(2*dz))], dim=1)  # (n, T + T*(2*dz))

        # ---- get constrained params (so α,β are in [min,max]) and slice to kept dims ----
        phi_params_full = self.get_phi_params()  # (n, T + T*(2*dy))
        psi_params_full = self.get_psi_params()
        phi_params_kept = slice_sum_beta_params(phi_params_full, T_full, dy, kept).detach()
        psi_params_kept = slice_sum_beta_params(psi_params_full, T_full, dy, kept).detach()

        # ---- build C[i,j,k,l] = ∫ φ_i φ_j ψ_k ψ_l dy over integrated dims only ----
        # If nothing to integrate, C is all ones.
        if len(dims_to_integrate) == 0:
            C = torch.ones((n, n, n, n), dtype=dtype, device=device)
        else:
            # Split full (dy) params to get the integrated part
            def split_params(params_flat, n_terms, dy_full):
                coeff = params_flat[:, :n_terms]                                  # (n,T)
                ab = params_flat[:, n_terms:].reshape(n, n_terms, 2*dy_full)      # (n,T,2*dy)
                alpha = ab[:, :, :dy_full]
                beta  = ab[:, :, dy_full:]
                return coeff, alpha, beta

            phi_c, phi_a, phi_b = split_params(phi_params_full, T_full, dy)  # (n,T), (n,T,dy), (n,T,dy)
            psi_c, psi_a, psi_b = split_params(psi_params_full, T_full, dy)

            # Select only the integrated dims (m = len(dims_to_integrate))
            m = len(dims_to_integrate)
            phi_a_int = phi_a[:, :, dims_to_integrate]  # (n,T,m)
            phi_b_int = phi_b[:, :, dims_to_integrate]  # (n,T,m)
            psi_a_int = psi_a[:, :, dims_to_integrate]  # (n,T,m)
            psi_b_int = psi_b[:, :, dims_to_integrate]  # (n,T,m)

            # Precompute lnB for the per-term/per-dim normalization
            lnB_phi = (torch.special.gammaln(phi_a_int)
                       + torch.special.gammaln(phi_b_int)
                       - torch.special.gammaln(phi_a_int + phi_b_int))           # (n,T,m)
            lnB_psi = (torch.special.gammaln(psi_a_int)
                       + torch.special.gammaln(psi_b_int)
                       - torch.special.gammaln(psi_a_int + psi_b_int))           # (n,T,m)

            # Broadcast to (n,n,n,n,T,T,T,T,m) and integrate like in gram_tensor, but only over m dims
            ca = phi_c[:, None, None, None, :, None, None, None]  # (n,n,n,n,T,1,1,1)
            cb = phi_c[None, :, None, None, None, :, None, None]  # (n,n,n,n,1,T,1,1)
            cc = psi_c[None, None, :, None, None, None, :, None]  # (n,n,n,n,1,1,T,1)
            cd = psi_c[None, None, None, :, None, None, None, :]  # (n,n,n,n,1,1,1,T)
            coeff_prod = ca * cb * cc * cd                        # (n,n,n,n,T,T,T,T)

            ai = phi_a_int[:, None, None, None, :, None, None, None, :]  # (n,n,n,n,T,1,1,1,m)
            bi = phi_b_int[:, None, None, None, :, None, None, None, :]
            aj = phi_a_int[None, :, None, None, None, :, None, None, :]
            bj = phi_b_int[None, :, None, None, None, :, None, None, :]
            ak = psi_a_int[None, None, :, None, None, None, :, None, :]
            bk = psi_b_int[None, None, :, None, None, None, :, None, :]
            al = psi_a_int[None, None, None, :, None, None, None, :, :]
            bl = psi_b_int[None, None, None, :, None, None, None, :, :]

            lnB_i = lnB_phi[:, None, None, None, :, None, None, None, :]  # (n,n,n,n,T,1,1,1,m)
            lnB_j = lnB_phi[None, :, None, None, None, :, None, None, :]
            lnB_k = lnB_psi[None, None, :, None, None, None, :, None, :]
            lnB_l = lnB_psi[None, None, None, :, None, None, None, :, :]

            # Per integrated dimension: ln B(sum α - 3, sum β - 3) - sum lnB(term)
            A_sum = ai + aj + ak + al - 3.0
            B_sum = bi + bj + bk + bl - 3.0

            log_num = (torch.special.gammaln(A_sum)
                       + torch.special.gammaln(B_sum)
                       - torch.special.gammaln(A_sum + B_sum))                 # (...,m)
            log_den = lnB_i + lnB_j + lnB_k + lnB_l
            log_I_dim = log_num - log_den                                      # (n,n,n,n,T,T,T,T,m)

            # Sum over the integrated dims (product across dims)
            log_I = log_I_dim.sum(dim=-1)                                      # (n,n,n,n,T,T,T,T)
            I = torch.exp(log_I)

            # Contract over all term indices (a,b,c,d) to get C[i,j,k,l]
            C = torch.einsum('ijklabcd,ijklabcd->ijkl', coeff_prod, I)         # (n,n,n,n)

        # ---- build T[i,j,k,l] = R[i,j] * Q[k,l] * C[i,j,k,l] ----
        Q, R = self.get_QR_matrices()  # Same as your Beta model
        T = (R[:, :, None, None] * Q[None, None, :, :]) * C                   # (n,n,n,n)

        return SumBetaMarginalSOSModel(
            phi_params=phi_params_kept,
            psi_params=psi_params_kept,
            coeff_tensor=T,
            n_terms=T_full,
            min_alpha_beta=self.min_alpha_beta,
            max_alpha_beta=self.max_alpha_beta,
        )


class SumBetaMarginalSOSModel(torch.nn.Module):
    """
    Marginalized SOS model for the sum-of-Beta basis.
    Stores reduced-dimension φ, ψ params (with coefficients and α/β per term)
    and a 4-tensor of coefficients T[i,j,k,l].

    Density:
        p(z) = sum_{i,j,k,l} T[i,j,k,l] *
               φ_i(z) * φ_j(z) * ψ_k(z) * ψ_l(z),
    where each φ_i, ψ_k is a *sum* of Beta products over the kept dims.
    """
    def __init__(self, phi_params, psi_params, coeff_tensor,
                 n_terms, min_alpha_beta=1.0, max_alpha_beta=50.0):
        super().__init__()
        # Store as buffers (fixed after construction), like your Beta marginal
        self.register_buffer("phi_params", phi_params)       # (n, T + T*(2*dz))
        self.register_buffer("psi_params", psi_params)       # (n, T + T*(2*dz))
        self.register_buffer("coeff_tensor", coeff_tensor)   # (n,n,n,n)
        self.n = phi_params.shape[0]
        self.n_terms = int(n_terms)
        # infer dz from layout
        total = phi_params.shape[1]
        # total = T + T*(2*dz) => (total - T) / (2*T) = dz
        self.dz = int((total - self.n_terms) // (2*self.n_terms))
        self.min_alpha_beta = min_alpha_beta
        self.max_alpha_beta = max_alpha_beta

    # ------- helpers to split params -------
    def _split(self, params):
        n, T, dz = self.n, self.n_terms, self.dz
        coeff = params[:, :T]                                   # (n,T)
        ab = params[:, T:].reshape(n, T, 2*dz)                  # (n,T,2*dz)
        alpha = ab[:, :, :dz]                                   # (n,T,dz)
        beta  = ab[:, :, dz:]                                   # (n,T,dz)
        return coeff, alpha, beta

    # ------- basis evals (sum of Beta pdf products over kept dims) -------
    def phi(self, z):
        """
        z: (p, dz) -> (p, n)
        φ_i(z) = sum_t c_{i,t} * Π_r Beta(z_r; α_{i,t,r}, β_{i,t,r})
        """
        coeff, alpha, beta = self._split(self.phi_params)       # (n,T), (n,T,dz), (n,T,dz)
        log_z   = torch.log(z.clamp_min(1e-8))[:, None, None, :]        # (p,1,1,dz)
        log_1mz = torch.log((1 - z).clamp_min(1e-8))[:, None, None, :]  # (p,1,1,dz)

        a = alpha[None, :, :, :]  # (1,n,T,dz)
        b = beta [None, :, :, :]  # (1,n,T,dz)

        lnB = torch.special.gammaln(a) + torch.special.gammaln(b) - torch.special.gammaln(a + b)
        log_pdf = (a - 1.0) * log_z + (b - 1.0) * log_1mz - lnB           # (p,n,T,dz)

        log_term = log_pdf.sum(dim=-1)                                     # (p,n,T)
        return (torch.exp(log_term) * coeff[None, :, :]).sum(dim=-1)       # (p,n)

    def psi(self, z):
        """
        z: (p, dz) -> (p, n)
        ψ_k(z) = sum_t c_{k,t} * Π_r Beta(z_r; α_{k,t,r}, β_{k,t,r})
        """
        coeff, alpha, beta = self._split(self.psi_params)
        log_z   = torch.log(z.clamp_min(1e-8))[:, None, None, :]
        log_1mz = torch.log((1 - z).clamp_min(1e-8))[:, None, None, :]

        a = alpha[None, :, :, :]  # (1,n,T,dz)
        b = beta [None, :, :, :]  # (1,n,T,dz)

        lnB = torch.special.gammaln(a) + torch.special.gammaln(b) - torch.special.gammaln(a + b)
        log_pdf = (a - 1.0) * log_z + (b - 1.0) * log_1mz - lnB           # (p,n,T,dz)

        log_term = log_pdf.sum(dim=-1)                                     # (p,n,T)
        return (torch.exp(log_term) * coeff[None, :, :]).sum(dim=-1)       # (p,n)

    def forward(self, z, return_log: bool = False):
        """
        Evaluate marginal density at z ∈ [0,1]^{dz}.
        z: (p, dz)
        """
        assert z.shape[1] == self.dz
        phi_z = self.phi(z)   # (p,n)
        psi_z = self.psi(z)   # (p,n)

        dens = torch.einsum("ijkl,pi,pj,pk,pl->p", self.coeff_tensor, phi_z, phi_z, psi_z, psi_z)
        if return_log:
            return torch.log(dens.clamp_min(1e-12))
        return dens