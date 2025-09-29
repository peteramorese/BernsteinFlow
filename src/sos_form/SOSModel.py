import torch
import copy
from torch.utils.data import DataLoader, TensorDataset
import time
import sys
import numpy as np
import cvxpy as cp


class SOSModel(torch.nn.Module):
    def __init__(self, dy : int, 
                dx : int, 
                n : int, 
                phi_param_dim : int, 
                psi_param_dim : int, 
                conditional : bool = True,
                reference_factor_model = None,
                mu : float = 1.0,
                npsd_penalty : float = 1.0,
                min_Q_eigval : float = 1e-2,
                fixed_phi_params = None,
                fixed_psi_params = None,
                fixed_Q = None,
                fixed_R = None,
                ):
        """
        SOS form conditional density model for p(y | x)
        Args:
            dy : dimension of the support
            dx : dimension of the conditioner variable
            n : number of basis functions 
            phi_param_dim : dimension of the phi (x-basis) parameters
            psi_param_dim : dimension of the psi (y-basis) parameters
            conditional : whether the model is conditional (if False, then dx = 0 and reference_factor_model is required)
            reference_factor_model : reference factor model for non-conditional models. The model is trained with the reference g-factor
            I
        """

        super().__init__()

        self.dy = dy
        self.dx = dx if conditional else dy
        self.n = n
        self.phi_param_dim = phi_param_dim
        self.psi_param_dim = psi_param_dim
        self.mu = mu
        self.npsd_penalty = npsd_penalty
        self.min_Q_eigval = min_Q_eigval
        self.conditional = conditional
        self.fixed_params = False

        if self.conditional:
            assert reference_factor_model is None, "reference_factor_model is currently not allowed for conditional models"
            self.R_uc = torch.nn.Parameter(torch.randn(self.n, self.n))
            self.phi_params_uc = torch.nn.Parameter(1 * torch.randn(self.n, phi_param_dim))
            self.psi_params_uc = torch.nn.Parameter(1 * torch.randn(self.n, psi_param_dim)) 
            self.Q_uc = torch.nn.Parameter(torch.randn(self.n, self.n))
        elif fixed_phi_params is not None and fixed_psi_params is not None and fixed_Q is not None and fixed_R is not None:
            self.register_buffer("fp_phi_params", fixed_phi_params)
            self.register_buffer("fp_psi_params", fixed_psi_params)
            self.register_buffer("fp_Q", fixed_Q)
            self.register_buffer("fp_R", fixed_R)
            self.fixed_params = True
        else:
            # If a reference factor model is provided, R and phi parameters are fixed
            assert reference_factor_model is not None, "reference_factor_model is required for non-conditional models"
            #self.reference_factor_model = reference_factor_model
            _, reference_R = reference_factor_model.get_QR_matrices()
            ref_R_evals = torch.linalg.eigvalsh(reference_R)
            if torch.any(ref_R_evals < 0):
                raise ValueError("Reference R matrix is not PSD")
            self.register_buffer("ref_R", reference_R.detach())
            self.register_buffer("ref_phi_params", reference_factor_model.get_phi_params().detach()) # Add as a buffer instead of trainable parameter
            self.psi_params_uc = torch.nn.Parameter(1 * torch.randn(self.n, psi_param_dim)) 
            self.Q_uc = torch.nn.Parameter(torch.randn(self.n, self.n))

        


    #def __get_phi(self, x : torch.Tensor):
    #    phi_x_vals = self.phi(x)
    #    phi_x_vals = torch.cat([torch.ones_like(phi_x_vals[:, :1]), phi_x_vals], dim=1)
    #    return 
        
    def constrained_phi_params(self):
        """
        Get the constrained parameters of the phi basis functions.
        """
        raise NotImplementedError()
    
    def constrained_psi_params(self):
        """
        Get the constrained parameters of the psi basis functions.
        """
        raise NotImplementedError()

    def get_phi_params(self):
        if self.conditional:
            return self.constrained_phi_params()
        elif self.fixed_params:
            return self.fp_phi_params
        else:
            return self.ref_phi_params

    def get_psi_params(self):
        if self.fixed_params:
            return self.fp_psi_params
        else:
            return self.constrained_psi_params()

    def phi(self, x : torch.Tensor):
        """
        Evaluate the phi basis function vector at x using self.phi_params. Must return a tensor of size (p, n)
        """
        raise NotImplementedError()

    def psi(self, y : torch.Tensor):
        """
        Evaluate the psi basis function vector at y using self.psi_params. Must return a tensor of size (p, n)
        """
        raise NotImplementedError()

    def gram_tensor(self, cross_gram_model = None):
        """
        Compute the inner product matrix of the psi basis functions. Must return a tensor of size (n*m, n*m)
        """
        raise NotImplementedError()
    
    #def regularization_loss(self):
    #    return torch.tensor(0.0, dtype=self.phi_params_uc.dtype, device=self.phi_params_uc.device)

    def get_QR_matrices(self, E4 : torch.Tensor = None):
        n = self.n
        if self.conditional:

            if E4 is None:
                E4 = self.gram_tensor()  # shape (n, n, n, n), E4[k, l, i, j]

            R_u_sym = self.R_uc + self.R_uc.T

            Q_unscaled =  self.Q_uc @ self.Q_uc.T + self.min_Q_eigval * torch.eye(n, dtype=self.Q_uc.dtype, device=self.Q_uc.device)

            #print("E4: \n", E4)

            # Expand Q so its (i,j) aligns with the LAST two dims of E4
            Q4_unscaled = Q_unscaled[None, None, :, :]   # shape (1, 1, n, n)

            # Elementwise multiply
            M4 = E4 * Q4_unscaled               # shape (n, n, n, n)

            # Reshape into (n^2, n^2)
            M = M4.permute(2, 3, 0, 1).reshape(n*n, n*n)

            if torch.any(torch.isnan(M)):
                print("M is nan")
                print("Q_unscaled: \n", Q_unscaled)
                #print("E4: \n", E4)
                input("...")
            # Compute the max eigenvalue of M to rescale Q. All e-vals of Q are guaranteed to be real and non-negative.
            lambda_M_vals = torch.real(torch.linalg.eigvals(M))
            lambda_M_max = torch.max(lambda_M_vals)
            
            #M_null_mask = (lambda_M_vals < 1e-8)
            #if
            #lambda_M_max = torch.max(torch.real(torch.linalg.eigvals(M)))

            #print("(prescale) M lambda vals: \n", torch.real(torch.linalg.eigvals(M)))

            # Rescale Q and M to make the nullspace of M non trivial
            Q = Q_unscaled / lambda_M_max
            M = M / lambda_M_max

            #print("(postscale) M lambda vals: \n", torch.real(torch.linalg.eigvals(M)))
            #print("Q_unscaled: \n", Q_unscaled)
            #print("lambda_M_max: \n", lambda_M_max)
            #print("Q: \n", Q)
            #input("...")

            I = torch.eye(n*n, dtype=Q.dtype, device=Q.device)
            A = I - M


            r_proj_vec = self._project_null(A, R_u_sym.reshape(-1))

            R_proj = r_proj_vec.reshape(n, n)
            #print("R_proj: \n", R_proj)
            #input("...")

            # Optional: symmetrize (can be omitted if E4, Q are guaranteed symmetric)
            R_proj = 0.5 * (R_proj + R_proj.T)

            return Q, R_proj #s_min
        elif self.fixed_params:
            return self.fp_Q, self.fp_R
        else:
            Q_unscaled = self.Q_uc @ self.Q_uc.T + self.min_Q_eigval * torch.eye(n, dtype=self.Q_uc.dtype, device=self.Q_uc.device)

            if E4 is None:
                E4 = self.gram_tensor()

            E4 = E4.permute(2, 3, 0, 1)
            
            normalization_constant = torch.einsum('ij,kl,ijkl->', Q_unscaled, self.ref_R, E4)

            Q = Q_unscaled / normalization_constant

            return Q, self.ref_R

    def _project_null(self, A : torch.Tensor, v : torch.Tensor, lam : float = 1e-8):
        Ax = A @ v                                 # (m,)
        G  = A @ A.T
        G  = G + lam * torch.eye(G.shape[-1], device=A.device, dtype=A.dtype)
        # Cholesky solve
        L = torch.linalg.cholesky(G)               # (m,m)
        y = torch.cholesky_solve(Ax.unsqueeze(-1), L).squeeze(-1)  # (m,)
        return v - A.T @ y

    def forward(self, yx : torch.Tensor, return_log_density : bool = False, Q = None, R = None):
        """
        Inference of density of y (given x if conditional)

        Args:
            yx : torch Tensor of size (p, dy + dx) if conditional, otherwise (p, dy)
        """

        if self.conditional:
            assert yx.shape[1] == self.dy + self.dx
            y = yx[:, :self.dy]
            x = yx[:, self.dy:]

            phi_x_vals = self.phi(x)
            phi_y_vals = self.phi(y)
            psi_y_vals = self.psi(y)

            if Q is None or R is None:
                Q, R = self.get_QR_matrices()

            phi_psi_vals = phi_x_vals * psi_y_vals

            f = torch.einsum("pi,ij,pj->p", phi_psi_vals, Q, phi_psi_vals)
            g_x = torch.einsum("pi,ij,pj->p", phi_x_vals, R, phi_x_vals)
            g_y = torch.einsum("pi,ij,pj->p", phi_y_vals, R, phi_y_vals)

            g_x = torch.clamp(g_x, min=1e-10)
            g_y = torch.clamp(g_y, min=1e-10)

            #print("g_x: ", g_x)
            #print("g_y: ", g_y)
            #print("f: ", f)

            # Compute density in log space
            log_density = torch.log(f + 1e-8) + torch.log(g_y + 1e-8) - torch.log(g_x + 1e-8) 
            #if torch.any(torch.isnan(log_density)) or torch.any(torch.isinf(log_density)):
            #    print("log_density is nan or inf")
            #    print("f: ", f)
            #    print("g_y: ", g_y)
            #    print("g_x: ", g_x)
            #    input("...")
        else:
            assert yx.shape[1] == self.dy
            y = yx

            phi_y_vals = self.phi(y)
            psi_y_vals = self.psi(y)

            if Q is None or R is None:
                Q, R = self.get_QR_matrices()

            f = torch.einsum("pi,ij,pj->p", psi_y_vals, Q, psi_y_vals)
            g_y = torch.einsum("pi,ij,pj->p", phi_y_vals, R, phi_y_vals)

            # Compute density in log space
            log_density = torch.log(f) + torch.log(g_y)

        if return_log_density:
            torch.clamp(log_density, min=-50, max=100)
            #print("log_density: ", log_density)
            return log_density
        else:
            density = torch.exp(log_density)
            return density

    def loss(self, yx : torch.Tensor):
        assert not self.fixed_params
        Q, R = self.get_QR_matrices()
        #density = self(yx, return_log_density=True)
        log_density = self(yx, return_log_density=True, Q=Q, R=R)
        #log_density = torch.log(density + 1e-10)
        nll_loss = -log_density.mean()

        logdet_loss = self.logdet_barrier_loss(Q=Q, R=R)

        loss = nll_loss + logdet_loss

        #print("loss: \n", loss)
        #input("...")
        #if torch.isinf(loss):
        #    print("loss is nan")
        #    input("...")
        return loss, nll_loss, logdet_loss
    
    def logdet_barrier_loss(self, Q = None, R = None):
        assert not self.fixed_params
        if Q is None or R is None:
            Q, R = self.get_QR_matrices()


        eigvals = torch.linalg.eigvalsh(R)
        #print(" in loss eigvals: ", eigvals)
        if torch.all(eigvals > 1e-8):
            #print("R is PSD")
            return torch.relu(-self.mu * torch.logdet(R))
        else:
            #print("R is INFEASIBLE")
            penalty = self.npsd_penalty * torch.sum(torch.relu(-eigvals + 2e-0)**2)
            return self.mu * penalty
        
    #def fixed_point_rank_loss(self, s_min):
    #    return self.npsd_penalty * (torch.exp(s_min) - 1.0)
    
    def is_psd(self):
        _, R = self.get_QR_matrices()
        #Q_eigvals = torch.linalg.eigvalsh(Q)
        R_eigvals = torch.linalg.eigvalsh(R)
        return torch.all(R_eigvals > 1e-10)
    

    def fixed_point_residual(self):
        """
        Checks the fixed-point condition:
            R[i,j] == Q[i,j] * sum_{k,l} R[k,l] * E4[k,l,i,j]
        where E4 has shape (n,n,n,n) with axes (k,l,i,j).

        Returns:
            resid      : (n,n) tensor = LHS - RHS
            abs_norm   : Frobenius norm of resid
            rel_norm   : ||resid||_F / (||R||_F + 1e-12)
            max_abs    : max absolute entrywise residual
            rhs        : the RHS matrix for inspection
        """
        # Contract over (k,l): tmp[i,j] = sum_{k,l} R[k,l] * E4[k,l,i,j]
        Q, R = self.get_QR_matrices()
        E4 = self.gram_tensor()
        tmp = torch.tensordot(R, E4, dims=([0, 1], [0, 1]))    # (n,n)
        rhs = Q * tmp                                           # (n,n)
        resid = R - rhs
        abs_norm = torch.linalg.norm(resid)                     # Frobenius
        rel_norm = abs_norm / (torch.linalg.norm(R) + 1e-12)
        max_abs = resid.abs().max()
        return resid, abs_norm, rel_norm, max_abs, rhs
    
    def propagate(self, belief_model):
        assert self.conditional 
        assert isinstance(belief_model, SOSModel)
        
        cross_E4 = self.gram_tensor(cross_gram_model=belief_model)

        # TEST
        cross_E4 = cross_E4.permute(2, 3, 0, 1)

        Q_belief, R = belief_model.get_QR_matrices()
        Q_self, _ = self.get_QR_matrices()
        
        contracted = torch.einsum('kl,klij->ij', Q_belief, cross_E4)
        Q_new = Q_self * contracted
        
        belief_type = type(belief_model)
        new_belief_model = belief_type(dy=self.dy, 
                                    dx=0, 
                                    n=self.n, 
                                    conditional=False, 
                                    fixed_phi_params=self.get_phi_params(), 
                                    fixed_psi_params=self.get_psi_params(), 
                                    fixed_Q=Q_new, 
                                    fixed_R=R)
        
        return new_belief_model


def optimize(model : SOSModel, data_loader : DataLoader, optimizer, 
             epochs=100, 
             log_buffer_size=20, 
             use_best=True):
    torch.autograd.set_detect_anomaly(True)

    def train_step(data):
        model.train()
        optimizer.zero_grad()
        # loss
        loss, nll_loss, logdet_loss = model.loss(data)
        #print("loss: ", loss)
        #input("...")
        loss.backward()
        optimizer.step()
        return loss.item(), nll_loss.item(), logdet_loss.item()

    stdout_buffer = []
    best_loss = float("inf")
    best_state = None

    for epoch in range(epochs):
        start_time = time.time()
        total_loss = 0.0
        nll_loss_val, constraint_loss_val = 0.0, 0.0

        for x_batch in data_loader:
            # Handle the case where DataLoader returns a list of tensors
            if isinstance(x_batch, list):
                x_batch = x_batch[0].to(next(model.parameters()).device)
            else:
                x_batch = x_batch.to(next(model.parameters()).device)
            
            # Ensure x_batch is 2D
            if x_batch.dim() == 1:
                x_batch = x_batch.unsqueeze(0)
            loss, nll_loss, constraint_loss = train_step(x_batch)
            total_loss += loss
            nll_loss_val = nll_loss  # just track last batch for logging
            constraint_loss_val = constraint_loss

        avg_loss = total_loss / len(data_loader)

        is_psd = model.is_psd()
        # --- Save best model in RAM ---
        if use_best and avg_loss < best_loss and is_psd:
            best_loss = avg_loss
            best_state = copy.deepcopy(model.state_dict())

        # --- Update lagrangians if needed ---
        # if (epoch + 1) % lagrangian_update_interval == 0:
        #     model.update_lagrangians()

        line = (f"Epoch {epoch+1}: Avg Loss = {avg_loss:.4f}, "
                f"NLL Loss = {nll_loss_val:.4f}, "
                f"LDB Loss = {constraint_loss_val:.4f}, "
                f"is_psd = {is_psd}, "
                f"time: {time.time() - start_time:.3f}")

        stdout_buffer.append(line)
        if len(stdout_buffer) <= log_buffer_size:
            print(line)
        else:
            stdout_buffer.pop(0)
            sys.stdout.write("\033[F" * len(stdout_buffer))
            for l in stdout_buffer:
                sys.stdout.write("\033[K")
                print(l)

    # --- Restore best model before returning ---
    if use_best and best_state is not None:
        model.load_state_dict(best_state)
        print(f"\n Restored best model (loss={best_loss:.4f})")

    return model
    