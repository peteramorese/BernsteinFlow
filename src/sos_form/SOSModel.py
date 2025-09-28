import torch
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
                mu : float = 1.0,
                npsd_penalty : float = 1.0,
                min_Q_eigval : float = 1e-2
                ):
        """
        SOS form conditional density model for p(y | x)
        Args:
            dy : dimension of the support
            dx : dimension of the conditioner variable
            n : number of basis functions 
            phi_param_dim : dimension of the phi (x-basis) parameters
            psi_param_dim : dimension of the psi (y-basis) parameters
        """

        super().__init__()

        self.dy = dy
        self.dx = dx
        self.n = n
        self.phi_param_dim = phi_param_dim
        self.psi_param_dim = psi_param_dim
        self.mu = mu
        self.npsd_penalty = npsd_penalty
        self.min_Q_eigval = min_Q_eigval
        
        self.phi_params = torch.nn.Parameter(1 * torch.randn(self.n, phi_param_dim))
        self.psi_params = torch.nn.Parameter(1 * torch.randn(self.n, psi_param_dim)) 

        self.R_uc = torch.nn.Parameter(torch.randn(self.n, self.n))
        self.Q_uc = torch.nn.Parameter(torch.randn(self.n, self.n))

    #def __get_phi(self, x : torch.Tensor):
    #    phi_x_vals = self.phi(x)
    #    phi_x_vals = torch.cat([torch.ones_like(phi_x_vals[:, :1]), phi_x_vals], dim=1)
    #    return 
        

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

    def gram_tensor(self):
        """
        Compute the inner product matrix of the psi basis functions. Must return a tensor of size (n*m, n*m)
        """
        raise NotImplementedError()

    def get_QR_matrices(self, E4 : torch.Tensor = None):
        n = self.n

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

        #U, S, Vh = torch.linalg.svd(A)

        #s_min = torch.min(S)

        #null_mask = (S < 1e-8)
        #if null_mask.sum() == 0:
        #    raise RuntimeError(f"M has trivial nullspace (min eval: {s_min})")
        #    #print("NULL MATRIX IS ZERO")
        #    #return torch.zeros_like(R_u_sym), s_min
        #    #return torch.eye(n, dtype=Q.dtype, device=Q.device), s_min
        ##print("NULL MATRIX IS NOT ZERO")

        #N = Vh[null_mask].T  # shape (n^2, k)

        #r_u_vec = R_u_sym.reshape(-1)
        #coeffs = N.T @ r_u_vec
        #r_proj_vec = N @ coeffs

        #A_pinv = torch.linalg.pinv(A)
        #pinv_projector = torch.eye(A.shape[-1], dtype=A.dtype, device=A.device) - A_pinv @ A
        #r_proj_vec = pinv_projector @ R_u_sym.reshape(-1)

        r_proj_vec = self._project_null(A, R_u_sym.reshape(-1))

        R_proj = r_proj_vec.reshape(n, n)
        #print("R_proj: \n", R_proj)
        #input("...")

        # Optional: symmetrize (can be omitted if E4, Q are guaranteed symmetric)
        R_proj = 0.5 * (R_proj + R_proj.T)

        return Q, R_proj #s_min
    
    def _project_null(self, A : torch.Tensor, v : torch.Tensor, lam : float = 1e-8):
        Ax = A @ v                                 # (m,)
        G  = A @ A.T
        G  = G + lam * torch.eye(G.shape[-1], device=A.device, dtype=A.dtype)
        # Cholesky solve
        L = torch.linalg.cholesky(G)               # (m,m)
        y = torch.cholesky_solve(Ax.unsqueeze(-1), L).squeeze(-1)  # (m,)
        return v - A.T @ y

    def forward(self, yx : torch.Tensor, return_log_density : bool = False):
        """
        Inference of density given y and x

        Args:
            yx : torch Tensor of size (p, dy + dx)
        """

        assert yx.shape[1] == self.dy + self.dx
        y = yx[:, :self.dy]
        x = yx[:, self.dy:]

        phi_x_vals = self.phi(x)
        phi_y_vals = self.phi(y)
        psi_y_vals = self.psi(y)

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
        log_density = torch.log(f) + torch.log(g_y) - torch.log(g_x) 
        #print("g_x: ", g_x)
        #print("g_y: ", g_y)
        if return_log_density:
            return log_density
        else:
            #print("density: ", density)
            density = torch.exp(log_density)
            return density

    def loss(self, yx : torch.Tensor):
        #density = self(yx, return_log_density=True)
        log_density = self(yx, return_log_density=True)
        #log_density = torch.log(density + 1e-10)
        nll_loss = -log_density.mean()

        logdet_loss = self.logdet_barrier_loss()

        loss = nll_loss + logdet_loss

        #print("loss: \n", loss)
        #input("...")
        #if torch.isinf(loss):
        #    print("loss is nan")
        #    input("...")
        return loss, nll_loss, logdet_loss
    
    def logdet_barrier_loss(self):
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
        Q, R = self.get_QR_matrices()
        Q_eigvals = torch.linalg.eigvalsh(Q)
        R_eigvals = torch.linalg.eigvalsh(R)
        return torch.all(Q_eigvals > 1e-8) and torch.all(R_eigvals > 1e-8)
    

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
    
def optimize(model : SOSModel, data_loader : DataLoader, optimizer, epochs=100, lagrangian_update_interval=10, log_buffer_size = 20):
    torch.autograd.set_detect_anomaly(True)
    def train_step(data):
        model.train()
        optimizer.zero_grad()
        #loss
        loss, nll_loss, logdet_loss = model.loss(data)
        loss.backward()

        #print("Grads:")
        #for name, p in model.named_parameters():
        #    if p.grad is not None:
        #        print(name, torch.isnan(p.grad).any(), p.grad.norm().item())
        #print("\n")

        optimizer.step()
        return loss.item(), nll_loss.item(), logdet_loss.item()

    stdout_buffer = []

    #with torch.no_grad():
    #    A = model.get_A_mat()
    #    eigvals = torch.linalg.eigvalsh(A)
    #    print("Init Eigvals: ", eigvals)
    #    input("...")

    for epoch in range(epochs):
        start_time = time.time()
        total_loss = 0.0
        for x_batch in data_loader:
            x_batch = x_batch[0].to(next(model.parameters()).device)
            loss, nll_loss, constraint_loss = train_step(x_batch)
            total_loss += loss
        avg_loss = total_loss / len(data_loader)

        ## DEBUG
        #print("res mat: ", model.get_residual_mat())
        #input("...")



        #if (epoch + 1) % lagrangian_update_interval == 0:
        #    #print("updating...")
        #    model.update_lagrangians()
        #    #input("...")
        
        is_psd = model.is_psd()
        line = f"Epoch {epoch+1}: Avg Loss = {avg_loss:.4f}, NLL Loss = {nll_loss:.4f}, LDB Loss = {constraint_loss:.4f}, is_psd = {is_psd}, time: {time.time() - start_time:.3f}"

        stdout_buffer.append(line)
        if len(stdout_buffer) <= log_buffer_size:
            print(line)
        else:
            stdout_buffer.pop(0)
            sys.stdout.write("\033[F" * len(stdout_buffer))
            for l in stdout_buffer:
                sys.stdout.write("\033[K")
                print(l)
