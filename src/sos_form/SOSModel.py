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
                min_Q_eigval : float = 1e-4
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
        
        self.phi_params = torch.nn.Parameter(0.1 * torch.randn(self.n, phi_param_dim))
        self.psi_params = torch.nn.Parameter(0.1 * torch.randn(self.n, psi_param_dim)) 

        self.Q_uc = torch.nn.Parameter(torch.randn(self.n, self.n))
        self.R_uc = torch.nn.Parameter(torch.randn(self.n, self.n))


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

        # Compute the max eigenvalue of M to rescale Q. All e-vals of Q are guaranteed to be real and non-negative.
        lambda_M_max = torch.max(torch.real(torch.linalg.eigvals(M)))

        # Rescale Q and M to make the nullspace of M non trivial
        Q = Q_unscaled / lambda_M_max
        M = M / lambda_M_max

        I = torch.eye(n*n, dtype=Q.dtype, device=Q.device)
        A = I - M

        U, S, Vh = torch.linalg.svd(A)

        s_min = torch.min(S)

        null_mask = (S < 1e-8)
        if null_mask.sum() == 0:
            raise RuntimeError(f"M has trivial nullspace (min eval: {s_min})")
            #print("NULL MATRIX IS ZERO")
            #return torch.zeros_like(R_u_sym), s_min
            #return torch.eye(n, dtype=Q.dtype, device=Q.device), s_min
        #print("NULL MATRIX IS NOT ZERO")

        N = Vh[null_mask].T  # shape (n^2, k)

        r_u_vec = R_u_sym.reshape(-1)
        coeffs = N.T @ r_u_vec
        r_proj_vec = N @ coeffs

        R_proj = r_proj_vec.reshape(n, n)
        #print("R_proj: \n", R_proj)
        #input("...")

        # Optional: symmetrize (can be omitted if E4, Q are guaranteed symmetric)
        R_proj = 0.5 * (R_proj + R_proj.T)

        return Q, R_proj #s_min

    def forward(self, yx : torch.Tensor):
        """
        Inference of density given y and x

        Args:
            yx : torch Tensor of size (p, dy + dx)
        """

        assert yx.shape[1] == self.dy + self.dx
        y = yx[:, :self.dy]
        x = yx[:, self.dy:]

        phi_vals = self.phi(x)
        psi_vals = self.psi(y)

        Q, R = self.get_QR_matrices()

        phi_psi_vals = phi_vals * psi_vals

        f = torch.einsum("pi,ij,pj->p", phi_psi_vals, Q, phi_psi_vals)
        g_x = torch.einsum("pi,ij,pj->p", phi_vals, R, phi_vals)
        g_y = torch.einsum("pi,ij,pj->p", psi_vals, R, psi_vals)

        #print("g_x: ", g_x)
        #print("g_y: ", g_y)
        #print("f: ", f)
        return torch.relu(f * g_y / g_x)


    def loss(self, yx : torch.Tensor):
        density = self(yx)
        log_density = torch.log(density + 1e-10)
        nll_loss = -log_density.mean()

        logdet_loss = self.logdet_barrier_loss()

        loss = nll_loss + logdet_loss

        #print("loss: \n", loss)
        return loss, nll_loss, logdet_loss
    
    def logdet_barrier_loss(self):
        Q, R = self.get_QR_matrices()


        eigvals = torch.linalg.eigvalsh(R)
        #print(" in loss eigvals: ", eigvals)
        if torch.all(eigvals > 1e-8):
            #print("R is PSD")
            return -self.mu * torch.logdet(R)
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
    

    
def optimize(model : SOSModel, data_loader : DataLoader, optimizer, epochs=100, lagrangian_update_interval=10, log_buffer_size = 20):
    def train_step(data):
        model.train()
        optimizer.zero_grad()
        #loss
        loss, nll_loss, logdet_loss = model.loss(data)
        loss.backward()
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
