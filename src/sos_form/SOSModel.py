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
                m : int, 
                phi_param_dim : int, 
                psi_param_dim : int, 
                gamma : float = 1.1, 
                eta : float = 0.25,
                sigma_init : float = 3.0,
                sigma_max : float = 100.0):
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
        self.sigma_max = sigma_max
        # Initialize with smaller values to prevent explosion
        self.phi_params = torch.nn.Parameter(0.1 * torch.randn(self.n - 1, phi_param_dim))
        self.psi_params = torch.nn.Parameter(0.1 * torch.randn(self.n*self.m, psi_param_dim)) 

        self.L_a = torch.nn.Parameter(0.1 * torch.randn(self.n*self.m, self.n*self.m))
        print("Coefficient matrix size: ", self.L_a.shape)

        # Augmented Lagrangian multipliers
        self.register_buffer("lagr_mult", torch.zeros(self.n, self.n)) # Linear penalty multipliers for each equality block
        self.sigma = sigma_init # Quadratic penalty multiplier
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

        #print("A mat:\n", A_mat)
        #print("Phi: ", phi_vec[0, :])
        #print("Psi: ", psi_vec[0, :])
        #print("basis vals: ", basis_vals)

        # Per-sample quadratic form: for each sample i, basis_vals[i]^T A basis_vals[i]
        density = torch.einsum("pi,ij,pj->p", basis_vals, A_mat, basis_vals)
        
        return density 
        
    def get_A_mat(self):
        return self.L_a @ self.L_a.T
    
    def get_residual_mat(self):
        psi_mat = self.psi_inner_product_mat()
        Gamma = psi_mat * self.get_A_mat() # Hadamard product between psi inner product mat and A

        # Reshape Gamma to group blocks and sum over each block
        # Gamma shape: (n, m, n, m) -> (n, n)
        Gamma_block_view = Gamma.view(self.n, self.m, self.n, self.m)

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
        #print("nll loss: ", nll_loss.item(),"aug lagrangian loss: ", aug_lagrangian_loss.item())
        loss = nll_loss + aug_lagrangian_loss
        #input("...")
        return loss, nll_loss, aug_lagrangian_loss
    
    def update_lagrangians(self):
        #print("Updating lagrangians...")
        with torch.no_grad():
            residuals = self.get_residual_mat()
            #print("residuals: ", residuals)

            # Lagrange update iteration
            v = torch.sum(residuals**2)
            #print("v: ", v)
            if v  < self.eta * self.v:
                self.lagr_mult -= self.sigma * residuals
            else:
                self.sigma = min(self.sigma_max, self.sigma *self.gamma)

            #print("sigma: ", self.sigma)

            self.v = v
    
    def get_v(self):
        return self.v

    def project_constraints(self):
        with torch.no_grad():
            A_proj, prob = project_psd_hadamard_blocksum(self.get_A_mat().cpu().numpy(), 
                                                        self.psi_inner_product_mat().cpu().numpy(), 
                                                        self.n, 
                                                        self.m,
                                                        verbose=True)
            #print("A proj: ", A_proj)
            eigvals, _ = np.linalg.eig(A_proj)
            print("Eigvals: ", eigvals)
            L = np.linalg.cholesky(A_proj)
            self.L_a.copy_(torch.tensor(L, dtype=self.L_a.dtype, device=self.L_a.device))


    
def optimize(model : SOSModel, data_loader : DataLoader, optimizer, epochs=100, lagrangian_update_interval=10, log_buffer_size = 20, constraints_only=False):
    def train_step(data):
        model.train()
        optimizer.zero_grad()
        if constraints_only:
            aug_lagrangian_loss = model.aug_lagrangian_loss()
            aug_lagrangian_loss.backward()
            optimizer.step()
            with torch.no_grad():
                loss, nll_loss, _ = model.loss(data)
            return loss.item(), nll_loss.item(),aug_lagrangian_loss.item()
        else:
            loss, nll_loss, aug_lagrangian_loss = model.loss(data)
            loss.backward()
            optimizer.step()
            return loss.item(), nll_loss.item(), aug_lagrangian_loss.item()

    stdout_buffer = []


    for epoch in range(epochs):
        start_time = time.time()
        total_loss = 0.0
        for x_batch in data_loader:
            x_batch = x_batch[0].to(next(model.parameters()).device)
            loss, nll_loss, aug_lagrangian_loss = train_step(x_batch)
            total_loss += loss
        avg_loss = total_loss / len(data_loader)

        if (epoch + 1) % lagrangian_update_interval == 0:
            #print("updating...")
            model.update_lagrangians()
            #input("...")
        
        line = f"Epoch {epoch+1}: Avg Loss = {avg_loss:.4f}, NLL Loss = {nll_loss:.4f}, AL Loss = {aug_lagrangian_loss:.4f}, v = {model.get_v():.6f}, sigma = {model.sigma:.4f}, time: {time.time() - start_time:.3f}"
        stdout_buffer.append(line)
        if len(stdout_buffer) <= log_buffer_size:
            print(line)
        else:
            stdout_buffer.pop(0)
            sys.stdout.write("\033[F" * len(stdout_buffer))
            for l in stdout_buffer:
                sys.stdout.write("\033[K")
                print(l)


def project_psd_hadamard_blocksum(A0: np.ndarray, Gamma: np.ndarray, n: int, m: int,
                                  verbose: bool = False):
    """
    Solve:
        minimize   ||A - A0||_F^2  
        subject to A is PSD (A >> 0)
                   For every (i,j) block of size m x m, sum( (A .* Gamma)[block] ) = 0

    Args
    ----
    A0 : (N,N) numpy array, target matrix (symmetric recommended)
    Gamma : (N,N) numpy array, fixed PSD matrix
    n : number of block rows (and block cols)
    m : block size (so N = n*m)
    verbose : print solver info

    Returns
    -------
    A_opt : optimal matrix A (numpy array) or None if infeasible
    prob  : the CVXPY problem object
    """
    N = n * m
    assert A0.shape == (N, N) and Gamma.shape == (N, N), "Shapes must be (n*m, n*m)"
    # Symmetrize inputs defensively
    A0 = 0.5 * (A0 + A0.T)
    Gamma = 0.5 * (Gamma + Gamma.T)

    # Decision variable
    A = cp.Variable((N, N), PSD=True)

    # PSD constraint
    #constraints = [A >> 0]
    constraints = []

    # Build linear equality constraints:
    # For each (i,j) block, sum of entries of (A ∘ Gamma) in that block equals 0 (except for the 1, 1 block which sums to 1)
    # This is a linear constraint:  <A, M_ij> = 0  where M_ij has Gamma's entries on that block, 0 elsewhere.
    for bi in range(n):
        for bj in range(n):
            rows = slice(bi*m, (bi+1)*m)
            cols = slice(bj*m, (bj+1)*m)
            M = np.zeros((N, N))
            M[rows, cols] = Gamma[rows, cols]
            
            sum_val = 0.0 if (bi > 0 or bj > 0) else 1.0
            constraints += [cp.sum(cp.multiply(A, M)) == sum_val]

    # Objective: keep A close to A0
    obj = cp.Minimize(cp.sum_squares(A - A0))

    prob = cp.Problem(obj, constraints)
    prob.solve(solver=cp.SCS, verbose=verbose, eps=1e-6)  # You can also try MOSEK/OSQP/SDPT3 if available

    if A.value is None:
        return None, prob
    # Symmetrize numerical noise
    A_opt = 0.5 * (A.value + A.value.T)
    return A_opt, prob