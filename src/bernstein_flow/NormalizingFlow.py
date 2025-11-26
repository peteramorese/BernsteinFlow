import torch
import torch.nn as nn
import torch.optim as optim
import time
import sys

from torch.utils.data import DataLoader, TensorDataset

from nflows.flows import Flow
from nflows.distributions import StandardNormal
from nflows.transforms import CompositeTransform, ReversePermutation
from nflows.transforms.autoregressive import MaskedAffineAutoregressiveTransform


# ------------------------------------------------------------
# Flow construction
# ------------------------------------------------------------

def create_conditional_transform(
    dim_y: int,
    dim_x: int,
    hidden_features: int = 64,
    num_blocks: int = 2,
):
    """
    Create a single conditional Masked Affine Autoregressive Transform T(y; x).

    Args:
        dim_y: dimension of target variable y.
        dim_x: dimension of conditioning variable x.
        hidden_features: width of hidden layers in the autoregressive net.
        num_blocks: number of residual blocks in the autoregressive net.

    Returns:
        A nflows MaskedAffineAutoregressiveTransform with context on x.
    """
    transform = MaskedAffineAutoregressiveTransform(
        features=dim_y,
        hidden_features=hidden_features,
        context_features=dim_x,
        num_blocks=num_blocks,
        use_batch_norm=False,
    )
    return transform


def create_conditional_flow(
    dim_x: int,
    dim_y: int,
    num_layers: int = 5,
    hidden_features: int = 64,
    device: str = "cpu",
) -> Flow:
    """
    Create a conditional normalizing flow modeling p(y | x).

    Args:
        dim_x: dimension of conditioning variable x.
        dim_y: dimension of target variable y.
        num_layers: number of autoregressive layers (plus permutations).
        hidden_features: width of hidden layers in each transform.
        device: "cpu" or "cuda".

    Returns:
        nflows Flow object.
    """
    transforms = []
    for _ in range(num_layers):
        transforms.append(
            create_conditional_transform(
                dim_y=dim_y,
                dim_x=dim_x,
                hidden_features=hidden_features,
                num_blocks=2,
            )
        )
        # Simple permutation between layers
        transforms.append(ReversePermutation(features=dim_y))

    transform = CompositeTransform(transforms)
    base_distribution = StandardNormal(shape=[dim_y])

    flow = Flow(transform=transform, distribution=base_distribution).to(device)
    return flow


# ------------------------------------------------------------
# Wrapper module
# ------------------------------------------------------------

class ConditionalNormalizingFlow(nn.Module):
    """
    Wrapper around nflows Flow to model p(y | x) and provide
    log_prob and sampling interfaces.

    y ~ flow(x)
    """

    def __init__(
        self,
        dim_x: int,
        dim_y: int,
        num_layers: int = 5,
        hidden_features: int = 64,
        device: str = "cpu",
    ):
        super().__init__()
        self.dim_x = dim_x
        self.dim_y = dim_y
        self.device = device

        self.flow = create_conditional_flow(
            dim_x=dim_x,
            dim_y=dim_y,
            num_layers=num_layers,
            hidden_features=hidden_features,
            device=device,
        )

    def log_prob(self, y: torch.Tensor, x: torch.Tensor) -> torch.Tensor:
        """
        Compute log p(y | x) for each pair (y_i, x_i).

        Args:
            y: tensor of shape (batch, dim_y)
            x: tensor of shape (batch, dim_x)

        Returns:
            log_prob: tensor of shape (batch,)
        """
        y = y.to(self.device)
        x = x.to(self.device)
        return self.flow.log_prob(inputs=y, context=x)

    def sample(self, x: torch.Tensor, num_samples_per_x: int = 1) -> torch.Tensor:
        """
        Sample y ~ p_theta(y | x).

        Args:
            x: tensor of shape (batch, dim_x)
            num_samples_per_x: number of samples per conditioning x.

        Returns:
            If num_samples_per_x == 1:
                (batch, dim_y)
            else:
                (batch, num_samples_per_x, dim_y)
        """
        x = x.to(self.device)

        # Ensure x is 2D: (batch, dim_x)
        if x.ndim == 1:
            x = x.unsqueeze(0)
        elif x.ndim > 2:
            x = x.view(-1, x.shape[-1])

        # nflows: (context_size, num_samples, features)
        y = self.flow.sample(num_samples=num_samples_per_x, context=x)
        # y shape: (batch, num_samples_per_x, dim_y)

        if num_samples_per_x == 1:
            # (batch, 1, dim_y) -> (batch, dim_y)
            y = y[:, 0, :]

        return y

# ------------------------------------------------------------
# Training functions
# ------------------------------------------------------------

def train_step(model, data_batch, optimizer):
    """
    Single training step for conditional normalizing flow.
    
    Args:
        model: ConditionalNormalizingFlow instance
        data_batch: tensor of shape (batch, dim_x + dim_y) containing [x, x'] concatenated
                   where x is conditioning variable and x' is target variable
        optimizer: optimizer instance
    """
    model.train()
    optimizer.zero_grad()
    
    # Split data into conditioning (x) and target (x')
    # Data format: [x, x'] for modeling p(x' | x)
    x_batch = data_batch[:, :model.dim_x].to(model.device)
    x_prime_batch = data_batch[:, model.dim_x:].to(model.device)
    
    # Compute negative log-likelihood: -log p(x' | x)
    log_prob = model.log_prob(x_prime_batch, x_batch)
    loss = -log_prob.mean()
    
    loss.backward()
    optimizer.step()
    
    return loss.item()


def optimize(model, data_loader: DataLoader, optimizer, epochs=100,
             log_buffer_size=20,
             print_interval=None,
             scheduler=None):
    """
    Optimize a conditional normalizing flow model.
    
    Args:
        model: ConditionalNormalizingFlow instance
        data_loader: DataLoader providing batches of shape (batch, dim_x + dim_y)
                     Data should be [x, x'] concatenated for modeling p(x' | x)
        optimizer: optimizer instance
        epochs: number of training epochs
        log_buffer_size: number of lines to keep in stdout buffer
        print_interval: if not None, print every print_interval epochs instead of using buffer
        scheduler: optional learning rate scheduler (e.g., ReduceLROnPlateau)
    
    Note:
        The data format assumes [x, x'] concatenated, where:
        - x (first dim_x columns) is the conditioning variable
        - x' (remaining dim_y columns) is the target variable
        This models p(x' | x).
    """
    stdout_buffer = []
    
    model.train()
    for epoch in range(epochs):
        start_time = time.time()
        total_loss = 0.0
        n_batches = 0
        for data_batch in data_loader:
            # Handle both TensorDataset format (tuple) and raw tensor format
            if isinstance(data_batch, (list, tuple)):
                data_batch = data_batch[0]
            data_batch = data_batch.to(next(model.parameters()).device)
            
            loss = train_step(model, data_batch, optimizer)
            total_loss += loss
            n_batches += 1
        
        avg_loss = total_loss / n_batches
        
        # Update learning rate scheduler if provided
        if scheduler is not None:
            if isinstance(scheduler, torch.optim.lr_scheduler.ReduceLROnPlateau):
                scheduler.step(avg_loss)
            else:
                scheduler.step()
        
        # Format line with learning rate if available
        lr_str = f", LR = {optimizer.param_groups[0]['lr']:.6f}" if scheduler is not None else ""
        line = f"Epoch {epoch+1}/{epochs}: Avg Loss = {avg_loss:.6f}{lr_str}, time: {time.time() - start_time:.3f}"
        
        if print_interval is not None:
            # Simple printing mode: print every print_interval epochs
            if (epoch + 1) % print_interval == 0 or epoch == 0:
                print(line)
        else:
            # Fancy stdout buffer rewriting mode
            stdout_buffer.append(line)
            if len(stdout_buffer) <= log_buffer_size:
                print(line)
            else:
                stdout_buffer.pop(0)
                sys.stdout.write("\033[F" * len(stdout_buffer))
                for l in stdout_buffer:
                    sys.stdout.write("\033[K")
                    print(l)
    
    model.eval()


def main():
    # Example dimensions
    dim_x = 5   # dimension of conditioning variable x
    dim_y = 3   # dimension of target variable y

    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Using device: {device}")

    # Create model
    model = ConditionalNormalizingFlow(
        dim_x=dim_x,
        dim_y=dim_y,
        num_layers=5,
        hidden_features=128,
        device=device,
    )

    optimizer = optim.Adam(model.parameters(), lr=1e-3)

    # --------------------------------------------------------
    # Create dummy training data (x, y) just for demonstration
    # --------------------------------------------------------
    N = 20000
    x_data = torch.randn(N, dim_x)

    # Some arbitrary nonlinear relationship y = f(x) + noise
    # (you would replace this with your real data)
    y_base = torch.stack(
        [
            x_data[:, 0] ** 2,
            torch.sin(x_data[:, 1]),
            x_data[:, 2] * x_data[:, 3],
        ],
        dim=-1,
    )
    y_data = y_base + 0.1 * torch.randn(N, dim_y)

    dataset = TensorDataset(x_data, y_data)
    loader = DataLoader(dataset, batch_size=256, shuffle=True)

    # --------------------------------------------------------
    # Training loop (maximum likelihood)
    # --------------------------------------------------------
    num_epochs = 10

    model.train()
    for epoch in range(num_epochs):
        total_loss = 0.0
        for x_batch, y_batch in loader:
            x_batch = x_batch.to(device)
            y_batch = y_batch.to(device)

            optimizer.zero_grad()
            # log p(y | x)
            log_prob = model.log_prob(y_batch, x_batch)  # (batch,)
            loss = -log_prob.mean()  # negative log-likelihood

            loss.backward()
            optimizer.step()

            total_loss += loss.item() * x_batch.size(0)

        avg_loss = total_loss / len(dataset)
        print(f"Epoch {epoch + 1}/{num_epochs}, NLL: {avg_loss:.4f}")

    # --------------------------------------------------------
    # Sampling from p(y | x)
    # --------------------------------------------------------
    model.eval()

    # Suppose we have some conditioning inputs x_cond
    x_cond = torch.randn(4, dim_x)  # e.g., 4 different x's

    # One sample per x
    with torch.no_grad():
        y_samples = model.sample(x_cond)  # shape: (4, dim_y)
        print("\nOne sample per x:")
        print(y_samples)

        # Multiple samples per x (e.g. Monte Carlo from p(y | x))
        y_samples_multi = model.sample(x_cond, num_samples_per_x=5)
        print("\nMultiple samples per x (shape):", y_samples_multi.shape)
        # shape: (4, 5, dim_y)


if __name__ == "__main__":
    main()