import itertools
import torch
import torch.fft
from functools import reduce
import operator

def integrate_weighted_product(p_list, q):
    """
    Compute R(y) = ∫_{x∈[0,1]^d} q(x) * ∏_i p_i(y,x) dx
    Returns R(y) as a coefficient tensor in y
    
    Args:
        p_list: list of d tensors of shape (y₁_deg+1,...,y_d_deg+1, x₁_deg+1,...,x_d_deg+1)
        q:      tensor of shape (x₁_deg+1, ..., x_d_deg+1)
    Returns:
        R: tensor of shape (sum of y-degrees + 1)
    """
    d = q.ndim
    for i, p in enumerate(p_list):
        if p.ndim != 2 * d:
            raise ValueError(f"Each p_i must have 2d dimensions; p_{i} has shape {p.shape}")

    # Compute total output degree in y: sum of degrees from each p_i
    y_shapes = [p.shape[:d] for p in p_list]
    out_y_shape = [sum(s[k] - 1 for s in y_shapes) + 1 for k in range(d)]

    # Compute convolution output shape in x
    x_shapes = [p.shape[d:] for p in p_list]
    q_shape = q.shape
    S = [sum(s[k] - 1 for s in x_shapes) + q_shape[k] for k in range(d)]

    #fft_dims = tuple(range(-2 * d, 0))
    fft_dims = tuple(range(0, 2 * d))
    fft_shape = out_y_shape + S
    print("fft shape: ", fft_shape)

    # Pad all p_i to full output y shape
    P_fft = []
    for p, y_shape in zip(p_list, y_shapes):
        # Pad in reverse order
        #pad_x = [] 
        pad_p = []
        for k in reversed(range(d)): # x dims
            cur = p.shape[-d + k]
            pad_p.extend([0, S[k] - cur])
        #pad_y = []
        for k in reversed(range(d)): # y dims
            cur = y_shape[k]
            pad_p.extend([0, out_y_shape[k] - cur])
        #pad = pad_x + pad_y 
        print("pad: ", pad_p)
        p_padded = torch.nn.functional.pad(p, pad_p)
        print("p padded:" , p_padded, " shape: ", p_padded.shape)
        P_fft.append(torch.fft.rfftn(p_padded, s=fft_shape, dim=fft_dims))

    # Extend q to be dimension of p
    q = q.view((1,) * d + q.shape)

    # Pad and FFT q
    pad_q = []
    for k in reversed(range(d)): # x dims
        cur = q_shape[k]
        pad_q.extend([0, S[k] - cur])
    for k in reversed(range(d)): # y dims
        pad_q.extend([0, out_y_shape[k] - 1])
    print("pad q: ", pad_q)
    q_padded = torch.nn.functional.pad(q, pad_q)
    print("q_padded:", q_padded)
    Q_fft = torch.fft.rfftn(q_padded, s=fft_shape, dim=fft_dims)

    # Multiply in frequency domain
    print("fft shape: ", Q_fft.shape)
    F = Q_fft
    for P in P_fft:
        F = F * P

    # Inverse FFT
    C = torch.fft.irfftn(F, s=fft_shape, dim=fft_dims)  # shape: (*out_y_shape, *S)
    print("output shape: ", C.shape)
    print("output: ", C.round().int())

    # Integration weights
    grids = torch.meshgrid(
        *[torch.arange(s, dtype=C.dtype, device=C.device) for s in S],
        indexing='ij'
    )
    W = torch.ones_like(C)
    for g in grids:
        W *= 1.0 / (g + 1.0)

    # Multiply and reduce over x-dims
    R = (C * W).sum(dim=tuple(range(-d, 0)))  # shape: out_y_shape
    return R

if __name__ == "__main__":
    p_list = [
        torch.tensor([[1, 2, 3], [4, 5, 6], [7, 8, 9]], dtype=torch.float32),
        torch.tensor([[7, 8, 9], [10, 11, 12], [13, 14, 15]], dtype=torch.float32)
    ]

    q = torch.tensor([1, 2, 3, 4, 5], dtype=torch.float32)

    r = integrate_weighted_product(p_list, q)
    print(r)