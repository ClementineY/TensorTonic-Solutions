import torch
import torch.nn.functional as F
import math

def scaled_dot_product_attention(Q: torch.Tensor, K: torch.Tensor, V: torch.Tensor) -> torch.Tensor:
    """
    Compute scaled dot-product attention.
    """
    a = torch.einsum("ntd, nsd -> nts", Q, K) / math.sqrt(Q.shape[-1])
        
    a = F.softmax(a, dim=-1)
    
    return torch.einsum("nts, nsd -> ntd", a, V)