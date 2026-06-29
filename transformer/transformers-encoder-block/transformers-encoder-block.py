import numpy as np

def softmax(x, axis=-1):
    """Provided: Softmax function."""
    e_x = np.exp(x - np.max(x, axis=axis, keepdims=True))
    return e_x / np.sum(e_x, axis=axis, keepdims=True)

def layer_norm(x: np.ndarray, gamma: np.ndarray, beta: np.ndarray, eps: float = 1e-6) -> np.ndarray:
    """
    Apply layer normalization.
    """
    mu = np.mean(x, axis = -1, keepdims = True)
    var = np.var(x, axis = -1, keepdims = True)
    return gamma * ((x-mu)/(var+eps)**0.5) + beta

def multi_head_attention(Q: np.ndarray, K: np.ndarray, V: np.ndarray,
                         W_q: np.ndarray, W_k: np.ndarray, W_v: np.ndarray,
                         W_o: np.ndarray, num_heads: int) -> np.ndarray:
    """
    Multi-head attention.
    """
    batch, seq_len, d_model = Q.shape
    d_k = d_model // num_heads
    Q_prime = Q @ W_q
    K_prime = K @ W_k
    V_prime = V @ W_v
    Q_trans = Q_prime.reshape(batch, seq_len, num_heads, d_k).transpose(0,2,1,3)
    K_trans = K_prime.reshape(batch, seq_len, num_heads, d_k).transpose(0,2,1,3)
    V_trans = V_prime.reshape(batch, seq_len, num_heads, d_k).transpose(0,2,1,3)
    scores = Q_trans @ K_trans.transpose(0,1,3,2)
    scaled_scores = scores / d_k**0.5
    head_out = softmax(scaled_scores) @ V_trans
    concat_out = head_out.transpose(0,2,1,3).reshape(batch, seq_len, d_model)
    out = concat_out @ W_o
    return out

def feed_forward(x: np.ndarray, W1: np.ndarray, b1: np.ndarray,
                 W2: np.ndarray, b2: np.ndarray) -> np.ndarray:
    """
    Position-wise feed-forward network.
    """
    hidden = x @ W1 + b1
    relu_o = np.maximum(0, hidden)
    o = relu_o @ W2 + b2
    return o

def encoder_block(x: np.ndarray, W_q: np.ndarray, W_k: np.ndarray, W_v: np.ndarray,
                  W_o: np.ndarray, W1: np.ndarray, b1: np.ndarray, W2: np.ndarray,
                  b2: np.ndarray, gamma1: np.ndarray, beta1: np.ndarray,
                  gamma2: np.ndarray, beta2: np.ndarray, num_heads: int) -> np.ndarray:
    """
    Complete encoder block: MHA + FFN with residuals and layer norms.
    """
    eps = 1e-6

    # 1. Multi-Head Attention
    Q_prime = x @ W_q
    K_prime = x @ W_k
    V_prime = x @ W_v
    
    batch, seq_len, d_model = Q_prime.shape  
    d_k = d_model // num_heads
    
    Q_trans = Q_prime.reshape(batch, seq_len, num_heads, d_k).transpose(0, 2, 1, 3)
    K_trans = K_prime.reshape(batch, seq_len, num_heads, d_k).transpose(0, 2, 1, 3)
    V_trans = V_prime.reshape(batch, seq_len, num_heads, d_k).transpose(0, 2, 1, 3)
    
    scores = Q_trans @ K_trans.transpose(0, 1, 3, 2)
    scaled_scores = scores / d_k**0.5
    head_out = softmax(scaled_scores) @ V_trans
    concat_out = head_out.transpose(0, 2, 1, 3).reshape(batch, seq_len, d_model)
    attn_out = concat_out @ W_o
    
    # Residual Connection 1 + Layer Norm 1
    x_residual1 = x + attn_out
    mu1 = np.mean(x_residual1, axis=-1, keepdims=True)
    var1 = np.var(x_residual1, axis=-1, keepdims=True)
    layer_norm1_out = gamma1 * ((x_residual1 - mu1) / (var1 + eps)**0.5) + beta1
    
    # 2. Feed-Forward Network
    hidden = layer_norm1_out @ W1 + b1
    relu_o = np.maximum(0, hidden)
    ffn_o = relu_o @ W2 + b2
    
    # Residual Connection 2 + Layer Norm 2 
    x_residual2 = layer_norm1_out + ffn_o
    mu2 = np.mean(x_residual2, axis=-1, keepdims=True)
    var2 = np.var(x_residual2, axis=-1, keepdims=True)
    layer_norm2_out = gamma2 * ((x_residual2 - mu2) / (var2 + eps)**0.5) + beta2
    
    return layer_norm2_out
    
    