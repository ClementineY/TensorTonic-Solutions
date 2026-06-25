import numpy as np

def softmax(x, axis=-1):
    e_x = np.exp(x - np.max(x, axis=axis, keepdims=True))
    return e_x / np.sum(e_x, axis=axis, keepdims=True)

def multi_head_attention(Q: np.ndarray, K: np.ndarray, V: np.ndarray,
                         W_q: np.ndarray, W_k: np.ndarray, W_v: np.ndarray,
                         W_o: np.ndarray, num_heads: int) -> np.ndarray:
    """
    Compute multi-head attention.
    """
    batch_size, n, dmodel = Q.shape
    h = dmodel // num_heads
    
    Q_prime = np.dot(Q, W_q)
    K_prime = np.dot(K, W_k)
    V_prime = np.dot(V, W_v)
    
    Q_heads = Q_prime.reshape(batch_size, n, num_heads, h).transpose(0,2,1,3)
    K_heads = K_prime.reshape(batch_size, n, num_heads, h).transpose(0,2,1,3)
    V_heads = V_prime.reshape(batch_size, n, num_heads, h).transpose(0,2,1,3)
    
    scores = (Q_heads @ K_heads.transpose(0, 1, 3, 2)) / (h ** 0.5)
    attention_weights = softmax(scores)
    head_outputs = attention_weights @ V_heads
    concat_out = head_outputs.transpose(0, 2, 1, 3).reshape(batch_size, n, dmodel)
    out = concat_out @ W_o
    return out
    
    