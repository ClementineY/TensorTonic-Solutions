import torch
import torch.nn as nn
import math

def create_embedding_layer(vocab_size: int, d_model: int) -> nn.Embedding:
    """
    Create an embedding layer.
    """
    embedding_layer = nn.Embedding(num_embeddings=vocab_size, embedding_dim=d_model)
    
    return embedding_layer

def embed_tokens(embedding: nn.Embedding, tokens: torch.Tensor, d_model: int) -> torch.Tensor:
    """
    Convert token indices to scaled embeddings.
    """
    embedded_output = embedding(tokens)
    scaled_output = embedded_output * math.sqrt(d_model)
    return scaled_output