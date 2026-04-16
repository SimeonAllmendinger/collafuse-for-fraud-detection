import torch
import math


def get_sinusoidal_embedding(timesteps, embedding_dim):
    """
    Returns sinusoidal embeddings for a batch of time steps.
    timesteps: Tensor of shape [batch_size]
    embedding_dim: Size of the output embedding vector
    """
    device = timesteps.device
    half_dim = embedding_dim // 2
    emb = math.log(10000) / (half_dim - 1)
    emb = torch.exp(torch.arange(half_dim, device=device) * -emb)
    emb = timesteps[:, None] * emb[None, :]  # [B, half_dim]
    emb = torch.cat([torch.sin(emb), torch.cos(emb)], dim=1)
    if embedding_dim % 2 == 1:  # zero pad if embedding_dim is odd
        emb = torch.cat([emb, torch.zeros(timesteps.size(0), 1, device=device)], dim=1)
    return emb
