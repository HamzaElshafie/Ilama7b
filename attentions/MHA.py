import torch
from torch import nn
from typing import Optional

class MultiHeadAttention(nn.Module):
    def __init__(self, d_model: int, num_heads: int, use_cache: Optional[bool] = False):
        super().__init__()
        self.d_model = d_model
        self.num_heads = num_heads
        self.use_cache = use_cache

        assert d_model % num_heads == 0, "d_model is not divisible by num_heads"

        self.head_dim  = d_model // num_heads

        self.q_proj = nn.Linear(d_model, d_model)
        self.k_proj = nn.Linear(d_model, d_model)
        self.v_proj = nn.Linear(d_model, d_model)
        self.o_proj = nn.Linear(d_model, d_model)

    def forward(self, x: torch.Tensor, causal_mask: Optional[torch.Tensor] = None):
        query_states = (
            self.q_proj(x) # (Batch, seq_len, d_model) --> (Batch, seq_len, d_model)
            .view(x.shape[0], x.shape[1], self.num_heads, self.head_dim) # (Batch, seq_len, d_model) --> (Batch, seq_len, num_heads, head_dim)
            .transpose(1, 2) # (Batch, seq_len, num_heads, head_dim) --> (Batch, c, seq_len, head_dim)
        )
        key_states = (
            self.k_proj(x) # (Batch, seq_len, d_model) --> (Batch, seq_len, d_model)
            .view(x.shape[0], x.shape[1], self.num_heads, self.head_dim) # (Batch, seq_len, d_model) --> (Batch, seq_len, num_heads, head_dim)
            .transpose(1, 2) # (Batch, seq_len, num_heads, head_dim) --> (Batch, c, seq_len, head_dim)
        )
        value_states = (
            self.v_proj(x) # (Batch, seq_len, d_model) --> (Batch, seq_len, d_model)
            .view(x.shape[0], x.shape[1], self.num_heads, self.head_dim) # (Batch, seq_len, d_model) --> (Batch, seq_len, num_heads, head_dim)
            .transpose(1, 2) # (Batch, seq_len, num_heads, head_dim) --> (Batch, c, seq_len, head_dim)
        )
        