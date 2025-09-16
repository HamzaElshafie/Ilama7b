import torch
import math
from torch import nn
from typing import Optional

def apply_rope(x: torch.Tensor):
    pass

def update_kv_cache(key_states: torch.Tensor, value_states: torch.Tensor):
    pass

class MultiHeadAttention(nn.Module):
    def __init__(self, d_model: int, num_heads: int, use_cache: Optional[bool] = False):
        super().__init__()
        self.d_model = d_model
        self.num_heads = num_heads
        self.use_cache = use_cache

        assert d_model % num_heads == 0, "d_model is not divisible by num_heads"

        self.head_dim  = d_model // num_heads
        self.scale = 1 / math.sqrt(self.head_dim)

        self.q_proj = nn.Linear(d_model, d_model)
        self.k_proj = nn.Linear(d_model, d_model)
        self.v_proj = nn.Linear(d_model, d_model)
        self.o_proj = nn.Linear(d_model, d_model)
    
    def attention(self, query_states, key_states, value_states, causal_mask: Optional[torch.Tensor] = None):
        attention_scores = query_states @ key_states.transpose(-2, -1) * self.scale # (Batch, num_heads, seq_len, seq_len)

        if causal_mask is not None:
            attention_scores = attention_scores.masked_fill(causal_mask == 0, -1e9)

        attention_scores = attention_scores.softmax(dim=-1)

        return attention_scores @ value_states

    def forward(self, x: torch.Tensor, causal_mask: Optional[torch.Tensor] = None):
        batch, seq_len, d_model = x.shape

        query_states = (
            self.q_proj(x) # (Batch, seq_len, d_model) --> (Batch, seq_len, d_model)
            .view(batch, seq_len, self.num_heads, self.head_dim) # (Batch, seq_len, d_model) --> (Batch, seq_len, num_heads, head_dim)
            .transpose(1, 2) # Reshape to match attention (Batch, seq_len, num_heads, head_dim) --> (Batch, num_heads, seq_len, head_dim)
        )
        key_states = (
            self.k_proj(x) # (Batch, seq_len, d_model) --> (Batch, seq_len, d_model)
            .view(batch, seq_len, self.num_heads, self.head_dim) # (Batch, seq_len, d_model) --> (Batch, seq_len, num_heads, head_dim)
            .transpose(1, 2) # Reshape to match attention (Batch, seq_len, num_heads, head_dim) --> (Batch, num_heads, seq_len, head_dim)
        )
        value_states = (
            self.v_proj(x) # (Batch, seq_len, d_model) --> (Batch, seq_len, d_model)
            .view(batch, seq_len, self.num_heads, self.head_dim) # (Batch, seq_len, d_model) --> (Batch, seq_len, num_heads, head_dim)
            .transpose(1, 2) # Reshape to match attention (Batch, seq_len, num_heads, head_dim) --> (Batch, num_heads, seq_len, head_dim)
        )

        # Apply RoPE
        query_states = apply_rope(query_states)
        key_states = apply_rope(key_states)

        if self.use_cache:
            # Update KV cache and get full tensors for attention
            key_states, value_states = update_kv_cache(key_states, value_states)

        x = self.attention(query_states, key_states, value_states, causal_mask) # (Batch, num_heads, seq_len, head_dim)

        # Concatenate heads
        x = self.o_proj(
            x.transpose(1, 2)
            .contiguous()
            .view(batch, seq_len, self.num_heads * self.head_dim)
        )

        return x