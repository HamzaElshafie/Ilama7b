import torch
import math
from torch import nn
from typing import Optional

def apply_rope(x: torch.Tensor):
    pass

class MultiQueryAttention(nn.Module):
    def __init__(self, d_model: int, num_heads: int, use_cache: Optional[bool] = False):
        super().__init__()
        self.d_model = d_model
        self.num_heads = num_heads
        self.use_cache = use_cache
        self.k_cache = None
        self.v_cache = None

        assert d_model % num_heads == 0, "d_model is not divisible by num_heads"

        self.head_dim  = d_model // num_heads
        self.scale = 1 / math.sqrt(self.head_dim)

        self.q_proj = nn.Linear(d_model, d_model)
        # Only one kv for all heads
        self.k_proj = nn.Linear(d_model, self.head_dim)
        self.v_proj = nn.Linear(d_model, self.head_dim)
        self.o_proj = nn.Linear(d_model, d_model)
    
    def update_kv_cache(self, key_states: torch.Tensor, value_states: torch.Tensor):
        if self.k_cache is None:
            self.k_cache = key_states
            self.v_cache = value_states
        else:
            self.k_cache = torch.cat([self.k_cache, key_states], dim=-2)
            self.v_cache = torch.cat([self.v_cache, value_states], dim=-2)

        return self.k_cache, self.v_cache

    def attention(self, query_states, key_states, value_states, causal_mask: Optional[torch.Tensor] = None):
        # Calculate attention scores
        attention_scores = query_states @ key_states.transpose(-2, -1) * self.scale

        if causal_mask is not None:
            attention_scores = attention_scores.masked_fill(causal_mask == 0, -1e9)

        # Apply softmax 
        attention_scores = attention_scores.softmax(-1)

        # logits @ values
        return attention_scores @ value_states

    def forward(self, x: torch.Tensor, causal_mask: Optional[torch.Tensor] = None):
        batch, seq_len, _, = x.shape

        query_states = (
            self.q_proj(x) # (Batch, seq_len, d_model) --> (Batch, seq_len, d_model)
            .view(batch, seq_len, self.num_heads, self.head_dim) # (Batch, seq_len, d_model) --> (Batch, seq_len, num_heads, head_dim)
            .transpose(1, 2) # Reshape to match attention (Batch, seq_len, num_heads, head_dim) --> (Batch, num_heads, seq_len, head_dim)
        )

        key_states = self.k_proj(x).unsqueeze(1) # (Batch, seq_len, d_model) --> (Batch, 1, seq_len, head_dim)
        value_states = self.v_proj(x).unsqueeze(1) # (Batch, seq_len, d_model) --> (Batch, 1, seq_len, head_dim)

        # Apply RoPE
        query_states = apply_rope(query_states)
        key_states = apply_rope(key_states)

        if self.use_cache:
            # Update KV cache and get full tensors for attention
            key_states, value_states = self.update_kv_cache(key_states, value_states)

        # Calculate attention
        x = self.attention(query_states, key_states, value_states, causal_mask) # (Batch, num_heads, seq_len, head_dim)

        # Concatenate heads and project
        x = self.o_proj(
            x.transpose(1, 2)
            .contiguous()
            .view(batch, seq_len, self.num_heads * self.head_dim)
        ) # (Batch, seq_len, d_model)

        return x