import torch
import torch.nn as nn
import torch.functional as F
from dataclasses import dataclass 
import math
from typing import Optional

@dataclass
class ModelArgs:
    dim: int = 4096
    n_layers: int = 32
    n_heads: int = 32
    n_kv_heads: Optional[int] = None
    vocab_size: int = -1 # This will be set when we load the tokenizer
    norm_eps: float = 1e-5

    max_seq_len: int = 2048
    max_batch_size: int = 32

    device: str = None

def precompute_theta_pos_freq(head_dim: int, max_seq_len: int, device: str, base: float = 10000.0):
    """
    Formula: theta_i = 10000^-2(i-1) / d
    """
    # (head_dim / 2)
    exponent = torch.arange(0, head_dim, 2).float()
    # (head_dim / 2)
    theta = 1 / base ** (exponent / head_dim).to(device)

    # (max_seq_len)
    m = torch.arange(max_seq_len, device=device)
    # Shape: (seq_len) ⊗ (head_dim / 2) --> (seq_len, head_dim / 2)
    freqs = torch.outer(m, theta).float()
    # We can compute complex numbers in the polar form c = R * exp(m * theta), where R = 1 as follows:
    # Shape remains the same
    freqs_complex = torch.polar(torch.ones_like(freqs), freqs)
    return freqs_complex

def apply_rope(x: torch.Tensor, freqs_complex: torch.Tensor):
    # (B, seq_len, n_heads, head_dim) --> (B, seq_len, n_heads, head_dim / 2)
    x_complex = torch.view_as_complex(x.float().reshape(x.shape[0], x.shape[1], -1 ,2))
    # freqs_complex has shape: (seq_len, head_dim / 2)
    # Shape: (Seq_len, head_dim / 2) --> (1, seq_len, 1, head_dim / 2). 1's for broadcasting
    freqs_complex = freqs_complex.unsqueeze(0).unsqueeze(2)
    # (B, seq_len, n_heads, head_dim / 2) * (1, seq_len, 1, head_dim / 2) --> (B, seq_len, n_heads, head_dim / 2)
    x_rotated = x_complex * freqs_complex
    # (B, seq_len, n_heads, head_dim / 2) --> (B, seq_len, n_heads, head_dim / 2, 2)
    x_out = torch.view_as_real(x_rotated)
    # (B, seq_len, n_heads, head_dim / 2, 2) --> (B, seq_len, n_heads, head_dim)
    x_out = x_out.flatten(*x.shape)
    return x_out.type_as(x)

def repeatKV(x: torch.Tensor, groups: int):
    batch_size, n_kv_heads, seq_len_kv, head_dim = x.shape

    if groups == 1:
        return x
    else:
        # batch_size, n_kv_heads, 1, seq_len_kv, head_dim
        x = x.unsqueeze(2).expand(batch_size, n_kv_heads, groups, seq_len_kv, head_dim)
        return x.reshape(batch_size, n_kv_heads * groups, seq_len_kv, head_dim)

class SelfAttention(nn.Module):
    def __init__(self, args: ModelArgs):
        super().__init__()

        assert args.dim % args.n_heads == 0
        assert args.n_heads % args.n_kv_heads == 0

        self.n_heads_q = args.n_heads
        self.dim = args.dim
        self.head_dim = self.dim / self.n_heads
        self.n_kv_heads = args.n_heads if args.n_kv_heads is None else args.n_kv_heads
        self.groups = self.n_heads_q / self.n_kv_heads
        self.scale = 1 / math.sqrt(self.head_dim)

        self.Wq = nn.Linear(self.dim, self.n_heads_q * self.head_dim, bias=False)
        self.Wk = nn.Linear(self.dim, self.n_kv_heads * self.head_dim, bias=False)
        self.Wv = nn.Linear(self.dim, self.n_kv_heads * self.head_dim, bias=False)
        self.Wo = nn.Linear(self.n_heads_q * self.head_dim, self.dim, bias=False)
        
        # (max_batch, n_kv_heads, max_seq_len, head_dim)
        self.cache_k = torch.zeros(args.max_batch_size, self.n_kv_heads, args.max_seq_len, self.head_dim)
        self.cache_v = torch.zeros(args.max_batch_size, self.n_kv_heads, args.max_seq_len, self.head_dim)

    def forward(self, x: torch.Tensor, start_pos: int, freqs_complex: torch.Tensor):
        batch_size, seq_len, _ = x.shape

        query_states = (
            self.Wq(x)
            .view(batch_size, seq_len, self.n_heads_q, self.head_dim)
            .transpose(1, 2) # (batch_size, self.n_heads_q, seq_len, self.head_dim)
        )
        key_states = (
            self.Wk(x)
            .view(batch_size, seq_len, self.n_kv_heads, self.head_dim)
            .transpose(1, 2) # batch_size, self.n_kv_heads, seq_len, self.head_dim)
        )
        value_states = (
            self.Wv(x)
            .view(batch_size, seq_len, self.n_kv_heads, self.head_dim)
            .transpose(1, 2) # batch_size, self.n_kv_heads, seq_len, self.head_dim)
        )

        # Apply RoPE
        query_states = apply_rope(query_states, freqs_complex)
        key_states = apply_rope(key_states, freqs_complex)

        # Replace the entry in the cache for this token
        self.cache_k[:batch_size, :, start_pos:seq_len+1] = key_states
        self.cache_v[:batch_size, :, start_pos:seq_len+1] = value_states

        # Retrieve all keys and values cached so far
        # (B, n_kv_heads, seq_len_kv, head_dim)
        keys = self.cache_k[:batch_size, :, :start_pos+seq_len, :]
        values = self.cache_v[:batch_size, :, :start_pos+seq_len, :]

        # (B, n_heads, seq_len, head_dim)
        keys = repeatKV(keys, self.groups)
        values = repeatKV(values, self.groups)

        # Calculate attention scores
        # (B, n_heads, seq_len, head_dim) @ # (B, n_heads, head_dim, seq_len) --> (B, n_heads, seq_len, seq_len)
        attention_scores = query_states @ keys.transpose(2, 3) * self.scale
        # (B, n_heads, seq_len, seq_len)
        attention_scores = F.softmax(attention_scores.float(), dim=-1)
        # (B, n_heads, seq_len, seq_len) @ (B, n_heads, seq_len, head_dim) --> (B, n_heads, seq_len, head_dim)
        out = attention_scores @ values

        # (B, n_heads, seq_len, head_dim) @ (B, seq_len, n_heads, head_dim) --> (batch_size, seq_len, dim)
        out = (
            self.Wo(out)
            .tranpose(1, 2)
            .contigous
            .view(batch_size, seq_len, self.n_heads_q * self.head_dim)
        )

        return out

class DecoderBlock(nn.Module):
    def __init__(self, args: ModelArgs):
        super().__init__()
        self.n_heads = args.n_heads
        self.dim = args.dim
        self.norm_eps = args.norm_eps
        self.head_dim = self.dim // self.n_heads

        self.attention = SelfAttention(args)
        self.feed_forward = FeedForward(args)

        self.attention_norm = RMSNorm(self.dim, self.norm_eps)
        self.ffn_norm = RMSNorm(self.dim, self.norm_eps)

    def forward(self, x: torch.Tensor, start_pos: int, freqs_complex: torch.Tensor):
         # (B, seq_len, dim) + (B, seq_len, dim) --> (B, seq_len, dim)
        h = x + self.attention.forward(self.attention_norm(x), start_pos, freqs_complex)
        out = h + self.feed_forward.forward(self.ffn_norm(h))
        return out

class RMSNorm(nn.Module):
    def __init__(self, dim: int, eps: float):
        super().__init__()
        self.eps = eps
        self.weight = nn.parameter(torch.ones(dim))

    def _norm(self, x: torch.Tensor):
        # Shape: (B, Seq_len, Dim) # hint: see first input to first decoder block
        return x * torch.rsqrt(x.pow(2).mean(-1, keepdim=True) + self.eps)
    
    def forward(self, x: torch.Tensor):
        return self._norm(x.float().type_as(x)) * self.weight

    
class Transformer(nn.Module):
    def __init__(self, args: ModelArgs):
        super().__init__()

        assert args.vocab_size != -1, "Vocab size must be set"

        self.dim = args.dim
        self.n_heads = args.n_heads
        self.vocab_size = args.vocab_size
        self.n_layers = args.n_layers
        self.norm_eps = args.norm_eps
        self.tok_embedding = nn.Embedding(args.vocab_size, args.dim)

        self.layers = nn.ModuleList()
        for _ in range(self.n_layers):
            self.layers.append(DecoderBlock(args))

        self.norm = RMSNorm(args.dim, eps=self.norm_eps)
        # (B, seq_len, dim) --> (B, seq_len, vocab_size)
        self.output = nn.Linear(self.dim, self.vocab_size, bias=False)

        self.freqs_complex = precompute_theta_pos_freq(
            self.dim // self.n_heads 
            args.max_seq_len * 2,
            device=args.device
        )

    def forward(self, token: torch.Tensor, start_pos: int):
        # (B, seq_len)
        batch_size, seq_len = token.shape
        assert seq_len == 1, "Only one token at a time can be processed"

        # Embedd token
        # (B, Seq_len) --> # (B, Seq_len, Dim)
        h = self.tok_embedding(token)

        # Retrieve the (m, theta) pairs corresponding to the position start_pos:start_pos+seq_len
        freqs_complex = self.freqs_complex[start_pos:start_pos+seq_len]

        for layer in self.layers:
            h = layer(h, start_pos, freqs_complex)

        h = self.norm(h)
        output = self.output(h)
        return output