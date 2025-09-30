import torch
import torch.nn as nn
import torch.nn.functional as F
import math
from dataclasses import dataclass
from typing import Optional

@dataclass
class ModelArgs:
    dim: int = 4096
    n_layers: int = 32
    n_heads: int = 32 # Num heads for queries
    n_kv_heads: Optional[int] = None # Num heads for K and V
    vocab_size: int = -1 # This will be set when we load the tokenizer
    multiple_of: int = 256
    ffn_dim_multiplier: Optional[float] = None
    norm_eps: float = 1e-5

    # Needed for KV cache
    max_batch_size: int = 32
    max_seq_len: int = 2048

    device: str = None

def precompute_theta_pos_frequencies(head_dim: int, seq_len: int, device: str, base: float = 10000.0):
    """
    Formula: θi = 1000^-2(i-1)/head_dim, where i = [1, 2, ..., head_dim/2]
    Applied here as:  θi = 1 / 1000^2(i-1)/head_dim, where i = [1, 2, ..., head_dim/2]
    """
    assert head_dim % 2 == 0, "RoPE cannot be applied to head dim thats odd"

    # Shape of theta params --> (head_dim / 2)
    theta_numerator = torch.arange(0, head_dim, 2).float()
    # Shape --> (head_dim / 2)
    theta = 1 / base ** (theta_numerator / head_dim).to(device)
    # Possible position "m" could be alot so we will give seq_len * 2 (for prompt)
    # Shape --> (seq_len)
    m = torch.arange(seq_len, device=device)
    # Multiply each theta by each pos using outer product 
    # Shape: (seq_len) ⊗ (head_dim / 2) --> (seq_len, head_dim / 2)
    freqs = torch.outer(m, theta).float()
    # We can compute complex numbers in the polar form c = R * exp(i * m * theta), where R = 1
    # (seq_len, head_dim / 2) --> (seq_len, head_dim / 2)
    freqs_complex = torch.polar(torch.ones_like(freqs), freqs)
    return freqs_complex

def apply_rotary_embeddings(x: torch.Tensor, freqs_complex: torch.Tensor, device: str):
    # Shape: (B, Seq_len, n_heads, head_dim)--> Shape: (B, Seq_len, n_heads, head_dim / 2)
    x_complex = torch.view_as_complex(x.float().reshape(x.shape[0], x.shape[1], -1, 2))
    # Shape: (Seq_len, head_dim / 2) --> (1, seq_len, 1, head_dim / 2). 1's for broadcasting
    freqs_complex = freqs_complex.unsqueeze(0).unsqueeze(2)
    # Shape: (B, seq_len, n_heads, head_dim / 2) * (1, seq_len, 1, head_dim / 2) --> (B, seq_len, n_heads, head_dim / 2)
    x_rotated = x_complex * freqs_complex
    # Shape: (B, seq_len, n_heads, head_dim / 2) --> (B, seq_len, n_heads, head_dim / 2, 2)
    x_out = torch.view_as_real(x_rotated)
    # Shape: (B, seq_len, n_heads, head_dim / 2, 2) --> (B, seq_len, n_heads, head_dim)
    x.out = x_out.reshape(*x.shape)
    return x_out.type_as(x).to(device)

def repeat_kv(x: torch.Tensor, n_groups: int):
    batch_size, seq_len, n_kv_heads, head_dim = x.shape
    if n_groups == 1:
        # Basically MHA
        return x
    else:
        # (B, n_kv_heads, 1, seq_len_kv, head_dim)
        x = x.unsqueeze(2).expand(batch_size, n_kv_heads, n_groups, seq_len, head_dim)
        return x.reshape(batch_size, n_kv_heads * n_groups, seq_len, head_dim)

class RMSNorm(nn.Module):
    def __init__(self, dim: int, eps: float):
        super().__init__()
        self.eps = eps
        self.weight = nn.Parameter(torch.ones(dim))

    def _norm(self, x: torch.Tensor):
        # Shape: (B, Seq_len, Dim) # hint: see first input to first decoder block
        return x * torch.rsqrt(x.pow(2).mean(-1, keepdim=True) + self.eps)
    
    def forward(self, x: torch.Tensor):
        return self.weight * self._norm(x.float()).type_as(x)
    
class FeedForward(nn.Module):
    def __init__(self, args: ModelArgs):
        super().__init__()

        self.dim = args.dim
        self.hidden_dim = 4 * self.dim
        self.hidden_dim = int(2 * self.hidden_dim / 3)
        self.multiple_of = args.multiple_of

        if args.ffn_dim_multiplier is not None:
            self.hidden_dim = int(args.ffn_dim_multiplier * self.hidden_dim)

        # Round the hidden_dim to the nearest multiple_of param 
        self.hidden_dim = self.multiple_of * (
            self.hidden_dim * ((self.hidden_dim + self.multiple_of - 1) // self.multiple_of))
        
        self.W = nn.Linear(self.dim, self.hidden_dim, bias=False)
        self.V = nn.Linear(self.dim, self.hidden_dim, bias=False)
        self.Wo = nn.Linear(self.hidden_dim, self.dim, bias=False)

    def forward(self, x: torch.Tensor):
        # (B, Seq_len, dim) --> (B, Seq_len, hidden_dim)
        a = self.W(x)
        # (B, Seq_len, hidden_dim)
        g = F.silu(a)
        # (B, Seq_len, dim) --> (B, Seq_len, hidden_dim)
        b = self.V(x)
        h = g * b
        # (B, Seq_len, hidden_dim) --> (B, Seq_len, dim) 
        return self.Wo(h)
    
class SelfAttention(nn.Module):
    def __init__(self, args: ModelArgs):
        super().__init__()

        assert args.n_heads % args.n_kv_heads == 0, "n_heads must be divisible by n_kv_heads"
        assert args.dim % args.n_heads == 0, "dim must be divisible by n_heads"

        # Indicates the number of heads for queries
        self.n_heads_q = args.n_heads
        # Indicates the number of heads for key and values 
        self.n_kv_heads = args.n_heads if args.n_kv_heads is None else args.n_kv_heads 
                        # If its same as n_heads it basically MHA, if n_kv_heads = 1 than its MQA
        # Indicates number of groups
        self.n_groups = self.n_heads_q // self.n_kv_heads
        # Indicates the dimension of each head
        self.head_dim = args.dim // self.n_heads
        self.scale = 1 / math.sqrt(self.head_dim)

        self.q_proj = nn.Linear(args.dim, self.head_dim * args.n_heads, bias=False)
        self.k_proj = nn.Linear(args.dim, self.n_kv_heads * self.head_dim, bias=False)
        self.v_proj = nn.Linear(args.dim, self.n_kv_heads * self.head_dim, bias=False)
        self.o_proj = nn.Linear(self.head_dim * args.n_heads, args.dim, bias=False)

        self.cache_k = torch.zeros((args.max_batch_size, self.n_kv_heads, args.max_seq_len, self.head_dim))
        self.cache_v = torch.zeros((args.max_batch_size, self.n_kv_heads, args.max_seq_len, self.head_dim))
    
    def forward(self, x: torch.Tensor, start_pos: int, freqs_complex: torch.Tensor):
        batch_size, seq_len, _ = x.shape

        query_states = (
            self.q_proj(x) # (B, 1, head_dim * n_heads_q)
            .view(batch_size, seq_len, self.n_heads_q, self.head_dim) # (B, 1, n_heads_q, head_dim)
            .transpose(1, 2) # (B, n_heads_q, 1, head_dim)
        )

        key_states = (
            self.k_proj(x) # (B, 1, n_kv_heads * head_dim)
            .view(batch_size, seq_len, self.n_kv_heads, self.head_dim)
            .transpose(1, 2)
        )

        value_states = (
            self.v_proj(x) # (B, 1, n_kv_heads * head_dim)
            .view(batch_size, seq_len, self.n_kv_heads, self.head_dim)
            .transpose(1, 2)
        )

        query_states = apply_rotary_embeddings(query_states, freqs_complex, device=x.device)
        key_states = apply_rotary_embeddings(key_states, freqs_complex, device=x.device)

        # Replace the entry in the cache for this token
        self.cache_k[: batch_size, :, start_pos:start_pos+seq_len] = key_states
        self.cache_v[: batch_size, :, start_pos:start_pos+seq_len] = value_states

        # Retrieve all the cached keys and values so far
        # Shape: (B, n_kv_heads, seq_len_kv, head_dim)
        keys = self.cache_k[:batch_size, :, :start_pos+seq_len, :]
        values = self.cache_v[:batch_size, :, :start_pos+seq_len, :]

        # Repeat the heads of the K and V to reach the number of heads of the queries
        # Shape: (B, n_kv_heads, seq_len_kv, head_dim) --> (B, n_q_heads, seq_len_kv, head_dim)
        keys = repeat_kv(keys, self.n_groups)
        values = repeat_kv(values, self.n_groups)

        # (B, n_heads_q, 1, head_dim) @ (B, n_q_heads, head_dim, seq_len_kv) --> (B, n_q_heads, 1, seq_len_kv)
        attention_scores = query_states @ keys.transpose(2, 3) * self.scale
        attention_scores = F.softmax(attention_scores.float(), dim=-1).type_as(query_states)
        # (B, n_q_heads, 1, seq_len_kv) @ (B, n_q_heads, seq_len_kv, head_dim) --> (B, n_q_heads, 1, head_dim)
        out = attention_scores @ values

        # Last step: multiply by out proj
        # (B, n_q_heads, 1, head_dim) --> (B, 1, n_q_heads, head_dim) --> (B, 1, dim) @ (B, 1, dim)
        out = self.o_proj(
            out.transpose(1, 2)
            .contiguous()
            .view(batch_size, seq_len, self.n_heads_q * self.head_dim)
        )

        return out
    
class DecoderBlock(nn.Module):
    def __init__(self, args: ModelArgs):
        super().__init__()

        self.n_heads = args.n_heads
        self.dim = args.dim
        self.head_dim = self.dim // self.n_heads

        self.attention = SelfAttention(args)
        self.feed_forward = FeedForward(args)

        # Normalisation before the self attention
        self.attention_norm = RMSNorm(args.dim, eps = args.norm_eps)
        # Normalisation before ffn block
        self.ffn_norm = RMSNorm(args.dim, eps = args.norm_eps)

    def forward(self, x: torch.Tensor, start_pos: int, freqs_complex: torch.Tensor):
        # (B, seq_len, dim) + (B, seq_len, dim) --> (B, seq_len, dim)
        h = x + self.attention.forward(self.attention_norm(x), start_pos, freqs_complex)
        out = h + self.feed_forward.forward(self.ffn_norm(h))
        return out

class Transformer(nn.Module):
    def __init__(self, args: ModelArgs) -> None:
        super().__init__()

        assert args.vocab_size != -1, "Vocab size must be set"

        self.args = args
        self.vocab_size = args.vocab_size
        self.n_layers = args.n_layers 
        self.tok_embeddings = nn.Embedding(self.vocab_size, args.dim)

        self.layers = nn.ModuleList()
        for _ in range(args.n_layers):
            self.layers.append(DecoderBlock(args))

        self.norm = RMSNorm(args.dim, eps=self.args.norm_eps)
        self.output = nn.Linear(args.dim, self.vocab_size, bias=False)

        self.freqs_complex = precompute_theta_pos_frequencies(
            self.args.dim // self.args.n_heads, 
            self.args.max_seq_len * 2, 
            device=self.args.device
        )

    def forward(self, tokens: torch.Tensor, start_pos: int):
        """
        Perform a forward pass through the Transformer model.

        Args:
            tokens (torch.Tensor): Input token indices.
            start_pos (int): Starting position for attention caching.

        Returns:
            torch.Tensor: Output logits after applying the Transformer model.

        Note: This model is aimed at inference not training. Otherwise we dont need the KV cache
            and assume seq_len == 1.

        """
        # (B, Seq_len)
        _, seq_len = tokens.shape
        assert seq_len == 1, "Only one token a time can be processed"

        # (B, Seq_len) --> # (B, Seq_len, Dim)
        h = self.tok_embeddings(tokens)

        # Retrieve the pairs (m, theta) corresponding to the positions [start_pos, start_pos + seq_len]
        freqs_complex = self.freqs_complex[start_pos:start_pos + seq_len]

        # Consecutively apply all the decoder layers
        for layer in self.layers:
            h = layer(h, start_pos, freqs_complex)
        h = self.norm(h)
        output = self.output(h).float()
        return output