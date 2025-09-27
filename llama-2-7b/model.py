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
    # We can compute tcomplex numbers in the polar form c = R * exp(i * m * theta), where R = 1
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
    x.out = x_out.flatten(*x.shape)
    return x_out.type_as(x).to(device)

class RMSNorm(nn.module):
    def __init__(self, dim: int, eps: float):
        super().__init__()
        self.eps = eps
        self.weight = nn.Parameter(torch.ones(dim))

    def _norm(self, x: torch.Tensor):
        # Shape: (B, Seq_len, Dim) # hint: see first input to first decoder block
        return x * torch.rsqrt(x.pow(2).mean(-1, keepdim=True) + self.eps)
    
    def forward(self, x: torch.Tensor):
        return self.weight * self._norm(x.float()).type_as(x)

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
        batch_size, seq_len = tokens.shape
        assert seq_len == 1, "Only one token a time can be processed"

        # (B, Seq_len) --> # (B, Seq_len, Dim)
        h = self.tok_embeddings(tokens)

        # Retrieve the pairs (m, theta) corresponding to te positions [start_pos, start_pos + seq_len]
        freqs_complex = self.freqs_complex[start_pos:start_pos + seq_len]

        # Consecutively apply all the decoder layers
        for layer in self.layers:
            h = layer(h, start_pos, freqs_complex)
        h = self.norm(h)
        output = self.output(h).float()
        return output