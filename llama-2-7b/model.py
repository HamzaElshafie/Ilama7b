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
    n_kv_heads: Op tional[int] = None # Num heads for K and V
    vocab_size: int = -1 # This will be set when we load the tokenizer
    multiple_of: int = 256
    ffn_dim_multiplier: Optional[float] = None
    norm_eps: float = 1e-5

    # Needed for KV cache
    max_batch_size: int = 32
    max_seq_len: int = 2048

    device: str = None

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