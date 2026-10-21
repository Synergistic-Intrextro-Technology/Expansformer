"""
================================================================================
CUSTOM TRANSFORMER LANGUAGE MODEL
From-scratch GPT-style architecture — full ownership, full control.
Train on any text corpus. Configurable everything.
================================================================================
"""

import math
import torch
import torch.nn as nn
import torch.nn.functional as F
from dataclasses import dataclass


@dataclass
class ModelConfig:
    """All architectural knobs in one place."""
    vocab_size: int = 32000
    context_length: int = 512
    n_layers: int = 6
    n_heads: int = 8
    embed_dim: int = 512
    ff_multiplier: int = 4       # feed-forward hidden = embed_dim * ff_multiplier
    dropout: float = 0.1
    bias: bool = False           # bias in linear layers and layernorms
    rope: bool = True            # rotary positional embeddings
    norm_eps: float = 1e-5


# ==============================================================================
# COMPONENTS
# ==============================================================================

class RMSNorm(nn.Module):
    """Root Mean Square Layer Normalization — no mean centering, lighter."""
    def __init__(self, dim: int, eps: float = 1e-5):
        super().__init__()
        self.eps = eps
        self.weight = nn.Parameter(torch.ones(dim))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        norm = x.float().pow(2).mean(-1, keepdim=True).add(self.eps).rsqrt()
        return (x.float() * norm).type_as(x) * self.weight


def precompute_rope_frequencies(dim: int, seq_len: int, theta: float = 10000.0) -> torch.Tensor:
    """Precompute rotary embedding frequency tensor."""
    freqs = 1.0 / (theta ** (torch.arange(0, dim, 2).float() / dim))
    t = torch.arange(seq_len).float()
    freqs = torch.outer(t, freqs)
    return torch.polar(torch.ones_like(freqs), freqs)  # complex64


def apply_rope(x: torch.Tensor, freqs: torch.Tensor) -> torch.Tensor:
    """Apply rotary positional embeddings to query/key tensors."""
    # x: (B, n_heads, T, head_dim) -> complex view: (B, n_heads, T, head_dim//2)
    x_complex = torch.view_as_complex(x.float().reshape(*x.shape[:-1], -1, 2))
    freqs = freqs[:x.shape[2], :]  # trim to actual sequence length
    freqs = freqs.unsqueeze(0).unsqueeze(0)  # (1, 1, T, head_dim//2)
    x_rotated = torch.view_as_real(x_complex * freqs).flatten(-2)
    return x_rotated.type_as(x)


class SwiGLU(nn.Module):
    """SwiGLU activation — gated linear unit with swish. Better than ReLU/GELU."""
    def __init__(self, embed_dim: int, ff_dim: int, bias: bool = False):
        super().__init__()
        self.w1 = nn.Linear(embed_dim, ff_dim, bias=bias)
        self.w2 = nn.Linear(ff_dim, embed_dim, bias=bias)
        self.w3 = nn.Linear(embed_dim, ff_dim, bias=bias)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.w2(F.silu(self.w1(x)) * self.w3(x))


class CausalSelfAttention(nn.Module):
    """Multi-head causal self-attention with optional RoPE."""

    def __init__(self, config: ModelConfig):
        super().__init__()
        assert config.embed_dim % config.n_heads == 0
        self.n_heads = config.n_heads
        self.head_dim = config.embed_dim // config.n_heads
        self.use_rope = config.rope

        self.qkv = nn.Linear(config.embed_dim, 3 * config.embed_dim, bias=config.bias)
        self.out_proj = nn.Linear(config.embed_dim, config.embed_dim, bias=config.bias)
        self.attn_dropout = nn.Dropout(config.dropout)
        self.resid_dropout = nn.Dropout(config.dropout)

        # Causal mask — registered as buffer, not parameter
        self.register_buffer(
            "causal_mask",
            torch.tril(torch.ones(config.context_length, config.context_length))
                 .view(1, 1, config.context_length, config.context_length)
        )

    def forward(self, x: torch.Tensor, rope_freqs: torch.Tensor | None = None) -> torch.Tensor:
        B, T, C = x.shape

        q, k, v = self.qkv(x).split(C, dim=-1)
        q = q.view(B, T, self.n_heads, self.head_dim).transpose(1, 2)
        k = k.view(B, T, self.n_heads, self.head_dim).transpose(1, 2)
        v = v.view(B, T, self.n_heads, self.head_dim).transpose(1, 2)

        # Apply RoPE to queries and keys
        if self.use_rope and rope_freqs is not None:
            q = apply_rope(q, rope_freqs)
            k = apply_rope(k, rope_freqs)

        # Scaled dot-product attention with causal mask
        scale = 1.0 / math.sqrt(self.head_dim)
        attn = (q @ k.transpose(-2, -1)) * scale
        attn = attn.masked_fill(self.causal_mask[:, :, :T, :T] == 0, float('-inf'))
        attn = F.softmax(attn, dim=-1)
        attn = self.attn_dropout(attn)

        out = (attn @ v).transpose(1, 2).contiguous().view(B, T, C)
        return self.resid_dropout(self.out_proj(out))


class TransformerBlock(nn.Module):
    """Pre-norm transformer block: RMSNorm -> Attention -> RMSNorm -> FFN."""

    def __init__(self, config: ModelConfig):
        super().__init__()
        self.norm1 = RMSNorm(config.embed_dim, eps=config.norm_eps)
        self.attn = CausalSelfAttention(config)
        self.norm2 = RMSNorm(config.embed_dim, eps=config.norm_eps)
        ff_dim = int(config.embed_dim * config.ff_multiplier * (2 / 3))  # SwiGLU sizing
        ff_dim = ((ff_dim + 63) // 64) * 64  # round to multiple of 64 for efficiency
        self.ffn = SwiGLU(config.embed_dim, ff_dim, bias=config.bias)

    def forward(self, x: torch.Tensor, rope_freqs: torch.Tensor | None = None) -> torch.Tensor:
        x = x + self.attn(self.norm1(x), rope_freqs)
        x = x + self.ffn(self.norm2(x))
        return x


# ==============================================================================
# THE MODEL
# ==============================================================================

class TransformerLM(nn.Module):
    """
    Complete autoregressive transformer language model.
    Architecture: token embed -> N x TransformerBlock -> RMSNorm -> linear head
    Features: RoPE, SwiGLU, RMSNorm, weight tying, mixed-precision friendly.
    """

    def __init__(self, config: ModelConfig):
        super().__init__()
        self.config = config

        self.token_embed = nn.Embedding(config.vocab_size, config.embed_dim)
        self.drop = nn.Dropout(config.dropout)
        self.blocks = nn.ModuleList([TransformerBlock(config) for _ in range(config.n_layers)])
        self.norm = RMSNorm(config.embed_dim, eps=config.norm_eps)
        self.lm_head = nn.Linear(config.embed_dim, config.vocab_size, bias=False)

        # Weight tying — embed and output projection share parameters
        self.lm_head.weight = self.token_embed.weight

        # Precompute RoPE frequencies
        if config.rope:
            head_dim = config.embed_dim // config.n_heads
            rope_freqs = precompute_rope_frequencies(head_dim, config.context_length)
            self.register_buffer("rope_freqs", rope_freqs)
        else:
            self.rope_freqs = None
            # Learned positional embeddings as fallback
            self.pos_embed = nn.Embedding(config.context_length, config.embed_dim)

        # Initialize weights
        self.apply(self._init_weights)
        # Scale residual projections per GPT-2 paper
        for pn, p in self.named_parameters():
            if pn.endswith('out_proj.weight') or pn.endswith('w2.weight'):
                nn.init.normal_(p, mean=0.0, std=0.02 / math.sqrt(2 * config.n_layers))

    def _init_weights(self, module: nn.Module):
        if isinstance(module, nn.Linear):
            nn.init.normal_(module.weight, mean=0.0, std=0.02)
            if module.bias is not None:
                nn.init.zeros_(module.bias)
        elif isinstance(module, nn.Embedding):
            nn.init.normal_(module.weight, mean=0.0, std=0.02)

    def forward(
        self,
        input_ids: torch.Tensor,
        targets: torch.Tensor | None = None,
    ) -> tuple[torch.Tensor, torch.Tensor | None]:
        """
        Args:
            input_ids: (B, T) token indices
            targets: (B, T) target token indices for loss computation
        Returns:
            logits: (B, T, vocab_size)
            loss: scalar if targets provided, else None
        """
        B, T = input_ids.shape
        assert T <= self.config.context_length, f"Sequence {T} exceeds context {self.config.context_length}"

        x = self.token_embed(input_ids)

        if not self.config.rope:
            positions = torch.arange(T, device=input_ids.device)
            x = x + self.pos_embed(positions)

        x = self.drop(x)

        rope_freqs = self.rope_freqs if self.config.rope else None
        for block in self.blocks:
            x = block(x, rope_freqs)

        x = self.norm(x)
        logits = self.lm_head(x)

        loss = None
        if targets is not None:
            loss = F.cross_entropy(
                logits.view(-1, logits.size(-1)),
                targets.view(-1),
                ignore_index=-1,
            )

        return logits, loss

    @torch.no_grad()
    def generate(
        self,
        input_ids: torch.Tensor,
        max_new_tokens: int = 200,
        temperature: float = 0.8,
        top_k: int = 50,
        top_p: float = 0.9,
    ) -> torch.Tensor:
        """Autoregressive generation with temperature, top-k, and nucleus sampling."""
        self.eval()
        for _ in range(max_new_tokens):
            # Crop to context length
            idx_cond = input_ids[:, -self.config.context_length:]
            logits, _ = self(idx_cond)
            logits = logits[:, -1, :] / max(temperature, 1e-8)

            # Top-k filtering
            if top_k > 0:
                topk_vals, _ = torch.topk(logits, min(top_k, logits.size(-1)))
                logits[logits < topk_vals[:, [-1]]] = float('-inf')

            # Top-p (nucleus) filtering
            if top_p < 1.0:
                sorted_logits, sorted_idx = torch.sort(logits, descending=True)
                cumulative_probs = torch.cumsum(F.softmax(sorted_logits, dim=-1), dim=-1)
                remove_mask = cumulative_probs - F.softmax(sorted_logits, dim=-1) >= top_p
                sorted_logits[remove_mask] = float('-inf')
                logits = sorted_logits.scatter(1, sorted_idx, sorted_logits)

            probs = F.softmax(logits, dim=-1)
            next_token = torch.multinomial(probs, num_samples=1)
            input_ids = torch.cat([input_ids, next_token], dim=1)

        return input_ids

    def count_parameters(self) -> dict:
        """Return parameter counts — total, trainable, and by component."""
        total = sum(p.numel() for p in self.parameters())
        trainable = sum(p.numel() for p in self.parameters() if p.requires_grad)
        embedding = sum(p.numel() for p in self.token_embed.parameters())
        attention = sum(
            p.numel()
            for block in self.blocks
            for p in block.attn.parameters()
        )
        ffn = sum(
            p.numel()
            for block in self.blocks
            for p in block.ffn.parameters()
        )
        return {
            "total": total,
            "trainable": trainable,
            "embedding": embedding,
            "attention": attention,
            "ffn": ffn,
            "total_M": f"{total / 1e6:.1f}M",
        }
