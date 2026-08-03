"""History-DINO intent token resampler for FastWAM_DINO."""

from __future__ import annotations

import torch
import torch.nn as nn


class PerceiverIntentBlock(nn.Module):
    """One lightweight Perceiver block over learnable intent queries."""

    def __init__(
        self,
        dim: int,
        num_heads: int,
        mlp_ratio: float = 4.0,
        dropout: float = 0.0,
    ):
        super().__init__()
        self.query_norm_cross = nn.LayerNorm(dim)
        self.context_norm = nn.LayerNorm(dim)
        self.cross_attn = nn.MultiheadAttention(
            embed_dim=dim,
            num_heads=num_heads,
            dropout=dropout,
            batch_first=True,
        )
        self.query_norm_self = nn.LayerNorm(dim)
        self.self_attn = nn.MultiheadAttention(
            embed_dim=dim,
            num_heads=num_heads,
            dropout=dropout,
            batch_first=True,
        )
        self.query_norm_mlp = nn.LayerNorm(dim)
        hidden_dim = int(dim * mlp_ratio)
        self.mlp = nn.Sequential(
            nn.Linear(dim, hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, dim),
            nn.Dropout(dropout),
        )

    def forward(self, queries: torch.Tensor, context: torch.Tensor) -> torch.Tensor:
        q = self.query_norm_cross(queries)
        kv = self.context_norm(context)
        cross_out, _ = self.cross_attn(q, kv, kv, need_weights=False)
        queries = queries + cross_out

        q = self.query_norm_self(queries)
        self_out, _ = self.self_attn(q, q, q, need_weights=False)
        queries = queries + self_out

        queries = queries + self.mlp(self.query_norm_mlp(queries))
        return queries


class DINOHistoryIntentResampler(nn.Module):
    """Compress short history DINO grids into a small set of text-dim intent tokens.

    Input history latents are expected in the same dense DINO layout as FastWAM_DINO:
    [B, D, T, H, W]. The module flattens the T/H/W axes, adds learnable temporal
    embeddings, and lets K learnable queries read the history through Perceiver-style
    cross-attention blocks.
    """

    def __init__(
        self,
        dino_dim: int,
        text_dim: int,
        num_intent_tokens: int = 8,
        resampler_dim: int = 1024,
        num_layers: int = 2,
        num_heads: int = 8,
        max_history_frames: int = 8,
        mlp_ratio: float = 4.0,
        dropout: float = 0.0,
    ):
        super().__init__()
        if num_intent_tokens <= 0:
            raise ValueError(f"`num_intent_tokens` must be positive, got {num_intent_tokens}")
        if num_layers <= 0:
            raise ValueError(f"`num_layers` must be positive, got {num_layers}")
        if max_history_frames <= 0:
            raise ValueError(f"`max_history_frames` must be positive, got {max_history_frames}")
        if resampler_dim % num_heads != 0:
            raise ValueError(
                f"`resampler_dim` must be divisible by `num_heads`, got "
                f"{resampler_dim} and {num_heads}"
            )

        self.dino_dim = int(dino_dim)
        self.text_dim = int(text_dim)
        self.num_intent_tokens = int(num_intent_tokens)
        self.resampler_dim = int(resampler_dim)
        self.max_history_frames = int(max_history_frames)

        self.input_norm = nn.LayerNorm(self.dino_dim)
        self.input_proj = nn.Linear(self.dino_dim, self.resampler_dim)
        self.temporal_embedding = nn.Parameter(
            torch.randn(1, self.max_history_frames, 1, self.resampler_dim) * 0.02
        )
        self.intent_queries = nn.Parameter(
            torch.randn(1, self.num_intent_tokens, self.resampler_dim) * 0.02
        )
        self.blocks = nn.ModuleList(
            [
                PerceiverIntentBlock(
                    dim=self.resampler_dim,
                    num_heads=num_heads,
                    mlp_ratio=mlp_ratio,
                    dropout=dropout,
                )
                for _ in range(num_layers)
            ]
        )
        self.output_norm = nn.LayerNorm(self.resampler_dim)
        self.output_proj = nn.Linear(self.resampler_dim, self.text_dim)

    def forward(self, history_dino_latents: torch.Tensor) -> torch.Tensor:
        if history_dino_latents.ndim != 5:
            raise ValueError(
                f"`history_dino_latents` must be [B,D,T,H,W], got "
                f"{tuple(history_dino_latents.shape)}"
            )
        batch_size, dino_dim, num_frames, height, width = history_dino_latents.shape
        if dino_dim != self.dino_dim:
            raise ValueError(
                f"History DINO dim mismatch: expected {self.dino_dim}, got {dino_dim}"
            )
        if num_frames > self.max_history_frames:
            raise ValueError(
                f"History frame count {num_frames} exceeds max_history_frames={self.max_history_frames}"
            )

        x = history_dino_latents.permute(0, 2, 3, 4, 1).reshape(
            batch_size, num_frames, height * width, dino_dim
        )
        x = self.input_proj(self.input_norm(x))
        x = x + self.temporal_embedding[:, :num_frames].to(dtype=x.dtype, device=x.device)
        x = x.reshape(batch_size, num_frames * height * width, self.resampler_dim)

        queries = self.intent_queries.to(dtype=x.dtype, device=x.device).expand(batch_size, -1, -1)
        for block in self.blocks:
            queries = block(queries, x)
        return self.output_proj(self.output_norm(queries))
