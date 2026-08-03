"""DINO-space Video DiT Expert for FastWAM_DINO.

This module implements a Video DiT that operates in DINO feature space instead of
VAE latent space. It shares the same MoT interface (pre_dit/post_dit) as the original
WanVideoDiT, but is adapted for DINO patch features as input/output.

Key differences from WanVideoDiT:
- in_dim = DINO feature_dim (1024 for ViT-L) instead of VAE channels (48)
- patch_size = [1, 1, 1] since DINO already produces spatial patches
- Optional learnable latent patch merge; defaults to a linear projection
- Simpler spatial structure: tokens are already at patch level
"""

import math
import os
from pathlib import Path
from typing import Any, Dict, Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F
from einops import rearrange

from fastwam.utils.logging_config import get_logger

from .helpers.gradient import gradient_checkpoint_forward
from .wan_video_dit import (
    DiTBlock,
    precompute_freqs_cis_3d,
    sinusoidal_embedding_1d,
)

logger = get_logger(__name__)


class DinoHead(nn.Module):
    """Output head for DINO-space Video DiT. Maps hidden_dim back to DINO feature_dim."""

    def __init__(self, hidden_dim: int, out_dim: int, eps: float = 1e-6):
        super().__init__()
        self.norm = nn.LayerNorm(hidden_dim, eps=eps, elementwise_affine=False)
        self.head = nn.Linear(hidden_dim, out_dim)
        self.modulation = nn.Parameter(torch.randn(1, 2, hidden_dim) / hidden_dim**0.5)

    def forward(self, x: torch.Tensor, t_mod: torch.Tensor) -> torch.Tensor:
        if len(t_mod.shape) == 3:
            # Per-token modulation: t_mod shape [B, seq_len, hidden_dim]
            shift, scale = (
                self.modulation.unsqueeze(0).to(dtype=t_mod.dtype, device=t_mod.device)
                + t_mod.unsqueeze(2)
            ).chunk(2, dim=2)
            x = self.head(self.norm(x) * (1 + scale.squeeze(2)) + shift.squeeze(2))
        else:
            shift, scale = (
                self.modulation.to(dtype=t_mod.dtype, device=t_mod.device) + t_mod
            ).chunk(2, dim=1)
            x = self.head(self.norm(x) * (1 + scale) + shift)
        return x


class DinoVideoDiT(nn.Module):
    """Video DiT Expert operating in DINO feature space.

    This module provides the same pre_dit/post_dit interface as WanVideoDiT,
    making it a drop-in replacement for the video expert in MoT.

    Architecture:
    - Input: [B, D_dino, T, H_grid, W_grid] DINO features (noisy during training)
    - Linear projection → hidden_dim tokens
    - N DiTBlocks with self-attention, cross-attention, FFN
    - DinoHead → [B, D_dino, T, H_grid, W_grid] predicted velocity

    Args:
        hidden_dim: Transformer hidden dimension (must satisfy num_heads * attn_head_dim).
        dino_dim: DINO feature dimension (e.g., 1024 for ViT-L).
        ffn_dim: Feed-forward network intermediate dimension.
        text_dim: Text encoder embedding dimension.
        freq_dim: Sinusoidal timestep embedding dimension.
        eps: LayerNorm epsilon.
        num_heads: Number of attention heads (must match ActionDiT for MoT).
        attn_head_dim: Per-head dimension (must match ActionDiT for MoT).
        num_layers: Number of DiT blocks (must match ActionDiT for MoT).
        video_attention_mask_mode: Attention mask mode for video tokens.
        use_gradient_checkpointing: Whether to use gradient checkpointing.
    """

    VIDEO_BACKBONE_SKIP_PREFIXES = (
        "input_projection.",
        "patch_embedding.",
        "view_embedding.",
        "head.",
        "freqs",
    )
    VIDEO_BACKBONE_META_KEYS = (
        "hidden_dim",
        "dino_dim",
        "ffn_dim",
        "num_layers",
        "num_heads",
        "attn_head_dim",
        "text_dim",
        "freq_dim",
        "eps",
    )

    def __init__(
        self,
        hidden_dim: int,
        dino_dim: int,
        ffn_dim: int,
        text_dim: int,
        freq_dim: int = 256,
        eps: float = 1e-6,
        num_heads: int = 24,
        attn_head_dim: int = 128,
        num_layers: int = 30,
        video_attention_mask_mode: str = "first_frame_causal",
        use_gradient_checkpointing: bool = False,
        latent_patch_size: Tuple[int, int, int] = (1, 1, 1),
        latent_patch_mode: str = "flat",
        latent_num_views: int = 1,
        output_patch_space: str = "dense",
    ):
        super().__init__()
        self.hidden_dim = hidden_dim
        self.dino_dim = dino_dim
        self.ffn_dim = ffn_dim
        self.text_dim = text_dim
        self.freq_dim = freq_dim
        self.num_heads = num_heads
        self.attn_head_dim = attn_head_dim
        self.num_layers = num_layers
        self.video_attention_mask_mode = video_attention_mask_mode
        self.use_gradient_checkpointing = use_gradient_checkpointing
        if len(latent_patch_size) != 3:
            raise ValueError(f"`latent_patch_size` must be a 3-tuple/list, got {latent_patch_size}")
        self.latent_patch_size = tuple(int(v) for v in latent_patch_size)
        if any(v <= 0 for v in self.latent_patch_size):
            raise ValueError(f"`latent_patch_size` values must be positive, got {self.latent_patch_size}")
        self.latent_patch_prod = math.prod(self.latent_patch_size)
        self.latent_patch_mode = str(latent_patch_mode)
        if self.latent_patch_mode not in {"flat", "view"}:
            raise ValueError(
                f"`latent_patch_mode` must be 'flat' or 'view', got {self.latent_patch_mode!r}"
            )
        self.latent_num_views = int(latent_num_views)
        if self.latent_num_views <= 0:
            raise ValueError(f"`latent_num_views` must be positive, got {self.latent_num_views}")
        if self.latent_patch_mode == "flat" and self.latent_num_views != 1:
            logger.warning(
                "DinoVideoDiT latent_patch_mode='flat' ignores latent_num_views=%s.",
                self.latent_num_views,
            )
        if self.latent_patch_size[0] != 1:
            raise ValueError(
                "DinoVideoDiT currently supports only temporal latent_patch_size=1. "
                "First-frame conditioning and per-token timestep masking assume that "
                "each DiT video token belongs to a single original frame; got "
                f"latent_patch_size={self.latent_patch_size}."
            )
        self.output_patch_space = str(output_patch_space).strip().lower()
        if self.output_patch_space not in {"dense", "merged"}:
            raise ValueError(
                f"`output_patch_space` must be 'dense' or 'merged', got {output_patch_space!r}"
            )

        if num_heads <= 0:
            raise ValueError(f"`num_heads` must be > 0, got {num_heads}")
        if attn_head_dim <= 0:
            raise ValueError(f"`attn_head_dim` must be > 0, got {attn_head_dim}")
        if attn_head_dim % 2 != 0:
            raise ValueError(f"`attn_head_dim` must be even for RoPE, got {attn_head_dim}")

        # Input patchifier: default is one DINO patch per token.  A larger
        # latent_patch_size is a learnable alternative to fixed DINO avg-pool.
        if self.latent_patch_size == (1, 1, 1):
            self.input_projection = nn.Linear(dino_dim, hidden_dim)
            self.patch_embedding = None
        else:
            self.input_projection = None
            self.patch_embedding = nn.Conv3d(
                dino_dim,
                hidden_dim,
                kernel_size=self.latent_patch_size,
                stride=self.latent_patch_size,
            )
        if self.latent_patch_mode == "view":
            self.view_embedding = nn.Parameter(torch.zeros(self.latent_num_views, hidden_dim))
            nn.init.normal_(self.view_embedding, mean=0.0, std=hidden_dim**-0.5)
        else:
            self.view_embedding = None

        # Text embedding
        self.text_embedding = nn.Sequential(
            nn.Linear(text_dim, hidden_dim),
            nn.GELU(approximate="tanh"),
            nn.Linear(hidden_dim, hidden_dim),
        )

        # Time embedding
        self.time_embedding = nn.Sequential(
            nn.Linear(freq_dim, hidden_dim),
            nn.SiLU(),
            nn.Linear(hidden_dim, hidden_dim),
        )
        self.time_projection = nn.Sequential(
            nn.SiLU(),
            nn.Linear(hidden_dim, hidden_dim * 6),
        )

        # Transformer blocks
        self.blocks = nn.ModuleList(
            [DiTBlock(hidden_dim, attn_head_dim, num_heads, ffn_dim, eps) for _ in range(num_layers)]
        )

        # Output head.  The default dense mode preserves the historical behavior:
        # each merged token predicts all DINO velocities inside its patch group.
        # The optional merged mode predicts one DINO velocity per merged token;
        # training then compares against a fixed patch-mean target in the same
        # token grid, avoiding dense DINO super-resolution.
        head_out_dim = dino_dim * self.latent_patch_prod
        if self.output_patch_space == "merged":
            head_out_dim = dino_dim
        self.head = DinoHead(hidden_dim, head_out_dim, eps)

        # RoPE frequency tables (3D: temporal + spatial H + spatial W)
        self.freqs = precompute_freqs_cis_3d(attn_head_dim)

        # For compatibility with fastwam.py first-frame fuse logic
        self.fuse_vae_embedding_in_latents = True
        self.seperated_timestep = True

        if use_gradient_checkpointing:
            logger.info("DinoVideoDiT: gradient checkpointing enabled.")
        if self.latent_patch_size != (1, 1, 1):
            logger.info(
                "DinoVideoDiT learnable latent patch merge enabled: latent_patch_size=%s, "
                "latent_patch_mode=%s, latent_num_views=%s, output_patch_space=%s",
                self.latent_patch_size,
                self.latent_patch_mode,
                self.latent_num_views,
                self.output_patch_space,
            )

    def init_from_wan_dit(self, wan_dit_state_dict: dict[str, torch.Tensor]) -> None:
        """Initialize DinoVideoDiT Transformer layers from a Wan2.2 Video DiT state_dict.

        Copies weights for: blocks.*, text_embedding.*, time_embedding.*,
        time_projection.*.  Skips input_projection, head, freqs (which are
        architecture-specific to the DINO variant).

        Args:
            wan_dit_state_dict: state_dict from a WanVideoDiT instance
                (e.g. loaded via ``load_wan22_ti2v_5b_components``).
        """
        transferable_prefixes = ("blocks.", "text_embedding.", "time_embedding.", "time_projection.")
        own_state = self.state_dict()

        copied, skipped_missing, skipped_shape = 0, 0, 0
        for key, wan_tensor in wan_dit_state_dict.items():
            if not any(key.startswith(p) for p in transferable_prefixes):
                continue
            if key not in own_state:
                skipped_missing += 1
                continue
            if own_state[key].shape != wan_tensor.shape:
                logger.warning(
                    f"Shape mismatch for '{key}': "
                    f"DinoVideoDiT={tuple(own_state[key].shape)}, "
                    f"WanVideoDiT={tuple(wan_tensor.shape)}. Skipping."
                )
                skipped_shape += 1
                continue
            own_state[key] = wan_tensor

        self.load_state_dict(own_state, strict=True)
        total_transferable = sum(
            1 for k in wan_dit_state_dict if any(k.startswith(p) for p in transferable_prefixes)
        )
        copied = total_transferable - skipped_missing - skipped_shape
        logger.info(
            f"DinoVideoDiT initialized from Wan2.2 DiT: "
            f"copied={copied}, skipped_missing={skipped_missing}, "
            f"skipped_shape={skipped_shape}"
        )

    @classmethod
    def backbone_key_set(cls, keys) -> set[str]:
        return {
            key
            for key in keys
            if not any(
                key == prefix.rstrip(".") or key.startswith(prefix)
                for prefix in cls.VIDEO_BACKBONE_SKIP_PREFIXES
            )
        }

    def load_preprocessed_backbone(self, video_dit_pretrained_path: str) -> None:
        """Load a preprocessed DinoVideoDiT backbone payload.

        The payload is produced by ``scripts/preprocess_dino_video_dit_backbone.py``.
        It initializes Transformer/text/time layers while keeping DINO-specific
        input/output projections random.
        """
        p = Path(video_dit_pretrained_path)
        if not p.is_absolute():
            p = Path(__file__).resolve().parents[4] / p
        if not os.path.isfile(p):
            raise FileNotFoundError(f"`video_dit_pretrained_path` does not exist: {p}")

        payload = torch.load(str(p), map_location="cpu")
        if not isinstance(payload, dict):
            raise ValueError(f"Invalid DinoVideoDiT backbone payload type from {p}: {type(payload)}")

        policy = payload.get("policy", {})
        if policy:
            logger.info("DinoVideoDiT backbone payload policy: %s", policy)

        meta = payload.get("meta")
        if not isinstance(meta, dict):
            raise ValueError(f"`meta` must be a dict in {p}, got {type(meta)}")
        expected_meta = {
            "hidden_dim": int(self.hidden_dim),
            "dino_dim": int(self.dino_dim),
            "ffn_dim": int(self.ffn_dim),
            "num_layers": int(self.num_layers),
            "num_heads": int(self.num_heads),
            "attn_head_dim": int(self.attn_head_dim),
            "text_dim": int(self.text_dim),
            "freq_dim": int(self.freq_dim),
        }
        for key in self.VIDEO_BACKBONE_META_KEYS:
            if key not in meta:
                raise ValueError(f"`meta.{key}` missing in {p}")
            if key == "eps":
                continue
            if int(meta[key]) != expected_meta[key]:
                raise ValueError(
                    f"`meta.{key}` mismatch in {p}: expected {expected_meta[key]}, got {meta[key]}"
                )

        backbone_state_dict = payload.get("backbone_state_dict")
        if not isinstance(backbone_state_dict, dict):
            raise ValueError(f"`backbone_state_dict` must be a dict in {p}, got {type(backbone_state_dict)}")

        own_state = self.state_dict()
        expected_backbone_keys = self.backbone_key_set(own_state.keys())
        provided_keys = set(backbone_state_dict.keys())
        missing_keys = sorted(expected_backbone_keys - provided_keys)
        unexpected_keys = sorted(provided_keys - expected_backbone_keys)
        if missing_keys or unexpected_keys:
            raise ValueError(
                "DinoVideoDiT backbone key mismatch in preprocessed payload. "
                f"missing={missing_keys[:10]}{'...' if len(missing_keys) > 10 else ''}, "
                f"unexpected={unexpected_keys[:10]}{'...' if len(unexpected_keys) > 10 else ''}"
            )

        merged_state = dict(own_state)
        for key in expected_backbone_keys:
            value = backbone_state_dict[key]
            if not isinstance(value, torch.Tensor):
                raise ValueError(f"`backbone_state_dict[{key}]` must be torch.Tensor in {p}, got {type(value)}")
            target = merged_state[key]
            if tuple(value.shape) != tuple(target.shape):
                raise ValueError(
                    f"Shape mismatch for `{key}` in {p}: expected {tuple(target.shape)}, got {tuple(value.shape)}"
                )
            merged_state[key] = value.to(device=target.device, dtype=target.dtype)

        self.load_state_dict(merged_state, strict=True)
        logger.info(
            "Loaded DinoVideoDiT backbone from %s (keys=%d; random_kept_prefixes=%s).",
            p,
            len(expected_backbone_keys),
            list(self.VIDEO_BACKBONE_SKIP_PREFIXES),
        )

    def build_video_to_video_mask(
        self,
        video_seq_len: int,
        video_tokens_per_frame: int,
        device: torch.device,
        video_prefix_len: int = 0,
    ) -> torch.Tensor:
        """Build attention mask for video self-attention within MoT."""
        if video_seq_len <= 0:
            raise ValueError(f"`video_seq_len` must be positive, got {video_seq_len}")
        if video_tokens_per_frame <= 0:
            raise ValueError(f"`video_tokens_per_frame` must be positive, got {video_tokens_per_frame}")
        video_prefix_len = int(video_prefix_len)
        if video_prefix_len < 0:
            raise ValueError(f"`video_prefix_len` must be non-negative, got {video_prefix_len}")
        if video_prefix_len > video_seq_len:
            raise ValueError(
                f"`video_prefix_len` cannot exceed `video_seq_len`, got "
                f"{video_prefix_len} and {video_seq_len}"
            )

        if self.video_attention_mask_mode == "bidirectional":
            return torch.ones((video_seq_len, video_seq_len), dtype=torch.bool, device=device)

        if video_prefix_len:
            original_seq_len = video_seq_len - video_prefix_len
            if original_seq_len <= 0:
                raise ValueError("Video prefix cannot consume the entire video sequence.")
            original_mask = self.build_video_to_video_mask(
                video_seq_len=original_seq_len,
                video_tokens_per_frame=video_tokens_per_frame,
                device=device,
                video_prefix_len=0,
            )
            mask = torch.zeros((video_seq_len, video_seq_len), dtype=torch.bool, device=device)
            mask[:video_prefix_len, :video_prefix_len] = True
            first_frame_end = video_prefix_len + min(video_tokens_per_frame, original_seq_len)
            mask[:video_prefix_len, video_prefix_len:first_frame_end] = True
            mask[video_prefix_len:, :video_prefix_len] = True
            mask[video_prefix_len:, video_prefix_len:] = original_mask
            return mask

        if self.video_attention_mask_mode == "per_frame_causal":
            if video_seq_len % video_tokens_per_frame != 0:
                raise ValueError(
                    f"`video_seq_len` must be divisible by `video_tokens_per_frame`, "
                    f"got {video_seq_len} and {video_tokens_per_frame}"
                )
            num_frames = video_seq_len // video_tokens_per_frame
            frame_causal = torch.tril(
                torch.ones((num_frames, num_frames), dtype=torch.bool, device=device)
            )
            return frame_causal.repeat_interleave(video_tokens_per_frame, dim=0).repeat_interleave(
                video_tokens_per_frame, dim=1
            )

        if self.video_attention_mask_mode == "first_frame_causal":
            video_mask = torch.ones((video_seq_len, video_seq_len), dtype=torch.bool, device=device)
            first_frame_tokens = min(video_tokens_per_frame, video_seq_len)
            video_mask[:first_frame_tokens, first_frame_tokens:] = False
            return video_mask

        raise ValueError(f"Unsupported video attention mask mode: {self.video_attention_mask_mode}")

    def _zero_time_modulation(
        self,
        batch_size: int,
        seq_len: int,
        device: torch.device,
        dtype: torch.dtype,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        zero_steps = torch.zeros((batch_size * seq_len,), dtype=dtype, device=device)
        t = self.time_embedding(sinusoidal_embedding_1d(self.freq_dim, zero_steps))
        t = t.reshape(batch_size, seq_len, self.hidden_dim)
        t_mod = self.time_projection(t).unflatten(2, (6, self.hidden_dim))
        return t, t_mod

    def prepend_video_prefix_tokens(
        self,
        pre_state: Dict[str, Any],
        prefix_tokens: Optional[torch.Tensor],
    ) -> Dict[str, Any]:
        """Prepend text-space condition tokens to the VideoDiT token sequence.

        The prefix is added after DINO grid patchification, so it never changes
        the stored grid metadata used by `post_dit` and video-loss targets.
        """
        if prefix_tokens is None:
            return pre_state
        if prefix_tokens.ndim != 3:
            raise ValueError(
                f"`prefix_tokens` must be 3D [B,K,D], got shape {tuple(prefix_tokens.shape)}"
            )

        tokens = pre_state["tokens"]
        batch_size, _, hidden_dim = tokens.shape
        if prefix_tokens.shape[0] != batch_size:
            raise ValueError(
                f"Batch mismatch between video tokens and prefix tokens: "
                f"{batch_size} vs {prefix_tokens.shape[0]}"
            )
        if prefix_tokens.shape[2] != self.text_dim:
            raise ValueError(
                f"Video prefix token dim must match text_dim={self.text_dim}, "
                f"got {prefix_tokens.shape[2]}"
            )

        prefix_len = int(prefix_tokens.shape[1])
        if prefix_len == 0:
            return pre_state
        prefix_emb = self.text_embedding(
            prefix_tokens.to(device=tokens.device, dtype=tokens.dtype)
        )
        if prefix_emb.shape[2] != hidden_dim:
            raise ValueError(
                f"Projected prefix hidden dim mismatch: {prefix_emb.shape[2]} vs {hidden_dim}"
            )

        freqs = pre_state["freqs"]
        prefix_freqs = torch.ones(
            (prefix_len, 1, freqs.shape[-1]),
            device=freqs.device,
            dtype=freqs.dtype,
        )

        t_mod = pre_state["t_mod"]
        _, prefix_t_mod = self._zero_time_modulation(
            batch_size=batch_size,
            seq_len=prefix_len,
            device=tokens.device,
            dtype=tokens.dtype,
        )
        if t_mod.ndim == 4:
            new_t_mod = torch.cat([prefix_t_mod.to(dtype=t_mod.dtype), t_mod], dim=1)
        elif t_mod.ndim == 3:
            expanded_t_mod = t_mod.unsqueeze(1).expand(
                -1, tokens.shape[1], -1, -1
            )
            new_t_mod = torch.cat(
                [prefix_t_mod.to(dtype=t_mod.dtype), expanded_t_mod],
                dim=1,
            )
        else:
            raise ValueError(f"Unexpected `t_mod` shape {tuple(t_mod.shape)}")

        context_mask = pre_state["context_mask"]
        prefix_context_mask = context_mask[:, :1, :].expand(-1, prefix_len, -1)

        pre_state = dict(pre_state)
        pre_state["tokens"] = torch.cat([prefix_emb, tokens], dim=1)
        pre_state["freqs"] = torch.cat([prefix_freqs, freqs], dim=0)
        pre_state["t_mod"] = new_t_mod
        pre_state["context_mask"] = torch.cat([prefix_context_mask, context_mask], dim=1)
        pre_state["meta"] = dict(pre_state["meta"])
        pre_state["meta"]["video_prefix_len"] = prefix_len
        return pre_state

    def patchify(self, x: torch.Tensor) -> torch.Tensor:
        """Convert DINO latents to DiT tokens.

        For the default patch size this is equivalent to the old per-patch
        linear projection.  For larger patch sizes this uses a learnable Conv3d
        patch embedding, matching the original WanVideoDiT pattern.
        """
        if self.latent_patch_size == (1, 1, 1):
            if self.latent_patch_mode == "view":
                x_tokens = self._view_rearrange_latents(x)
                x_tokens = rearrange(x_tokens, "b v d t h w -> b (t v h w) d")
            else:
                x_tokens = rearrange(x, "b d t h w -> b (t h w) d")
            assert self.input_projection is not None
            return self.input_projection(x_tokens)
        assert self.patch_embedding is not None
        if self.latent_patch_mode == "view":
            x = self._view_rearrange_latents(x)
            batch_size, num_views = x.shape[:2]
            x = rearrange(x, "b v d t h w -> (b v) d t h w")
            x = self.patch_embedding(x)
            x = rearrange(x, "(b v) d t h w -> b (t v h w) d", b=batch_size, v=num_views)
        else:
            x = self.patch_embedding(x)
            x = rearrange(x, "b d t h w -> b (t h w) d")
        return x

    def unpatchify(self, x_tokens: torch.Tensor, grid_size: Tuple[int, int, int]) -> torch.Tensor:
        f, h, w = grid_size
        pt, ph, pw = self.latent_patch_size
        if self.latent_patch_mode == "view":
            num_views = self.latent_num_views
            if self.latent_patch_size == (1, 1, 1):
                return rearrange(
                    x_tokens,
                    "b (t v h w) d -> b d t h (v w)",
                    v=num_views,
                    t=f,
                    h=h,
                    w=w,
                )
            return rearrange(
                x_tokens,
                "b (t v h w) (d pt ph pw) -> b d (t pt) (h ph) (v w pw)",
                v=num_views,
                t=f,
                h=h,
                w=w,
                pt=pt,
                ph=ph,
                pw=pw,
            )
        if self.latent_patch_size == (1, 1, 1):
            return rearrange(x_tokens, "b (t h w) d -> b d t h w", t=f, h=h, w=w)
        return rearrange(
            x_tokens,
            "b (t h w) (d pt ph pw) -> b d (t pt) (h ph) (w pw)",
            t=f,
            h=h,
            w=w,
            pt=pt,
            ph=ph,
            pw=pw,
        )

    def unpatchify_merged(self, x_tokens: torch.Tensor, grid_size: Tuple[int, int, int]) -> torch.Tensor:
        """Convert one prediction per merged token to a compact DINO grid."""
        f, h, w = grid_size
        if self.latent_patch_mode == "view":
            return rearrange(
                x_tokens,
                "b (t v h w) d -> b d t h (v w)",
                v=self.latent_num_views,
                t=f,
                h=h,
                w=w,
            )
        return rearrange(x_tokens, "b (t h w) d -> b d t h w", t=f, h=h, w=w)

    def target_to_output_space(self, x: torch.Tensor) -> torch.Tensor:
        """Map dense DINO velocity targets to the configured output space.

        Dense mode returns the target unchanged.  Merged mode computes a fixed
        average over each latent patch group, producing one DINO target vector
        per merged DiT token.  This keeps the supervision space aligned with
        the compressed token grid without adding a learnable teacher branch.
        """
        if self.output_patch_space == "dense" or self.latent_patch_size == (1, 1, 1):
            return x
        if x.ndim != 5:
            raise ValueError(f"`x` must be 5D [B,D,T,H,W], got shape {tuple(x.shape)}")
        pt, ph, pw = self.latent_patch_size
        _, _, num_frames, height_grid, width_grid = x.shape
        patch_width_grid = width_grid
        if self.latent_patch_mode == "view":
            if width_grid % self.latent_num_views != 0:
                raise ValueError(
                    "DINO latent width must be divisible by latent_num_views for merged target, "
                    f"got width={width_grid} and latent_num_views={self.latent_num_views}."
                )
            patch_width_grid = width_grid // self.latent_num_views
        if num_frames % pt != 0 or height_grid % ph != 0 or patch_width_grid % pw != 0:
            raise ValueError(
                "DINO latent target grid must be divisible by latent_patch_size, "
                f"got grid={(num_frames, height_grid, patch_width_grid)} "
                f"(original width={width_grid}, mode={self.latent_patch_mode}) and "
                f"latent_patch_size={self.latent_patch_size}."
            )
        if self.latent_patch_mode == "view":
            return rearrange(
                x,
                "b d (t pt) (h ph) (v w pw) -> b d t h (v w) (pt ph pw)",
                v=self.latent_num_views,
                pt=pt,
                ph=ph,
                pw=pw,
            ).mean(dim=-1)
        return rearrange(
            x,
            "b d (t pt) (h ph) (w pw) -> b d t h w (pt ph pw)",
            pt=pt,
            ph=ph,
            pw=pw,
        ).mean(dim=-1)

    def _view_rearrange_latents(self, x: torch.Tensor) -> torch.Tensor:
        if x.shape[-1] % self.latent_num_views != 0:
            raise ValueError(
                "DINO latent width must be divisible by latent_num_views for view-aware patching, "
                f"got width={x.shape[-1]} and latent_num_views={self.latent_num_views}."
            )
        return rearrange(x, "b d t h (v w) -> b v d t h w", v=self.latent_num_views)

    def _add_view_embedding(
        self,
        tokens: torch.Tensor,
        patched_frames: int,
        patched_h: int,
        patched_w: int,
    ) -> torch.Tensor:
        if self.latent_patch_mode != "view":
            return tokens
        assert self.view_embedding is not None
        batch_size, seq_len, hidden_dim = tokens.shape
        expected_seq_len = patched_frames * self.latent_num_views * patched_h * patched_w
        if seq_len != expected_seq_len:
            raise ValueError(
                "View-aware token sequence length mismatch, "
                f"got seq_len={seq_len}, expected={expected_seq_len} from "
                f"grid={(patched_frames, self.latent_num_views, patched_h, patched_w)}."
            )
        view_emb = self.view_embedding.to(device=tokens.device, dtype=tokens.dtype)
        view_emb = view_emb.view(1, 1, self.latent_num_views, 1, 1, hidden_dim)
        tokens = tokens.reshape(
            batch_size,
            patched_frames,
            self.latent_num_views,
            patched_h,
            patched_w,
            hidden_dim,
        )
        tokens = tokens + view_emb
        return tokens.reshape(batch_size, seq_len, hidden_dim)

    def _build_rope_freqs(
        self,
        patched_frames: int,
        patched_h: int,
        patched_w: int,
        device: torch.device,
    ) -> torch.Tensor:
        freqs = torch.cat(
            [
                self.freqs[0][:patched_frames].view(patched_frames, 1, 1, -1).expand(
                    patched_frames, patched_h, patched_w, -1
                ),
                self.freqs[1][:patched_h].view(1, patched_h, 1, -1).expand(
                    patched_frames, patched_h, patched_w, -1
                ),
                self.freqs[2][:patched_w].view(1, 1, patched_w, -1).expand(
                    patched_frames, patched_h, patched_w, -1
                ),
            ],
            dim=-1,
        )
        if self.latent_patch_mode == "view":
            freqs = freqs.unsqueeze(1).expand(
                patched_frames,
                self.latent_num_views,
                patched_h,
                patched_w,
                -1,
            )
            return freqs.reshape(
                patched_frames * self.latent_num_views * patched_h * patched_w,
                1,
                -1,
            ).to(device)
        freqs = freqs.reshape(patched_frames * patched_h * patched_w, 1, -1)
        return freqs.to(device)

    def pre_dit(
        self,
        x: torch.Tensor,
        timestep: torch.Tensor,
        context: torch.Tensor,
        context_mask: Optional[torch.Tensor] = None,
        action: Optional[torch.Tensor] = None,
        fuse_vae_embedding_in_latents: bool = False,
        control_camera_latents_input: Optional[torch.Tensor] = None,
    ) -> Dict[str, Any]:
        """Prepare tokens, freqs, t_mod, context for MoT forward.

        Args:
            x: [B, D_dino, T, H_grid, W_grid] noisy DINO features (or clean first frame).
            timestep: [B] diffusion timestep.
            context: [B, L, text_dim] text embeddings.
            context_mask: [B, L] boolean mask.
            action: Not used for DINO video expert (action conditioning via cross-attn is optional).
            fuse_vae_embedding_in_latents: If True, first frame has timestep=0 (clean conditioning).

        Returns:
            Dict with keys: tokens, freqs, t_mod, context, context_mask, meta.
        """
        if x.ndim != 5:
            raise ValueError(f"`x` must be 5D [B, D, T, H, W], got shape {tuple(x.shape)}")

        batch_size, dino_dim, num_frames, height_grid, width_grid = x.shape
        patch_t, patch_h, patch_w = self.latent_patch_size
        patch_width_grid = width_grid
        if self.latent_patch_mode == "view":
            if width_grid % self.latent_num_views != 0:
                raise ValueError(
                    "DINO latent width must be divisible by latent_num_views for view-aware patching, "
                    f"got width={width_grid} and latent_num_views={self.latent_num_views}."
                )
            patch_width_grid = width_grid // self.latent_num_views
        if num_frames % patch_t != 0 or height_grid % patch_h != 0 or patch_width_grid % patch_w != 0:
            raise ValueError(
                "DINO latent grid must be divisible by latent_patch_size, "
                f"got grid={(num_frames, height_grid, patch_width_grid)} "
                f"(original width={width_grid}, mode={self.latent_patch_mode}) and "
                f"latent_patch_size={self.latent_patch_size}."
            )
        patched_frames = num_frames // patch_t
        patched_h = height_grid // patch_h
        patched_w = patch_width_grid // patch_w
        tokens_per_frame = patched_h * patched_w
        if self.latent_patch_mode == "view":
            tokens_per_frame *= self.latent_num_views

        if context.ndim != 3:
            raise ValueError(f"`context` must be 3D [B, L, D], got shape {tuple(context.shape)}")
        if timestep.ndim != 1:
            raise ValueError(f"`timestep` must be 1D [B], got shape {tuple(timestep.shape)}")

        if context_mask is None:
            context_mask = torch.ones(
                (context.shape[0], context.shape[1]), dtype=torch.bool, device=context.device
            )

        if timestep.shape[0] == 1 and batch_size > 1:
            timestep = timestep.expand(batch_size)

        # Time embedding with per-token timestep (first frame = 0 if fuse mode)
        if fuse_vae_embedding_in_latents:
            token_timesteps = torch.ones(
                (batch_size, patched_frames, tokens_per_frame),
                dtype=timestep.dtype,
                device=timestep.device,
            ) * timestep.view(batch_size, 1, 1)
            token_timesteps[:, 0, :] = 0  # First frame is clean (no noise)
            token_timesteps = token_timesteps.reshape(batch_size, -1)
            token_t_emb = sinusoidal_embedding_1d(self.freq_dim, token_timesteps.reshape(-1))
            t = self.time_embedding(token_t_emb).reshape(batch_size, -1, self.hidden_dim)
            t_mod = self.time_projection(t).unflatten(2, (6, self.hidden_dim))
        else:
            t_emb = sinusoidal_embedding_1d(self.freq_dim, timestep)
            t = self.time_embedding(t_emb)
            t_mod = self.time_projection(t).unflatten(1, (6, self.hidden_dim))

        x_tokens = self.patchify(x)
        x_tokens = self._add_view_embedding(
            x_tokens,
            patched_frames=patched_frames,
            patched_h=patched_h,
            patched_w=patched_w,
        )

        # Text embedding
        context_emb = self.text_embedding(context)

        # Expand context mask for cross-attention: [B, L] → [B, seq_len, L]
        seq_len = patched_frames * tokens_per_frame
        context_mask_expanded = context_mask.unsqueeze(1).expand(-1, seq_len, -1)

        # RoPE frequencies
        freqs = self._build_rope_freqs(
            patched_frames=patched_frames,
            patched_h=patched_h,
            patched_w=patched_w,
            device=x_tokens.device,
        )

        return {
            "tokens": x_tokens,
            "freqs": freqs,
            "t": t,
            "t_mod": t_mod,
            "context": context_emb,
            "context_mask": context_mask_expanded,
            "meta": {
                "grid_size": (patched_frames, patched_h, patched_w),
                "original_grid_size": (num_frames, height_grid, width_grid),
                "latent_patch_size": self.latent_patch_size,
                "latent_patch_mode": self.latent_patch_mode,
                "latent_num_views": self.latent_num_views,
                "tokens_per_frame": tokens_per_frame,
                "batch_size": batch_size,
            },
        }

    def post_dit(self, x_tokens: torch.Tensor, pre_state: Dict[str, Any]) -> torch.Tensor:
        """Convert DiT output tokens back to DINO feature space.

        Args:
            x_tokens: [B, seq_len, hidden_dim] transformer output tokens.
            pre_state: Dict from pre_dit containing metadata.

        Returns:
            [B, D_dino, T, H_grid, W_grid] predicted velocity in DINO space.
        """
        prefix_len = int(pre_state.get("meta", {}).get("video_prefix_len", 0))
        if prefix_len:
            if x_tokens.shape[1] <= prefix_len:
                raise ValueError(
                    f"Cannot strip video prefix of length {prefix_len} from "
                    f"token sequence length {x_tokens.shape[1]}."
                )
            x_tokens = x_tokens[:, prefix_len:]
        grid_size = pre_state["meta"]["grid_size"]

        # Apply output head: [B, seq_len, hidden_dim] → configured DINO output space.
        x = self.head(x_tokens, pre_state["t"])

        if self.output_patch_space == "merged":
            return self.unpatchify_merged(x, grid_size)
        return self.unpatchify(x, grid_size)

    def forward(
        self,
        x: torch.Tensor,
        timestep: torch.Tensor,
        context: torch.Tensor,
        context_mask: Optional[torch.Tensor] = None,
        fuse_vae_embedding_in_latents: bool = False,
    ) -> torch.Tensor:
        """Full forward pass (standalone, without MoT)."""
        pre_state = self.pre_dit(
            x=x,
            timestep=timestep,
            context=context,
            context_mask=context_mask,
            fuse_vae_embedding_in_latents=fuse_vae_embedding_in_latents,
        )
        x_tokens = pre_state["tokens"]
        context_emb = pre_state["context"]
        t_mod = pre_state["t_mod"]
        freqs = pre_state["freqs"]
        context_attn_mask = pre_state["context_mask"]

        self_attn_mask = self.build_video_to_video_mask(
            video_seq_len=x_tokens.shape[1],
            video_tokens_per_frame=int(pre_state["meta"]["tokens_per_frame"]),
            device=x_tokens.device,
        ) if self.video_attention_mask_mode != "bidirectional" else None

        for block in self.blocks:
            if self.use_gradient_checkpointing:
                x_tokens = gradient_checkpoint_forward(
                    block,
                    self.use_gradient_checkpointing,
                    x_tokens, context_emb, t_mod, freqs,
                    context_mask=context_attn_mask, self_attn_mask=self_attn_mask,
                )
            else:
                x_tokens = block(
                    x_tokens, context_emb, t_mod, freqs,
                    context_mask=context_attn_mask, self_attn_mask=self_attn_mask,
                )

        return self.post_dit(x_tokens, pre_state)
