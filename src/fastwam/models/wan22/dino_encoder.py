"""Frozen DINOv3 encoder for extracting semantic patch features from video frames.

This module wraps a pretrained DINOv3 ViT model and provides a clean interface
for encoding video tensors into per-frame patch features suitable for
diffusion-based dynamics learning in DINO latent space.

Supports loading from:
- Local safetensors file (recommended for DINOv3)
- torch.hub (DINOv2/v3)
- HuggingFace Transformers AutoModel
"""

import os
from typing import Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F

from fastwam.utils.logging_config import get_logger

logger = get_logger(__name__)


def _rotate_half(x: torch.Tensor) -> torch.Tensor:
    x1 = x[..., : x.shape[-1] // 2]
    x2 = x[..., x.shape[-1] // 2 :]
    return torch.cat((-x2, x1), dim=-1)


def _get_patches_center_coordinates(
    num_patches_h: int,
    num_patches_w: int,
    dtype: torch.dtype,
    device: torch.device,
) -> torch.Tensor:
    coords_h = torch.arange(0.5, num_patches_h, dtype=dtype, device=device) / num_patches_h
    coords_w = torch.arange(0.5, num_patches_w, dtype=dtype, device=device) / num_patches_w
    coords = torch.stack(torch.meshgrid(coords_h, coords_w, indexing="ij"), dim=-1)
    return coords.flatten(0, 1).mul_(2.0).sub_(1.0)


def _build_rope_embeddings(
    height: int,
    width: int,
    patch_size: int,
    head_dim: int,
    dtype: torch.dtype,
    device: torch.device,
    rope_theta: float = 100.0,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Build DINOv3 2D RoPE embeddings for patch tokens."""
    num_patches_h = height // patch_size
    num_patches_w = width // patch_size
    with torch.autocast(device_type=device.type if device.type != "mps" else "cpu", enabled=False):
        coords = _get_patches_center_coordinates(
            num_patches_h,
            num_patches_w,
            dtype=torch.float32,
            device=device,
        )
        inv_freq = 1.0 / rope_theta ** torch.arange(
            0,
            1,
            4 / head_dim,
            dtype=torch.float32,
            device=device,
        )
        angles = 2 * torch.pi * coords[:, :, None] * inv_freq[None, None, :]
        angles = angles.flatten(1, 2).tile(2)
        cos = torch.cos(angles).to(dtype=dtype)
        sin = torch.sin(angles).to(dtype=dtype)
    return cos, sin


def _apply_rotary_pos_emb(
    q: torch.Tensor,
    k: torch.Tensor,
    cos: torch.Tensor,
    sin: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Apply RoPE to patch tokens while leaving CLS/register prefix tokens untouched."""
    num_tokens = q.shape[-2]
    num_patches = sin.shape[-2]
    num_prefix_tokens = num_tokens - num_patches

    q_prefix, q_patches = q.split((num_prefix_tokens, num_patches), dim=-2)
    k_prefix, k_patches = k.split((num_prefix_tokens, num_patches), dim=-2)

    cos = cos[None, None, :, :]
    sin = sin[None, None, :, :]
    q_patches = (q_patches * cos) + (_rotate_half(q_patches) * sin)
    k_patches = (k_patches * cos) + (_rotate_half(k_patches) * sin)
    return torch.cat((q_prefix, q_patches), dim=-2), torch.cat((k_prefix, k_patches), dim=-2)


# ---------------------------------------------------------------------------
# Lightweight ViT backbone — structure mirrors the DINOv3 safetensors exactly
# ---------------------------------------------------------------------------
# Key naming in this module matches the checkpoint 1-to-1 so that no key
# remapping is needed at load time.  Verified against the header of
# dinov3_vitl16_pretrain_lvd1689m.safetensors (415 keys, 24 layers).
#
# Checkpoint key structure:
#   embeddings.cls_token                   [1, 1, D]
#   embeddings.mask_token                  [1, 1, D]
#   embeddings.patch_embeddings.weight     [D, 3, P, P]
#   embeddings.patch_embeddings.bias       [D]
#   embeddings.register_tokens             [1, 4, D]
#   layer.{i}.norm1.weight/bias            [D]
#   layer.{i}.attention.q_proj.weight      [D, D]   (has bias)
#   layer.{i}.attention.q_proj.bias        [D]
#   layer.{i}.attention.k_proj.weight      [D, D]   (NO bias)
#   layer.{i}.attention.v_proj.weight      [D, D]   (has bias)
#   layer.{i}.attention.v_proj.bias        [D]
#   layer.{i}.attention.o_proj.weight      [D, D]   (has bias)
#   layer.{i}.attention.o_proj.bias        [D]
#   layer.{i}.layer_scale1.lambda1         [D]
#   layer.{i}.norm2.weight/bias            [D]
#   layer.{i}.mlp.up_proj.weight           [4D, D]  (GELU MLP)
#   layer.{i}.mlp.up_proj.bias             [4D]
#   layer.{i}.mlp.down_proj.weight         [D, 4D]
#   layer.{i}.mlp.down_proj.bias           [D]
#   layer.{i}.layer_scale2.lambda1         [D]
#   norm.weight/bias                       [D]


class _Attention(nn.Module):
    """Multi-head attention with separate Q/K/V projections (DINOv3 style).

    K projection has NO bias; Q, V, and O projections have bias.
    """

    def __init__(self, dim: int, num_heads: int):
        super().__init__()
        self.num_heads = num_heads
        self.head_dim = dim // num_heads
        self.q_proj = nn.Linear(dim, dim, bias=True)
        self.k_proj = nn.Linear(dim, dim, bias=False)
        self.v_proj = nn.Linear(dim, dim, bias=True)
        self.o_proj = nn.Linear(dim, dim, bias=True)

    def forward(
        self,
        x: torch.Tensor,
        position_embeddings: tuple[torch.Tensor, torch.Tensor],
    ) -> torch.Tensor:
        batch_size, seq_len, _ = x.shape
        query = self.q_proj(x).reshape(batch_size, seq_len, self.num_heads, self.head_dim).transpose(1, 2)
        key = self.k_proj(x).reshape(batch_size, seq_len, self.num_heads, self.head_dim).transpose(1, 2)
        value = self.v_proj(x).reshape(batch_size, seq_len, self.num_heads, self.head_dim).transpose(1, 2)
        query, key = _apply_rotary_pos_emb(query, key, *position_embeddings)
        attn_out = F.scaled_dot_product_attention(query, key, value)
        attn_out = attn_out.transpose(1, 2).reshape(batch_size, seq_len, -1)
        return self.o_proj(attn_out)


class _MLP(nn.Module):
    """Standard GELU MLP with up_proj / down_proj naming (matching DINOv3 checkpoint)."""

    def __init__(self, dim: int, mlp_ratio: float = 4.0):
        super().__init__()
        hidden = int(dim * mlp_ratio)
        self.up_proj = nn.Linear(dim, hidden)
        self.act = nn.GELU()
        self.down_proj = nn.Linear(hidden, dim)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.down_proj(self.act(self.up_proj(x)))


class _Block(nn.Module):
    """Single transformer block matching DINOv3 checkpoint layout.

    Uses ParameterDict for layer_scale so that the state_dict key becomes
    ``layer_scale1.lambda1`` / ``layer_scale2.lambda1`` — exactly matching
    the checkpoint.
    """

    def __init__(self, dim: int, num_heads: int, mlp_ratio: float = 4.0,
                 layer_scale_init: Optional[float] = None):
        super().__init__()
        self.norm1 = nn.LayerNorm(dim, eps=1e-6)
        self.attention = _Attention(dim, num_heads)
        self.norm2 = nn.LayerNorm(dim, eps=1e-6)
        self.mlp = _MLP(dim, mlp_ratio)

        # Layer scale — stored as ParameterDict so key is "layer_scale1.lambda1"
        if layer_scale_init is not None:
            self.layer_scale1 = nn.ParameterDict({
                "lambda1": nn.Parameter(torch.full((dim,), layer_scale_init))
            })
            self.layer_scale2 = nn.ParameterDict({
                "lambda1": nn.Parameter(torch.full((dim,), layer_scale_init))
            })
        else:
            self.layer_scale1 = None
            self.layer_scale2 = None

    def forward(
        self,
        x: torch.Tensor,
        position_embeddings: tuple[torch.Tensor, torch.Tensor],
    ) -> torch.Tensor:
        attn_out = self.attention(self.norm1(x), position_embeddings)
        if self.layer_scale1 is not None:
            x = x + attn_out * self.layer_scale1["lambda1"]
        else:
            x = x + attn_out
        mlp_out = self.mlp(self.norm2(x))
        if self.layer_scale2 is not None:
            x = x + mlp_out * self.layer_scale2["lambda1"]
        else:
            x = x + mlp_out
        return x


class _Embeddings(nn.Module):
    """Patch embedding + special tokens, matching DINOv3 checkpoint naming.

    State-dict keys produced:
      embeddings.cls_token, embeddings.mask_token,
      embeddings.patch_embeddings.weight/bias, embeddings.register_tokens
    """

    def __init__(self, in_channels: int, embed_dim: int, patch_size: int,
                 num_register_tokens: int = 4):
        super().__init__()
        self.cls_token = nn.Parameter(torch.zeros(1, 1, embed_dim))
        self.mask_token = nn.Parameter(torch.zeros(1, 1, embed_dim))
        self.patch_embeddings = nn.Conv2d(
            in_channels, embed_dim, kernel_size=patch_size, stride=patch_size
        )
        self.register_tokens = nn.Parameter(torch.zeros(1, num_register_tokens, embed_dim))


class _DinoViT(nn.Module):
    """Minimal ViT whose state-dict keys mirror the DINOv3 safetensors exactly.

    Structure:
      embeddings.*          — cls_token, mask_token, patch_embeddings, register_tokens
      layer.{i}.*           — transformer blocks (norm1, attention, norm2, mlp, layer_scale)
      norm.*                — final layer norm
    """

    def __init__(
        self,
        image_size: int = 224,
        patch_size: int = 16,
        in_channels: int = 3,
        embed_dim: int = 1024,
        num_heads: int = 16,
        num_layers: int = 24,
        mlp_ratio: float = 4.0,
        num_register_tokens: int = 4,
        layer_scale_init: Optional[float] = None,
    ):
        super().__init__()
        self.patch_size = patch_size
        self.embed_dim = embed_dim
        self.num_patches = (image_size // patch_size) ** 2
        self.num_register_tokens = num_register_tokens
        self.rope_theta = 100.0

        self.embeddings = _Embeddings(in_channels, embed_dim, patch_size, num_register_tokens)

        # Use nn.ModuleList named "layer" so keys become layer.0.*, layer.1.*, ...
        self.layer = nn.ModuleList([
            _Block(embed_dim, num_heads, mlp_ratio, layer_scale_init=layer_scale_init)
            for _ in range(num_layers)
        ])
        self.norm = nn.LayerNorm(embed_dim, eps=1e-6)

    def forward_features(self, x: torch.Tensor) -> dict:
        batch_size = x.shape[0]
        _, _, height, width = x.shape

        # Patch embedding
        patch_tokens = self.embeddings.patch_embeddings(x).flatten(2).transpose(1, 2)  # [B, N, D]

        # Match DINOv3 token order: CLS | register tokens | patch tokens.
        cls_tokens = self.embeddings.cls_token.expand(batch_size, -1, -1)
        if self.num_register_tokens > 0:
            reg_tokens = self.embeddings.register_tokens.expand(batch_size, -1, -1)
            x = torch.cat([cls_tokens, reg_tokens, patch_tokens], dim=1)
        else:
            x = torch.cat([cls_tokens, patch_tokens], dim=1)

        position_embeddings = _build_rope_embeddings(
            height=height,
            width=width,
            patch_size=self.patch_size,
            head_dim=self.embed_dim // self.layer[0].attention.num_heads,
            dtype=x.dtype,
            device=x.device,
            rope_theta=self.rope_theta,
        )

        for block in self.layer:
            x = block(x, position_embeddings)
        x = self.norm(x)

        # Split: CLS | register_tokens | patch_tokens. Register tokens are not returned.
        cls_out = x[:, 0]
        patch_tokens = x[:, 1 + self.num_register_tokens:]

        return {
            "x_norm_clstoken": cls_out,
            "x_norm_patchtokens": patch_tokens,
        }


# ---------------------------------------------------------------------------
# Model configs
# ---------------------------------------------------------------------------

_DINO_CONFIGS = {
    "vits14": dict(patch_size=14, embed_dim=384, num_heads=6, num_layers=12, mlp_ratio=4.0),
    "vitb14": dict(patch_size=14, embed_dim=768, num_heads=12, num_layers=12, mlp_ratio=4.0),
    "vitl14": dict(patch_size=14, embed_dim=1024, num_heads=16, num_layers=24, mlp_ratio=4.0),
    "vitg14": dict(patch_size=14, embed_dim=1536, num_heads=24, num_layers=40, mlp_ratio=4.0),
    "vits16": dict(patch_size=16, embed_dim=384, num_heads=6, num_layers=12, mlp_ratio=4.0),
    "vitb16": dict(patch_size=16, embed_dim=768, num_heads=12, num_layers=12, mlp_ratio=4.0),
    "vitl16": dict(patch_size=16, embed_dim=1024, num_heads=16, num_layers=24, mlp_ratio=4.0),
}


def _build_dino_vit(
    variant: str = "vitl16",
    image_size: int = 224,
    num_register_tokens: int = 4,
    layer_scale_init: Optional[float] = None,
) -> _DinoViT:
    """Build a _DinoViT for a given variant string."""
    if variant not in _DINO_CONFIGS:
        raise ValueError(f"Unknown DINO variant '{variant}'. Available: {sorted(_DINO_CONFIGS)}")
    config = _DINO_CONFIGS[variant]
    return _DinoViT(
        image_size=image_size,
        num_register_tokens=num_register_tokens,
        layer_scale_init=layer_scale_init,
        **config,
    )


def _remap_timm_dinov3_state_dict(
    state_dict: dict[str, torch.Tensor],
    model: _DinoViT,
) -> dict[str, torch.Tensor]:
    """Convert timm DINOv3 ViT checkpoint keys to the lightweight HF-style keys."""
    remapped: dict[str, torch.Tensor] = {}
    model_state = model.state_dict()

    for key, value in state_dict.items():
        if key == "cls_token":
            remapped["embeddings.cls_token"] = value
        elif key == "reg_token":
            remapped["embeddings.register_tokens"] = value
        elif key == "patch_embed.proj.weight":
            remapped["embeddings.patch_embeddings.weight"] = value
        elif key == "patch_embed.proj.bias":
            remapped["embeddings.patch_embeddings.bias"] = value
        elif key == "norm.weight":
            remapped["norm.weight"] = value
        elif key == "norm.bias":
            remapped["norm.bias"] = value
        elif key.startswith("blocks."):
            parts = key.split(".")
            if len(parts) < 3:
                continue
            block_idx = parts[1]
            suffix = ".".join(parts[2:])
            prefix = f"layer.{block_idx}"

            if suffix == "attn.qkv.weight":
                q_weight, k_weight, v_weight = value.chunk(3, dim=0)
                remapped[f"{prefix}.attention.q_proj.weight"] = q_weight
                remapped[f"{prefix}.attention.k_proj.weight"] = k_weight
                remapped[f"{prefix}.attention.v_proj.weight"] = v_weight
            elif suffix == "attn.qkv.bias":
                q_bias, k_bias, v_bias = value.chunk(3, dim=0)
                remapped[f"{prefix}.attention.q_proj.bias"] = q_bias
                remapped[f"{prefix}.attention.v_proj.bias"] = v_bias
                if k_bias.abs().max().item() > 0:
                    logger.warning(
                        "Ignoring non-zero timm DINOv3 K bias for block %s; lightweight model uses bias=False.",
                        block_idx,
                    )
            elif suffix == "attn.proj.weight":
                remapped[f"{prefix}.attention.o_proj.weight"] = value
            elif suffix == "attn.proj.bias":
                remapped[f"{prefix}.attention.o_proj.bias"] = value
            elif suffix == "mlp.fc1.weight":
                remapped[f"{prefix}.mlp.up_proj.weight"] = value
            elif suffix == "mlp.fc1.bias":
                remapped[f"{prefix}.mlp.up_proj.bias"] = value
            elif suffix == "mlp.fc2.weight":
                remapped[f"{prefix}.mlp.down_proj.weight"] = value
            elif suffix == "mlp.fc2.bias":
                remapped[f"{prefix}.mlp.down_proj.bias"] = value
            elif suffix == "gamma_1":
                remapped[f"{prefix}.layer_scale1.lambda1"] = value
            elif suffix == "gamma_2":
                remapped[f"{prefix}.layer_scale2.lambda1"] = value
            elif suffix.startswith("norm1.") or suffix.startswith("norm2."):
                remapped[f"{prefix}.{suffix}"] = value

    # timm DINOv3 checkpoints omit all-zero QKV bias tensors and mask_token.
    # Keep the lightweight model numerically aligned by explicitly filling the
    # omitted Q/V biases with zeros; K has bias=False.
    for key, value in model_state.items():
        if key.endswith("attention.q_proj.bias") or key.endswith("attention.v_proj.bias"):
            remapped.setdefault(key, torch.zeros_like(value))
        elif key == "embeddings.mask_token":
            remapped.setdefault(key, torch.zeros_like(value))

    return remapped


# ---------------------------------------------------------------------------
# Weight loading
# ---------------------------------------------------------------------------

def _load_dino_from_safetensors(
    weights_path: str,
    variant: str = "vitl16",
    image_size: int = 224,
    num_register_tokens: int = 4,
    device: torch.device = torch.device("cpu"),
    dtype: torch.dtype = torch.float32,
) -> _DinoViT:
    """Load a _DinoViT from a DINOv3 safetensors checkpoint.

    Native DINOv3 checkpoints mirror the lightweight model keys directly;
    timm checkpoints are detected and remapped before loading.
    """
    from safetensors.torch import load_file

    model = _build_dino_vit(
        variant, image_size,
        num_register_tokens=num_register_tokens,
        layer_scale_init=1e-5,
    )

    state_dict = load_file(weights_path, device="cpu")
    if any(key.startswith("blocks.") for key in state_dict):
        logger.info("Detected timm DINOv3 checkpoint format; remapping keys.")
        state_dict = _remap_timm_dinov3_state_dict(state_dict, model)
    missing, unexpected = model.load_state_dict(state_dict, strict=False)

    if unexpected:
        logger.info(
            f"Unexpected keys in checkpoint (ignored, {len(unexpected)}): "
            f"{unexpected[:5]}{'...' if len(unexpected) > 5 else ''}"
        )
    if missing:
        logger.warning(
            f"Missing keys when loading DINO weights ({len(missing)}): "
            f"{missing[:10]}{'...' if len(missing) > 10 else ''}"
        )
    else:
        logger.info("All DINOv3 checkpoint keys loaded successfully.")

    return model.to(device=device, dtype=dtype)


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

class DinoVideoEncoder(nn.Module):
    """Frozen DINOv3 encoder that extracts patch-level features from video frames.

    The encoder processes each frame independently through a frozen DINOv3 ViT,
    producing dense patch features that serve as the latent representation for
    diffusion-based dynamics prediction.

    Args:
        model_name: Model identifier for variant detection (e.g. "dinov3-vitl16").
        model_path: Path to local .safetensors weights. If provided, uses the
            built-in ViT loader. Otherwise falls back to torch.hub / transformers.
        input_resolution: Expected input image resolution (H, W).
        patch_size: ViT patch size (16 for DINOv3).
        feature_dim: Output feature dimension per patch (1024 for ViT-L).
        use_cls_token: Whether to prepend the CLS token to patch features.
        normalize_features: Whether to L2-normalize output features.
        latent_spatial_pool: Optional average-pooling factor applied to encoded
            DINO latents as (height_pool, width_pool). This can compress wide
            multi-camera layouts after DINO encoding while preserving both views.
        encode_microbatch_size: Maximum number of frames to encode per DINO
            forward call. Keeps memory bounded while avoiding per-frame Python
            loops. Set to <=0 to encode all frames at once.
    """

    def __init__(
        self,
        model_name: str = "dinov3-vitl16",
        model_path: Optional[str] = None,
        input_resolution: Tuple[int, int] = (224, 224),
        patch_size: int = 16,
        feature_dim: int = 1024,
        use_cls_token: bool = False,
        normalize_features: bool = False,
        latent_spatial_pool: Tuple[int, int] = (1, 1),
        encode_microbatch_size: int = 72,
    ):
        super().__init__()
        self.model_name = model_name
        self.model_path = model_path
        self.input_resolution = input_resolution
        self.patch_size = patch_size
        self.feature_dim = feature_dim
        self.use_cls_token = use_cls_token
        self.normalize_features = normalize_features
        if len(latent_spatial_pool) != 2:
            raise ValueError(
                f"`latent_spatial_pool` must be a 2-tuple/list, got {latent_spatial_pool}"
            )
        self.latent_spatial_pool = (int(latent_spatial_pool[0]), int(latent_spatial_pool[1]))
        if self.latent_spatial_pool[0] <= 0 or self.latent_spatial_pool[1] <= 0:
            raise ValueError(f"`latent_spatial_pool` values must be positive, got {self.latent_spatial_pool}")
        self.encode_microbatch_size = int(encode_microbatch_size)

        self.grid_size = (
            input_resolution[0] // patch_size,
            input_resolution[1] // patch_size,
        )
        self.num_patches = self.grid_size[0] * self.grid_size[1]

        self.backbone = None
        self._loaded = False

    def _detect_variant(self) -> str:
        """Detect ViT variant string from model_name."""
        name_lower = self.model_name.lower()
        for variant in ["vitl16", "vitb16", "vits16", "vitl14", "vitb14", "vits14", "vitg14"]:
            if variant in name_lower:
                return variant
        # Infer from patch_size and feature_dim
        size_map = {384: "s", 768: "b", 1024: "l", 1536: "g"}
        size_letter = size_map.get(self.feature_dim, "l")
        return f"vit{size_letter}{self.patch_size}"

    def load_backbone(
        self,
        device: torch.device = torch.device("cpu"),
        dtype: torch.dtype = torch.float32,
    ):
        """Load the pretrained DINO backbone."""
        if self._loaded:
            return

        if self.model_path and os.path.isfile(self.model_path):
            variant = self._detect_variant()
            logger.info(f"Loading DINO from safetensors: {self.model_path} (variant={variant})")
            self.backbone = _load_dino_from_safetensors(
                weights_path=self.model_path,
                variant=variant,
                image_size=self.input_resolution[0],
                device=device,
                dtype=dtype,
            )
        else:
            hub_name = self._hub_model_name()
            try:
                logger.info(f"Loading DINO via torch.hub: {hub_name}")
                self.backbone = torch.hub.load("facebookresearch/dinov2", hub_name)
            except Exception:
                logger.info(f"torch.hub failed, trying transformers for {self.model_name}")
                from transformers import AutoModel
                self.backbone = AutoModel.from_pretrained(self.model_name)
            self.backbone = self.backbone.to(device=device, dtype=dtype)

        self.backbone.eval()
        for param in self.backbone.parameters():
            param.requires_grad = False

        self._loaded = True
        logger.info(
            f"DinoVideoEncoder loaded: model={self.model_name}, "
            f"path={self.model_path}, "
            f"resolution={self.input_resolution}, grid={self.grid_size}, "
            f"feature_dim={self.feature_dim}, num_patches={self.num_patches}, "
            f"latent_spatial_pool={self.latent_spatial_pool}, "
            f"encode_microbatch_size={self.encode_microbatch_size}"
        )

    def _hub_model_name(self) -> str:
        """Map model_name to torch.hub entry name."""
        name_map = {
            "facebook/dinov2-small": "dinov2_vits14",
            "facebook/dinov2-base": "dinov2_vitb14",
            "facebook/dinov2-large": "dinov2_vitl14",
            "facebook/dinov2-giant": "dinov2_vitg14",
        }
        if self.model_name in name_map:
            return name_map[self.model_name]
        return self.model_name.split("/")[-1]

    @property
    def temporal_downsample_factor(self) -> int:
        """DINO processes every frame independently, no temporal downsampling."""
        return 1

    @property
    def spatial_downsample_factor(self) -> int:
        return self.patch_size

    @torch.no_grad()
    def encode_frames(self, frames: torch.Tensor) -> torch.Tensor:
        """Encode a batch of frames into DINO patch features.

        Args:
            frames: [B, 3, H, W] tensor of RGB frames, normalized to [-1, 1].

        Returns:
            features: [B, N_patches, D] tensor of patch features.
                If use_cls_token=True, shape is [B, 1+N_patches, D].
        """
        if not self._loaded:
            raise RuntimeError("Backbone not loaded. Call `load_backbone()` first.")

        # Resize to expected resolution if needed
        if frames.shape[2:] != self.input_resolution:
            frames = F.interpolate(
                frames, size=self.input_resolution, mode="bilinear", align_corners=False,
            )

        # Normalize from [-1, 1] to ImageNet normalization
        mean = torch.tensor([0.485, 0.456, 0.406], device=frames.device, dtype=frames.dtype).view(1, 3, 1, 1)
        std = torch.tensor([0.229, 0.224, 0.225], device=frames.device, dtype=frames.dtype).view(1, 3, 1, 1)
        frames_01 = (frames + 1.0) * 0.5
        frames_norm = (frames_01 - mean) / std

        # Extract features
        if hasattr(self.backbone, "forward_features"):
            output = self.backbone.forward_features(frames_norm)
            if isinstance(output, dict):
                patch_tokens = output["x_norm_patchtokens"]
                if self.use_cls_token:
                    cls_token = output["x_norm_clstoken"].unsqueeze(1)
                    patch_tokens = torch.cat([cls_token, patch_tokens], dim=1)
            else:
                patch_tokens = output[:, 1:]
                if self.use_cls_token:
                    patch_tokens = torch.cat([output[:, :1], patch_tokens], dim=1)
        else:
            # Transformers API fallback
            output = self.backbone(frames_norm, output_hidden_states=False)
            num_register_tokens = int(getattr(self.backbone.config, "num_register_tokens", 0))
            patch_start = 1 + num_register_tokens
            patch_tokens = output.last_hidden_state[:, patch_start:]
            if self.use_cls_token:
                cls_token = output.last_hidden_state[:, :1]
                patch_tokens = torch.cat([cls_token, patch_tokens], dim=1)

        if self.normalize_features:
            patch_tokens = F.normalize(patch_tokens, dim=-1)

        return patch_tokens

    @torch.no_grad()
    def encode_video_to_latent(self, video: torch.Tensor) -> torch.Tensor:
        """Encode a video tensor into DINO latent features in spatial format.

        Flattens the temporal dimension and processes frames in configurable
        microbatches through the frozen DINO encoder, then reshapes the output
        into a 5D tensor compatible with the DinoVideoDiT.

        Args:
            video: [B, 3, T, H, W] RGB video, values in [-1, 1].

        Returns:
            [B, D_dino, T, H_grid, W_grid] DINO feature latents, where
            H_grid = H // patch_size, W_grid = W // patch_size.
        """
        if video.ndim != 5:
            raise ValueError(f"`video` must be 5D [B, 3, T, H, W], got shape {tuple(video.shape)}")
        if self.use_cls_token:
            raise ValueError(
                "`encode_video_to_latent` expects patch-only DINO features. "
                "Set `use_cls_token=false` for video latent training."
            )

        batch_size, channels, num_frames, height, width = video.shape
        if channels != 3:
            raise ValueError(f"`video` channel dimension must be 3, got {channels}")
        if height % self.patch_size != 0 or width % self.patch_size != 0:
            raise ValueError(
                "`video` spatial size must be divisible by DINO patch_size="
                f"{self.patch_size}, got {(height, width)}"
            )
        height_grid = height // self.patch_size
        width_grid = width // self.patch_size
        num_patches = height_grid * width_grid

        # [B, 3, T, H, W] -> [B*T, 3, H, W].  This is much faster than a
        # Python loop over frames while still allowing bounded microbatches.
        frames = video.permute(0, 2, 1, 3, 4).reshape(batch_size * num_frames, channels, height, width)
        microbatch_size = self.encode_microbatch_size
        if microbatch_size <= 0:
            microbatch_size = frames.shape[0]

        frame_features = []
        for start in range(0, frames.shape[0], microbatch_size):
            features = self.encode_frames(frames[start:start + microbatch_size])
            frame_features.append(features)

        # [B*T, N_patches, D] -> [B, T, N_patches, D]
        stacked = torch.cat(frame_features, dim=0)
        if stacked.shape[1] != num_patches:
            raise ValueError(
                "DINO patch count mismatch: "
                f"expected {num_patches} patches for video size {(height, width)}, "
                f"got {stacked.shape[1]}."
            )
        stacked = stacked.reshape(batch_size, num_frames, num_patches, -1)

        # Reshape to spatial: [B, T, H_g*W_g, D] → [B, D, T, H_g, W_g]
        stacked = stacked.reshape(batch_size, num_frames, height_grid, width_grid, -1)
        latents = stacked.permute(0, 4, 1, 2, 3).contiguous()  # [B, D, T, H_g, W_g]
        pool_h, pool_w = self.latent_spatial_pool
        if pool_h != 1 or pool_w != 1:
            if height_grid % pool_h != 0 or width_grid % pool_w != 0:
                raise ValueError(
                    "`latent_spatial_pool` must divide the DINO grid exactly, "
                    f"got grid={(height_grid, width_grid)} and pool={self.latent_spatial_pool}."
                )
            latents = F.avg_pool3d(
                latents,
                kernel_size=(1, pool_h, pool_w),
                stride=(1, pool_h, pool_w),
            )

        return latents
