import argparse
import sys
from pathlib import Path
from typing import Any

import torch
import torch.nn.functional as F
from omegaconf import OmegaConf

REPO_ROOT = Path(__file__).resolve().parents[1]
SRC_DIR = REPO_ROOT / "src"
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

from fastwam.models.wan22.dino_video_dit import DinoVideoDiT
from fastwam.models.wan22.helpers.loader import load_wan22_ti2v_5b_components


def _parse_dtype(name: str) -> torch.dtype:
    value = str(name).strip().lower()
    if value == "float32":
        return torch.float32
    if value == "float16":
        return torch.float16
    if value == "bfloat16":
        return torch.bfloat16
    raise ValueError(f"Unsupported dtype: {name}. Expected one of: float32, float16, bfloat16.")


def _parse_bool(name: str) -> bool:
    value = str(name).strip().lower()
    if value in {"1", "true", "yes", "y"}:
        return True
    if value in {"0", "false", "no", "n"}:
        return False
    raise ValueError(f"Cannot parse bool value: {name}")


def _is_unresolved_interpolation(value: Any) -> bool:
    return isinstance(value, str) and "${" in value and "}" in value


def _require_int_config(cfg: dict[str, Any], key: str) -> int:
    value = cfg.get(key)
    if _is_unresolved_interpolation(value):
        raise ValueError(f"`{key}` is unresolved interpolation: {value}")
    return int(value)


def _interpolate_last_dim(tensor: torch.Tensor, new_size: int, apply_alpha_scaling: bool = False) -> torch.Tensor:
    """Interpolate tensor's last dimension to new_size using linear interpolation.
    
    Args:
        tensor: Source tensor to resize
        new_size: Target size for the last dimension
        apply_alpha_scaling: Whether to apply alpha scaling (sqrt(src_dim/target_dim))
    """
    if tensor.shape[-1] == new_size:
        return tensor
    
    original_dtype = tensor.dtype
    flat = tensor.reshape(-1, 1, tensor.shape[-1]).to(torch.float32)
    flat = F.interpolate(flat, size=new_size, mode="linear", align_corners=True)
    result = flat.reshape(*tensor.shape[:-1], new_size)
    
    # Apply alpha scaling if the last dimension changed
    if apply_alpha_scaling and tensor.shape[-1] != new_size:
        alpha = (float(tensor.shape[-1]) / float(new_size)) ** 0.5
        result = result * alpha
    
    return result.to(dtype=original_dtype)


def _resize_tensor_to_shape(src: torch.Tensor, target_shape: tuple[int, ...], apply_alpha_scaling: bool = False) -> torch.Tensor:
    """Resize tensor to target shape using interpolation on mismatched dimensions.
    
    Args:
        src: Source tensor
        target_shape: Target shape tuple
        apply_alpha_scaling: Whether to apply alpha scaling when resizing the last dimension
    """
    if tuple(src.shape) == tuple(target_shape):
        return src

    out = src.to(torch.float32)
    while out.ndim < len(target_shape):
        out = out.unsqueeze(0)
    while out.ndim > len(target_shape):
        if out.shape[0] != 1:
            raise ValueError(
                f"Cannot reduce tensor rank for resize: src shape={tuple(src.shape)}, target={target_shape}"
            )
        out = out.squeeze(0)

    last_dim_resized = False
    for dim, new_size in enumerate(target_shape):
        current_size = out.shape[dim]
        if current_size == new_size:
            continue
        
        # Track if we're resizing the last dimension (for alpha scaling)
        if dim == out.ndim - 1:
            last_dim_resized = True
        
        # Permute the target dimension to the end for interpolation
        perm = [i for i in range(out.ndim) if i != dim] + [dim]
        inv_perm = [0] * out.ndim
        for i, p in enumerate(perm):
            inv_perm[p] = i
        
        # Permute, interpolate, and restore original order
        out_perm = out.permute(*perm).contiguous()
        prefix_shape = out_perm.shape[:-1]
        out_perm = _interpolate_last_dim(out_perm, new_size)
        out_perm = out_perm.reshape(*prefix_shape, new_size)
        out = out_perm.permute(*inv_perm).contiguous()

    if tuple(out.shape) != tuple(target_shape):
        raise ValueError(
            f"Resize produced wrong shape for tensor. src={tuple(src.shape)}, target={target_shape}, got={tuple(out.shape)}"
        )
    
    # Apply alpha scaling only once for the last dimension resize
    if apply_alpha_scaling and last_dim_resized and src.shape[-1] != target_shape[-1]:
        alpha = (float(src.shape[-1]) / float(target_shape[-1])) ** 0.5
        out = out.to(torch.float32) * alpha
    
    return out.to(dtype=src.dtype)


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Preprocess DinoVideoDiT backbone weights from WanVideoDiT and save as .pt payload."
    )
    parser.add_argument(
        "--model-config", 
        required=True, 
        help="Path to model yaml with video_dit_config, e.g. configs/model/fastwam_dino_s_smallvideo.yaml"
    )
    parser.add_argument(
        "--output", 
        required=True, 
        help="Output .pt path for preprocessed DinoVideoDiT backbone."
    )
    parser.add_argument(
        "--device", 
        default="cuda", 
        help="Device for loading model and preprocessing."
    )
    parser.add_argument(
        "--dtype", 
        default="bfloat16", 
        choices=["float32", "float16", "bfloat16"]
    )
    parser.add_argument(
        "--apply-alpha-scaling",
        default="true",
        help="Whether to apply alpha=sqrt(dv/da) when dimensions are resized (true/false). Default: true.",
    )
    parser.add_argument(
        "--wan-model-id",
        default="Wan-AI/Wan2.2-TI2V-5B",
        help="Wan2.2 model ID to load for weight extraction.",
    )
    args = parser.parse_args()

    model_config_path = Path(args.model_config)
    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    apply_alpha_scaling = _parse_bool(args.apply_alpha_scaling)

    # Load model config
    cfg = OmegaConf.load(str(model_config_path))
    if "video_dit_config" not in cfg:
        raise ValueError(f"`{model_config_path}` must contain `video_dit_config` at top level.")

    video_cfg = OmegaConf.to_container(cfg.video_dit_config, resolve=False)
    if not isinstance(video_cfg, dict):
        raise ValueError("`video_dit_config` must resolve to a dict.")

    torch_dtype = _parse_dtype(args.dtype)

    # Resolve key fields
    int_fields = ["hidden_dim", "ffn_dim", "num_layers", "num_heads", "attn_head_dim", "text_dim", "freq_dim"]
    for key in int_fields:
        if key in video_cfg:
            video_cfg[key] = _require_int_config(video_cfg, key)
    
    if "eps" in video_cfg:
        video_cfg["eps"] = float(video_cfg["eps"])

    print(f"[INFO] Loaded model config from {model_config_path}")
    print(f"[INFO] DinoVideoDiT config:")
    for key in ["hidden_dim", "ffn_dim", "num_layers", "num_heads", "attn_head_dim"]:
        if key in video_cfg:
            print(f"  {key}: {video_cfg[key]}")
    print(f"[INFO] Preprocessing with dtype={torch_dtype} on device={args.device}")
    print(f"[INFO] apply_alpha_scaling={apply_alpha_scaling}")

    # Load Wan2.2 DiT components
    wan_dit_config = {
        "has_image_input": False,
        "patch_size": [1, 2, 2],
        "in_dim": 48,
        "hidden_dim": 3072,  # Wan2.2-5B default
        "ffn_dim": 14336,    # Wan2.2-5B default
        "freq_dim": 256,
        "text_dim": 4096,
        "out_dim": 48,
        "num_heads": 24,
        "attn_head_dim": 128,
        "num_layers": 30,
        "eps": 1e-6,
        "seperated_timestep": True,
        "require_clip_embedding": False,
        "require_vae_embedding": False,
        "fuse_vae_embedding_in_latents": True,
        "video_attention_mask_mode": "first_frame_causal",
        "action_conditioned": False,
    }
    
    print(f"\n[INFO] Loading Wan2.2 DiT from '{args.wan_model_id}'...")
    components = load_wan22_ti2v_5b_components(
        device=args.device,
        torch_dtype=torch_dtype,
        model_id=args.wan_model_id,
        tokenizer_model_id=cfg.get("tokenizer_model_id", "Wan-AI/Wan2.1-T2V-1.3B"),
        redirect_common_files=True,
        dit_config=wan_dit_config,
        skip_dit_load_from_pretrain=False,
        load_text_encoder=False,
        load_vae=False,
    )
    wan_dit = components.dit
    if wan_dit is None:
        raise RuntimeError("Wan DiT initialization requested but loader returned no DiT.")
    
    print(f"[INFO] Wan2.2 DiT loaded successfully")

    # Create target DinoVideoDiT with small video config
    dino_dim = cfg.get("dino_config", {}).get("feature_dim", 1024) if "dino_config" in cfg else 1024
    
    # For preprocessing, we use a simplified config (no text encoder needed)
    target_video_cfg = {
        "hidden_dim": video_cfg.get("hidden_dim", 1024),
        "dino_dim": dino_dim,
        "ffn_dim": video_cfg.get("ffn_dim", 4096),
        "text_dim": video_cfg.get("text_dim", 4096),
        "freq_dim": video_cfg.get("freq_dim", 256),
        "eps": video_cfg.get("eps", 1e-6),
        "num_heads": video_cfg.get("num_heads", 24),
        "attn_head_dim": video_cfg.get("attn_head_dim", 128),
        "num_layers": video_cfg.get("num_layers", 30),
        "video_attention_mask_mode": "first_frame_causal",
        "use_gradient_checkpointing": False,
    }
    
    print(f"\n[INFO] Creating target DinoVideoDiT with dino_dim={dino_dim}...")
    dino_video_dit = DinoVideoDiT(**target_video_cfg).to(device=args.device, dtype=torch_dtype)

    # Extract backbone weights from Wan2.2 DiT
    wan_state = wan_dit.state_dict()
    target_state = dino_video_dit.state_dict()
    
    # Transferable prefixes: blocks.*, text_embedding.*, time_embedding.*, time_projection.*
    transferable_prefixes = ("blocks.", "text_embedding.", "time_embedding.", "time_projection.")
    
    backbone_state_dict: dict[str, torch.Tensor] = {}
    copied = 0
    interpolated = 0
    skipped_shape = 0
    skipped_missing = 0

    print(f"\n[INFO] Extracting backbone weights from Wan2.2 DiT...")
    for key in sorted(target_state.keys()):
        if not any(key.startswith(p) for p in transferable_prefixes):
            continue
        
        if key not in wan_state:
            skipped_missing += 1
            continue
        
        src = wan_state[key]
        target = target_state[key]
        
        if tuple(src.shape) == tuple(target.shape):
            value = src
            copied += 1
        else:
            # Need to interpolate
            value = _resize_tensor_to_shape(src, tuple(target.shape), apply_alpha_scaling)
            interpolated += 1
            if tuple(value.shape) != tuple(target.shape):
                print(f"  [WARN] Shape mismatch after resize for '{key}': src={tuple(src.shape)}, target={tuple(target.shape)}, got={tuple(value.shape)}")
                skipped_shape += 1
                continue
        
        backbone_state_dict[key] = value.detach().to(dtype=target.dtype, device="cpu").contiguous()

    # Load the extracted backbone into DinoVideoDiT
    own_state = dino_video_dit.state_dict()
    for key in backbone_state_dict:
        if key in own_state:
            own_state[key] = backbone_state_dict[key].to(device=args.device, dtype=torch_dtype)
    
    dino_video_dit.load_state_dict(own_state, strict=True)

    # Save the payload
    payload = {
        "policy": {
            "transferable_prefixes": list(transferable_prefixes),
            "alpha_scaling": bool(apply_alpha_scaling),
            "interpolation": "sequential_1d_linear_align_corners_true",
        },
        "backbone_state_dict": {k: v.cpu() for k, v in backbone_state_dict.items()},
        "meta": {
            "hidden_dim": int(target_video_cfg["hidden_dim"]),
            "dino_dim": int(dino_dim),
            "ffn_dim": int(target_video_cfg["ffn_dim"]),
            "num_layers": int(target_video_cfg["num_layers"]),
            "num_heads": int(target_video_cfg["num_heads"]),
            "attn_head_dim": int(target_video_cfg["attn_head_dim"]),
            "text_dim": int(target_video_cfg["text_dim"]),
            "freq_dim": int(target_video_cfg["freq_dim"]),
            "eps": float(target_video_cfg["eps"]),
        },
    }
    
    torch.save(payload, str(output_path))

    print(f"\n[INFO] Saved DinoVideoDiT backbone payload to {output_path}")
    print(f"  copied={copied}, interpolated={interpolated}, skipped_missing={skipped_missing}, skipped_shape={skipped_shape}")
    print(f"  total backbone keys: {len(backbone_state_dict)}")


if __name__ == "__main__":
    main()
