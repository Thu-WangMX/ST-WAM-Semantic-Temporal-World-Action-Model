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

from fastwam.models.wan22.helpers.loader import load_wan22_ti2v_5b_components
from fastwam.models.wan22.wan_video_dit import WanVideoDiT


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


def _require_float_config(cfg: dict[str, Any], key: str) -> float:
    value = cfg.get(key)
    if _is_unresolved_interpolation(value):
        raise ValueError(f"`{key}` is unresolved interpolation: {value}")
    return float(value)


def _resolve_bool_config(cfg: dict[str, Any], key: str, default: bool = False) -> bool:
    value = cfg.get(key, default)
    if _is_unresolved_interpolation(value):
        return bool(default)
    if isinstance(value, bool):
        return value
    return _parse_bool(str(value))


def _interpolate_last_dim(tensor: torch.Tensor, new_size: int) -> torch.Tensor:
    if tensor.shape[-1] == new_size:
        return tensor
    flat = tensor.reshape(-1, 1, tensor.shape[-1]).to(torch.float32)
    flat = F.interpolate(flat, size=new_size, mode="linear", align_corners=True)
    return flat.reshape(*tensor.shape[:-1], new_size)


def _resize_tensor_to_shape(
    src: torch.Tensor,
    target_shape: tuple[int, ...],
    apply_alpha_scaling: bool = False,
) -> torch.Tensor:
    if tuple(src.shape) == tuple(target_shape):
        return src

    out = src.to(torch.float32)
    while out.ndim < len(target_shape):
        out = out.unsqueeze(0)
    while out.ndim > len(target_shape):
        if out.shape[0] != 1:
            raise ValueError(f"Cannot reduce tensor rank: src={tuple(src.shape)}, target={target_shape}")
        out = out.squeeze(0)

    last_dim_resized = False
    for dim, new_size in enumerate(target_shape):
        current_size = out.shape[dim]
        if current_size == new_size:
            continue
        if dim == out.ndim - 1:
            last_dim_resized = True

        perm = [i for i in range(out.ndim) if i != dim] + [dim]
        inv_perm = [0] * out.ndim
        for i, p in enumerate(perm):
            inv_perm[p] = i

        out_perm = out.permute(*perm).contiguous()
        prefix_shape = out_perm.shape[:-1]
        out_perm = _interpolate_last_dim(out_perm, new_size)
        out_perm = out_perm.reshape(*prefix_shape, new_size)
        out = out_perm.permute(*inv_perm).contiguous()

    if tuple(out.shape) != tuple(target_shape):
        raise ValueError(
            f"Resize produced wrong shape. src={tuple(src.shape)}, target={target_shape}, got={tuple(out.shape)}"
        )

    if apply_alpha_scaling and last_dim_resized and src.shape[-1] != target_shape[-1]:
        alpha = (float(src.shape[-1]) / float(target_shape[-1])) ** 0.5
        out = out.to(torch.float32) * alpha

    return out.to(dtype=src.dtype)


def _load_target_video_cfg(model_config_path: Path) -> tuple[dict[str, Any], dict[str, Any]]:
    cfg = OmegaConf.load(str(model_config_path))
    if "video_dit_config" not in cfg:
        raise ValueError(f"`{model_config_path}` must contain `video_dit_config`.")
    video_cfg = OmegaConf.to_container(cfg.video_dit_config, resolve=False)
    if not isinstance(video_cfg, dict):
        raise ValueError("`video_dit_config` must resolve to a dict.")

    int_fields = [
        "hidden_dim",
        "in_dim",
        "out_dim",
        "ffn_dim",
        "num_layers",
        "num_heads",
        "attn_head_dim",
        "text_dim",
        "freq_dim",
    ]
    for key in int_fields:
        video_cfg[key] = _require_int_config(video_cfg, key)
    video_cfg["eps"] = _require_float_config(video_cfg, "eps")
    video_cfg["patch_size"] = [int(x) for x in video_cfg["patch_size"]]
    if _is_unresolved_interpolation(video_cfg.get("action_dim")):
        print("[WARN] `video_dit_config.action_dim` is unresolved; defaulting to 7 for preprocessing.")
        video_cfg["action_dim"] = 7
    elif "action_dim" in video_cfg:
        video_cfg["action_dim"] = int(video_cfg["action_dim"])
    video_cfg["use_gradient_checkpointing"] = _resolve_bool_config(
        video_cfg, "use_gradient_checkpointing", default=False
    )

    return video_cfg, OmegaConf.to_container(cfg, resolve=False)


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Preprocess a small WanVideoDiT from Wan2.2-5B weights for VAE-latent FastWAM."
    )
    parser.add_argument("--model-config", required=True, help="Path to model yaml with target video_dit_config.")
    parser.add_argument("--output", required=True, help="Output .pt path for the preprocessed small WanVideoDiT.")
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--dtype", default="bfloat16", choices=["float32", "float16", "bfloat16"])
    parser.add_argument("--apply-alpha-scaling", default="true")
    parser.add_argument("--wan-model-id", default="Wan-AI/Wan2.2-TI2V-5B")
    args = parser.parse_args()

    model_config_path = Path(args.model_config)
    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    torch_dtype = _parse_dtype(args.dtype)
    apply_alpha_scaling = _parse_bool(args.apply_alpha_scaling)

    target_video_cfg, cfg = _load_target_video_cfg(model_config_path)
    redirect_common_files = _parse_bool(cfg.get("redirect_common_files", True))

    source_wan_cfg = {
        "has_image_input": False,
        "patch_size": [1, 2, 2],
        "in_dim": 48,
        "hidden_dim": 3072,
        "ffn_dim": 14336,
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

    print(f"[INFO] Target small WanVideoDiT config from {model_config_path}:")
    for key in ["hidden_dim", "in_dim", "out_dim", "ffn_dim", "num_layers", "num_heads", "attn_head_dim", "patch_size"]:
        print(f"  {key}: {target_video_cfg[key]}")
    print(f"[INFO] Loading source Wan2.2 DiT from {args.wan_model_id} with dtype={torch_dtype} on {args.device}")

    components = load_wan22_ti2v_5b_components(
        device=args.device,
        torch_dtype=torch_dtype,
        model_id=args.wan_model_id,
        tokenizer_model_id=cfg.get("tokenizer_model_id", "Wan-AI/Wan2.1-T2V-1.3B"),
        redirect_common_files=redirect_common_files,
        dit_config=source_wan_cfg,
        skip_dit_load_from_pretrain=False,
        load_text_encoder=False,
        load_vae=False,
    )
    source_dit = components.dit
    if source_dit is None:
        raise RuntimeError("Source WanVideoDiT failed to load.")

    print("[INFO] Creating target small WanVideoDiT...")
    target_dit = WanVideoDiT(**target_video_cfg).to(device=args.device, dtype=torch_dtype)

    source_state = source_dit.state_dict()
    target_state = target_dit.state_dict()
    resized_state: dict[str, torch.Tensor] = {}
    copied = 0
    interpolated = 0
    for key in sorted(target_state.keys()):
        if key not in source_state:
            raise ValueError(f"Target key `{key}` missing from source WanVideoDiT state dict.")
        src = source_state[key]
        target = target_state[key]
        if tuple(src.shape) == tuple(target.shape):
            value = src
            copied += 1
        else:
            value = _resize_tensor_to_shape(src, tuple(target.shape), apply_alpha_scaling)
            interpolated += 1
        if tuple(value.shape) != tuple(target.shape):
            raise ValueError(
                f"Shape mismatch after resize for `{key}`: src={tuple(src.shape)}, "
                f"target={tuple(target.shape)}, got={tuple(value.shape)}"
            )
        resized_state[key] = value.detach().to(dtype=target.dtype, device="cpu").contiguous()

    payload = {
        "policy": {
            "source": "Wan-AI/Wan2.2-TI2V-5B WanVideoDiT",
            "alpha_scaling": bool(apply_alpha_scaling),
            "interpolation": "sequential_1d_linear_align_corners_true",
            "scope": "full_wan_video_dit_state_dict",
        },
        "state_dict": resized_state,
        "meta": {
            "hidden_dim": int(target_video_cfg["hidden_dim"]),
            "in_dim": int(target_video_cfg["in_dim"]),
            "out_dim": int(target_video_cfg["out_dim"]),
            "ffn_dim": int(target_video_cfg["ffn_dim"]),
            "num_layers": int(target_video_cfg["num_layers"]),
            "num_heads": int(target_video_cfg["num_heads"]),
            "attn_head_dim": int(target_video_cfg["attn_head_dim"]),
            "text_dim": int(target_video_cfg["text_dim"]),
            "freq_dim": int(target_video_cfg["freq_dim"]),
            "patch_size": [int(x) for x in target_video_cfg["patch_size"]],
            "eps": float(target_video_cfg["eps"]),
        },
    }
    torch.save(payload, str(output_path))
    print(f"[INFO] Saved small WanVideoDiT payload to {output_path}")
    print(f"  keys={len(resized_state)}, copied={copied}, interpolated={interpolated}")


if __name__ == "__main__":
    main()
