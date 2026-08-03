import json
import logging
import os
import sys
import time
from datetime import timedelta
from pathlib import Path
from typing import Any

import hydra
import numpy as np
import torch
import torch.distributed as dist
from hydra.utils import instantiate
from omegaconf import DictConfig, OmegaConf
from torch.utils.data import DataLoader, Dataset
from tqdm import tqdm

REPO_ROOT = Path(__file__).resolve().parents[1]
SRC_DIR = REPO_ROOT / "src"
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

from fastwam.models.wan22.helpers.loader import load_wan22_ti2v_5b_components
from fastwam.runtime import _mixed_precision_to_model_dtype, _normalize_mixed_precision
from fastwam.utils.config_resolvers import register_default_resolvers
from fastwam.utils.logging_config import get_logger, setup_logging

register_default_resolvers()
logger = get_logger(__name__)


def _init_distributed():
    world_size = int(os.environ.get("WORLD_SIZE", "1"))
    if world_size <= 1:
        return False, 0, 1, 0

    local_rank = int(os.environ.get("LOCAL_RANK", "0"))
    backend = "nccl" if torch.cuda.is_available() else "gloo"
    if torch.cuda.is_available():
        torch.cuda.set_device(local_rank)

    timeout_seconds = int(os.environ.get("VAE_DIST_TIMEOUT_SECONDS", "86400"))
    if timeout_seconds <= 0:
        raise ValueError(f"VAE_DIST_TIMEOUT_SECONDS must be positive, got {timeout_seconds}")

    if not dist.is_initialized():
        dist.init_process_group(
            backend=backend,
            init_method="env://",
            timeout=timedelta(seconds=timeout_seconds),
        )
    return True, dist.get_rank(), dist.get_world_size(), local_rank


def _to_bool(value: Any) -> bool:
    if isinstance(value, bool):
        return value
    if isinstance(value, str):
        text = value.strip().lower()
        if text in {"1", "true", "yes", "y"}:
            return True
        if text in {"0", "false", "no", "n"}:
            return False
    raise ValueError(f"Cannot parse bool value: {value}")


def _atomic_json_save(payload: dict[str, Any], output_path: Path):
    output_path.parent.mkdir(parents=True, exist_ok=True)
    tmp_path = output_path.parent / f".{output_path.name}.tmp.{os.getpid()}.{time.time_ns()}"
    with tmp_path.open("w", encoding="utf-8") as f:
        json.dump(payload, f, indent=2, sort_keys=True)
    os.replace(tmp_path, output_path)


class _IndexedVideoDataset(Dataset):
    def __init__(self, dataset, indices: list[int]):
        self.dataset = dataset
        self.indices = indices

    def __len__(self) -> int:
        return len(self.indices)

    def __getitem__(self, item: int) -> dict[str, Any]:
        idx = int(self.indices[item])
        sample = self.dataset._get(idx) if hasattr(self.dataset, "_get") else self.dataset[idx]
        resolved_idx = int(sample.get("dataset_idx", idx))
        if resolved_idx != idx:
            raise RuntimeError(
                "VAE latent precompute requires deterministic dataset indexing, "
                f"but requested idx={idx} resolved to idx={resolved_idx}."
            )
        video = sample["video"]
        if video.ndim != 4:
            raise ValueError(
                f"`sample['video']` must be [C,T,H,W], got {tuple(video.shape)} "
                f"for requested idx={idx}, resolved idx={resolved_idx}"
            )
        return {
            "dataset_idx": torch.tensor(resolved_idx, dtype=torch.long),
            "video": video.contiguous(),
        }


def _resolve_cache_dir(cfg: DictConfig) -> Path:
    cache_dir = cfg.get("vae_latent_cache_dir")
    if cache_dir is None and cfg.get("data") is not None and cfg.data.get("train") is not None:
        cache_dir = cfg.data.train.get("vae_latent_cache_dir")
    if cache_dir is None or str(cache_dir).strip() == "":
        raise ValueError(
            "Missing VAE latent cache dir. Pass for example: "
            "vae_latent_cache_dir=./data/vae_latents_cache/libero_wan22vae38_2cam224_window_mmap"
        )
    return Path(str(cache_dir)).expanduser()


def _resolve_cache_mode(cfg: DictConfig) -> str:
    mode = cfg.get("vae_latent_cache_mode")
    if mode is None and cfg.get("data") is not None and cfg.data.get("train") is not None:
        mode = cfg.data.train.get("vae_latent_cache_mode", "window_mmap")
    mode = str(mode or "window_mmap").strip().lower()
    if mode != "window_mmap":
        raise ValueError(f"`vae_latent_cache_mode` currently supports only 'window_mmap', got {mode}")
    return mode


def _build_dataset(cfg: DictConfig):
    if cfg.get("data") is None or cfg.data.get("train") is None:
        raise ValueError("`cfg.data.train` is required.")
    data_train_cfg = OmegaConf.create(OmegaConf.to_container(cfg.data.train, resolve=True))
    OmegaConf.set_struct(data_train_cfg, False)
    data_train_cfg.load_text_context = False
    data_train_cfg.dino_latent_cache_dir = None
    data_train_cfg.dino_latent_cache_required = False
    data_train_cfg.load_history_dino_latents = False
    data_train_cfg.history_dino_latent_cache_required = False
    data_train_cfg.vae_latent_cache_dir = None
    data_train_cfg.vae_latent_cache_required = False
    data_train_cfg.skip_video_load_if_latent_cached = False
    return instantiate(data_train_cfg)


def _build_vae(cfg: DictConfig, device: str, torch_dtype: torch.dtype):
    model_cfg = cfg.get("model")
    if model_cfg is None:
        raise ValueError("`cfg.model` is required for VAE latent precompute.")
    components = load_wan22_ti2v_5b_components(
        device=device,
        torch_dtype=torch_dtype,
        model_id=str(model_cfg.get("model_id", "Wan-AI/Wan2.2-TI2V-5B")),
        tokenizer_model_id=str(model_cfg.get("tokenizer_model_id", "Wan-AI/Wan2.1-T2V-1.3B")),
        redirect_common_files=_to_bool(model_cfg.get("redirect_common_files", True)),
        load_dit=False,
        load_text_encoder=False,
        load_vae=True,
    )
    if components.vae is None:
        raise RuntimeError("Wan VAE failed to load.")
    components.vae.eval().requires_grad_(False)
    return components.vae


def _dtype_from_name(dtype_name: str):
    name = str(dtype_name).strip().lower()
    if name in {"bf16", "bfloat16"}:
        return torch.bfloat16, np.uint16, "uint16", "bf16"
    if name in {"fp16", "float16"}:
        return torch.float16, np.float16, "float16", "fp16"
    if name in {"fp32", "float32"}:
        return torch.float32, np.float32, "float32", "fp32"
    raise ValueError("`vae_latent_cache_dtype` must be one of bf16/fp16/fp32.")


def _latents_to_numpy(latents: torch.Tensor, save_dtype: torch.dtype) -> np.ndarray:
    latents = latents.detach().to(device="cpu", dtype=save_dtype).contiguous()
    if save_dtype is torch.bfloat16:
        return latents.view(torch.uint16).numpy()
    return latents.numpy()


@hydra.main(config_path="../configs", config_name="train", version_base="1.3")
def main(cfg: DictConfig):
    setup_logging(log_level=logging.INFO)

    is_distributed, rank, world_size, local_rank = _init_distributed()
    if is_distributed and rank == 0:
        logger.info("Distributed VAE precompute enabled: world_size=%d", world_size)

    if torch.cuda.is_available():
        device = f"cuda:{local_rank}" if is_distributed else "cuda"
    else:
        device = "cpu"

    mixed_precision = _normalize_mixed_precision(str(cfg.get("mixed_precision", "bf16")))
    torch_dtype = _mixed_precision_to_model_dtype(mixed_precision)
    save_dtype, np_dtype, storage_dtype, save_dtype_name = _dtype_from_name(
        str(cfg.get("vae_latent_cache_dtype", "bf16"))
    )
    cache_dir = _resolve_cache_dir(cfg)
    cache_mode = _resolve_cache_mode(cfg)
    overwrite = _to_bool(cfg.get("overwrite", False))
    batch_size = int(cfg.get("vae_precompute_batch_size", 4))
    if batch_size <= 0:
        raise ValueError(f"`vae_precompute_batch_size` must be positive, got {batch_size}")
    num_workers = int(cfg.get("vae_precompute_num_workers", 8))
    if num_workers < 0:
        raise ValueError(f"`vae_precompute_num_workers` must be non-negative, got {num_workers}")

    dataset = _build_dataset(cfg)
    total = len(dataset)
    data_cfg = OmegaConf.to_container(cfg.data.train, resolve=True)
    video_sample_count = len(getattr(dataset, "video_sample_indices", []))
    if video_sample_count <= 1:
        raise ValueError(f"Invalid video_sample_count={video_sample_count}")

    vae = _build_vae(cfg, device=device, torch_dtype=torch_dtype)
    temporal_factor = int(vae.temporal_downsample_factor)
    latent_t = (video_sample_count - 1) // temporal_factor + 1
    latent_h = int(data_cfg.get("video_size")[0]) // int(vae.upsampling_factor)
    latent_w = int(data_cfg.get("video_size")[1]) // int(vae.upsampling_factor)
    latent_shape = (int(vae.z_dim), latent_t, latent_h, latent_w)

    cache_dir.mkdir(parents=True, exist_ok=True)
    complete_marker = cache_dir / ".complete"
    mmap_file = "latents.bf16.bin" if save_dtype is torch.bfloat16 else f"latents.{save_dtype_name}.bin"
    mmap_path = cache_dir / mmap_file
    metadata_path = cache_dir / "metadata.json"

    if complete_marker.exists() and not overwrite:
        if rank == 0:
            logger.info("VAE latent cache already complete: %s", cache_dir)
        if is_distributed and dist.is_initialized():
            dist.barrier()
            dist.destroy_process_group()
        return

    metadata = {
        "format_version": 1,
        "cache_mode": cache_mode,
        "mmap_file": mmap_file,
        "total_samples": total,
        "latent_shape": list(latent_shape),
        "save_dtype": save_dtype_name,
        "storage_dtype": storage_dtype,
        "model_id": str(cfg.model.get("model_id", "Wan-AI/Wan2.2-TI2V-5B")),
        "vae_class": type(vae).__name__,
        "vae_z_dim": int(vae.z_dim),
        "vae_temporal_downsample_factor": temporal_factor,
        "vae_spatial_downsample_factor": int(vae.upsampling_factor),
        "data_video_size": data_cfg.get("video_size"),
        "num_frames": data_cfg.get("num_frames"),
        "action_video_freq_ratio": data_cfg.get("action_video_freq_ratio"),
        "video_sample_count": video_sample_count,
        "concat_multi_camera": data_cfg.get("concat_multi_camera"),
        "complete": False,
    }

    if (not is_distributed) or rank == 0:
        if mmap_path.exists() and not overwrite:
            raise FileExistsError(
                f"Found existing incomplete VAE mmap cache at {mmap_path}. "
                "Pass overwrite=true to rebuild it."
            )
        if overwrite:
            complete_marker.unlink(missing_ok=True)
            mmap_path.unlink(missing_ok=True)
        _atomic_json_save(metadata, metadata_path)
        mmap = np.memmap(mmap_path, mode="w+", dtype=np_dtype, shape=(total, *latent_shape))
        mmap.flush()
        del mmap

    if is_distributed:
        dist.barrier()

    logger.info(
        "Precomputing VAE latents to %s | shape=(%d,%s) device=%s model_dtype=%s save_dtype=%s "
        "batch_size=%d num_workers=%d",
        cache_dir,
        total,
        ",".join(str(v) for v in latent_shape),
        device,
        torch_dtype,
        save_dtype,
        batch_size,
        num_workers,
    )

    local_indices = list(range(rank, total, world_size))
    loader = DataLoader(
        _IndexedVideoDataset(dataset, local_indices),
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=torch.cuda.is_available(),
        drop_last=False,
        persistent_workers=(num_workers > 0),
    )
    mmap = np.memmap(mmap_path, mode="r+", dtype=np_dtype, shape=(total, *latent_shape))
    written = 0

    pbar = tqdm(
        total=len(local_indices),
        desc=f"VAE latents rank {rank}/{world_size}" if is_distributed else "VAE latents",
        unit="sample",
        dynamic_ncols=True,
        disable=is_distributed and rank != 0,
    )
    with torch.no_grad():
        for batch in loader:
            videos = batch["video"].to(device=device, dtype=torch_dtype, non_blocking=True)
            latents = vae.encode(videos, device=device, tiled=False)
            latents_np = _latents_to_numpy(latents, save_dtype=save_dtype)
            if tuple(latents_np.shape[1:]) != latent_shape:
                raise ValueError(
                    f"Encoded VAE latent shape mismatch: got {tuple(latents_np.shape[1:])}, "
                    f"expected {latent_shape}."
                )
            indices = [int(x) for x in batch["dataset_idx"].detach().cpu().tolist()]
            mmap[indices] = latents_np
            written += len(indices)
            pbar.update(len(indices))
    mmap.flush()
    pbar.close()

    if is_distributed:
        reduce_device = torch.device(device) if str(device).startswith("cuda") else torch.device("cpu")
        count_tensor = torch.tensor([written], device=reduce_device, dtype=torch.long)
        dist.all_reduce(count_tensor, op=dist.ReduceOp.SUM)
        written = int(count_tensor[0].item())
        dist.barrier()

    if (not is_distributed) or rank == 0:
        metadata["complete"] = True
        _atomic_json_save(metadata, metadata_path)
        complete_marker.write_text(time.strftime("%Y-%m-%d %H:%M:%S\n"), encoding="utf-8")
        logger.info("Finished VAE latent precompute: written=%d total=%d cache=%s", written, total, cache_dir)

    if is_distributed and dist.is_initialized():
        dist.barrier()
        dist.destroy_process_group()


if __name__ == "__main__":
    main()
