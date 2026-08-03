import json
import logging
import os
import sys
import uuid
from datetime import timedelta
from pathlib import Path
from typing import Any

import hydra
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

from fastwam.models.wan22.dino_encoder import DinoVideoEncoder
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

    timeout_seconds = int(os.environ.get("DINO_DIST_TIMEOUT_SECONDS", "86400"))
    if timeout_seconds <= 0:
        raise ValueError(f"DINO_DIST_TIMEOUT_SECONDS must be positive, got {timeout_seconds}")

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


def _atomic_torch_save(payload: dict[str, Any], output_path: Path):
    output_path.parent.mkdir(parents=True, exist_ok=True)
    tmp_path = output_path.parent / f".{output_path.name}.tmp.{uuid.uuid4().hex}"
    torch.save(payload, str(tmp_path))
    os.replace(tmp_path, output_path)


def _atomic_json_save(payload: dict[str, Any], output_path: Path):
    output_path.parent.mkdir(parents=True, exist_ok=True)
    tmp_path = output_path.parent / f".{output_path.name}.tmp.{uuid.uuid4().hex}"
    with tmp_path.open("w", encoding="utf-8") as f:
        json.dump(payload, f, indent=2, sort_keys=True)
    os.replace(tmp_path, output_path)


def _cache_path(cache_dir: Path, idx: int) -> Path:
    return cache_dir / f"{int(idx):08d}.pt"


def _frame_cache_path(cache_dir: Path, idx: int) -> Path:
    return cache_dir / "frames" / f"{int(idx):08d}.pt"


class _IndexedVideoDataset(Dataset):
    def __init__(self, dataset, indices: list[int]):
        self.dataset = dataset
        self.indices = indices

    def __len__(self) -> int:
        return len(self.indices)

    def __getitem__(self, item: int) -> dict[str, Any]:
        idx = int(self.indices[item])
        if hasattr(self.dataset, "_get"):
            sample = self.dataset._get(idx)
        else:
            sample = self.dataset[idx]
        resolved_idx = int(sample.get("dataset_idx", idx))
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


class _FrameVideoDataset(Dataset):
    def __init__(self, robot_dataset, indices: list[int]):
        self.robot_dataset = robot_dataset
        self.indices = indices
        self.multi_dataset = robot_dataset.lerobot_dataset.multi_dataset

        self.dataset_offsets: list[int] = []
        offset = 0
        for dataset in self.multi_dataset._datasets:
            self.dataset_offsets.append(offset)
            offset += len(dataset)

    def __len__(self) -> int:
        return len(self.indices)

    def _resolve_dataset_and_local_idx(self, global_idx: int):
        for dataset_idx, dataset in enumerate(self.multi_dataset._datasets):
            start = self.dataset_offsets[dataset_idx]
            end = start + len(dataset)
            if start <= global_idx < end:
                return dataset, global_idx - start
        raise IndexError(f"Global frame index {global_idx} out of bounds.")

    def _load_camera_frame(self, dataset, local_idx: int, lerobot_key: str) -> torch.Tensor:
        item = dataset.hf_dataset[local_idx]
        if lerobot_key in dataset.meta.video_keys:
            timestamp = float(item["timestamp"].item() if hasattr(item["timestamp"], "item") else item["timestamp"])
            episode_index = int(item["episode_index"].item() if hasattr(item["episode_index"], "item") else item["episode_index"])
            frame = dataset._query_videos({lerobot_key: [timestamp]}, episode_index)[lerobot_key]
        else:
            frame = item[lerobot_key]
            if frame.ndim == 4:
                frame = frame[0]
        if frame.ndim != 3:
            raise ValueError(f"Expected camera frame [C,H,W] for {lerobot_key}, got {tuple(frame.shape)}")
        return frame.to(dtype=torch.float32)

    def _format_frame(self, global_idx: int) -> torch.Tensor:
        dataset, local_idx = self._resolve_dataset_and_local_idx(global_idx)
        processor = self.robot_dataset.lerobot_dataset.processor
        if processor is None:
            raise ValueError("Frame-level DINO precompute requires RobotVideoDataset processor to be set.")
        transforms = processor.train_transforms if processor.is_train else processor.val_transforms

        camera_frames = []
        for meta in self.robot_dataset.lerobot_dataset.image_meta:
            frame = self._load_camera_frame(dataset, local_idx, meta["lerobot_key"])

            # Match BaseLerobotDataset._get_image + FastWAMProcessor image
            # transforms for a single timestamp. This keeps frame cache inputs
            # bit-aligned with online DINO training without loading a full
            # 33-step window for every cached frame.
            if frame.dtype != torch.uint8:
                frame = (frame * 255).to(torch.uint8)
            image = frame.unsqueeze(0)  # [1,C,H,W]
            current_transforms = transforms[meta["key"]] if isinstance(transforms, dict) else transforms
            for trans in current_transforms:
                image = trans(image)
            expected_shape = tuple(meta["shape"])
            if tuple(image.shape[1:]) != expected_shape:
                raise ValueError(
                    f"Frame transform shape mismatch for {meta['key']}: "
                    f"got {tuple(image.shape[1:])}, expected {expected_shape}"
                )
            camera_frames.append(image[0])

        if len(camera_frames) == 1:
            image = camera_frames[0]
        elif self.robot_dataset.concat_multi_camera == "robotwin":
            if len(camera_frames) != 3:
                raise ValueError(
                    f"`concat_multi_camera='robotwin'` requires exactly 3 cameras, got {len(camera_frames)}."
                )
            cam_top = torch.nn.functional.interpolate(
                camera_frames[0].unsqueeze(0),
                size=(256, 320),
                mode="bilinear",
                align_corners=False,
                antialias=True,
            )[0]
            cam_left = torch.nn.functional.interpolate(
                camera_frames[1].unsqueeze(0),
                size=(128, 160),
                mode="bilinear",
                align_corners=False,
                antialias=True,
            )[0]
            cam_right = torch.nn.functional.interpolate(
                camera_frames[2].unsqueeze(0),
                size=(128, 160),
                mode="bilinear",
                align_corners=False,
                antialias=True,
            )[0]
            bottom = torch.cat([cam_left, cam_right], dim=-1)
            image = torch.cat([cam_top, bottom], dim=-2)
        elif self.robot_dataset.concat_multi_camera == "horizontal":
            image = torch.cat(camera_frames, dim=-1)
        elif self.robot_dataset.concat_multi_camera == "vertical":
            image = torch.cat(camera_frames, dim=-2)
        else:
            raise ValueError(
                "Frame-level DINO precompute currently supports concat_multi_camera "
                f"horizontal/vertical/robotwin, got {self.robot_dataset.concat_multi_camera}."
            )

        image = self.robot_dataset.resize_transform(image)
        image = self.robot_dataset.crop_transform(image)
        image = self.robot_dataset.normalize_transform(image)
        return image.contiguous()  # [C,H,W], range [-1,1]

    def __getitem__(self, item: int) -> dict[str, Any]:
        idx = int(self.indices[item])
        return {
            "frame_idx": torch.tensor(idx, dtype=torch.long),
            "frame": self._format_frame(idx),
        }


def _resolve_cache_dir(cfg: DictConfig) -> Path:
    cache_dir = cfg.get("dino_latent_cache_dir")
    if cache_dir is None and cfg.get("data") is not None and cfg.data.get("train") is not None:
        cache_dir = cfg.data.train.get("dino_latent_cache_dir")
    if cache_dir is None or str(cache_dir).strip() == "":
        raise ValueError(
            "Missing DINO latent cache dir. Pass for example: "
            "dino_latent_cache_dir=./data/dino_latents_cache/libero_dino_s_2cam224 "
            "and then train with data.train.dino_latent_cache_dir pointing to the same path."
        )
    return Path(str(cache_dir)).expanduser()


def _resolve_cache_mode(cfg: DictConfig) -> str:
    mode = cfg.get("dino_latent_cache_mode")
    if mode is None and cfg.get("data") is not None and cfg.data.get("train") is not None:
        mode = cfg.data.train.get("dino_latent_cache_mode", "window")
    mode = str(mode or "window").strip().lower()
    if mode not in {"window", "frame"}:
        raise ValueError(f"`dino_latent_cache_mode` must be 'window' or 'frame', got {mode}")
    return mode


def _build_dino_encoder(model_cfg: DictConfig, device: str, torch_dtype: torch.dtype) -> DinoVideoEncoder:
    if model_cfg is None:
        raise ValueError("`cfg.model` is required.")
    dino_config = model_cfg.get("dino_config")
    if dino_config is None:
        raise ValueError("`cfg.model.dino_config` is required for DINO latent precompute.")
    if isinstance(dino_config, DictConfig):
        dino_config = OmegaConf.to_container(dino_config, resolve=True)
    if not isinstance(dino_config, dict):
        raise ValueError(f"`dino_config` must resolve to a dict, got {type(dino_config)}")

    encoder = DinoVideoEncoder(
        model_name=dino_config.get("model_name", "dinov3-vitl16"),
        model_path=dino_config.get("model_path", None),
        input_resolution=tuple(dino_config.get("input_resolution", [224, 224])),
        patch_size=dino_config.get("patch_size", 16),
        feature_dim=dino_config.get("feature_dim", 1024),
        use_cls_token=dino_config.get("use_cls_token", False),
        normalize_features=dino_config.get("normalize_features", False),
        latent_spatial_pool=tuple(dino_config.get("latent_spatial_pool", [1, 1])),
        encode_microbatch_size=dino_config.get("encode_microbatch_size", 72),
    )
    encoder.load_backbone(device=torch.device(device), dtype=torch_dtype)
    encoder.eval()
    return encoder


def _build_dataset(cfg: DictConfig):
    if cfg.get("data") is None or cfg.data.get("train") is None:
        raise ValueError("`cfg.data.train` is required.")
    data_train_cfg = OmegaConf.create(OmegaConf.to_container(cfg.data.train, resolve=True))
    OmegaConf.set_struct(data_train_cfg, False)
    data_train_cfg.load_text_context = False
    data_train_cfg.dino_latent_cache_dir = None
    data_train_cfg.dino_latent_cache_required = False
    return instantiate(data_train_cfg)


def _encode_and_save_batch(
    *,
    encoder: DinoVideoEncoder,
    batch_indices: list[int],
    batch_videos: list[torch.Tensor],
    cache_dir: Path,
    device: str,
    torch_dtype: torch.dtype,
    save_dtype: torch.dtype,
    overwrite: bool,
) -> tuple[int, int]:
    if not batch_indices:
        return 0, 0

    video = torch.stack(batch_videos, dim=0).to(device=device, dtype=torch_dtype, non_blocking=True)
    latents = encoder.encode_video_to_latent(video).detach().to(device="cpu", dtype=save_dtype).contiguous()

    written = 0
    skipped = 0
    for local_i, idx in enumerate(batch_indices):
        path = _cache_path(cache_dir, idx)
        if path.exists() and not overwrite:
            skipped += 1
            continue
        latent_i = latents[local_i].clone().contiguous()
        payload = {
            "dino_latents": latent_i,
            "idx": int(idx),
            "shape": list(latent_i.shape),
        }
        _atomic_torch_save(payload, path)
        written += 1
    return written, skipped


def _encode_and_save_tensor_batch(
    *,
    encoder: DinoVideoEncoder,
    indices: torch.Tensor,
    videos: torch.Tensor,
    cache_dir: Path,
    device: str,
    torch_dtype: torch.dtype,
    save_dtype: torch.dtype,
    overwrite: bool,
) -> tuple[int, int]:
    if videos.ndim != 5:
        raise ValueError(f"`videos` must be [B,C,T,H,W], got {tuple(videos.shape)}")

    videos = videos.to(device=device, dtype=torch_dtype, non_blocking=True)
    latents = encoder.encode_video_to_latent(videos).detach().to(device="cpu", dtype=save_dtype).contiguous()

    indices_list = [int(x) for x in indices.detach().cpu().tolist()]
    written = 0
    skipped = 0
    for local_i, idx in enumerate(indices_list):
        path = _cache_path(cache_dir, idx)
        if path.exists() and not overwrite:
            skipped += 1
            continue
        latent_i = latents[local_i].clone().contiguous()
        payload = {
            "dino_latents": latent_i,
            "idx": int(idx),
            "shape": list(latent_i.shape),
        }
        _atomic_torch_save(payload, path)
        written += 1
    return written, skipped


def _encode_and_save_frame_tensor_batch(
    *,
    encoder: DinoVideoEncoder,
    indices: torch.Tensor,
    frames: torch.Tensor,
    cache_dir: Path,
    device: str,
    torch_dtype: torch.dtype,
    save_dtype: torch.dtype,
    overwrite: bool,
) -> tuple[int, int]:
    if frames.ndim != 4:
        raise ValueError(f"`frames` must be [B,C,H,W], got {tuple(frames.shape)}")

    frames = frames.to(device=device, dtype=torch_dtype, non_blocking=True)
    video = frames.unsqueeze(2)  # [B,C,1,H,W]
    latents = encoder.encode_video_to_latent(video).detach().to(device="cpu", dtype=save_dtype)
    latents = latents[:, :, 0].contiguous()  # [B,D,Hg,Wg]

    indices_list = [int(x) for x in indices.detach().cpu().tolist()]
    written = 0
    skipped = 0
    for local_i, idx in enumerate(indices_list):
        path = _frame_cache_path(cache_dir, idx)
        if path.exists() and not overwrite:
            skipped += 1
            continue
        latent_i = latents[local_i].clone().contiguous()
        payload = {
            "dino_latent": latent_i,
            "idx": int(idx),
            "shape": list(latent_i.shape),
        }
        _atomic_torch_save(payload, path)
        written += 1
    return written, skipped


@hydra.main(config_path="../configs", config_name="train", version_base="1.3")
def main(cfg: DictConfig):
    setup_logging(log_level=logging.INFO)

    is_distributed, rank, world_size, local_rank = _init_distributed()
    if is_distributed and rank == 0:
        logger.info("Distributed DINO precompute enabled: world_size=%d", world_size)

    if torch.cuda.is_available():
        device = f"cuda:{local_rank}" if is_distributed else "cuda"
    else:
        device = "cpu"

    mixed_precision = _normalize_mixed_precision(str(cfg.get("mixed_precision", "bf16")))
    torch_dtype = _mixed_precision_to_model_dtype(mixed_precision)
    save_dtype_name = str(cfg.get("dino_latent_cache_dtype", "bf16")).strip().lower()
    if save_dtype_name in {"bf16", "bfloat16"}:
        save_dtype = torch.bfloat16
    elif save_dtype_name in {"fp16", "float16"}:
        save_dtype = torch.float16
    elif save_dtype_name in {"fp32", "float32"}:
        save_dtype = torch.float32
    else:
        raise ValueError("`dino_latent_cache_dtype` must be one of bf16/fp16/fp32.")

    cache_dir = _resolve_cache_dir(cfg)
    cache_mode = _resolve_cache_mode(cfg)
    cache_dir.mkdir(parents=True, exist_ok=True)
    overwrite = _to_bool(cfg.get("overwrite", False))
    batch_size = int(cfg.get("dino_precompute_batch_size", 8))
    if batch_size <= 0:
        raise ValueError(f"`dino_precompute_batch_size` must be positive, got {batch_size}")
    num_workers = int(cfg.get("dino_precompute_num_workers", 8))
    if num_workers < 0:
        raise ValueError(f"`dino_precompute_num_workers` must be non-negative, got {num_workers}")

    logger.info(
        "Precomputing DINO latents to %s | mode=%s device=%s model_dtype=%s save_dtype=%s overwrite=%s batch_size=%d num_workers=%d",
        cache_dir,
        cache_mode,
        device,
        torch_dtype,
        save_dtype,
        overwrite,
        batch_size,
        num_workers,
    )

    dataset = _build_dataset(cfg)
    encoder = _build_dino_encoder(cfg.model, device=device, torch_dtype=torch_dtype)
    total = len(dataset)
    local_indices = list(range(rank, total, world_size))
    cache_path_fn = _frame_cache_path if cache_mode == "frame" else _cache_path
    indices_to_process = [
        idx for idx in local_indices
        if overwrite or not cache_path_fn(cache_dir, idx).exists()
    ]

    dino_cfg = OmegaConf.to_container(cfg.model.dino_config, resolve=True)
    data_cfg = OmegaConf.to_container(cfg.data.train, resolve=True)
    if (not is_distributed) or rank == 0:
        _atomic_json_save(
            {
                "total_samples": total,
                "cache_mode": cache_mode,
                "dino_config": dino_cfg,
                "data_video_size": data_cfg.get("video_size"),
                "num_frames": data_cfg.get("num_frames"),
                "action_video_freq_ratio": data_cfg.get("action_video_freq_ratio"),
                "concat_multi_camera": data_cfg.get("concat_multi_camera"),
                "save_dtype": save_dtype_name,
            },
            cache_dir / "metadata.json",
        )

    written = 0
    skipped = len(local_indices) - len(indices_to_process)
    if cache_mode == "frame":
        loader_dataset = _FrameVideoDataset(dataset, indices_to_process)
    else:
        loader_dataset = _IndexedVideoDataset(dataset, indices_to_process)
    loader = DataLoader(
        loader_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=torch.cuda.is_available(),
        drop_last=False,
        persistent_workers=(num_workers > 0),
    )

    pbar = tqdm(
        total=len(local_indices),
        initial=skipped,
        desc=f"DINO latents rank {rank}/{world_size}" if is_distributed else "DINO latents",
        unit="sample",
        dynamic_ncols=True,
        disable=is_distributed and rank != 0,
    )
    with torch.no_grad():
        for batch in loader:
            if cache_mode == "frame":
                new_written, new_skipped = _encode_and_save_frame_tensor_batch(
                    encoder=encoder,
                    indices=batch["frame_idx"],
                    frames=batch["frame"],
                    cache_dir=cache_dir,
                    device=device,
                    torch_dtype=torch_dtype,
                    save_dtype=save_dtype,
                    overwrite=overwrite,
                )
                batch_count = int(batch["frame"].shape[0])
            else:
                new_written, new_skipped = _encode_and_save_tensor_batch(
                    encoder=encoder,
                    indices=batch["dataset_idx"],
                    videos=batch["video"],
                    cache_dir=cache_dir,
                    device=device,
                    torch_dtype=torch_dtype,
                    save_dtype=save_dtype,
                    overwrite=overwrite,
                )
                batch_count = int(batch["video"].shape[0])
            written += new_written
            skipped += new_skipped
            pbar.update(batch_count)

    pbar.close()

    if is_distributed:
        reduce_device = torch.device(device) if str(device).startswith("cuda") else torch.device("cpu")
        counts = torch.tensor([written, skipped], device=reduce_device, dtype=torch.long)
        dist.all_reduce(counts, op=dist.ReduceOp.SUM)
        written = int(counts[0].item())
        skipped = int(counts[1].item())

    if (not is_distributed) or rank == 0:
        logger.info("Finished DINO latent precompute: written=%d skipped=%d total=%d", written, skipped, total)

    if is_distributed and dist.is_initialized():
        dist.barrier()
        dist.destroy_process_group()


if __name__ == "__main__":
    main()
