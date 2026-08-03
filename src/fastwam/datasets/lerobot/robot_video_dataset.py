import hashlib
import json
import os
from typing import Optional
import time
import numpy as np
import traceback
import torch
import torchvision.transforms.functional as transforms_F
from contextlib import contextmanager

from omegaconf import DictConfig, OmegaConf

from hydra.utils import instantiate
from .base_lerobot_dataset import BaseLerobotDataset
from .utils.normalizer import save_dataset_stats_to_json, load_dataset_stats_from_json
from ..dataset_utils import ResizeSmallestSideAspectPreserving, CenterCrop, Normalize
from fastwam.utils.logging_config import get_logger
from fastwam.utils import misc, pytorch_utils
from accelerate import PartialState
logger = get_logger(__name__)


DEFAULT_PROMPT = "A video recorded from a robot's point of view executing the following instruction: {task}"

class RobotVideoDataset(torch.utils.data.Dataset):
    def __init__(
        self,
        dataset_dirs,
        shape_meta,
        num_frames=33,
        video_size=[384, 640],
        camera_key=None,
        processor=None,
        text_embedding_cache_dir=None,
        context_len=128,
        pretrained_norm_stats=None,
        val_set_proportion=0.05,
        is_training_set=False,
        global_sample_stride=1,
        episode_filter: Optional[str] = None,
        robotwin_split_block_size: int = 550,
        robotwin_clean_episodes_per_block: int = 50,
        action_video_freq_ratio: int = 1,
        skip_padding_as_possible: bool = False,
        max_padding_retry: int = 3,
        concat_multi_camera: str = "horizontal", # "horizontal", "vertical", "robotwin", or None
        override_instruction: Optional[str] = None, # whether to hardcode a specific instruction for all samples, for debugging
        load_text_context: bool = True,
        dino_latent_cache_dir: Optional[str] = None,
        dino_latent_cache_mode: str = "window",
        dino_latent_cache_required: bool = False,
        vae_latent_cache_dir: Optional[str] = None,
        vae_latent_cache_mode: str = "window_mmap",
        vae_latent_cache_required: bool = False,
        skip_video_load_if_latent_cached: bool = False,
        allow_latent_cache_with_episode_filter: bool = False,
        load_history_dino_latents: bool = False,
        history_dino_frame_offsets: Optional[list[int]] = None,
        history_dino_latent_cache_required: Optional[bool] = None,
        load_history_dino_video: bool = False,
        load_history_vae_video: bool = False,
        history_vae_frame_offsets: Optional[list[int]] = None,
        load_semantic_image: bool = False,
        semantic_frame_offset: int = 0,
    ):
        self.lerobot_dataset = BaseLerobotDataset(
            dataset_dirs=dataset_dirs,
            shape_meta=OmegaConf.to_container(shape_meta, resolve=True),
            obs_size=num_frames,
            action_size=num_frames - 1,
            val_set_proportion=val_set_proportion,
            is_training_set=is_training_set,
            global_sample_stride=global_sample_stride,
            episode_filter=episode_filter,
            robotwin_split_block_size=robotwin_split_block_size,
            robotwin_clean_episodes_per_block=robotwin_clean_episodes_per_block,
        )
    
        self.num_frames = num_frames
        self.action_video_freq_ratio = action_video_freq_ratio
        
        assert (num_frames - 1) % self.action_video_freq_ratio == 0, \
            f"num_frames-1 must be divisible by action_video_freq_ratio, got {num_frames - 1} and {self.action_video_freq_ratio}"
        assert ((num_frames - 1) // self.action_video_freq_ratio) % 4 == 0, \
            f"video frames must be divisible by 4 for tokenization, got {(num_frames - 1) // self.action_video_freq_ratio}"
        self.video_sample_indices = list(range(0, num_frames, self.action_video_freq_ratio))

        self.camera_key = camera_key

        self.video_size = video_size
        self.text_embedding_cache_dir = text_embedding_cache_dir
        self.context_len = context_len
        self.skip_padding_as_possible = skip_padding_as_possible
        self.max_padding_retry = max_padding_retry
        self.concat_multi_camera = concat_multi_camera
        self.override_instruction = override_instruction
        self.load_text_context = load_text_context
        self.vae_latent_cache_dir = vae_latent_cache_dir
        self.vae_latent_cache_mode = str(vae_latent_cache_mode).strip().lower()
        if self.vae_latent_cache_mode not in {"window", "window_mmap"}:
            raise ValueError(
                f"`vae_latent_cache_mode` must be 'window' or 'window_mmap', got {vae_latent_cache_mode}"
            )
        self.vae_latent_cache_required = bool(vae_latent_cache_required)
        self.skip_video_load_if_latent_cached = bool(skip_video_load_if_latent_cached)
        if self.skip_video_load_if_latent_cached:
            if self.vae_latent_cache_dir is None or str(self.vae_latent_cache_dir).strip() == "":
                raise ValueError(
                    "`skip_video_load_if_latent_cached=true` requires `vae_latent_cache_dir`."
                )
            if not self.vae_latent_cache_required:
                raise ValueError(
                    "`skip_video_load_if_latent_cached=true` requires "
                    "`vae_latent_cache_required=true`."
                )
        self.lerobot_dataset._set_return_images(not self.skip_video_load_if_latent_cached)
        self.dino_latent_cache_dir = dino_latent_cache_dir
        self.dino_latent_cache_mode = str(dino_latent_cache_mode).strip().lower()
        if self.dino_latent_cache_mode not in {"window", "frame", "frame_mmap"}:
            raise ValueError(
                f"`dino_latent_cache_mode` must be 'window', 'frame', or 'frame_mmap', got {dino_latent_cache_mode}"
            )
        self.dino_latent_cache_required = dino_latent_cache_required
        self.allow_latent_cache_with_episode_filter = bool(allow_latent_cache_with_episode_filter)
        self._assert_episode_filter_cache_safe()
        self.load_history_dino_latents = bool(load_history_dino_latents)
        if history_dino_frame_offsets is None:
            history_dino_frame_offsets = [-8, -4, 0]
        self.history_dino_frame_offsets = [int(offset) for offset in history_dino_frame_offsets]
        if self.load_history_dino_latents and len(self.history_dino_frame_offsets) == 0:
            raise ValueError(
                "`history_dino_frame_offsets` must contain at least one offset when "
                "`load_history_dino_latents=true`."
            )
        self.load_history_dino_video = bool(load_history_dino_video)
        if self.load_history_dino_video and len(self.history_dino_frame_offsets) == 0:
            raise ValueError(
                "`history_dino_frame_offsets` must contain at least one offset when "
                "`load_history_dino_video=true`."
            )
        self.load_history_vae_video = bool(load_history_vae_video)
        self.load_semantic_image = bool(load_semantic_image)
        self.semantic_frame_offset = int(semantic_frame_offset)
        if history_vae_frame_offsets is None:
            history_vae_frame_offsets = self.history_dino_frame_offsets
        self.history_vae_frame_offsets = [int(offset) for offset in history_vae_frame_offsets]
        if self.load_history_vae_video and len(self.history_vae_frame_offsets) == 0:
            raise ValueError(
                "`history_vae_frame_offsets` must contain at least one offset when "
                "`load_history_vae_video=true`."
            )
        self.history_dino_latent_cache_required = (
            bool(dino_latent_cache_required)
            if history_dino_latent_cache_required is None
            else bool(history_dino_latent_cache_required)
        )
        if self.load_history_dino_latents and self.dino_latent_cache_mode == "window":
            raise ValueError(
                "`load_history_dino_latents=true` requires dino_latent_cache_mode='frame' "
                "or 'frame_mmap'. Window caches contain the prediction window, not arbitrary "
                "past frames."
            )
        self._dino_frame_mmap = None
        self._dino_frame_mmap_path = None
        self._dino_frame_mmap_np_dtype = None
        self._dino_frame_mmap_torch_dtype = None
        self._dino_frame_mmap_shape = None
        self._vae_window_mmap = None
        self._vae_window_mmap_path = None
        self._vae_window_mmap_np_dtype = None
        self._vae_window_mmap_torch_dtype = None
        self._vae_window_mmap_shape = None

        self.resize_transform = ResizeSmallestSideAspectPreserving(
            args={"img_w": self.video_size[1], "img_h": self.video_size[0]},
        )
        self.crop_transform = CenterCrop(
            args={"img_w": self.video_size[1], "img_h": self.video_size[0]},
        )
        self.normalize_transform = Normalize(
            args={"mean": 0.5, "std": 0.5},
        )
        if processor is not None:
            if isinstance(processor, DictConfig):
                processor = instantiate(processor)
            if not pretrained_norm_stats:
                if not is_training_set:
                    raise ValueError("pretrained_norm_stats must be provided for validation/test sets since we don't want to calculate stats on them.")
                if PartialState().is_main_process:
                    logger.info("Calculating dataset stats for normalization...")
                    dataset_stats = self.lerobot_dataset.get_dataset_stats(processor)
                    work_dir = misc.get_work_dir()
                    save_dataset_stats_to_json(dataset_stats, os.path.join(work_dir, "dataset_stats.json"))
                else:
                    dataset_stats = None
                if torch.distributed.is_available() and torch.distributed.is_initialized():
                    obj_list = [dataset_stats]
                    torch.distributed.broadcast_object_list(obj_list, src=0)
                    dataset_stats = obj_list[0]
            else:
                dataset_stats = load_dataset_stats_from_json(pretrained_norm_stats)
                logger.info(f"Using dataset stats: {pretrained_norm_stats}")
                if PartialState().is_main_process:
                    work_dir = misc.get_work_dir()
                    save_dataset_stats_to_json(dataset_stats, os.path.join(work_dir, "dataset_stats.json"))

            processor.set_normalizer_from_stats(dataset_stats)
            self.lerobot_dataset.set_processor(processor)

    def _assert_episode_filter_cache_safe(self) -> None:
        episode_filter = getattr(self.lerobot_dataset, "episode_filter", None)
        if episode_filter is None or self.allow_latent_cache_with_episode_filter:
            return
        active_cache_dirs = []
        if self.dino_latent_cache_dir is not None and str(self.dino_latent_cache_dir).strip() != "":
            active_cache_dirs.append("dino_latent_cache_dir")
        if self.vae_latent_cache_dir is not None and str(self.vae_latent_cache_dir).strip() != "":
            active_cache_dirs.append("vae_latent_cache_dir")
        if not active_cache_dirs:
            return
        raise ValueError(
            f"`episode_filter={episode_filter}` reindexes dataset frame ids, so full-dataset "
            f"latent caches can become misaligned: {active_cache_dirs}. Use online encoding, "
            "or regenerate caches for the same filtered dataset and set "
            "`allow_latent_cache_with_episode_filter=true` explicitly."
        )
        
    def __len__(self):
        return len(self.lerobot_dataset)

    def _get_dino_latent_cache_path(self, idx: int) -> Optional[str]:
        if self.dino_latent_cache_dir is None or str(self.dino_latent_cache_dir).strip() == "":
            return None
        return os.path.join(str(self.dino_latent_cache_dir), f"{int(idx):08d}.pt")

    def _get_dino_frame_cache_path(self, idx: int) -> Optional[str]:
        if self.dino_latent_cache_dir is None or str(self.dino_latent_cache_dir).strip() == "":
            return None
        return os.path.join(str(self.dino_latent_cache_dir), "frames", f"{int(idx):08d}.pt")

    def _get_vae_latent_cache_path(self, idx: int) -> Optional[str]:
        if self.vae_latent_cache_dir is None or str(self.vae_latent_cache_dir).strip() == "":
            return None
        return os.path.join(str(self.vae_latent_cache_dir), f"{int(idx):08d}.pt")

    def _get_global_indices_for_offsets(self, idx: int, offsets: list[int]) -> list[int]:
        episode_to = self.lerobot_dataset.episode_data_index["to"]
        episode_idx = int(torch.searchsorted(episode_to, torch.tensor(int(idx)), right=True).item())
        episode_start = int(self.lerobot_dataset.episode_data_index["from"][episode_idx].item())
        episode_end = int(self.lerobot_dataset.episode_data_index["to"][episode_idx].item())
        return [
            max(episode_start, min(episode_end - 1, int(idx) + int(offset)))
            for offset in offsets
        ]

    def _get_video_global_indices(self, idx: int) -> list[int]:
        return self._get_global_indices_for_offsets(idx, self.video_sample_indices)

    def _get_history_dino_global_indices(self, idx: int) -> list[int]:
        return self._get_global_indices_for_offsets(idx, self.history_dino_frame_offsets)

    def _get_history_vae_global_indices(self, idx: int) -> list[int]:
        return self._get_global_indices_for_offsets(idx, self.history_vae_frame_offsets)

    @staticmethod
    def _mmap_dtype_from_metadata(metadata: dict, metadata_path: str, cache_name: str):
        save_dtype = str(metadata.get("save_dtype", "bf16")).strip().lower()
        storage_dtype = str(metadata.get("storage_dtype", "")).strip().lower()
        if save_dtype in {"bf16", "bfloat16"}:
            if storage_dtype and storage_dtype != "uint16":
                raise ValueError(f"{cache_name} bf16 mmap cache must use uint16 storage, got {storage_dtype}")
            return np.uint16, torch.bfloat16
        if save_dtype in {"fp16", "float16"}:
            return np.float16, torch.float16
        if save_dtype in {"fp32", "float32"}:
            return np.float32, torch.float32
        raise ValueError(f"Unsupported {cache_name} mmap save_dtype={save_dtype!r} in {metadata_path}")

    def _load_cached_dino_latents(self, idx: int) -> Optional[torch.Tensor]:
        if self.dino_latent_cache_mode == "frame":
            return self._load_cached_dino_frame_latents(idx)
        if self.dino_latent_cache_mode == "frame_mmap":
            return self._load_cached_dino_frame_mmap_latents(idx)

        cache_path = self._get_dino_latent_cache_path(idx)
        if cache_path is None:
            return None
        if not os.path.exists(cache_path):
            if self.dino_latent_cache_required:
                raise FileNotFoundError(
                    f"Missing DINO latent cache for dataset idx={idx}: {cache_path}. "
                    "Run scripts/precompute_dino_latents.py first, or disable "
                    "`dino_latent_cache_required`."
                )
            return None

        payload = torch.load(cache_path, map_location="cpu")
        if isinstance(payload, dict):
            latents = payload.get("dino_latents", payload.get("latents"))
        else:
            latents = payload
        if latents is None:
            raise KeyError(f"DINO latent cache missing `dino_latents`: {cache_path}")
        if not torch.is_tensor(latents):
            raise TypeError(f"Cached DINO latents must be a tensor, got {type(latents)} in {cache_path}")
        if latents.ndim != 4:
            raise ValueError(
                f"Cached DINO latents must be [D,T,H,W] for one sample, "
                f"got shape {tuple(latents.shape)} in {cache_path}"
            )
        return latents.contiguous()

    def _load_cached_dino_frame_latents(self, idx: int) -> Optional[torch.Tensor]:
        frame_indices = self._get_video_global_indices(idx)
        return self._load_cached_dino_frame_latents_for_indices(
            frame_indices,
            idx_for_error=idx,
            required=self.dino_latent_cache_required,
        )

    def _load_cached_dino_frame_latents_for_indices(
        self,
        frame_indices: list[int],
        idx_for_error: int,
        required: bool,
    ) -> Optional[torch.Tensor]:
        latents = []
        missing_paths = []
        for frame_idx in frame_indices:
            cache_path = self._get_dino_frame_cache_path(frame_idx)
            if cache_path is None:
                return None
            if not os.path.exists(cache_path):
                missing_paths.append(cache_path)
                continue
            payload = torch.load(cache_path, map_location="cpu")
            latent = payload.get("dino_latent", payload.get("latent")) if isinstance(payload, dict) else payload
            if latent is None:
                raise KeyError(f"DINO frame cache missing `dino_latent`: {cache_path}")
            if not torch.is_tensor(latent):
                raise TypeError(f"Cached DINO frame latent must be a tensor, got {type(latent)} in {cache_path}")
            if latent.ndim != 3:
                raise ValueError(
                    f"Cached DINO frame latent must be [D,H,W], got shape {tuple(latent.shape)} in {cache_path}"
                )
            latents.append(latent.contiguous())

        if missing_paths:
            if required:
                preview = ", ".join(missing_paths[:3])
                raise FileNotFoundError(
                    f"Missing {len(missing_paths)} DINO frame cache files for dataset idx={idx_for_error}: {preview}. "
                    "Run scripts/precompute_dino_latents.py with dino_latent_cache_mode=frame first."
                )
            return None
        if not latents:
            return None
        return torch.stack(latents, dim=1).contiguous()  # [D, T, H, W]

    def _open_dino_frame_mmap(self):
        if self._dino_frame_mmap is not None:
            return
        if self.dino_latent_cache_dir is None or str(self.dino_latent_cache_dir).strip() == "":
            return

        metadata_path = os.path.join(str(self.dino_latent_cache_dir), "metadata.json")
        if not os.path.exists(metadata_path):
            if self.dino_latent_cache_required:
                raise FileNotFoundError(f"Missing DINO frame mmap metadata: {metadata_path}")
            return

        with open(metadata_path, "r", encoding="utf-8") as f:
            metadata = json.load(f)
        if str(metadata.get("cache_mode", "")).strip().lower() != "frame_mmap":
            raise ValueError(
                f"DINO mmap cache metadata must have cache_mode='frame_mmap', got "
                f"{metadata.get('cache_mode')!r} in {metadata_path}"
            )

        mmap_file = metadata.get("mmap_file", "frames.bin")
        mmap_path = os.path.join(str(self.dino_latent_cache_dir), str(mmap_file))
        if not os.path.exists(mmap_path):
            if self.dino_latent_cache_required:
                raise FileNotFoundError(f"Missing DINO frame mmap payload: {mmap_path}")
            return

        total_frames = int(metadata["total_frames"])
        latent_shape = tuple(int(v) for v in metadata["latent_shape"])
        save_dtype = str(metadata.get("save_dtype", "bf16")).strip().lower()
        storage_dtype = str(metadata.get("storage_dtype", "")).strip().lower()
        if save_dtype in {"bf16", "bfloat16"}:
            np_dtype = np.uint16
            torch_dtype = torch.bfloat16
            if storage_dtype and storage_dtype != "uint16":
                raise ValueError(f"bf16 mmap cache must use uint16 storage, got {storage_dtype}")
        elif save_dtype in {"fp16", "float16"}:
            np_dtype = np.float16
            torch_dtype = torch.float16
        elif save_dtype in {"fp32", "float32"}:
            np_dtype = np.float32
            torch_dtype = torch.float32
        else:
            raise ValueError(f"Unsupported DINO mmap save_dtype={save_dtype!r} in {metadata_path}")

        self._dino_frame_mmap_path = mmap_path
        self._dino_frame_mmap_np_dtype = np_dtype
        self._dino_frame_mmap_torch_dtype = torch_dtype
        self._dino_frame_mmap_shape = (total_frames, *latent_shape)
        self._dino_frame_mmap = np.memmap(
            mmap_path,
            mode="r",
            dtype=np_dtype,
            shape=self._dino_frame_mmap_shape,
        )

    def _load_cached_dino_frame_mmap_latents(self, idx: int) -> Optional[torch.Tensor]:
        frame_indices = self._get_video_global_indices(idx)
        return self._load_cached_dino_frame_mmap_latents_for_indices(frame_indices, idx_for_error=idx)

    def _load_cached_dino_frame_mmap_latents_for_indices(
        self,
        frame_indices: list[int],
        idx_for_error: int,
    ) -> Optional[torch.Tensor]:
        self._open_dino_frame_mmap()
        if self._dino_frame_mmap is None:
            return None

        total_frames = int(self._dino_frame_mmap_shape[0])
        invalid = [frame_idx for frame_idx in frame_indices if frame_idx < 0 or frame_idx >= total_frames]
        if invalid:
            raise IndexError(
                f"DINO frame mmap index out of bounds for dataset idx={idx_for_error}: "
                f"indices={invalid[:3]}, total_frames={total_frames}"
            )

        latent_np = np.asarray(self._dino_frame_mmap[frame_indices])
        if self._dino_frame_mmap_torch_dtype is torch.bfloat16:
            latent = torch.from_numpy(np.array(latent_np, copy=True)).view(torch.bfloat16)
        else:
            latent = torch.from_numpy(np.array(latent_np, copy=True)).to(dtype=self._dino_frame_mmap_torch_dtype)
        if latent.ndim != 4:
            raise ValueError(
                f"DINO frame mmap slice must be [T,D,H,W], got {tuple(latent.shape)} "
                f"from {self._dino_frame_mmap_path}"
            )
        return latent.permute(1, 0, 2, 3).contiguous()  # [D, T, H, W]

    def _load_cached_vae_latents(self, idx: int) -> Optional[torch.Tensor]:
        if self.vae_latent_cache_mode == "window_mmap":
            return self._load_cached_vae_window_mmap_latents(idx)

        cache_path = self._get_vae_latent_cache_path(idx)
        if cache_path is None:
            return None
        if not os.path.exists(cache_path):
            if self.vae_latent_cache_required:
                raise FileNotFoundError(
                    f"Missing VAE latent cache for dataset idx={idx}: {cache_path}. "
                    "Run scripts/precompute_vae_latents.py first, or disable "
                    "`vae_latent_cache_required`."
                )
            return None

        payload = torch.load(cache_path, map_location="cpu")
        latents = payload.get("vae_latents", payload.get("latents")) if isinstance(payload, dict) else payload
        if latents is None:
            raise KeyError(f"VAE latent cache missing `vae_latents`: {cache_path}")
        if not torch.is_tensor(latents):
            raise TypeError(f"Cached VAE latents must be a tensor, got {type(latents)} in {cache_path}")
        if latents.ndim != 4:
            raise ValueError(
                f"Cached VAE latents must be [C,T,H,W] for one sample, "
                f"got shape {tuple(latents.shape)} in {cache_path}"
            )
        return latents.contiguous()

    def _open_vae_window_mmap(self):
        if self._vae_window_mmap is not None:
            return
        if self.vae_latent_cache_dir is None or str(self.vae_latent_cache_dir).strip() == "":
            return

        metadata_path = os.path.join(str(self.vae_latent_cache_dir), "metadata.json")
        if not os.path.exists(metadata_path):
            if self.vae_latent_cache_required:
                raise FileNotFoundError(f"Missing VAE window mmap metadata: {metadata_path}")
            return

        with open(metadata_path, "r", encoding="utf-8") as f:
            metadata = json.load(f)
        if str(metadata.get("cache_mode", "")).strip().lower() != "window_mmap":
            raise ValueError(
                f"VAE mmap cache metadata must have cache_mode='window_mmap', got "
                f"{metadata.get('cache_mode')!r} in {metadata_path}"
            )
        complete_marker = os.path.join(str(self.vae_latent_cache_dir), ".complete")
        if not bool(metadata.get("complete", False)) or not os.path.exists(complete_marker):
            if self.vae_latent_cache_required:
                raise RuntimeError(
                    f"VAE mmap cache is incomplete: metadata_complete={metadata.get('complete')!r}, "
                    f"complete_marker_exists={os.path.exists(complete_marker)} in {self.vae_latent_cache_dir}"
                )
            return

        expected = {
            "num_frames": int(self.num_frames),
            "action_video_freq_ratio": int(self.action_video_freq_ratio),
            "video_sample_count": len(self.video_sample_indices),
            "concat_multi_camera": self.concat_multi_camera,
        }
        if "data_video_size" in metadata:
            expected["data_video_size"] = [int(self.video_size[0]), int(self.video_size[1])]
        for key, expected_value in expected.items():
            if key not in metadata:
                continue
            actual_value = metadata[key]
            if actual_value != expected_value:
                raise ValueError(
                    f"VAE mmap metadata mismatch for {key}: cache={actual_value!r}, "
                    f"dataset={expected_value!r} in {metadata_path}"
                )

        mmap_file = metadata.get("mmap_file", "latents.bf16.bin")
        mmap_path = os.path.join(str(self.vae_latent_cache_dir), str(mmap_file))
        if not os.path.exists(mmap_path):
            if self.vae_latent_cache_required:
                raise FileNotFoundError(f"Missing VAE window mmap payload: {mmap_path}")
            return

        total_samples = int(metadata["total_samples"])
        latent_shape = tuple(int(v) for v in metadata["latent_shape"])
        if total_samples < len(self.lerobot_dataset):
            raise ValueError(
                f"VAE mmap cache has fewer samples than dataset: cache={total_samples}, "
                f"dataset={len(self.lerobot_dataset)} in {metadata_path}"
            )
        np_dtype, torch_dtype = self._mmap_dtype_from_metadata(metadata, metadata_path, "VAE")
        expected_bytes = int(total_samples * np.prod(latent_shape) * np.dtype(np_dtype).itemsize)
        actual_bytes = os.path.getsize(mmap_path)
        if actual_bytes != expected_bytes:
            raise ValueError(
                f"VAE mmap payload size mismatch: file={actual_bytes} bytes, "
                f"expected={expected_bytes} bytes from shape={(total_samples, *latent_shape)} "
                f"and dtype={np_dtype} in {mmap_path}"
            )

        self._vae_window_mmap_path = mmap_path
        self._vae_window_mmap_np_dtype = np_dtype
        self._vae_window_mmap_torch_dtype = torch_dtype
        self._vae_window_mmap_shape = (total_samples, *latent_shape)
        self._vae_window_mmap = np.memmap(
            mmap_path,
            mode="r",
            dtype=np_dtype,
            shape=self._vae_window_mmap_shape,
        )

    def _load_cached_vae_window_mmap_latents(self, idx: int) -> Optional[torch.Tensor]:
        self._open_vae_window_mmap()
        if self._vae_window_mmap is None:
            return None

        total_samples = int(self._vae_window_mmap_shape[0])
        if idx < 0 or idx >= total_samples:
            raise IndexError(
                f"VAE window mmap index out of bounds for dataset idx={idx}: total_samples={total_samples}"
            )

        latent_np = np.asarray(self._vae_window_mmap[int(idx)])
        if self._vae_window_mmap_torch_dtype is torch.bfloat16:
            latent = torch.from_numpy(np.array(latent_np, copy=True)).view(torch.bfloat16)
        else:
            latent = torch.from_numpy(np.array(latent_np, copy=True)).to(dtype=self._vae_window_mmap_torch_dtype)
        if latent.ndim != 4:
            raise ValueError(
                f"VAE window mmap slice must be [C,T,H,W], got {tuple(latent.shape)} "
                f"from {self._vae_window_mmap_path}"
            )
        return latent.contiguous()

    def _load_cached_history_dino_latents(self, idx: int) -> Optional[torch.Tensor]:
        if not self.load_history_dino_latents:
            return None
        frame_indices = self._get_history_dino_global_indices(idx)
        if self.dino_latent_cache_mode == "frame":
            history = self._load_cached_dino_frame_latents_for_indices(
                frame_indices,
                idx_for_error=idx,
                required=self.history_dino_latent_cache_required,
            )
        elif self.dino_latent_cache_mode == "frame_mmap":
            history = self._load_cached_dino_frame_mmap_latents_for_indices(
                frame_indices,
                idx_for_error=idx,
            )
        else:
            raise ValueError(
                "`load_history_dino_latents=true` only supports frame/frame_mmap DINO caches."
            )
        if history is None and self.history_dino_latent_cache_required:
            raise FileNotFoundError(
                f"Missing history DINO latents for dataset idx={idx}, "
                f"frame_indices={frame_indices}, cache_dir={self.dino_latent_cache_dir}"
            )
        return history

    def _format_pixel_values_video(self, video: torch.Tensor, frame_indices: list[int]) -> torch.Tensor:
        """Format processor pixel_values into [C,T,H,W] in the same layout as main video."""
        num_cameras = 1
        if video.ndim == 5:
            video = video[:, frame_indices, :, :, :]  # [num_cameras, T, C, H, W]
            num_cameras, t_video, c, h, w = video.shape
        else:
            assert video.ndim == 4, f"Expected pixel_values [T,C,H,W] or [N,T,C,H,W], got {video.shape}"
            video = video[frame_indices, :, :, :]
            t_video, c, h, w = video.shape
        video = video.view(num_cameras, t_video, c, h, w)

        if self.concat_multi_camera == "robotwin":
            if num_cameras != 3:
                raise ValueError(
                    f"`concat_multi_camera='robotwin'` requires exactly 3 cameras, got {num_cameras}"
                )
            cam_top = transforms_F.resize(
                video[0],
                size=[256, 320],
                interpolation=transforms_F.InterpolationMode.BILINEAR,
                antialias=True,
            )
            cam_left = transforms_F.resize(
                video[1],
                size=[128, 160],
                interpolation=transforms_F.InterpolationMode.BILINEAR,
                antialias=True,
            )
            cam_right = transforms_F.resize(
                video[2],
                size=[128, 160],
                interpolation=transforms_F.InterpolationMode.BILINEAR,
                antialias=True,
            )
            bottom = torch.cat([cam_left, cam_right], dim=-1)
            video = torch.cat([cam_top, bottom], dim=-2)
        elif num_cameras > 1:
            if self.concat_multi_camera == "horizontal":
                video = torch.cat([video[i] for i in range(num_cameras)], dim=-1)
            elif self.concat_multi_camera == "vertical":
                video = torch.cat([video[i] for i in range(num_cameras)], dim=-2)
            else:
                raise ValueError(
                    f"Invalid concat_multi_camera: {self.concat_multi_camera}. "
                    "Expected one of: horizontal, vertical, robotwin."
                )
        else:
            video = video.squeeze(0)

        video = self.resize_transform(video)
        video = self.crop_transform(video)
        video = self.normalize_transform(video)
        return video.permute(1, 0, 2, 3).contiguous()

    def _load_history_video_frames(self, idx: int, frame_indices: list[int], history_name: str) -> torch.Tensor:
        previous_return_images = bool(self.lerobot_dataset.return_images)
        frames = []
        try:
            if not previous_return_images:
                self.lerobot_dataset._set_return_images(True)
            for frame_idx in frame_indices:
                history_sample = self.lerobot_dataset[int(frame_idx)]
                resolved_idx = int(history_sample.get("idx", frame_idx))
                if resolved_idx != int(frame_idx):
                    raise RuntimeError(
                        f"{history_name} loading requires deterministic dataset indexing, "
                        f"but requested idx={frame_idx} resolved to idx={resolved_idx}."
                    )
                if "pixel_values" not in history_sample:
                    raise KeyError(
                        f"{history_name} loading requires `pixel_values`; "
                        "failed to temporarily enable image loading."
                    )
                frame_video = self._format_pixel_values_video(history_sample["pixel_values"], [0])
                if frame_video.shape[1] != 1:
                    raise ValueError(
                        f"Expected one formatted history frame, got T={frame_video.shape[1]}"
                    )
                frames.append(frame_video[:, 0])
        finally:
            if not previous_return_images:
                self.lerobot_dataset._set_return_images(False)
        if len(frames) != len(frame_indices):
            raise RuntimeError(
                f"Loaded {len(frames)} {history_name} frames, expected {len(frame_indices)} for idx={idx}."
            )
        return torch.stack(frames, dim=1).contiguous()  # [C, T_history, H, W]

    def _load_history_dino_video(self, idx: int) -> Optional[torch.Tensor]:
        if not self.load_history_dino_video:
            return None
        frame_indices = self._get_history_dino_global_indices(idx)
        return self._load_history_video_frames(idx, frame_indices, "History DINO video")

    def _load_history_vae_video(self, idx: int) -> Optional[torch.Tensor]:
        if not self.load_history_vae_video:
            return None
        frame_indices = self._get_history_vae_global_indices(idx)
        return self._load_history_video_frames(idx, frame_indices, "History VAE video")

    def _get(self, idx):
        sample_idx = idx
        sample = None
        for attempt in range(self.max_padding_retry + 1):
            sample = self.lerobot_dataset[sample_idx]

            if not self.skip_padding_as_possible:
                break

            action_is_pad = sample["action_is_pad"]
            image_is_pad = sample["image_is_pad"]
            proprio_is_pad = sample["proprio_is_pad"]
            has_pad = False
            if bool(action_is_pad.any().item()):
                has_pad = True
            if bool(image_is_pad.any().item()):
                has_pad = True
            if bool(proprio_is_pad.any().item()):
                has_pad = True

            if not has_pad or attempt >= self.max_padding_retry:
                break

            sample_idx = np.random.randint(len(self.lerobot_dataset))

        resolved_sample_idx = int(sample.get("idx", sample_idx))
        
        image_is_pad = sample["image_is_pad"][self.video_sample_indices]
        video = None
        video_frame_count = len(self.video_sample_indices)

        if not self.skip_video_load_if_latent_cached:
            video = sample["pixel_values"]  # [T, C, H, W] or [num_cameras, T, C, H, W]
            num_cameras = 1
            if video.ndim == 5:
                video = video[:, self.video_sample_indices, :, :, :] # [num_cameras, T_video, C, H, W]
                num_cameras, T_video, C, H, W = video.shape
            else:
                assert video.ndim == 4, f"Expected video to have shape [T, C, H, W], but got {video.shape}"
                video = video[self.video_sample_indices, :, :, :] # [T_video, C, H, W]
                T_video, C, H, W = video.shape
            video_frame_count = int(T_video)

            video = video.view(num_cameras, T_video, C, H, W)  # [num_cameras, T_video, C, H, W]
            if self.concat_multi_camera == "robotwin":
                if num_cameras != 3:
                    raise ValueError(
                        f"`concat_multi_camera='robotwin'` requires exactly 3 cameras, got {num_cameras}"
                    )
                cam_top = transforms_F.resize(
                    video[0],
                    size=[256, 320],
                    interpolation=transforms_F.InterpolationMode.BILINEAR,
                    antialias=True,
                )  # [T_video, C, 256, 320]
                cam_left = transforms_F.resize(
                    video[1],
                    size=[128, 160],
                    interpolation=transforms_F.InterpolationMode.BILINEAR,
                    antialias=True,
                )  # [T_video, C, 128, 160]
                cam_right = transforms_F.resize(
                    video[2],
                    size=[128, 160],
                    interpolation=transforms_F.InterpolationMode.BILINEAR,
                    antialias=True,
                )  # [T_video, C, 128, 160]
                bottom = torch.cat([cam_left, cam_right], dim=-1)  # [T_video, C, 128, 320]
                video = torch.cat([cam_top, bottom], dim=-2)  # [T_video, C, 384, 320]
            elif num_cameras > 1:
                if self.concat_multi_camera == "horizontal":
                    video = torch.cat([video[i] for i in range(num_cameras)], dim=-1)  # [T_video, C, H, num_cameras*W]
                elif self.concat_multi_camera == "vertical":
                    video = torch.cat([video[i] for i in range(num_cameras)], dim=-2)  # [T_video, C, num_cameras*H, W]
                else:
                    raise ValueError(
                        f"Invalid concat_multi_camera: {self.concat_multi_camera}. "
                        "Expected one of: horizontal, vertical, robotwin."
                    )
            else:
                video = video.squeeze(0)  # [T_video, C, H, W]

            # final resize and normalization
            video = self.resize_transform(video)
            video = self.crop_transform(video)
            video = self.normalize_transform(video)  # [T_video, C, H, W]

            video = video.permute(1, 0, 2, 3) # [C, T_video, H, W], range [-1, 1]

        # Proxy (from lerobot): 
        #   action: [num_frames-1, action_dim] # start from t0, except the last frame
        #   proprio: [num_frames, proprio_dim] # start from t0 to the last frame, aligned with video frames
        action = sample["action"] # [T-1, action_dim]
        proprio = sample["proprio"][:-1, :] # [T-1, state_dim]， to align with action
        if video_frame_count <= 1:
            raise ValueError(f"`video` must have at least 2 frames, got T={video_frame_count}")
        if action.shape[0] % (video_frame_count - 1) != 0:
            raise ValueError(
                f"`action` horizon must be divisible by `video` transitions, got "
                f"{action.shape[0]} and {video_frame_count - 1}"
            )

        task = sample["instruction"]
        
        # FIXME
        if self.override_instruction is not None:
            task = self.override_instruction
        instruction = DEFAULT_PROMPT.format(task=task)

        if self.load_text_context:
            context, context_mask = self._get_cached_text_context(instruction)
            # NOTE: to keep consistent with wan2.2's behavior
            context[~context_mask] = 0.0
            context_mask = torch.ones_like(context_mask)
        
        data = {
            "dataset_idx": resolved_sample_idx,
            "video_frame_count": video_frame_count,
            "action": action,
            "proprio": proprio,
            "prompt": instruction,
            "image_is_pad": image_is_pad,
            "action_is_pad": sample["action_is_pad"],
            "proprio_is_pad": sample["proprio_is_pad"],
        }
        if video is not None:
            data["video"] = video
        if self.load_semantic_image:
            if video is not None:
                data["semantic_image"] = video[:, 0].contiguous()
            else:
                semantic_idx = self._get_global_indices_for_offsets(
                    resolved_sample_idx,
                    [self.semantic_frame_offset],
                )
                semantic_video = self._load_history_video_frames(
                    resolved_sample_idx,
                    semantic_idx,
                    "Semantic current image",
                )
                if semantic_video.shape[1] != 1:
                    raise ValueError(
                        f"Expected one semantic image frame, got T={semantic_video.shape[1]}"
                    )
                data["semantic_image"] = semantic_video[:, 0].contiguous()
        if self.load_text_context:
            data["context"] = context
            data["context_mask"] = context_mask

        vae_latents = self._load_cached_vae_latents(resolved_sample_idx)
        if vae_latents is not None:
            data["vae_latents"] = vae_latents

        dino_latents = self._load_cached_dino_latents(resolved_sample_idx)
        if dino_latents is not None:
            if dino_latents.shape[1] != video_frame_count:
                raise ValueError(
                    f"Cached DINO latent temporal length mismatch for idx={resolved_sample_idx}: "
                    f"cache T={dino_latents.shape[1]}, video T={video_frame_count}"
                )
            data["dino_latents"] = dino_latents
        history_dino_latents = self._load_cached_history_dino_latents(resolved_sample_idx)
        if history_dino_latents is not None:
            expected_t = len(self.history_dino_frame_offsets)
            if history_dino_latents.shape[1] != expected_t:
                raise ValueError(
                    f"History DINO latent temporal length mismatch for idx={resolved_sample_idx}: "
                    f"cache T={history_dino_latents.shape[1]}, expected T={expected_t}"
                )
            data["history_dino_latents"] = history_dino_latents

        history_video = self._load_history_dino_video(resolved_sample_idx)
        history_offsets = self.history_dino_frame_offsets
        history_name = "History DINO video"
        if history_video is None:
            history_video = self._load_history_vae_video(resolved_sample_idx)
            history_offsets = self.history_vae_frame_offsets
            history_name = "History VAE video"
        if history_video is not None:
            expected_t = len(history_offsets)
            if history_video.shape[1] != expected_t:
                raise ValueError(
                    f"{history_name} temporal length mismatch for idx={resolved_sample_idx}: "
                    f"video T={history_video.shape[1]}, expected T={expected_t}"
                )
            data["history_video"] = history_video
        return data

    def _get_cached_text_context(self, prompt: str):
        if self.text_embedding_cache_dir is None:
            raise ValueError("text_embedding_cache_dir is not set.")
        cache_dir = self.text_embedding_cache_dir
        os.makedirs(cache_dir, exist_ok=True)
        hashed = hashlib.sha256(prompt.encode("utf-8")).hexdigest()
        cache_path = os.path.join(cache_dir, f"{hashed}.t5_len{self.context_len}.wan22ti2v5b.pt")
        if not os.path.exists(cache_path):
            raise FileNotFoundError(
                f"Missing text embedding cache: {cache_path}. "
                "Run scripts/precompute_text_embeds.py first."
            )
        payload = torch.load(cache_path, map_location="cpu")
        context = payload["context"]
        context_mask = payload["mask"].bool()
        if context.ndim != 2:
            raise ValueError(
                f"Cached `context` must be 2D [L, D], got shape {tuple(context.shape)} in {cache_path}"
            )
        if context_mask.ndim != 1:
            raise ValueError(
                f"Cached `mask` must be 1D [L], got shape {tuple(context_mask.shape)} in {cache_path}"
            )
        if context.shape[0] != self.context_len:
            raise ValueError(
                f"Cached context_len mismatch: expected {self.context_len}, got {context.shape[0]} in {cache_path}"
            )
        if context_mask.shape[0] != self.context_len:
            raise ValueError(
                f"Cached mask_len mismatch: expected {self.context_len}, got {context_mask.shape[0]} in {cache_path}"
            )

        return context, context_mask

    def __getitem__(self, idx):
        try:
            data = self._get(idx)
        except Exception as e:
            if self.dino_latent_cache_required or self.vae_latent_cache_required:
                raise
            print(f"Error processing sample idx {idx}: {e}. Returning a random sample instead.")
            # trace back
            print(traceback.format_exc())
            random_idx = np.random.randint(len(self))
            data = self._get(random_idx)
        return data
