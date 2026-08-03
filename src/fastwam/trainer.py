import logging
import json
import inspect
import os
import re
from math import ceil
from pathlib import Path
import time

import numpy as np
import torch
from accelerate import Accelerator
from omegaconf import DictConfig
from PIL import Image
from torch.optim.lr_scheduler import ConstantLR, CosineAnnealingLR, LinearLR, SequentialLR
from torch.utils.data import DataLoader

from .utils.fs import ensure_dir
from .utils.logging_config import get_logger, setup_logging
from .utils.pytorch_utils import set_global_seed
from .utils.samplers import ResumableEpochSampler
from .utils.video_io import save_mp4
from .utils.video_metrics import pil_frames_to_video_tensor, video_psnr, video_ssim

logger = get_logger(__name__)


class Wan22Trainer:
    def __init__(self, model, train_dataset, val_dataset=None, *, cfg: DictConfig):
        self.model = model
        self.train_dataset = train_dataset
        self.val_dataset = val_dataset
        self.cfg = cfg
        self.output_dir = str(cfg.output_dir)
        self.learning_rate = float(cfg.learning_rate)
        self.weight_decay = float(cfg.weight_decay)
        self.batch_size = int(cfg.batch_size)
        self.num_workers = int(cfg.num_workers)
        self.prefetch_factor = int(getattr(cfg, "prefetch_factor", 2))
        self.persistent_workers = bool(getattr(cfg, "persistent_workers", False))
        self.pin_memory = bool(getattr(cfg, "pin_memory", torch.cuda.is_available()))
        self.num_epochs = int(cfg.num_epochs)
        max_steps = cfg.max_steps
        self.max_steps = int(max_steps) if max_steps is not None else None
        self.log_every = int(cfg.log_every)
        self.save_every = int(cfg.save_every)
        self.eval_every = int(cfg.eval_every)
        self.eval_num_inference_steps = int(cfg.eval_num_inference_steps)
        self.gradient_accumulation_steps = int(cfg.gradient_accumulation_steps)
        self.max_grad_norm = float(cfg.max_grad_norm)
        self.safe_checkpoint_serialization = bool(getattr(cfg, "safe_checkpoint_serialization", False))
        self.save_accelerate_state = bool(getattr(cfg, "save_accelerate_state", True))
        self.seed = int(cfg.seed)
        
        self.resume = cfg.resume
        self.mixed_precision = str(cfg.mixed_precision).strip().lower()
        if self.mixed_precision not in {"no", "fp16", "bf16"}:
            raise ValueError(
                f"Unsupported mixed_precision: {cfg.mixed_precision}. "
                "Expected one of: ['no', 'fp16', 'bf16']."
            )
        self.wandb_enabled = bool(cfg.wandb.enabled)

        self.accelerator = Accelerator(
            gradient_accumulation_steps=self.gradient_accumulation_steps,
            mixed_precision=self.mixed_precision,
            step_scheduler_with_optimizer=False,
        )
        
        logger.info(
            "Accelerate training: distributed_type=%s zero_stage=%s world_size=%d process_index=%d cfg_mixed_precision=%s accelerator_mixed_precision=%s grad_accum=%d grad_clip=%.4f",
            self.accelerator.distributed_type,
            self.accelerator.state.deepspeed_plugin.deepspeed_config.get("zero_optimization", {}).get("stage", "unknown"),
            self.accelerator.num_processes,
            self.accelerator.process_index,
            self.mixed_precision,
            self.accelerator.mixed_precision,
            self.gradient_accumulation_steps,
            self.max_grad_norm,
        )
        logger.info("using accelerator.device=%s", self.accelerator.device)
        worker_init_fn = set_global_seed(self.seed, get_worker_init_fn=True)
        self._assert_dataset_length_consistent(self.train_dataset, "train_dataset")
        if self.val_dataset is not None:
            self._assert_dataset_length_consistent(self.val_dataset, "val_dataset")
        self._assert_history_intent_offsets_consistent()
        self._assert_semantic_history_offsets_consistent()

        self._weight_checkpoint_loaded_pre_prepare = False
        self._load_weight_checkpoint_before_prepare()

        # Freeze non-trainable modules before optimizer/deepspeed initialization.
        # This keeps DiT (+ optional small context heads) as trainable when ZeRO builds optimizer state.
        self._apply_dit_only_train_mode(self.model)
        trainable_params = list(self.model.dit.parameters())
        proprio_encoder = getattr(self.model, "proprio_encoder", None)
        if proprio_encoder is not None:
            trainable_params.extend(list(proprio_encoder.parameters()))
        intent_encoder = getattr(self.model, "intent_encoder", None)
        if intent_encoder is not None:
            trainable_params.extend(list(intent_encoder.parameters()))
        semantic_history_encoder = getattr(self.model, "semantic_history_encoder", None)
        if semantic_history_encoder is not None:
            trainable_params.extend(list(semantic_history_encoder.trainable_parameters()))
        self.optimizer = torch.optim.AdamW(
            trainable_params,
            lr=self.learning_rate,
            weight_decay=self.weight_decay,
            betas=(0.9, 0.95),
        )
        
        self.train_loader = self._build_loader(self.train_dataset, worker_init_fn=worker_init_fn)
        total_train_steps = self._estimate_total_train_steps()
        self.max_steps = total_train_steps
        warmup_steps = int(total_train_steps * 0.05)
        self.scheduler = self._build_scheduler(
            scheduler_type=cfg.lr_scheduler_type,
            total_train_steps=total_train_steps,
            warmup_steps=warmup_steps,
        )
        self.global_step = 0
        self.epoch = 0
        self.batch_in_epoch = 0

        self.checkpoint_root = os.path.join(self.output_dir, "checkpoints")
        self.weights_dir = os.path.join(self.checkpoint_root, "weights")
        self.state_dir = os.path.join(self.checkpoint_root, "state")
        self.eval_dir = os.path.join(self.output_dir, "eval")

        ensure_dir(self.output_dir)
        ensure_dir(self.checkpoint_root)
        ensure_dir(self.weights_dir)
        ensure_dir(self.state_dir)
        ensure_dir(self.eval_dir)

        self.model, self.optimizer, self.train_loader, self.scheduler = self.accelerator.prepare(
            self.model, self.optimizer, self.train_loader, self.scheduler
        )
        self.optimizer.zero_grad(set_to_none=True)
        self.wandb_run = None
        self._init_wandb()
        self._resume_or_load_checkpoint()

        val_size = len(self.val_dataset) if self.val_dataset is not None else len(self.train_dataset)
        logger.info("Train/val dataset size: %d/%d", len(self.train_dataset), val_size)

    def _init_wandb(self):
        if not self.wandb_enabled or not self.accelerator.is_main_process:
            return
        try:
            import wandb
        except ImportError as e:
            raise ImportError(
                "wandb logging is enabled in config (`wandb.enabled=true`) but wandb is not installed."
            ) from e

        self.wandb_run = wandb.init(
            entity=self.cfg.wandb.workspace,
            project=self.cfg.wandb.project,
            name=self.cfg.wandb.name,
            group=None if self.cfg.wandb.group in (None, "null", "") else str(self.cfg.wandb.group),
            mode=self.cfg.wandb.mode,
            dir=self.output_dir,
        )
        logger.info(
            "Initialized wandb run: workspace=%s project=%s name=%s",
            self.cfg.wandb.workspace,
            self.cfg.wandb.project,
            self.cfg.wandb.name,
        )

    def _wandb_log(self, payload: dict):
        if self.wandb_run is None:
            return
        self.wandb_run.log(payload, step=self.global_step)

    def _finish_wandb(self):
        if self.wandb_run is None:
            return
        self.wandb_run.finish()
        self.wandb_run = None

    def _build_loader(self, dataset, worker_init_fn=None):
        self.train_sampler = ResumableEpochSampler(
            dataset=dataset,
            seed=self.seed,
            batch_size=self.batch_size,
            num_processes=self.accelerator.num_processes,
        )
        loader_kwargs = {
            "batch_size": self.batch_size,
            "shuffle": False,
            "sampler": self.train_sampler,
            "num_workers": self.num_workers,
            "pin_memory": self.pin_memory,
            "worker_init_fn": worker_init_fn,
        }
        if self.num_workers > 0:
            loader_kwargs["prefetch_factor"] = self.prefetch_factor
            loader_kwargs["persistent_workers"] = self.persistent_workers
        return DataLoader(
            dataset,
            **loader_kwargs,
        )

    def _assert_dataset_length_consistent(self, dataset, dataset_name: str):
        if not hasattr(dataset, "__len__"):
            raise TypeError(f"`{dataset_name}` must implement __len__ for rank consistency checks.")

        local_length = len(dataset)
        gathered_lengths = self.accelerator.gather(
            torch.tensor([local_length], device=self.accelerator.device, dtype=torch.int64)
        ).reshape(-1)
        if torch.all(gathered_lengths == gathered_lengths[0]):
            return

        if self.accelerator.is_main_process:
            print(f"[dataset-check] {dataset_name} length mismatch across ranks after initialization:")
            for rank, rank_length in enumerate(gathered_lengths.cpu().tolist()):
                print(f"rank {rank}: {rank_length}")
        self.accelerator.wait_for_everyone()
        raise RuntimeError(
            f"{dataset_name} length mismatch across ranks: {gathered_lengths.cpu().tolist()}"
        )

    def _assert_history_intent_offsets_consistent(self):
        intent_encoder = getattr(self.model, "intent_encoder", None)
        if intent_encoder is None:
            return

        intent_config = getattr(self.model, "intent_config", None) or {}
        model_offsets = intent_config.get(
            "history_offsets",
            intent_config.get("history_dino_frame_offsets", None),
        )
        if model_offsets is None:
            raise ValueError(
                "`model.intent_config.history_offsets` must be set when "
                "`model.intent_config.enabled=true`."
            )
        model_offsets = [int(offset) for offset in model_offsets]
        max_history_frames = int(intent_config.get("max_history_frames", len(model_offsets)))
        if max_history_frames < len(model_offsets):
            raise ValueError(
                "`model.intent_config.max_history_frames` must be >= len(history_offsets), "
                f"got {max_history_frames} and {len(model_offsets)}."
            )

        intent_source = str(intent_config.get("source", "dino")).strip().lower()
        if intent_source not in {"dino", "vae"}:
            raise ValueError(
                f"Unsupported `model.intent_config.source={intent_source!r}`; "
                "expected one of: dino, vae."
            )

        datasets = [("train_dataset", self.train_dataset)]
        if self.val_dataset is not None:
            datasets.append(("val_dataset", self.val_dataset))
        for dataset_name, dataset in datasets:
            if intent_source == "vae":
                load_history = bool(getattr(dataset, "load_history_vae_video", False))
                if not load_history:
                    raise ValueError(
                        f"{dataset_name} must set `load_history_vae_video=true` when "
                        "`model.intent_config.enabled=true` and `model.intent_config.source='vae'`."
                    )
                dataset_offsets = getattr(dataset, "history_vae_frame_offsets", None)
                offset_attr = "history_vae_frame_offsets"
                intent_name = "Short-VAE-Intent"
            else:
                load_history_latents = bool(getattr(dataset, "load_history_dino_latents", False))
                load_history_video = bool(getattr(dataset, "load_history_dino_video", False))
                legacy_load_history_video = bool(getattr(dataset, "load_history_vae_video", False))
                history_source_count = sum(
                    int(flag)
                    for flag in (load_history_latents, load_history_video, legacy_load_history_video)
                )
                if history_source_count != 1:
                    raise ValueError(
                        f"{dataset_name} should provide exactly one DINO intent history source: "
                        "`load_history_dino_latents=true` for cached latents, or "
                        "`load_history_dino_video=true` for online DINO encoding."
                    )
                if load_history_latents:
                    dataset_offsets = getattr(dataset, "history_dino_frame_offsets", None)
                    offset_attr = "history_dino_frame_offsets"
                elif load_history_video:
                    dataset_offsets = getattr(dataset, "history_dino_frame_offsets", None)
                    offset_attr = "history_dino_frame_offsets"
                elif legacy_load_history_video:
                    dataset_offsets = getattr(dataset, "history_vae_frame_offsets", None)
                    offset_attr = "history_vae_frame_offsets"
                else:
                    raise ValueError(
                        f"{dataset_name} must set `load_history_dino_latents=true` for cached "
                        "history DINO latents, or `load_history_dino_video=true` for online "
                        "history DINO encoding, when `model.intent_config.enabled=true` and "
                        "`model.intent_config.source='dino'`."
                    )
                intent_name = "Short-DINO-Intent"

            if dataset_offsets is None:
                raise ValueError(
                    f"{dataset_name} has no `{offset_attr}`; cannot verify "
                    f"{intent_name} train/infer consistency."
                )
            dataset_offsets = [int(offset) for offset in dataset_offsets]
            if dataset_offsets != model_offsets:
                raise ValueError(
                    f"{intent_name} history offset mismatch between model and dataset: "
                    f"model.intent_config.history_offsets={model_offsets}, "
                    f"{dataset_name}.{offset_attr}={dataset_offsets}."
                )

        if self.accelerator.is_main_process:
            logger.info(
                "Short-%s-Intent history offsets verified: offsets=%s max_history_frames=%d",
                intent_source.upper(),
                model_offsets,
                max_history_frames,
            )

    def _assert_semantic_history_offsets_consistent(self):
        semantic_history_encoder = getattr(self.model, "semantic_history_encoder", None)
        if semantic_history_encoder is None:
            return

        semantic_config = getattr(self.model, "semantic_history_config", None) or {}
        use_history = bool(semantic_config.get("use_history", True))
        history_source = str(semantic_config.get("source", "dino")).strip().lower()
        if history_source not in {"dino", "vae"}:
            raise ValueError(
                f"Unsupported model.semantic_history_config.source={history_source!r}; "
                "expected one of: dino, vae."
            )
        datasets = [("train_dataset", self.train_dataset)]
        if self.val_dataset is not None:
            datasets.append(("val_dataset", self.val_dataset))
        for dataset_name, dataset in datasets:
            if not bool(getattr(dataset, "load_semantic_image", False)):
                raise ValueError(
                    f"{dataset_name} must set `load_semantic_image=true` when "
                    "`model.semantic_history_config.enabled=true`."
                )

        if not use_history:
            if self.accelerator.is_main_process:
                logger.info("Qwen current-semantic adapter verified without history memory.")
            return

        model_offsets = semantic_config.get("history_offsets", None)
        if model_offsets is None:
            raise ValueError(
                "`model.semantic_history_config.history_offsets` must be set when "
                "`model.semantic_history_config.enabled=true`."
            )
        model_offsets = [int(offset) for offset in model_offsets]
        max_history_frames = int(semantic_config.get("max_history_frames", len(model_offsets)))
        if max_history_frames < len(model_offsets):
            raise ValueError(
                "`model.semantic_history_config.max_history_frames` must be >= len(history_offsets), "
                f"got {max_history_frames} and {len(model_offsets)}."
            )

        for dataset_name, dataset in datasets:
            if history_source == "vae":
                if not bool(getattr(dataset, "load_history_vae_video", False)):
                    raise ValueError(
                        f"{dataset_name} must set `load_history_vae_video=true` when "
                        "`model.semantic_history_config.source='vae'`."
                    )
                if bool(getattr(dataset, "load_history_dino_latents", False)) or bool(
                    getattr(dataset, "load_history_dino_video", False)
                ):
                    raise ValueError(
                        f"{dataset_name} must not load DINO history when semantic history source is VAE."
                    )
                dataset_offsets = getattr(dataset, "history_vae_frame_offsets", None)
                offset_attr = "history_vae_frame_offsets"
            else:
                load_history_latents = bool(getattr(dataset, "load_history_dino_latents", False))
                load_history_video = bool(getattr(dataset, "load_history_dino_video", False))
                if int(load_history_latents) + int(load_history_video) != 1:
                    raise ValueError(
                        f"{dataset_name} should provide exactly one semantic-history DINO source: "
                        "`load_history_dino_latents=true` for cached latents, or "
                        "`load_history_dino_video=true` for online DINO encoding."
                    )
                if bool(getattr(dataset, "load_history_vae_video", False)):
                    raise ValueError(
                        f"{dataset_name} must not load VAE history when semantic history source is DINO."
                    )
                dataset_offsets = getattr(dataset, "history_dino_frame_offsets", None)
                offset_attr = "history_dino_frame_offsets"
            if dataset_offsets is None:
                raise ValueError(
                    f"{dataset_name} has no `{offset_attr}`; cannot verify "
                    "semantic-history train/infer consistency."
                )
            dataset_offsets = [int(offset) for offset in dataset_offsets]
            if dataset_offsets != model_offsets:
                raise ValueError(
                    f"Semantic-history {history_source.upper()} offset mismatch between model and dataset: "
                    f"model.semantic_history_config.history_offsets={model_offsets}, "
                    f"{dataset_name}.{offset_attr}={dataset_offsets}."
                )

        if self.accelerator.is_main_process:
            logger.info(
                "Qwen semantic-history verified: source=%s offsets=%s max_history_frames=%d",
                history_source,
                model_offsets,
                max_history_frames,
            )

    def _estimate_total_train_steps(self) -> int:
        if self.max_steps is not None:
            return max(int(self.max_steps), 1)

        if not hasattr(self.train_dataset, "__len__"):
            raise TypeError("`train_dataset` must implement __len__ when `max_steps` is None.")

        num_processes = max(int(self.accelerator.num_processes), 1)
        global_batch_size = max(self.batch_size * num_processes, 1)
        micro_steps_per_epoch = max(ceil(len(self.train_dataset) / global_batch_size), 1)
        opt_steps_per_epoch = max(
            ceil(micro_steps_per_epoch / self.gradient_accumulation_steps),
            1,
        )
        return max(opt_steps_per_epoch * self.num_epochs, 1)

    def _build_scheduler(self, scheduler_type, total_train_steps: int, warmup_steps: int = 0):
        scheduler_type = str(scheduler_type).strip().lower()
        total_train_steps = max(int(total_train_steps), 1)
        warmup_steps = min(max(int(warmup_steps), 0), total_train_steps - 1)

        remaining_steps = max(total_train_steps - warmup_steps, 1)
        if scheduler_type == "cosine":
            main_scheduler = CosineAnnealingLR(
                self.optimizer,
                T_max=remaining_steps,
                eta_min=self.learning_rate * 0.01,
            )
        elif scheduler_type == "constant":
            main_scheduler = ConstantLR(self.optimizer, factor=1.0, total_iters=remaining_steps)
        else:
            raise ValueError(
                f"Unsupported lr_scheduler_type: {scheduler_type}. "
                "Expected one of: ['cosine', 'constant']."
            )

        if warmup_steps <= 0:
            return main_scheduler

        warmup_scheduler = LinearLR(
            self.optimizer,
            start_factor=1.0 / warmup_steps,
            end_factor=1.0,
            total_iters=warmup_steps,
        )
        return SequentialLR(
            self.optimizer,
            schedulers=[warmup_scheduler, main_scheduler],
            milestones=[warmup_steps],
        )
    
    def _estimate_eta(self):
        elapsed = max(time.perf_counter() - self.run_start_time, 1e-6)
        done_steps = max(self.global_step - self.run_start_step, 1)
        steps_per_sec = done_steps / elapsed
        remaining_steps = max(self.max_steps - self.global_step, 0)
        eta_seconds = int(remaining_steps / max(steps_per_sec, 1e-9))
        eta_h, eta_rem = divmod(eta_seconds, 3600)
        eta_m, eta_s = divmod(eta_rem, 60)
        return f"{eta_h:02d}:{eta_m:02d}:{eta_s:02d}", steps_per_sec

    def _load_weight_checkpoint_before_prepare(self):
        resume = self.resume
        if not resume:
            return
        resume_path = Path(str(resume))
        if resume_path.is_dir():
            return
        if not resume_path.exists():
            raise FileNotFoundError(f"Resume checkpoint not found: {resume}")
        logger.info("Loading weight checkpoint before accelerator.prepare: %s", resume)
        self.model.load_checkpoint(str(resume_path), optimizer=None)
        self._weight_checkpoint_loaded_pre_prepare = True

    def _resume_or_load_checkpoint(self):
        resume = self.resume
        if not resume:
            return
        resume_path = Path(str(resume))
        if resume_path.is_dir():
            logger.info("Resuming full training state from directory: %s", resume)
            self.load_training_state(str(resume_path))
            return
        if not resume_path.exists():
            raise FileNotFoundError(f"Resume checkpoint not found: {resume}")
        if self._weight_checkpoint_loaded_pre_prepare:
            logger.warning(
                "Loaded .pt weights only before optimizer/DeepSpeed initialization; "
                "optimizer/scheduler/step were not restored."
            )
            return
        logger.info("Loading weight checkpoint only: %s", resume)
        self.accelerator.unwrap_model(self.model).load_checkpoint(str(resume_path), optimizer=None)
        logger.warning("Loaded .pt weights only; optimizer/scheduler/step were not restored.")

    def _set_dit_only_train_mode(self):
        # Match DiffSynth's freeze_except("dit"): only DiT stays trainable/in-train-mode.
        logger.info("Setting DiT to train mode and freezing other model components.")
        model = self.accelerator.unwrap_model(self.model)
        self._apply_dit_only_train_mode(model)

    @staticmethod
    def _apply_dit_only_train_mode(model):
        model.eval()
        model.requires_grad_(False)
        model.dit.train()
        model.dit.requires_grad_(True)
        proprio_encoder = getattr(model, "proprio_encoder", None)
        if proprio_encoder is not None:
            proprio_encoder.train()
            proprio_encoder.requires_grad_(True)
        intent_encoder = getattr(model, "intent_encoder", None)
        if intent_encoder is not None:
            intent_encoder.train()
            intent_encoder.requires_grad_(True)
        semantic_history_encoder = getattr(model, "semantic_history_encoder", None)
        if semantic_history_encoder is not None:
            semantic_history_encoder.set_adapter_train_mode()

    @staticmethod
    def _to_batched_eval_sample(sample):
        video = sample.get("video", None)
        prompt = sample.get("prompt", None)
        action = sample.get("action", None)
        proprio = sample.get("proprio", None)
        context = sample.get("context", None)
        context_mask = sample.get("context_mask", None)
        vae_latents = sample.get("vae_latents", None)
        dino_latents = sample.get("dino_latents", None)
        history_dino_latents = sample.get("history_dino_latents", None)
        history_vae_latents = sample.get("history_vae_latents", None)
        history_video = sample.get("history_video", None)
        semantic_image = sample.get("semantic_image", sample.get("vlm_image", None))
        action_is_pad = sample.get("action_is_pad", None)
        image_is_pad = sample.get("image_is_pad", None)

        batch_size = None
        num_video_frames = None
        if video is not None:
            if not isinstance(video, torch.Tensor):
                raise TypeError(
                    f"Expected tensor video for evaluation, got {type(video)}. "
                    "Evaluation now expects `video` with shape [3,T,H,W] or [B,3,T,H,W]."
                )
            if video.ndim == 4:
                video = video.unsqueeze(0)
            if video.ndim != 5:
                raise ValueError(f"Expected video shape [3,T,H,W] or [B,3,T,H,W], got {tuple(video.shape)}")
            batch_size = int(video.shape[0])
            num_video_frames = int(video.shape[2])
            if num_video_frames <= 1:
                raise ValueError(
                    f"`sample['video']` must have at least 2 frames for action evaluation, got {num_video_frames}"
                )
        elif "video_frame_count" in sample:
            num_video_frames = int(sample["video_frame_count"])
            if num_video_frames <= 1:
                raise ValueError(f"`video_frame_count` must be at least 2, got {num_video_frames}")

        if prompt is not None:
            if isinstance(prompt, str):
                prompt = [prompt]
            elif isinstance(prompt, tuple):
                prompt = list(prompt)
            elif not isinstance(prompt, list):
                raise TypeError(f"Expected prompt type str/list[str], got {type(prompt)}")
        
        action_horizon = None
        action = None
        if "action" in sample:
            action = sample["action"]
            if not isinstance(action, torch.Tensor):
                raise TypeError(
                    f"`sample['action']` must be a torch.Tensor, got {type(action)}"
                )
            if action.ndim == 2:
                action = action.unsqueeze(0)
            if action.ndim != 3:
                raise ValueError(f"`sample['action']` must be 3D [B, T, a_dim], got shape {tuple(action.shape)}")
            if batch_size is None:
                batch_size = int(action.shape[0])
            elif action.shape[0] != batch_size:
                raise ValueError(f"Action batch mismatch: action={action.shape[0]} vs batch={batch_size}")
            action_horizon = int(action.shape[1])

        proprio = None
        if "proprio" in sample:
            proprio = sample["proprio"]
            if not isinstance(proprio, torch.Tensor):
                raise TypeError(f"`sample['proprio']` must be a torch.Tensor, got {type(proprio)}")
            if proprio.ndim == 2:
                proprio = proprio.unsqueeze(0)
            if proprio.ndim != 3:
                raise ValueError(f"`sample['proprio']` must be 3D [B, T, d], got shape {tuple(proprio.shape)}")
            if batch_size is None:
                batch_size = int(proprio.shape[0])
            elif proprio.shape[0] != batch_size:
                raise ValueError(f"Proprio batch mismatch: proprio={proprio.shape[0]} vs batch={batch_size}")

        if context is not None or context_mask is not None:
            if context is None or context_mask is None:
                raise ValueError("`context` and `context_mask` must both exist in eval sample.")
            if context.ndim == 2:
                context = context.unsqueeze(0)
            if context_mask.ndim == 1:
                context_mask = context_mask.unsqueeze(0)
            if context.ndim != 3 or context_mask.ndim != 2:
                raise ValueError(
                    f"`context/context_mask` must be [B,L,D]/[B,L], got {tuple(context.shape)} and {tuple(context_mask.shape)}"
                )
            if batch_size is None:
                batch_size = int(context.shape[0])
            elif context.shape[0] != batch_size or context_mask.shape[0] != batch_size:
                raise ValueError(
                    "Context batch mismatch: "
                    f"context={context.shape[0]} mask={context_mask.shape[0]} vs batch={batch_size}"
                )

        if vae_latents is not None:
            if not isinstance(vae_latents, torch.Tensor):
                raise TypeError(f"`sample['vae_latents']` must be a torch.Tensor, got {type(vae_latents)}")
            if vae_latents.ndim == 4:
                vae_latents = vae_latents.unsqueeze(0)
            if vae_latents.ndim != 5:
                raise ValueError(
                    f"`sample['vae_latents']` must be [C,T,H,W] or [B,C,T,H,W], "
                    f"got {tuple(vae_latents.shape)}"
                )
            if batch_size is None:
                batch_size = int(vae_latents.shape[0])
            elif vae_latents.shape[0] != batch_size:
                raise ValueError(
                    f"Eval VAE latent batch mismatch: vae={tuple(vae_latents.shape)} batch={batch_size}"
                )

        if dino_latents is not None:
            if not isinstance(dino_latents, torch.Tensor):
                raise TypeError(f"`sample['dino_latents']` must be a torch.Tensor, got {type(dino_latents)}")
            if dino_latents.ndim == 4:
                dino_latents = dino_latents.unsqueeze(0)
            if dino_latents.ndim != 5:
                raise ValueError(
                    f"`sample['dino_latents']` must be [D,T,H,W] or [B,D,T,H,W], "
                    f"got {tuple(dino_latents.shape)}"
                )
            if batch_size is None:
                batch_size = int(dino_latents.shape[0])
            elif dino_latents.shape[0] != batch_size:
                raise ValueError(
                    f"Eval DINO latent batch mismatch: dino={tuple(dino_latents.shape)} batch={batch_size}"
                )
            if num_video_frames is None:
                num_video_frames = int(dino_latents.shape[2])
            elif dino_latents.shape[2] != num_video_frames:
                raise ValueError(
                    "Eval DINO latent/video temporal mismatch: "
                    f"dino={tuple(dino_latents.shape)} video_T={num_video_frames}"
                )

        if prompt is not None and batch_size is not None and len(prompt) != batch_size:
            raise ValueError(f"Prompt batch mismatch: len(prompt)={len(prompt)} vs batch={batch_size}")
        if action is not None and num_video_frames is not None and action.shape[1] % (num_video_frames - 1) != 0:
            raise ValueError(
                "`sample['action']` temporal dimension must be divisible by "
                f"video frames-1={num_video_frames - 1}, got {action.shape[1]}"
            )

        if history_dino_latents is not None:
            if not isinstance(history_dino_latents, torch.Tensor):
                raise TypeError(
                    f"`sample['history_dino_latents']` must be a torch.Tensor, "
                    f"got {type(history_dino_latents)}"
                )
            if history_dino_latents.ndim == 4:
                history_dino_latents = history_dino_latents.unsqueeze(0)
            if history_dino_latents.ndim != 5:
                raise ValueError(
                    "`sample['history_dino_latents']` must be [D,T,H,W] or [B,D,T,H,W], "
                    f"got {tuple(history_dino_latents.shape)}"
                )
            if batch_size is None:
                batch_size = int(history_dino_latents.shape[0])
            elif history_dino_latents.shape[0] != batch_size:
                raise ValueError(
                    "Eval history DINO latent batch mismatch: "
                    f"history={tuple(history_dino_latents.shape)} batch={batch_size}"
                )

        if history_vae_latents is not None:
            if not isinstance(history_vae_latents, torch.Tensor):
                raise TypeError(
                    f"`sample['history_vae_latents']` must be a torch.Tensor, "
                    f"got {type(history_vae_latents)}"
                )
            if history_vae_latents.ndim == 4:
                history_vae_latents = history_vae_latents.unsqueeze(0)
            if history_vae_latents.ndim != 5:
                raise ValueError(
                    "`sample['history_vae_latents']` must be [C,T,H,W] or [B,C,T,H,W], "
                    f"got {tuple(history_vae_latents.shape)}"
                )
            if batch_size is None:
                batch_size = int(history_vae_latents.shape[0])
            elif history_vae_latents.shape[0] != batch_size:
                raise ValueError(
                    "Eval history VAE latent batch mismatch: "
                    f"history={tuple(history_vae_latents.shape)} batch={batch_size}"
                )

        if history_video is not None:
            if not isinstance(history_video, torch.Tensor):
                raise TypeError(
                    f"`sample['history_video']` must be a torch.Tensor, got {type(history_video)}"
                )
            if history_video.ndim == 4:
                history_video = history_video.unsqueeze(0)
            if history_video.ndim != 5:
                raise ValueError(
                    "`sample['history_video']` must be [C,T,H,W] or [B,C,T,H,W], "
                    f"got {tuple(history_video.shape)}"
                )
            if history_video.shape[1] != 3:
                raise ValueError(
                    f"`sample['history_video']` channel dim must be 3 after batching, "
                    f"got {tuple(history_video.shape)}"
                )
            if batch_size is None:
                batch_size = int(history_video.shape[0])
            elif history_video.shape[0] != batch_size:
                raise ValueError(
                    "Eval history video batch mismatch: "
                    f"history={tuple(history_video.shape)} batch={batch_size}"
                )

        if semantic_image is not None:
            if not isinstance(semantic_image, torch.Tensor):
                raise TypeError(
                    f"`sample['semantic_image']` must be a torch.Tensor, got {type(semantic_image)}"
                )
            if semantic_image.ndim == 3:
                semantic_image = semantic_image.unsqueeze(0)
            if semantic_image.ndim != 4 or semantic_image.shape[1] != 3:
                raise ValueError(
                    "`sample['semantic_image']` must be [3,H,W] or [B,3,H,W], "
                    f"got {tuple(semantic_image.shape)}"
                )
            if batch_size is None:
                batch_size = int(semantic_image.shape[0])
            elif semantic_image.shape[0] != batch_size:
                raise ValueError(
                    "Eval semantic image batch mismatch: "
                    f"semantic={tuple(semantic_image.shape)} batch={batch_size}"
                )

        if action_is_pad is not None:
            if not isinstance(action_is_pad, torch.Tensor):
                action_is_pad = torch.as_tensor(action_is_pad, dtype=torch.bool)
            if action_is_pad.ndim == 1:
                action_is_pad = action_is_pad.unsqueeze(0)
            if action_is_pad.ndim != 2:
                raise ValueError(f"`sample['action_is_pad']` must be [T] or [B,T], got {tuple(action_is_pad.shape)}")

        if image_is_pad is not None:
            if not isinstance(image_is_pad, torch.Tensor):
                image_is_pad = torch.as_tensor(image_is_pad, dtype=torch.bool)
            if image_is_pad.ndim == 1:
                image_is_pad = image_is_pad.unsqueeze(0)
            if image_is_pad.ndim != 2:
                raise ValueError(f"`sample['image_is_pad']` must be [T] or [B,T], got {tuple(image_is_pad.shape)}")

        return {
            "video": video,
            "prompt": prompt,
            "action": action,
            "proprio": proprio,
            "context": context,
            "context_mask": context_mask,
            "vae_latents": vae_latents,
            "dino_latents": dino_latents,
            "history_dino_latents": history_dino_latents,
            "history_vae_latents": history_vae_latents,
            "history_video": history_video,
            "semantic_image": semantic_image,
            "action_is_pad": action_is_pad,
            "image_is_pad": image_is_pad,
            "action_horizon": action_horizon,
            "video_frame_count": num_video_frames,
        }

    def _compute_eval_action_metrics(self, sample, pred_action, gt_action):
        if gt_action is None or pred_action is None:
            return {}
        if sample["proprio"] is None:
            raise ValueError("Eval sample must contain `proprio` for action denormalization.")

        proprio = sample["proprio"].detach().to(device="cpu", dtype=torch.float32)
        processor = self.val_dataset.lerobot_dataset.processor

        denorm_actions = {}
        action_meta = processor.shape_meta["action"]
        state_meta = processor.shape_meta["state"]
        for action_name, raw_action in (("pred", pred_action), ("gt", gt_action)):
            if not isinstance(raw_action, torch.Tensor):
                raise TypeError(f"{action_name} action must be a torch.Tensor, got {type(raw_action)}")
            if raw_action.ndim == 2:
                action_btd = raw_action.unsqueeze(0)
            elif raw_action.ndim == 3 and raw_action.shape[0] == 1:
                action_btd = raw_action
            else:
                raise ValueError(
                    f"{action_name} action must have shape [T, D] or [1, T, D], got {tuple(raw_action.shape)}"
                )
            action_btd = action_btd.detach().to(device="cpu", dtype=torch.float32)

            batch = {
                "action": action_btd,
                "state": proprio,
            }
            batch = processor.action_state_merger.backward(batch)
            batch = processor.normalizer.backward(batch)
            merged_batch = {
                "action": {meta["key"]: batch["action"][meta["key"]].squeeze(0) for meta in action_meta},
                "state": {meta["key"]: batch["state"][meta["key"]].squeeze(0) for meta in state_meta},
            }
            merged_batch = processor.action_state_merger.forward(merged_batch)
            denorm_action = merged_batch["action"].unsqueeze(0)
            if denorm_action.ndim != 3 or denorm_action.shape[0] != 1:
                raise ValueError(
                    f"Denormalized {action_name} action must have shape [1, T, D], got {tuple(denorm_action.shape)}"
                )
            denorm_actions[action_name] = denorm_action

        pred_action_denorm = denorm_actions["pred"]
        gt_action_denorm = denorm_actions["gt"]

        if pred_action_denorm.shape != gt_action_denorm.shape:
            raise ValueError(
                "Predicted action/GT action shape mismatch after denormalization: "
                f"pred={tuple(pred_action_denorm.shape)} vs gt={tuple(gt_action_denorm.shape)}"
            )
        action_diff = pred_action_denorm - gt_action_denorm
        return {
            "action_l1": action_diff.abs().mean().item(),
            "action_l2": action_diff.pow(2).mean().item(),
        }

    @torch.no_grad()
    def evaluate(self):
        if self.val_dataset is None:
            return None

        model = self.accelerator.unwrap_model(self.model)
        was_dit_training = model.dit.training
        model.eval()

        # eval_index = (self.global_step + self.accelerator.process_index) % len(self.val_dataset)
        rng = torch.Generator(device="cpu").manual_seed(self.global_step + self.accelerator.process_index)
        eval_index = torch.randint(0, len(self.val_dataset), (1,), generator=rng).item()
        sample = self._to_batched_eval_sample(self.val_dataset[eval_index])

        # 1. training loss
        with self.accelerator.autocast():
            val_loss, _ = model.training_loss(sample)
            val_loss = val_loss.float().item()

        dino_encoder = getattr(model, "dino_encoder", None)
        dino_backbone_loaded = True
        if dino_encoder is not None:
            dino_backbone_loaded = bool(getattr(dino_encoder, "_loaded", False))
        if sample["video"] is None or (hasattr(model, "infer_action") and not dino_backbone_loaded):
            local_metrics = torch.tensor(
                [float(val_loss)],
                device=self.accelerator.device,
                dtype=torch.float32,
            ).unsqueeze(0)
            gathered_metrics = self.accelerator.gather_for_metrics(local_metrics)
            mean_metrics = gathered_metrics.mean(dim=0)

            if was_dit_training:
                self._set_dit_only_train_mode()

            return {"val_loss": float(mean_metrics[0].item())}
        
        prompt = sample["prompt"][0]
        video0 = sample["video"][0] # Tensor [3, T, H, W] in (-1, 1)
        action = sample["action"][0] if "action" in sample and sample["action"] is not None else None
        proprio = sample["proprio"][0, 0] if "proprio" in sample and sample["proprio"] is not None else None # from [1, T, d] to [d]
        input_image = video0[:, 0].unsqueeze(0)
        _, num_frames, _, _ = video0.shape

        action_only_inference = bool(getattr(model, "action_only_inference", False))
        if (action_only_inference or not hasattr(model, "infer")) and hasattr(model, "infer_action"):
            infer_kwargs = {
                "input_image": input_image,
                "action_horizon": sample["action_horizon"],
                "proprio": proprio,
                "num_inference_steps": self.eval_num_inference_steps,
                "seed": 42,
                "tiled": False,
            }
            if sample["context"] is not None:
                infer_kwargs["prompt"] = None
                infer_kwargs["context"] = sample["context"][0]
                infer_kwargs["context_mask"] = sample["context_mask"][0]
            else:
                infer_kwargs["prompt"] = prompt
            if sample.get("history_dino_latents") is not None:
                infer_kwargs["history_dino_latents"] = sample["history_dino_latents"][0]
            elif sample.get("history_vae_latents") is not None:
                infer_kwargs["history_vae_latents"] = sample["history_vae_latents"][0]
            elif sample.get("history_video") is not None:
                infer_kwargs["history_video"] = sample["history_video"]
            if sample.get("semantic_image") is not None:
                infer_kwargs["semantic_image"] = sample["semantic_image"][0]
            if getattr(model, "semantic_history_encoder", None) is not None:
                infer_kwargs["semantic_prompt"] = prompt

            pred = model.infer_action(**infer_kwargs)
            action_metrics = self._compute_eval_action_metrics(sample, pred.get("action"), action)

            local_metrics = torch.tensor(
                [
                    float(val_loss),
                    float(action_metrics["action_l2"]) if "action_l2" in action_metrics else -1.0,
                    float(action_metrics["action_l1"]) if "action_l1" in action_metrics else -1.0,
                ],
                device=self.accelerator.device,
                dtype=torch.float32,
            ).unsqueeze(0)
            gathered_metrics = self.accelerator.gather_for_metrics(local_metrics)
            mean_metrics = gathered_metrics.mean(dim=0)

            if was_dit_training:
                self._set_dit_only_train_mode()

            result = {"val_loss": float(mean_metrics[0].item())}
            if "action_l2" in action_metrics:
                result["action_l2"] = float(mean_metrics[1].item())
            if "action_l1" in action_metrics:
                result["action_l1"] = float(mean_metrics[2].item())
            return result

        # 2. inference and video saving
        infer_kwargs = {
            "input_image": input_image,
            "num_frames": num_frames,
            "action": action,
            "action_horizon": sample['action_horizon'],
            "proprio": proprio,
            "text_cfg_scale": 1.0,
            "action_cfg_scale": 1.0,
            "num_inference_steps": self.eval_num_inference_steps,
            "seed": 42,
            "tiled": False,
        }
        if sample["context"] is not None:
            infer_kwargs["prompt"] = None
            infer_kwargs["context"] = sample["context"][0]
            infer_kwargs["context_mask"] = sample["context_mask"][0]
        else:
            infer_kwargs["prompt"] = prompt

        pred = model.infer(
            **infer_kwargs,
        )
        
        pred_video = pred["video"]
        pred_action = pred.get("action", None)

        # 3. inference metrics against GT video
        pred_video_tensor = pil_frames_to_video_tensor(pred_video)
        gt_video_tensor = ((video0.detach().float().cpu().clamp(-1.0, 1.0) + 1.0) * 0.5).contiguous()

        assert pred_video_tensor.shape == gt_video_tensor.shape, (
            "Eval infer prediction/GT shape mismatch: "
            f"pred={tuple(pred_video_tensor.shape)} vs gt={tuple(gt_video_tensor.shape)}"
        )

        psnr_rollout_vs_gt = video_psnr(pred=pred_video_tensor, target=gt_video_tensor)
        ssim_rollout_vs_gt = video_ssim(pred=pred_video_tensor, target=gt_video_tensor)

        action_metrics = self._compute_eval_action_metrics(sample, pred_action, action)

        # 4. VAE reconstruction metrics against GT video
        gt_video_batch = video0.unsqueeze(0).to(device=model.device, dtype=model.torch_dtype)
        vae_latents = model._encode_video_latents(gt_video_batch, tiled=False)
        vae_recon_video = model._decode_latents(vae_latents, tiled=False)
        vae_video_tensor = pil_frames_to_video_tensor(vae_recon_video)

        assert vae_video_tensor.shape == gt_video_tensor.shape, (
            "Eval VAE reconstruction/GT shape mismatch: "
            f"vae={tuple(vae_video_tensor.shape)} vs gt={tuple(gt_video_tensor.shape)}"
        )

        psnr_decode_vs_gt = video_psnr(pred=vae_video_tensor, target=gt_video_tensor)
        ssim_decode_vs_gt = video_ssim(pred=vae_video_tensor, target=gt_video_tensor)

        psnr_rollout_vs_decode = video_psnr(pred=pred_video_tensor, target=vae_video_tensor)
        ssim_rollout_vs_decode = video_ssim(pred=pred_video_tensor, target=vae_video_tensor)

        stitched_video_tensor = torch.cat(
            [pred_video_tensor, vae_video_tensor, gt_video_tensor],
            dim=2,
        ).contiguous()
        stitched_frames = []
        for t in range(stitched_video_tensor.shape[1]):
            frame = (stitched_video_tensor[:, t].permute(1, 2, 0).clamp(0.0, 1.0).numpy() * 255.0).astype(np.uint8)
            stitched_frames.append(Image.fromarray(frame))

        video_path = os.path.join(
            self.eval_dir,
            f"step_{self.global_step:06d}_rank_{self.accelerator.process_index:03d}.mp4",
        )
        save_mp4(stitched_frames, video_path, fps=8)

        local_metrics = torch.tensor(
            [
                float(val_loss),
                float(psnr_rollout_vs_gt),
                float(ssim_rollout_vs_gt),
                float(psnr_rollout_vs_decode),
                float(ssim_rollout_vs_decode),
                float(psnr_decode_vs_gt),
                float(ssim_decode_vs_gt),
                float(action_metrics["action_l2"]) if "action_l2" in action_metrics else -1.0,
                float(action_metrics["action_l1"]) if "action_l1" in action_metrics else -1.0,
            ],
            device=self.accelerator.device,
            dtype=torch.float32,
        ).unsqueeze(0)
        gathered_metrics = self.accelerator.gather_for_metrics(local_metrics)
        mean_metrics = gathered_metrics[:, :7].mean(dim=0)
        action_l2_mean = gathered_metrics[:, 7].mean().item() if "action_l2" in action_metrics else None
        action_l1_mean = gathered_metrics[:, 8].mean().item() if "action_l1" in action_metrics else None

        if was_dit_training:
            self._set_dit_only_train_mode()

        result = {
            "val_loss": float(mean_metrics[0].item()),
            "psnr_rg": float(mean_metrics[1].item()),
            "ssim_rg": float(mean_metrics[2].item()),
            "psnr_rd": float(mean_metrics[3].item()),
            "ssim_rd": float(mean_metrics[4].item()),
            "psnr_dg": float(mean_metrics[5].item()),
            "ssim_dg": float(mean_metrics[6].item()),
            "video_path": video_path,
        }
        if action_l2_mean is not None:
            result["action_l2"] = float(action_l2_mean)
        if action_l1_mean is not None:
            result["action_l1"] = float(action_l1_mean)
        return result

    def _save_weights_checkpoint(self, step_tag: str):
        model = self.accelerator.unwrap_model(self.model)
        ckpt_path = os.path.join(self.weights_dir, f"{step_tag}.pt")
        save_kwargs = {
            "optimizer": None,
            "step": self.global_step,
        }
        if self.safe_checkpoint_serialization:
            save_kwargs["safe_serialization"] = True
        model.save_checkpoint(ckpt_path, **save_kwargs)
        return ckpt_path

    def _save_trainer_state(self, state_path: str, *, full_state: bool):
        state_file = os.path.join(state_path, "trainer_state.json")
        payload = {
            "global_step": int(self.global_step),
            "epoch": int(self.epoch),
            "batch_in_epoch": int(self.batch_in_epoch),
            "full_state": bool(full_state),
        }
        with open(state_file, "w", encoding="utf-8") as f:
            json.dump(payload, f, ensure_ascii=True, indent=2)

    def save_checkpoint(self):
        step_tag = f"step_{self.global_step:06d}"

        self.accelerator.wait_for_everyone()
        ckpt_path = None
        if self.accelerator.is_main_process:
            ckpt_path = self._save_weights_checkpoint(step_tag=step_tag)
        self.accelerator.wait_for_everyone()

        state_path = os.path.join(self.state_dir, step_tag)
        ensure_dir(state_path)
        if self.save_accelerate_state:
            self.accelerator.save_state(output_dir=state_path)
        elif self.accelerator.is_main_process:
            logger.warning(
                "Skipping Accelerate/DeepSpeed state save at step=%d; only the verified weight checkpoint is resumable.",
                self.global_step,
            )
        if self.accelerator.is_main_process:
            self._save_trainer_state(state_path, full_state=self.save_accelerate_state)
        self.accelerator.wait_for_everyone()

        return {"weights_path": ckpt_path, "state_path": state_path}

    def load_training_state(self, state_dir: str):
        state_file = Path(state_dir) / "trainer_state.json"
        if state_file.exists():
            with open(state_file, "r", encoding="utf-8") as f:
                payload = json.load(f)
            if not bool(payload.get("full_state", True)):
                raise ValueError(
                    f"Training state at {state_dir} contains no Accelerate/DeepSpeed state. "
                    "Resume from the corresponding weights/step_*.pt checkpoint instead."
                )
        self.accelerator.load_state(input_dir=state_dir)
        if state_file.exists():
            with open(state_file, "r", encoding="utf-8") as f:
                payload = json.load(f)
            self.global_step = int(payload["global_step"])

            if "epoch" in payload and "batch_in_epoch" in payload:
                self.epoch = int(payload["epoch"])
                self.batch_in_epoch = int(payload["batch_in_epoch"])
                self.train_sampler.set_epoch(self.epoch)
                self.train_sampler.set_epoch_offset(0)
                self.train_sampler.set_resume_batch_offset(self.batch_in_epoch)
                logger.info(
                    "Restored dataloader progress: epoch=%d batch_in_epoch=%d sample_offset=%d",
                    self.epoch,
                    self.batch_in_epoch,
                    self.batch_in_epoch * self.batch_size * self.accelerator.num_processes,
                )
            else:
                self.epoch = 0
                self.batch_in_epoch = 0
                self.train_sampler.clear_resume_batch_offset()
                logger.warning(
                    "State file does not contain `epoch`/`batch_in_epoch`; "
                    "optimizer/scheduler were restored, but dataloader progress resume is skipped."
                )
            self.accelerator.wait_for_everyone()
            return

        match = re.search(r"step[_-](\d+)$", str(state_dir).rstrip("/"))
        if match:
            self.global_step = int(match.group(1))
        else:
            self.global_step = 0
        self.epoch = 0
        self.batch_in_epoch = 0
        self.train_sampler.clear_resume_batch_offset()
        self.accelerator.wait_for_everyone()
        logger.info("Loaded accelerate training state from %s at step=%d", state_dir, self.global_step)
        logger.warning(
            "State file `%s` is missing; dataloader progress resume is skipped.",
            state_file,
        )

    def train(self):
        self._set_dit_only_train_mode()

        unwrapped_model = self.accelerator.unwrap_model(self.model)

        if self.max_steps is None:
            raise ValueError("`max_steps` must be set before entering the while-step training loop.")

        logger.info("Starting training with max_steps=%d.", self.max_steps)
        self.train_sampler.set_epoch(self.epoch)
        data_iter = iter(self.train_loader)
        self.run_start_step = self.global_step
        self.run_start_time = time.perf_counter()

        while self.global_step < self.max_steps:
            try:
                sample = next(data_iter)
                self.batch_in_epoch += 1
            except StopIteration:
                self.epoch += 1
                self.batch_in_epoch = 0
                self.train_sampler.clear_resume_batch_offset()
                self.train_sampler.set_epoch(self.epoch)
                data_iter = iter(self.train_loader)
                continue

            with self.accelerator.accumulate(self.model):
                train_model = self.model if hasattr(self.model, "training_loss") else self.accelerator.unwrap_model(self.model)

                with self.accelerator.autocast():
                    loss, loss_dict = train_model.training_loss(sample)
                self.accelerator.backward(loss)

                if self.accelerator.sync_gradients:
                    grad_norm = self.accelerator.clip_grad_norm_(self.model.parameters(), self.max_grad_norm)
                    self.optimizer.step()
                    if not self.accelerator.optimizer_step_was_skipped:
                        self.scheduler.step()
                    self.optimizer.zero_grad(set_to_none=True)
                    self.global_step += 1
                    global_loss = float(
                        self.accelerator.gather(loss.detach().float().reshape(1)).mean().item()
                    )
                    global_loss_metrics = {}
                    for key, value in loss_dict.items():
                        metric_tensor = torch.tensor(float(value), device=loss.device, dtype=torch.float32).reshape(1)
                        global_loss_metrics[key] = float(
                            self.accelerator.gather(metric_tensor).mean().item()
                        )
                    grad_norm_tensor = torch.tensor(grad_norm, device=loss.device, dtype=torch.float32)
                    global_grad_norm = float(self.accelerator.gather(grad_norm_tensor).mean().item())

                    current_lr = float(self.optimizer.param_groups[0]["lr"])

                    if self.log_every > 0 and self.global_step % self.log_every == 0 and self.accelerator.is_main_process:
                        eta_str, steps_per_sec = self._estimate_eta()
                        description = "[train] epoch=%d step=%d/%d loss=%.4f " % (
                            self.epoch,
                            self.global_step,
                            self.max_steps,
                            global_loss,
                        )
                        if global_loss_metrics:
                            detail_str = " ".join([f"{k}={v:.4f}" for k, v in sorted(global_loss_metrics.items())])
                            description += detail_str + " "
                        description += "lr=%.2e speed=%.2f step/s, %.2f samples/s eta=%s" % (
                            current_lr,
                            steps_per_sec,
                            steps_per_sec * self.batch_size * self.accelerator.num_processes,
                            eta_str,
                        )
                        logger.info(description)

                        wandb_payload = {
                            "train/loss": global_loss,
                            "train/grad_norm": global_grad_norm,
                            "train/lr": current_lr,
                            "performance/steps_per_sec": steps_per_sec,
                            "performance/samples_per_sec": steps_per_sec * self.batch_size * self.accelerator.num_processes,
                        }
                        for key, value in global_loss_metrics.items():
                            wandb_payload[f"train/{key}"] = value
                        self._wandb_log(wandb_payload)

                    if (
                        self.eval_every > 0
                        and self.val_dataset is not None
                        and self.global_step % self.eval_every == 0
                    ):
                        metrics = self.evaluate()
                        self.accelerator.wait_for_everyone()
                        if metrics is not None and self.accelerator.is_main_process:
                            description = "[eval] step=%d val_loss=%.4f" % (
                                self.global_step,
                                metrics["val_loss"],
                            )
                            if "psnr_rd" in metrics and "ssim_rd" in metrics:
                                description += " infer_psnr=%.4f infer_ssim=%.4f" % (
                                    metrics["psnr_rd"],
                                    metrics["ssim_rd"],
                                )
                            if "action_l2" in metrics:
                                description += " action_l2=%.4f" % metrics["action_l2"]
                            if "action_l1" in metrics:
                                description += " action_l1=%.4f" % metrics["action_l1"]
                            logger.info(description)
                            eval_payload = {
                                "eval/val_loss": float(metrics["val_loss"]),
                            }
                            for key in ("psnr_rg", "ssim_rg", "psnr_rd", "ssim_rd", "psnr_dg", "ssim_dg"):
                                if key in metrics:
                                    eval_payload[f"eval/{key}"] = float(metrics[key])
                            if "action_l2" in metrics:
                                eval_payload["eval/action_l2"] = float(metrics["action_l2"])
                            if "action_l1" in metrics:
                                eval_payload["eval/action_l1"] = float(metrics["action_l1"])
                            self._wandb_log(eval_payload)

                    if self.save_every > 0 and self.global_step % self.save_every == 0:
                        ckpt_info = self.save_checkpoint()
                        if self.accelerator.is_main_process:
                            logger.info(
                                "[ckpt] step=%d weights=%s state=%s",
                                self.global_step,
                                ckpt_info["weights_path"],
                                ckpt_info["state_path"],
                            )

                    if self.global_step >= self.max_steps:
                        ckpt_info = self.save_checkpoint()
                        if self.accelerator.is_main_process:
                            logger.info(
                                "[done] max_steps reached step=%d weights=%s state=%s",
                                self.global_step,
                                ckpt_info["weights_path"],
                                ckpt_info["state_path"],
                            )
                        return

        ckpt_info = self.save_checkpoint()
        if self.accelerator.is_main_process:
            logger.info(
                "[done] training finished step=%d weights=%s state=%s",
                self.global_step,
                ckpt_info["weights_path"],
                ckpt_info["state_path"],
            )
        
