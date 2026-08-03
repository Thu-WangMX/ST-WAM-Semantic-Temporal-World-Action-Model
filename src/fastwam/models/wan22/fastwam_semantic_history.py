"""Two-expert FastWAM with action-only Qwen/DINO-history readout."""

from __future__ import annotations

from typing import Any, Optional, Sequence, Union

import torch
import torch.nn.functional as F

from fastwam.utils.logging_config import get_logger

from .dino_encoder import DinoVideoEncoder
from .fastwam import FastWAM
from .semantic_history import QwenDINOHistoryActionAdapter

logger = get_logger(__name__)


class FastWAMSemanticHistory(FastWAM):
    """FastWAM with frozen Qwen current semantics reading frozen DINO history.

    The semantic adapter is intentionally visible only to the action expert.
    The VAE video expert remains the original two-expert FastWAM path.
    """

    action_only_inference = True

    def __init__(
        self,
        *args,
        dino_encoder: Optional[DinoVideoEncoder] = None,
        semantic_history_encoder: Optional[QwenDINOHistoryActionAdapter] = None,
        semantic_history_config: Optional[dict[str, Any]] = None,
        **kwargs,
    ):
        self.dino_encoder = dino_encoder
        self.semantic_history_encoder = semantic_history_encoder
        self.semantic_history_config = dict(semantic_history_config or {})
        self.semantic_history_frame_count = None
        self.semantic_history_dino_dim = None
        super().__init__(*args, **kwargs)

    @classmethod
    def from_wan22_pretrained(
        cls,
        *,
        dino_config: dict[str, Any],
        semantic_history_config: dict[str, Any],
        **kwargs,
    ):
        model = super().from_wan22_pretrained(**kwargs)
        if not bool(semantic_history_config.get("enabled", False)):
            raise ValueError(
                "FastWAMSemanticHistory requires `semantic_history_config.enabled=true`."
            )

        config = dict(semantic_history_config)
        injection_mode = str(config.get("injection_mode", "action_context_after_proprio"))
        if injection_mode != "action_context_after_proprio":
            raise ValueError(
                "FastWAMSemanticHistory supports only "
                "`semantic_history_config.injection_mode=action_context_after_proprio`."
            )
        history_offsets = [int(offset) for offset in config.get("history_offsets", [-24, -16, -8, -1])]
        if not history_offsets:
            raise ValueError("`semantic_history_config.history_offsets` must contain at least one offset.")
        max_history_frames = int(config.get("max_history_frames", len(history_offsets)))
        if max_history_frames < len(history_offsets):
            raise ValueError(
                "`semantic_history_config.max_history_frames` must be >= len(history_offsets), "
                f"got {max_history_frames} vs {len(history_offsets)}."
            )

        dino_encoder = DinoVideoEncoder(
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
        if bool(dino_config.get("load_backbone", True)):
            dino_encoder.load_backbone(device=model.device, dtype=model.torch_dtype)
        else:
            logger.info(
                "DINO backbone loading is disabled; training and inference must provide cached "
                "history_dino_latents."
            )

        config.update(
            {
                "enabled": True,
                "use_history": True,
                "injection_mode": injection_mode,
                "history_offsets": history_offsets,
                "max_history_frames": max_history_frames,
            }
        )
        adapter = QwenDINOHistoryActionAdapter(
            vlm_model_name_or_path=str(config["vlm_model_name_or_path"]),
            vlm_family=str(config.get("vlm_family", "qwen3_vl")),
            trust_remote_code=bool(config.get("trust_remote_code", True)),
            freeze_vlm=bool(config.get("freeze_vlm", True)),
            processor_min_pixels=config.get("processor_min_pixels", None),
            processor_max_pixels=config.get("processor_max_pixels", None),
            attn_implementation=config.get("attn_implementation", None),
            dino_dim=int(dino_config.get("feature_dim", 1024)),
            text_dim=model.text_dim,
            history_offsets=history_offsets,
            max_history_frames=max_history_frames,
            num_output_tokens=int(config.get("num_output_tokens", 8)),
            resampler_dim=int(config.get("resampler_dim", 1024)),
            num_layers=int(config.get("num_resampler_layers", 2)),
            num_heads=int(config.get("num_heads", 8)),
            dropout=float(config.get("dropout", 0.0)),
            use_history=True,
            torch_dtype=model.torch_dtype,
        ).to(device=model.device, dtype=model.torch_dtype)

        model.dino_encoder = dino_encoder
        model.semantic_history_encoder = adapter
        model.semantic_history_config = config
        model.semantic_history_frame_count = len(history_offsets)
        model.semantic_history_dino_dim = int(dino_config.get("feature_dim", 1024))
        model.to(model.device)
        model.model_paths["semantic_history_dino"] = dino_config.get("model_path", None)
        return model

    def to(self, *args, **kwargs):
        super().to(*args, **kwargs)
        dino_encoder = getattr(self, "dino_encoder", None)
        if dino_encoder is not None and bool(getattr(dino_encoder, "_loaded", False)):
            dino_encoder.backbone.to(*args, **kwargs)
        semantic_history_encoder = getattr(self, "semantic_history_encoder", None)
        if semantic_history_encoder is not None:
            semantic_history_encoder.to(*args, **kwargs)
        return self

    @staticmethod
    def _append_semantic_tokens(
        context: torch.Tensor,
        context_mask: torch.Tensor,
        semantic_tokens: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        if semantic_tokens.ndim != 3:
            raise ValueError(
                f"Semantic tokens must be [B,L,D], got {tuple(semantic_tokens.shape)}."
            )
        if semantic_tokens.shape[0] != context.shape[0] or semantic_tokens.shape[2] != context.shape[2]:
            raise ValueError(
                "Semantic token shape mismatch: "
                f"tokens={tuple(semantic_tokens.shape)} context={tuple(context.shape)}."
            )
        semantic_tokens = semantic_tokens.to(device=context.device, dtype=context.dtype)
        semantic_mask = torch.ones(
            (context_mask.shape[0], semantic_tokens.shape[1]),
            device=context_mask.device,
            dtype=torch.bool,
        )
        return (
            torch.cat([context, semantic_tokens], dim=1),
            torch.cat([context_mask, semantic_mask], dim=1),
        )

    @staticmethod
    def _normalize_history_video(history_video: torch.Tensor) -> torch.Tensor:
        if history_video.ndim == 4:
            if history_video.shape[0] == 3:
                return history_video.unsqueeze(0).contiguous()
            if history_video.shape[1] == 3:
                return history_video.permute(1, 0, 2, 3).unsqueeze(0).contiguous()
        elif history_video.ndim == 5:
            if history_video.shape[1] == 3:
                return history_video
            if history_video.shape[2] == 3:
                return history_video.permute(0, 2, 1, 3, 4).contiguous()
        raise ValueError(
            "`history_video` must be [3,T,H,W], [T,3,H,W], [B,3,T,H,W], or [B,T,3,H,W], "
            f"got {tuple(history_video.shape)}."
        )

    def _encode_semantic_history_tokens(
        self,
        *,
        prompts: str | Sequence[str],
        semantic_image: torch.Tensor,
        history_dino_latents: Optional[torch.Tensor] = None,
        history_video: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        if self.semantic_history_encoder is None or self.dino_encoder is None:
            raise RuntimeError("Semantic-history components are not initialized.")
        if prompts is None:
            raise ValueError(
                "`prompt`/`semantic_prompt` is required for Qwen semantic-history encoding."
            )
        if not torch.is_tensor(semantic_image):
            raise TypeError(f"`semantic_image` must be a tensor, got {type(semantic_image)}.")
        if semantic_image.ndim == 3:
            semantic_image = semantic_image.unsqueeze(0)
        if semantic_image.ndim != 4 or semantic_image.shape[1] != 3:
            raise ValueError(
                f"`semantic_image` must be [B,3,H,W] or [3,H,W], got {tuple(semantic_image.shape)}."
            )
        batch_size = int(semantic_image.shape[0])
        if history_dino_latents is not None and history_video is not None:
            raise ValueError("Provide only one of `history_dino_latents` or `history_video`.")
        if history_dino_latents is None:
            if history_video is None:
                raise ValueError(
                    "`history_dino_latents` or `history_video` is required for semantic-history readout."
                )
            if not bool(getattr(self.dino_encoder, "_loaded", False)):
                raise ValueError(
                    "Online DINO history encoding requires `model.dino_config.load_backbone=true`."
                )
            history_video = self._normalize_history_video(history_video)
            if history_video.shape[0] != batch_size:
                raise ValueError(
                    f"Batch mismatch between semantic_image ({batch_size}) and history_video ({history_video.shape[0]})."
                )
            history_dino_latents = self.dino_encoder.encode_video_to_latent(
                history_video.to(device=self.device, dtype=self.torch_dtype, non_blocking=True)
            )
        elif history_dino_latents.ndim == 4:
            history_dino_latents = history_dino_latents.unsqueeze(0)
        if history_dino_latents.ndim != 5:
            raise ValueError(
                f"`history_dino_latents` must be [B,D,T,H,W], got {tuple(history_dino_latents.shape)}."
            )
        if history_dino_latents.shape[0] != batch_size:
            raise ValueError(
                f"Batch mismatch between semantic_image ({batch_size}) and history ({history_dino_latents.shape[0]})."
            )
        if history_dino_latents.shape[1] != self.semantic_history_dino_dim:
            raise ValueError(
                f"History DINO dim mismatch: got {history_dino_latents.shape[1]}, "
                f"expected {self.semantic_history_dino_dim}."
            )
        if history_dino_latents.shape[2] != self.semantic_history_frame_count:
            raise ValueError(
                f"History DINO T mismatch: got {history_dino_latents.shape[2]}, "
                f"expected {self.semantic_history_frame_count}."
            )
        return self.semantic_history_encoder(
            prompts=prompts,
            semantic_image=semantic_image,
            history_dino_latents=history_dino_latents.to(
                device=self.device,
                dtype=self.torch_dtype,
                non_blocking=True,
            ),
        )

    def build_inputs(self, sample, tiled: bool = False):
        inputs = super().build_inputs(sample, tiled=tiled)
        semantic_image = sample.get("semantic_image", sample.get("vlm_image", None))
        semantic_prompt = sample.get("semantic_prompt", sample.get("prompt", None))
        if semantic_image is None:
            raise ValueError(
                "`semantic_image` is required for FastWAMSemanticHistory. "
                "Set `data.train.load_semantic_image=true`."
            )
        semantic_tokens = self._encode_semantic_history_tokens(
            prompts=semantic_prompt,
            semantic_image=semantic_image,
            history_dino_latents=sample.get("history_dino_latents", None),
            history_video=sample.get("history_video", None),
        )
        context_action, context_action_mask = self._append_semantic_tokens(
            inputs["context"], inputs["context_mask"], semantic_tokens
        )
        inputs["context_action"] = context_action
        inputs["context_action_mask"] = context_action_mask
        return inputs

    def training_loss(self, sample, tiled: bool = False):
        inputs = self.build_inputs(sample, tiled=tiled)
        input_latents = inputs["input_latents"]
        batch_size = input_latents.shape[0]
        context = inputs["context"]
        context_mask = inputs["context_mask"]
        context_action = inputs["context_action"]
        context_action_mask = inputs["context_action_mask"]
        action = inputs["action"]
        action_is_pad = inputs["action_is_pad"]
        image_is_pad = inputs["image_is_pad"]

        noise_video = torch.randn_like(input_latents)
        timestep_video = self.train_video_scheduler.sample_training_t(
            batch_size=batch_size, device=self.device, dtype=input_latents.dtype
        )
        latents = self.train_video_scheduler.add_noise(input_latents, noise_video, timestep_video)
        target_video = self.train_video_scheduler.training_target(input_latents, noise_video, timestep_video)
        if inputs["first_frame_latents"] is not None:
            latents[:, :, 0:1] = inputs["first_frame_latents"]

        noise_action = torch.randn_like(action)
        timestep_action = self.train_action_scheduler.sample_training_t(
            batch_size=batch_size, device=self.device, dtype=action.dtype
        )
        noisy_action = self.train_action_scheduler.add_noise(action, noise_action, timestep_action)
        target_action = self.train_action_scheduler.training_target(action, noise_action, timestep_action)

        video_pre = self.video_expert.pre_dit(
            x=latents,
            timestep=timestep_video,
            context=context,
            context_mask=context_mask,
            action=action,
            fuse_vae_embedding_in_latents=inputs["fuse_vae_embedding_in_latents"],
        )
        action_pre = self.action_expert.pre_dit(
            action_tokens=noisy_action,
            timestep=timestep_action,
            context=context_action,
            context_mask=context_action_mask,
        )
        attention_mask = self._build_mot_attention_mask(
            video_seq_len=video_pre["tokens"].shape[1],
            action_seq_len=action_pre["tokens"].shape[1],
            video_tokens_per_frame=int(video_pre["meta"]["tokens_per_frame"]),
            device=video_pre["tokens"].device,
        )
        tokens_out = self.mot(
            embeds_all={"video": video_pre["tokens"], "action": action_pre["tokens"]},
            attention_mask=attention_mask,
            freqs_all={"video": video_pre["freqs"], "action": action_pre["freqs"]},
            context_all={
                "video": {"context": video_pre["context"], "mask": video_pre["context_mask"]},
                "action": {"context": action_pre["context"], "mask": action_pre["context_mask"]},
            },
            t_mod_all={"video": video_pre["t_mod"], "action": action_pre["t_mod"]},
        )
        pred_video = self.video_expert.post_dit(tokens_out["video"], video_pre)
        pred_action = self.action_expert.post_dit(tokens_out["action"], action_pre)
        include_initial_video_step = inputs["first_frame_latents"] is None
        if not include_initial_video_step:
            pred_video = pred_video[:, :, 1:]
            target_video = target_video[:, :, 1:]
        loss_video_per_sample = self._compute_video_loss_per_sample(
            pred_video=pred_video,
            target_video=target_video,
            image_is_pad=image_is_pad,
            include_initial_video_step=include_initial_video_step,
        )
        video_weight = self.train_video_scheduler.training_weight(timestep_video).to(
            loss_video_per_sample.device, dtype=loss_video_per_sample.dtype
        )
        loss_video = (loss_video_per_sample * video_weight).mean()
        action_loss_token = F.mse_loss(pred_action.float(), target_action.float(), reduction="none").mean(dim=2)
        if action_is_pad is not None:
            valid = (~action_is_pad).to(device=action_loss_token.device, dtype=action_loss_token.dtype)
            action_loss_per_sample = (action_loss_token * valid).sum(dim=1) / valid.sum(dim=1).clamp(min=1.0)
        else:
            action_loss_per_sample = action_loss_token.mean(dim=1)
        action_weight = self.train_action_scheduler.training_weight(timestep_action).to(
            action_loss_per_sample.device, dtype=action_loss_per_sample.dtype
        )
        loss_action = (action_loss_per_sample * action_weight).mean()
        loss_total = self.loss_lambda_video * loss_video + self.loss_lambda_action * loss_action
        return loss_total, {
            "loss_video": self.loss_lambda_video * float(loss_video.detach().item()),
            "loss_action": self.loss_lambda_action * float(loss_action.detach().item()),
        }

    @torch.no_grad()
    def infer_action(
        self,
        prompt: Optional[str],
        input_image: torch.Tensor,
        action_horizon: int,
        proprio: Optional[torch.Tensor] = None,
        context: Optional[torch.Tensor] = None,
        context_mask: Optional[torch.Tensor] = None,
        negative_prompt: Optional[str] = None,
        text_cfg_scale: float = 1.0,
        num_inference_steps: int = 20,
        sigma_shift: Optional[float] = None,
        seed: Optional[int] = None,
        rand_device: str = "cpu",
        tiled: bool = False,
        history_dino_latents: Optional[torch.Tensor] = None,
        history_video: Optional[torch.Tensor] = None,
        semantic_image: Optional[torch.Tensor] = None,
        semantic_prompt: Optional[str] = None,
    ) -> dict[str, Any]:
        self.eval()
        if str(getattr(self.video_expert, "video_attention_mask_mode", "")) != "first_frame_causal":
            raise ValueError("`infer_action` requires `video_attention_mask_mode='first_frame_causal'.")
        if input_image.ndim == 3:
            input_image = input_image.unsqueeze(0)
        if input_image.ndim != 4 or input_image.shape[0] != 1 or input_image.shape[1] != 3:
            raise ValueError(f"`input_image` must be [1,3,H,W] or [3,H,W], got {tuple(input_image.shape)}.")
        if input_image.shape[-2] % 16 != 0 or input_image.shape[-1] % 16 != 0:
            raise ValueError(
                f"`input_image` must be resized before infer, got HxW={tuple(input_image.shape[-2:])}."
            )
        if proprio is not None:
            if self.proprio_dim is None:
                raise ValueError("`proprio` was provided but `proprio_dim=None`.")
            if proprio.ndim == 1:
                proprio = proprio.unsqueeze(0)
            if proprio.ndim != 2 or proprio.shape[0] != 1 or proprio.shape[1] != self.proprio_dim:
                raise ValueError(f"`proprio` must be [D] or [1,D={self.proprio_dim}], got {tuple(proprio.shape)}.")
            proprio = proprio.to(device=self.device, dtype=self.torch_dtype)

        use_prompt = prompt is not None
        use_context = context is not None or context_mask is not None
        if use_prompt and use_context:
            raise ValueError("`prompt` and `context/context_mask` are mutually exclusive.")
        if not use_prompt and not use_context:
            raise ValueError("Either `prompt` or both `context/context_mask` must be provided.")
        if use_prompt:
            context, context_mask = self.encode_prompt(prompt)
        else:
            if context is None or context_mask is None:
                raise ValueError("`context` and `context_mask` must be both provided together.")
            if context.ndim == 2:
                context = context.unsqueeze(0)
            if context_mask.ndim == 1:
                context_mask = context_mask.unsqueeze(0)
            if context.ndim != 3 or context_mask.ndim != 2:
                raise ValueError(
                    f"`context/context_mask` must be [B,L,D]/[B,L], got {tuple(context.shape)} and {tuple(context_mask.shape)}."
                )
            context = context.to(device=self.device, dtype=self.torch_dtype, non_blocking=True)
            context_mask = context_mask.to(device=self.device, dtype=torch.bool, non_blocking=True)
        if proprio is not None:
            context, context_mask = self._append_proprio_to_context(context, context_mask, proprio)

        input_image = input_image.to(device=self.device, dtype=self.torch_dtype)
        if semantic_image is None:
            semantic_image = input_image
        if semantic_prompt is None:
            semantic_prompt = prompt
        semantic_tokens = self._encode_semantic_history_tokens(
            prompts=semantic_prompt,
            semantic_image=semantic_image,
            history_dino_latents=history_dino_latents,
            history_video=history_video,
        )
        context_action, context_action_mask = self._append_semantic_tokens(
            context, context_mask, semantic_tokens
        )

        generator = None if seed is None else torch.Generator(device=rand_device).manual_seed(seed)
        latents_action = torch.randn(
            (1, action_horizon, self.action_expert.action_dim),
            generator=generator,
            device=rand_device,
            dtype=torch.float32,
        ).to(device=self.device, dtype=self.torch_dtype)
        first_frame_latents = self._encode_input_image_latents_tensor(input_image=input_image, tiled=tiled)
        fuse_flag = bool(getattr(self.video_expert, "fuse_vae_embedding_in_latents", False))
        timestep_video = torch.zeros(
            (first_frame_latents.shape[0],), dtype=first_frame_latents.dtype, device=self.device
        )
        video_pre = self.video_expert.pre_dit(
            x=first_frame_latents,
            timestep=timestep_video,
            context=context,
            context_mask=context_mask,
            action=None,
            fuse_vae_embedding_in_latents=fuse_flag,
        )
        video_seq_len = int(video_pre["tokens"].shape[1])
        attention_mask = self._build_mot_attention_mask(
            video_seq_len=video_seq_len,
            action_seq_len=latents_action.shape[1],
            video_tokens_per_frame=int(video_pre["meta"]["tokens_per_frame"]),
            device=video_pre["tokens"].device,
        )
        video_kv_cache = self.mot.prefill_video_cache(
            video_tokens=video_pre["tokens"],
            video_freqs=video_pre["freqs"],
            video_t_mod=video_pre["t_mod"],
            video_context_payload={"context": video_pre["context"], "mask": video_pre["context_mask"]},
            video_attention_mask=attention_mask[:video_seq_len, :video_seq_len],
        )
        infer_timesteps, infer_deltas = self.infer_action_scheduler.build_inference_schedule(
            num_inference_steps=num_inference_steps,
            device=self.device,
            dtype=latents_action.dtype,
            shift_override=sigma_shift,
        )
        for step_t, step_delta in zip(infer_timesteps, infer_deltas):
            timestep_action = step_t.unsqueeze(0).to(dtype=latents_action.dtype, device=self.device)
            pred_action = self._predict_action_noise_with_cache(
                latents_action=latents_action,
                timestep_action=timestep_action,
                context=context_action,
                context_mask=context_action_mask,
                video_kv_cache=video_kv_cache,
                attention_mask=attention_mask,
                video_seq_len=video_seq_len,
            )
            latents_action = self.infer_action_scheduler.step(pred_action, step_delta, latents_action)
        return {"action": latents_action[0].detach().to(device="cpu", dtype=torch.float32)}

    def infer_joint(self, *args, **kwargs):
        raise RuntimeError(
            "FastWAMSemanticHistory is action-only: use `infer_action` with semantic image and history inputs. "
            "This explicit error prevents semantic tokens from leaking into the VAE video branch."
        )

    def save_checkpoint(self, path, optimizer=None, step=None):
        payload = {
            "mot": self.mot.state_dict(),
            "step": step,
            "torch_dtype": str(self.torch_dtype),
            "semantic_history_encoder": self.semantic_history_encoder.adapter_state_dict(),
            "semantic_history_config": self.semantic_history_config,
        }
        if self.proprio_encoder is not None:
            payload["proprio_encoder"] = self.proprio_encoder.state_dict()
        if optimizer is not None:
            payload["optimizer"] = optimizer.state_dict()
        torch.save(payload, path)

    def load_checkpoint(self, path, optimizer=None):
        payload = torch.load(path, map_location="cpu")
        checkpoint_config = dict(payload.get("semantic_history_config") or {})
        if not checkpoint_config:
            raise ValueError("Checkpoint has no semantic-history adapter state for FastWAMSemanticHistory.")
        for key in ("history_offsets", "max_history_frames", "num_output_tokens", "vlm_model_name_or_path"):
            if key in checkpoint_config and key in self.semantic_history_config:
                if checkpoint_config[key] != self.semantic_history_config[key]:
                    raise ValueError(
                        f"Checkpoint semantic_history_config.{key} mismatch: "
                        f"checkpoint={checkpoint_config[key]!r}, current={self.semantic_history_config[key]!r}."
                    )
        step = super().load_checkpoint(path, optimizer=optimizer)
        if "semantic_history_encoder" not in payload:
            raise ValueError("Checkpoint is missing `semantic_history_encoder` weights.")
        self.semantic_history_encoder.load_adapter_state_dict(
            payload["semantic_history_encoder"], strict=True
        )
        return step
