"""FastWAM VAE+DINO three-branch MoT.

This opt-in variant keeps the original VAE/Wan video branch as the primary
world-model target and adds a DINO video branch as semantic auxiliary
supervision.  Action tokens see only current/first-frame VAE+DINO condition
tokens plus action tokens, never noisy future video tokens.
"""

import os
from pathlib import Path
from typing import Any, Optional, Sequence, Union
import zipfile

import torch
import torch.nn as nn
import torch.nn.functional as F

from fastwam.utils.logging_config import get_logger

from .action_dit import ActionDiT
from .dino_encoder import DinoVideoEncoder
from .dino_intent import DINOHistoryIntentResampler
from .dino_video_dit import DinoVideoDiT
from .helpers.loader import load_wan22_ti2v_5b_components
from .mot import MoT
from .schedulers.scheduler_continuous import WanContinuousFlowMatchScheduler
from .semantic_history import QwenDINOHistoryActionAdapter

logger = get_logger(__name__)

INTENT_INJECTION_MODES = {"context_after_proprio"}
DINO_FUTURE_MODES = {"predict", "condition_only"}


class FastWAMVAEDinoMoT(nn.Module):
    """Three-branch MoT: VAE video + DINO video + action.

    Expert order is fixed as ``video`` (VAE), ``dino`` (DINO), ``action`` so
    that the block attention mask has stable segment boundaries.
    """

    def __init__(
        self,
        video_expert,
        dino_video_expert: DinoVideoDiT,
        action_expert: ActionDiT,
        mot: MoT,
        vae,
        dino_encoder: DinoVideoEncoder,
        intent_encoder: Optional[DINOHistoryIntentResampler] = None,
        semantic_history_encoder: Optional[QwenDINOHistoryActionAdapter] = None,
        intent_config: Optional[dict[str, Any]] = None,
        semantic_history_config: Optional[dict[str, Any]] = None,
        text_encoder=None,
        tokenizer=None,
        text_dim: Optional[int] = None,
        proprio_dim: Optional[int] = None,
        device: str = "cpu",
        torch_dtype: torch.dtype = torch.float32,
        video_train_shift: float = 5.0,
        video_infer_shift: float = 5.0,
        video_num_train_timesteps: int = 1000,
        action_train_shift: float = 5.0,
        action_infer_shift: float = 5.0,
        action_num_train_timesteps: int = 1000,
        loss_lambda_video: float = 1.0,
        loss_lambda_dino: float = 0.02,
        loss_lambda_action: float = 1.0,
        dino_future_mode: str = "predict",
    ):
        super().__init__()
        self.video_expert = video_expert
        self.dino_video_expert = dino_video_expert
        self.action_expert = action_expert
        self.mot = mot
        self.dit = self.mot

        self.vae = vae
        self.dino_encoder = dino_encoder
        self.intent_encoder = intent_encoder
        self.semantic_history_encoder = semantic_history_encoder
        if self.intent_encoder is not None and self.semantic_history_encoder is not None:
            raise ValueError(
                "FastWAMVAEDinoMoT supports either legacy intent_config or "
                "semantic_history_config in one model, not both. Use a separate task "
                "variant for Qwen semantic-history action-only injection."
            )

        self.intent_config = dict(intent_config or {})
        self.semantic_history_config = dict(semantic_history_config or {})
        self.dino_future_mode = str(dino_future_mode).strip().lower()
        if self.dino_future_mode not in DINO_FUTURE_MODES:
            raise ValueError(
                f"Unsupported `dino_future_mode={self.dino_future_mode!r}`; "
                f"expected one of {sorted(DINO_FUTURE_MODES)}."
            )
        self.intent_source = str(self.intent_config.get("source", "dino")).strip().lower()
        if self.intent_source not in {"dino", "vae"}:
            raise ValueError(
                f"Unsupported `intent_config.source={self.intent_source!r}`; "
                "expected one of: dino, vae."
            )
        self.intent_injection_mode = str(
            self.intent_config.get("injection_mode", "context_after_proprio")
        )
        if self.intent_injection_mode not in INTENT_INJECTION_MODES:
            raise ValueError(
                f"Unsupported `intent_config.injection_mode={self.intent_injection_mode!r}` for "
                "FastWAMVAEDinoMoT. This variant currently supports only "
                "`context_after_proprio`."
            )
        if self.intent_encoder is not None:
            self.intent_config["injection_mode"] = self.intent_injection_mode
            self.intent_config["source"] = self.intent_source
        history_offsets = self.intent_config.get("history_offsets", None)
        self.intent_history_frame_count = (
            len(history_offsets)
            if self.intent_encoder is not None and history_offsets is not None
            else None
        )
        self.semantic_history_uses_history = bool(
            self.semantic_history_config.get("use_history", True)
        )
        self.semantic_history_source = str(
            self.semantic_history_config.get("source", "dino")
        ).strip().lower()
        if self.semantic_history_source not in {"dino", "vae"}:
            raise ValueError(
                f"Unsupported semantic_history_config.source={self.semantic_history_source!r}; "
                "expected one of: dino, vae."
            )
        semantic_offsets = self.semantic_history_config.get("history_offsets", None)
        self.semantic_history_frame_count = (
            len(semantic_offsets)
            if (
                self.semantic_history_encoder is not None
                and self.semantic_history_uses_history
                and semantic_offsets is not None
            )
            else None
        )
        if self.semantic_history_encoder is not None:
            self.semantic_history_config["injection_mode"] = str(
                self.semantic_history_config.get("injection_mode", "action_context_after_proprio")
            )
            self.semantic_history_config["use_history"] = self.semantic_history_uses_history
            self.semantic_history_config["source"] = self.semantic_history_source
        self.text_encoder = text_encoder
        self.tokenizer = tokenizer
        if text_dim is None:
            if self.text_encoder is None:
                raise ValueError("`text_dim` is required when `text_encoder` is not loaded.")
            text_dim = int(self.text_encoder.dim)
        self.text_dim = int(text_dim)

        self.proprio_dim = None if proprio_dim is None else int(proprio_dim)
        if self.proprio_dim is not None:
            self.proprio_encoder = nn.Linear(self.proprio_dim, self.text_dim).to(torch_dtype)
        else:
            self.proprio_encoder = None

        self.train_video_scheduler = WanContinuousFlowMatchScheduler(
            num_train_timesteps=video_num_train_timesteps,
            shift=video_train_shift,
        )
        self.infer_video_scheduler = WanContinuousFlowMatchScheduler(
            num_train_timesteps=video_num_train_timesteps,
            shift=video_infer_shift,
        )
        self.train_action_scheduler = WanContinuousFlowMatchScheduler(
            num_train_timesteps=action_num_train_timesteps,
            shift=action_train_shift,
        )
        self.infer_action_scheduler = WanContinuousFlowMatchScheduler(
            num_train_timesteps=action_num_train_timesteps,
            shift=action_infer_shift,
        )
        self.train_scheduler = self.train_video_scheduler
        self.infer_scheduler = self.infer_video_scheduler

        self.device = torch.device(device)
        self.torch_dtype = torch_dtype
        self.loss_lambda_video = float(loss_lambda_video)
        self.loss_lambda_dino = float(loss_lambda_dino)
        self.loss_lambda_action = float(loss_lambda_action)

        self.to(self.device)

    @classmethod
    def from_config(
        cls,
        dino_config: dict[str, Any],
        vae_video_dit_config: dict[str, Any],
        dino_video_dit_config: dict[str, Any],
        action_dit_config: dict[str, Any],
        device: str = "cuda",
        torch_dtype: torch.dtype = torch.bfloat16,
        model_id: str = "Wan-AI/Wan2.2-TI2V-5B",
        tokenizer_model_id: str = "Wan-AI/Wan2.1-T2V-1.3B",
        tokenizer_max_len: int = 512,
        load_text_encoder: bool = False,
        proprio_dim: Optional[int] = None,
        redirect_common_files: bool = True,
        action_dit_pretrained_path: Optional[str] = None,
        vae_video_dit_pretrained_path: Optional[str] = None,
        dino_video_dit_pretrained_path: Optional[str] = None,
        skip_dit_load_from_pretrain: bool = False,
        mot_checkpoint_mixed_attn: bool = True,
        video_train_shift: float = 5.0,
        video_infer_shift: float = 5.0,
        video_num_train_timesteps: int = 1000,
        action_train_shift: float = 5.0,
        action_infer_shift: float = 5.0,
        action_num_train_timesteps: int = 1000,
        loss_lambda_video: float = 1.0,
        loss_lambda_dino: float = 0.02,
        loss_lambda_action: float = 1.0,
        intent_config: Optional[dict[str, Any]] = None,
        semantic_history_config: Optional[dict[str, Any]] = None,
        dino_future_mode: str = "predict",
    ):
        components = load_wan22_ti2v_5b_components(
            device=device,
            torch_dtype=torch_dtype,
            model_id=model_id,
            tokenizer_model_id=tokenizer_model_id,
            tokenizer_max_len=tokenizer_max_len,
            redirect_common_files=redirect_common_files,
            dit_config=vae_video_dit_config,
            skip_dit_load_from_pretrain=(
                skip_dit_load_from_pretrain or bool(vae_video_dit_pretrained_path)
            ),
            load_text_encoder=load_text_encoder,
            load_vae=True,
        )
        video_expert = components.dit
        if video_expert is None:
            raise RuntimeError("Wan component loader returned no VAE video DiT.")
        if vae_video_dit_pretrained_path:
            video_expert.load_preprocessed_weights(str(vae_video_dit_pretrained_path))

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
            dino_encoder.load_backbone(device=torch.device(device), dtype=torch_dtype)
        else:
            logger.info(
                "DINO backbone loading disabled; training must provide `dino_latents` "
                "and inference will require enabling the backbone."
            )

        text_dim = int(dino_video_dit_config.get("text_dim", vae_video_dit_config.get("text_dim", 4096)))
        dino_video_expert = DinoVideoDiT(
            hidden_dim=dino_video_dit_config["hidden_dim"],
            dino_dim=dino_config.get("feature_dim", 1024),
            ffn_dim=dino_video_dit_config["ffn_dim"],
            text_dim=text_dim,
            freq_dim=dino_video_dit_config.get("freq_dim", 256),
            eps=dino_video_dit_config.get("eps", 1e-6),
            num_heads=dino_video_dit_config["num_heads"],
            attn_head_dim=dino_video_dit_config["attn_head_dim"],
            num_layers=dino_video_dit_config["num_layers"],
            video_attention_mask_mode=dino_video_dit_config.get(
                "video_attention_mask_mode", "first_frame_causal"
            ),
            use_gradient_checkpointing=dino_video_dit_config.get("use_gradient_checkpointing", True),
            latent_patch_size=tuple(dino_video_dit_config.get("latent_patch_size", [1, 1, 1])),
            latent_patch_mode=dino_video_dit_config.get("latent_patch_mode", "flat"),
            latent_num_views=dino_video_dit_config.get("latent_num_views", 1),
            output_patch_space=dino_video_dit_config.get("output_patch_space", "dense"),
        ).to(device=device, dtype=torch_dtype)
        if dino_video_dit_pretrained_path:
            dino_video_expert.load_preprocessed_backbone(str(dino_video_dit_pretrained_path))

        intent_encoder = None
        intent_config = dict(intent_config or {})
        if bool(intent_config.get("enabled", False)):
            injection_mode = str(intent_config.get("injection_mode", "context_after_proprio"))
            if injection_mode not in INTENT_INJECTION_MODES:
                raise ValueError(
                    f"Unsupported `intent_config.injection_mode={injection_mode!r}` for "
                    "FastWAMVAEDinoMoT. This variant currently supports only "
                    "`context_after_proprio`."
                )
            intent_config["injection_mode"] = injection_mode
            intent_source = str(intent_config.get("source", "dino")).strip().lower()
            if intent_source not in {"dino", "vae"}:
                raise ValueError(
                    f"Unsupported `intent_config.source={intent_source!r}` for FastWAMVAEDinoMoT; "
                    "expected one of: dino, vae."
                )
            intent_config["source"] = intent_source
            history_offsets = intent_config.get(
                "history_offsets",
                intent_config.get("history_dino_frame_offsets", [-8, -4, 0]),
            )
            history_offsets = [int(offset) for offset in history_offsets]
            if not history_offsets:
                raise ValueError("`intent_config.history_offsets` must contain at least one frame offset.")
            intent_config["history_offsets"] = history_offsets
            max_history_frames = int(
                intent_config.get("max_history_frames", max(1, len(history_offsets)))
            )
            if max_history_frames < len(history_offsets):
                raise ValueError(
                    "`intent_config.max_history_frames` must be >= len(history_offsets), "
                    f"got {max_history_frames} vs {len(history_offsets)}."
                )
            intent_config["max_history_frames"] = max_history_frames
            intent_input_dim = (
                int(getattr(components.vae, "z_dim", 48))
                if intent_source == "vae"
                else int(dino_config.get("feature_dim", 1024))
            )
            intent_encoder = DINOHistoryIntentResampler(
                dino_dim=intent_input_dim,
                text_dim=text_dim,
                num_intent_tokens=int(intent_config.get("num_intent_tokens", 8)),
                resampler_dim=int(intent_config.get("resampler_dim", 1024)),
                num_layers=int(intent_config.get("num_resampler_layers", 2)),
                num_heads=int(intent_config.get("num_heads", 8)),
                max_history_frames=max_history_frames,
                mlp_ratio=float(intent_config.get("mlp_ratio", 4.0)),
                dropout=float(intent_config.get("dropout", 0.0)),
            ).to(device=device, dtype=torch_dtype)

        semantic_history_encoder = None
        semantic_history_config = dict(semantic_history_config or {})
        if bool(semantic_history_config.get("enabled", False)):
            if intent_encoder is not None:
                raise ValueError(
                    "`semantic_history_config.enabled=true` cannot be combined with "
                    "`intent_config.enabled=true` in the same FastWAMVAEDinoMoT model."
                )
            injection_mode = str(
                semantic_history_config.get("injection_mode", "action_context_after_proprio")
            )
            if injection_mode != "action_context_after_proprio":
                raise ValueError(
                    "Unsupported `semantic_history_config.injection_mode`: "
                    f"{injection_mode!r}. Expected 'action_context_after_proprio'."
                )
            semantic_history_config["injection_mode"] = injection_mode
            use_history = bool(semantic_history_config.get("use_history", True))
            semantic_history_config["use_history"] = use_history
            history_source = str(semantic_history_config.get("source", "dino")).strip().lower()
            if history_source not in {"dino", "vae"}:
                raise ValueError(
                    f"Unsupported semantic_history_config.source={history_source!r}; "
                    "expected one of: dino, vae."
                )
            semantic_history_config["source"] = history_source
            history_offsets = semantic_history_config.get(
                "history_offsets", [-24, -16, -8, -1] if use_history else []
            )
            history_offsets = [int(offset) for offset in history_offsets]
            if use_history and not history_offsets:
                raise ValueError("`semantic_history_config.history_offsets` must contain at least one offset.")
            semantic_history_config["history_offsets"] = history_offsets
            max_history_frames = int(
                semantic_history_config.get("max_history_frames", max(1, len(history_offsets)))
            )
            semantic_history_config["max_history_frames"] = max_history_frames
            history_input_dim = (
                int(getattr(components.vae, "z_dim", 48))
                if history_source == "vae"
                else int(dino_config.get("feature_dim", 1024))
            )
            semantic_history_encoder = QwenDINOHistoryActionAdapter(
                vlm_model_name_or_path=str(semantic_history_config["vlm_model_name_or_path"]),
                vlm_family=str(semantic_history_config.get("vlm_family", "qwen3_vl")),
                trust_remote_code=bool(semantic_history_config.get("trust_remote_code", True)),
                freeze_vlm=bool(semantic_history_config.get("freeze_vlm", True)),
                processor_min_pixels=semantic_history_config.get("processor_min_pixels", None),
                processor_max_pixels=semantic_history_config.get("processor_max_pixels", None),
                attn_implementation=semantic_history_config.get("attn_implementation", None),
                dino_dim=history_input_dim,
                text_dim=text_dim,
                history_offsets=history_offsets,
                max_history_frames=max_history_frames,
                use_history=use_history,
                num_output_tokens=int(semantic_history_config.get("num_output_tokens", 8)),
                resampler_dim=int(semantic_history_config.get("resampler_dim", 1024)),
                num_layers=int(semantic_history_config.get("num_resampler_layers", 2)),
                num_heads=int(semantic_history_config.get("num_heads", 8)),
                dropout=float(semantic_history_config.get("dropout", 0.0)),
                torch_dtype=torch_dtype,
                history_source=history_source,
            ).to(device=device, dtype=torch_dtype)

        action_expert = ActionDiT.from_pretrained(
            action_dit_config=action_dit_config,
            action_dit_pretrained_path=action_dit_pretrained_path,
            skip_dit_load_from_pretrain=skip_dit_load_from_pretrain,
            device=device,
            torch_dtype=torch_dtype,
        )

        for name, expert in (("dino", dino_video_expert), ("action", action_expert)):
            if int(expert.num_heads) != int(video_expert.num_heads):
                raise ValueError(f"{name} expert `num_heads` must match VAE video expert.")
            if int(expert.attn_head_dim) != int(video_expert.attn_head_dim):
                raise ValueError(f"{name} expert `attn_head_dim` must match VAE video expert.")
            if int(len(expert.blocks)) != int(len(video_expert.blocks)):
                raise ValueError(f"{name} expert `num_layers` must match VAE video expert.")

        mot = MoT(
            mixtures={
                "video": video_expert,
                "dino": dino_video_expert,
                "action": action_expert,
            },
            mot_checkpoint_mixed_attn=mot_checkpoint_mixed_attn,
        )

        model = cls(
            video_expert=video_expert,
            dino_video_expert=dino_video_expert,
            action_expert=action_expert,
            mot=mot,
            vae=components.vae,
            dino_encoder=dino_encoder,
            intent_encoder=intent_encoder,
            semantic_history_encoder=semantic_history_encoder,
            intent_config=intent_config,
            semantic_history_config=semantic_history_config,
            text_encoder=components.text_encoder,
            tokenizer=components.tokenizer,
            text_dim=text_dim,
            proprio_dim=proprio_dim,
            device=device,
            torch_dtype=torch_dtype,
            video_train_shift=video_train_shift,
            video_infer_shift=video_infer_shift,
            video_num_train_timesteps=video_num_train_timesteps,
            action_train_shift=action_train_shift,
            action_infer_shift=action_infer_shift,
            action_num_train_timesteps=action_num_train_timesteps,
            loss_lambda_video=loss_lambda_video,
            loss_lambda_dino=loss_lambda_dino,
            loss_lambda_action=loss_lambda_action,
            dino_future_mode=dino_future_mode,
        )
        model.model_paths = {
            "vae_video_dit": components.dit_path,
            "vae_video_dit_preprocessed": vae_video_dit_pretrained_path,
            "dino_video_dit_preprocessed": dino_video_dit_pretrained_path,
            "vae": components.vae_path,
            "text_encoder": components.text_encoder_path,
            "tokenizer": components.tokenizer_path,
            "action_dit_backbone": (
                "SKIPPED_PRETRAIN" if skip_dit_load_from_pretrain else action_dit_pretrained_path
            ),
        }
        return model

    def to(self, *args, **kwargs):
        super().to(*args, **kwargs)
        self.mot.to(*args, **kwargs)
        if self.text_encoder is not None:
            self.text_encoder.to(*args, **kwargs)
        self.vae.to(*args, **kwargs)
        if self.dino_encoder is not None and self.dino_encoder._loaded:
            self.dino_encoder.backbone.to(*args, **kwargs)
        if self.semantic_history_encoder is not None:
            self.semantic_history_encoder.to(*args, **kwargs)
        return self

    @torch.no_grad()
    def encode_prompt(self, prompt: Union[str, Sequence[str]]):
        if self.text_encoder is None or self.tokenizer is None:
            raise ValueError(
                "Prompt encoding requires loaded text encoder/tokenizer. "
                "Set `load_text_encoder=true` or provide cached `context/context_mask`."
            )
        ids, mask = self.tokenizer(prompt, return_mask=True, add_special_tokens=True)
        ids = ids.to(self.device)
        mask = mask.to(self.device, dtype=torch.bool)
        prompt_emb = self.text_encoder(ids, mask)
        seq_lens = mask.gt(0).sum(dim=1).long()
        for i, v in enumerate(seq_lens):
            prompt_emb[i, v:] = 0
        mask = torch.ones_like(mask)
        return prompt_emb.to(device=self.device), mask

    def _append_proprio_to_context(
        self,
        context: torch.Tensor,
        context_mask: torch.Tensor,
        proprio: Optional[torch.Tensor],
    ) -> tuple[torch.Tensor, torch.Tensor]:
        if self.proprio_encoder is None or proprio is None:
            return context, context_mask
        if proprio.ndim != 2:
            raise ValueError(f"`proprio` must be 2D [B, D], got shape {tuple(proprio.shape)}")
        if self.proprio_dim is None or proprio.shape[1] != self.proprio_dim:
            raise ValueError(
                f"`proprio` last dim must be {self.proprio_dim}, got {proprio.shape[1]}"
            )
        proprio_token = self.proprio_encoder(
            proprio.to(device=self.device, dtype=context.dtype).unsqueeze(1)
        ).to(dtype=context.dtype)
        proprio_mask = torch.ones((context_mask.shape[0], 1), dtype=torch.bool, device=context_mask.device)
        return (
            torch.cat([context, proprio_token], dim=1),
            torch.cat([context_mask, proprio_mask], dim=1),
        )

    def _append_intent_to_context(
        self,
        context: torch.Tensor,
        context_mask: torch.Tensor,
        intent_tokens: Optional[torch.Tensor],
    ) -> tuple[torch.Tensor, torch.Tensor]:
        if intent_tokens is None:
            return context, context_mask
        if intent_tokens.ndim != 3:
            raise ValueError(
                f"`intent_tokens` must be [B,K,D], got shape {tuple(intent_tokens.shape)}"
            )
        if intent_tokens.shape[0] != context.shape[0]:
            raise ValueError(
                f"Batch mismatch between context and intent tokens: "
                f"{context.shape[0]} vs {intent_tokens.shape[0]}"
            )
        if intent_tokens.shape[-1] != context.shape[-1]:
            raise ValueError(
                f"Intent token dim {intent_tokens.shape[-1]} must match context dim {context.shape[-1]}"
            )
        intent_tokens = intent_tokens.to(device=context.device, dtype=context.dtype)
        intent_mask = torch.ones(
            (context_mask.shape[0], intent_tokens.shape[1]),
            dtype=torch.bool,
            device=context_mask.device,
        )
        return (
            torch.cat([context, intent_tokens], dim=1),
            torch.cat([context_mask, intent_mask], dim=1),
        )

    def _encode_history_intent(self, history_latents: torch.Tensor) -> torch.Tensor:
        if self.intent_encoder is None:
            raise ValueError("`intent_encoder` is not enabled for this model.")
        if history_latents.ndim != 5:
            raise ValueError(
                f"History intent latents must be [B,D,T,H,W], got {tuple(history_latents.shape)}"
            )
        if (
            self.intent_history_frame_count is not None
            and history_latents.shape[2] != self.intent_history_frame_count
        ):
            raise ValueError(
                "History intent temporal length does not match `intent_config.history_offsets`: "
                f"got T={history_latents.shape[2]}, expected {self.intent_history_frame_count}."
            )
        if self.intent_source == "vae":
            self._validate_vae_history_latent_shape(history_latents)
        else:
            self._validate_dino_latent_shape(history_latents)
        history_latents = history_latents.to(
            device=self.device,
            dtype=self.torch_dtype,
            non_blocking=True,
        )
        return self.intent_encoder(history_latents)

    def _encode_semantic_history_tokens(
        self,
        *,
        prompts,
        semantic_image: torch.Tensor,
        history_dino_latents: Optional[torch.Tensor] = None,
        history_vae_latents: Optional[torch.Tensor] = None,
        history_video: Optional[torch.Tensor] = None,
        tiled: bool = False,
    ) -> torch.Tensor:
        if self.semantic_history_encoder is None:
            raise ValueError("`semantic_history_encoder` is not enabled for this model.")
        if prompts is None:
            raise ValueError(
                "`prompt`/`semantic_prompt` is required when semantic_history_config.enabled=true. "
                "Cached T5 context alone is not enough for Qwen-VL semantic encoding."
            )
        if not torch.is_tensor(semantic_image):
            raise TypeError(f"`semantic_image` must be a tensor, got {type(semantic_image)}")
        if semantic_image.ndim == 3:
            semantic_image = semantic_image.unsqueeze(0)
        if semantic_image.ndim != 4 or semantic_image.shape[1] != 3:
            raise ValueError(
                f"`semantic_image` must be [B,3,H,W] or [3,H,W], got {tuple(semantic_image.shape)}"
            )
        batch_size = int(semantic_image.shape[0])
        if not self.semantic_history_uses_history:
            if history_dino_latents is not None or history_vae_latents is not None or history_video is not None:
                raise ValueError(
                    "This semantic adapter has use_history=false, so history inputs must be omitted."
                )
            return self.semantic_history_encoder(
                prompts=prompts,
                semantic_image=semantic_image,
                history_dino_latents=None,
            )
        if history_dino_latents is not None and history_vae_latents is not None:
            raise ValueError("Provide only one of `history_dino_latents` or `history_vae_latents`.")
        history_latents = (
            history_vae_latents if self.semantic_history_source == "vae" else history_dino_latents
        )
        wrong_history_latents = (
            history_dino_latents if self.semantic_history_source == "vae" else history_vae_latents
        )
        if wrong_history_latents is not None:
            raise ValueError(
                f"semantic_history_config.source={self.semantic_history_source!r} received "
                "latents from the other history encoder."
            )
        if history_latents is not None and history_video is not None:
            raise ValueError("Provide only cached history latents or `history_video`, not both.")
        if history_latents is None:
            if history_video is None:
                raise ValueError(
                    "History latents or `history_video` are required when "
                    "semantic_history_config.enabled=true."
                )
            if history_video.ndim == 4:
                if history_video.shape[0] == 3:
                    history_video = history_video.unsqueeze(0).contiguous()
                elif history_video.shape[1] == 3:
                    history_video = history_video.permute(1, 0, 2, 3).unsqueeze(0).contiguous()
                else:
                    raise ValueError(
                        f"`history_video` must be [3,T,H,W] or [T,3,H,W], got {tuple(history_video.shape)}"
                    )
            elif history_video.ndim == 5:
                if history_video.shape[1] == 3:
                    pass
                elif history_video.shape[2] == 3:
                    history_video = history_video.permute(0, 2, 1, 3, 4).contiguous()
                else:
                    raise ValueError(
                        f"`history_video` must be [B,3,T,H,W] or [B,T,3,H,W], got {tuple(history_video.shape)}"
                    )
            else:
                raise ValueError(f"`history_video` must be 4D or 5D, got {tuple(history_video.shape)}")
            if history_video.shape[0] != batch_size:
                raise ValueError(
                    f"Batch mismatch between semantic_image and history_video: "
                    f"{batch_size} vs {history_video.shape[0]}"
                )
            if self.semantic_history_source == "vae":
                history_latents = self._encode_history_vae_video(history_video, tiled=tiled)
            else:
                if not bool(getattr(self.dino_encoder, "_loaded", False)):
                    raise ValueError(
                        "Online semantic-history DINO encoding requires `model.dino_config.load_backbone=true`."
                    )
                history_latents = self._encode_video_dino(
                    history_video.to(device=self.device, dtype=self.torch_dtype, non_blocking=True)
                )
        elif history_latents.ndim == 4:
            history_latents = history_latents.unsqueeze(0)
        if history_latents.ndim != 5:
            raise ValueError(
                f"History latents must be 5D [B,D,T,H,W], got {tuple(history_latents.shape)}"
            )
        if history_latents.shape[0] != batch_size:
            raise ValueError(
                f"Batch mismatch between semantic_image and history latents: "
                f"{batch_size} vs {history_latents.shape[0]}"
            )
        if (
            self.semantic_history_frame_count is not None
            and history_latents.shape[2] != self.semantic_history_frame_count
        ):
            raise ValueError(
                "Semantic history temporal length does not match semantic_history_config.history_offsets: "
                f"got T={history_latents.shape[2]}, expected {self.semantic_history_frame_count}."
            )
        if self.semantic_history_source == "vae":
            self._validate_vae_history_latent_shape(history_latents)
        else:
            self._validate_dino_latent_shape(history_latents)
        return self.semantic_history_encoder(
            prompts=prompts,
            semantic_image=semantic_image,
            history_dino_latents=(
                history_latents.to(device=self.device, dtype=self.torch_dtype, non_blocking=True)
                if self.semantic_history_source == "dino"
                else None
            ),
            history_vae_latents=(
                history_latents.to(device=self.device, dtype=self.torch_dtype, non_blocking=True)
                if self.semantic_history_source == "vae"
                else None
            ),
        )

    @torch.no_grad()
    def _encode_video_latents(self, video_tensor, tiled=False, tile_size=(30, 52), tile_stride=(15, 26)):
        return self.vae.encode(
            video_tensor,
            device=self.device,
            tiled=tiled,
            tile_size=tile_size,
            tile_stride=tile_stride,
        )

    def _validate_vae_latent_shape(
        self,
        latents: torch.Tensor,
        *,
        video_frame_count: Optional[int] = None,
    ) -> None:
        if latents.ndim != 5:
            raise ValueError(f"VAE latents must be 5D [B,C,T,H,W], got shape {tuple(latents.shape)}")
        expected_dim = getattr(self.vae, "z_dim", None)
        if expected_dim is None:
            vae_model = getattr(self.vae, "model", None)
            expected_dim = getattr(vae_model, "z_dim", latents.shape[1])
        expected_dim = int(expected_dim)
        if latents.shape[1] != expected_dim:
            raise ValueError(
                f"VAE latent channel mismatch: cache C={latents.shape[1]}, model C={expected_dim}."
            )
        if video_frame_count is not None:
            temporal_factor = int(self.vae.temporal_downsample_factor)
            if (int(video_frame_count) - 1) % temporal_factor != 0:
                raise ValueError(
                    "Selected video frame count is incompatible with Wan VAE latent downsampling: "
                    f"T={video_frame_count}, temporal_downsample_factor={temporal_factor}."
                )
            expected_t = (int(video_frame_count) - 1) // temporal_factor + 1
            if latents.shape[2] != expected_t:
                raise ValueError(
                    f"VAE latent temporal mismatch: cache T={latents.shape[2]}, expected T={expected_t} "
                    f"from selected video T={video_frame_count}."
                )
        if self.dino_encoder is not None:
            input_h, input_w = self.dino_encoder.input_resolution
            vae_factor = int(getattr(self.vae, "upsampling_factor", 16))
            expected_hw = (input_h // vae_factor, input_w // vae_factor)
            if tuple(latents.shape[-2:]) != expected_hw:
                raise ValueError(
                    "VAE latent spatial shape mismatch: "
                    f"cache={(latents.shape[-2], latents.shape[-1])}, expected={expected_hw}."
                )

    def _validate_vae_history_latent_shape(self, latents: torch.Tensor) -> None:
        self._validate_vae_latent_shape(latents, video_frame_count=None)

    @torch.no_grad()
    def _encode_history_vae_video(self, history_video: torch.Tensor, tiled: bool = False) -> torch.Tensor:
        if history_video.ndim != 5:
            raise ValueError(
                f"`history_video` must be [B,3,T,H,W] for VAE intent, got {tuple(history_video.shape)}"
            )
        if history_video.shape[1] != 3:
            raise ValueError(f"`history_video` channel dim must be 3, got {history_video.shape[1]}")
        latents = []
        history_video = history_video.to(device=self.device, dtype=self.torch_dtype, non_blocking=True)
        for frame_idx in range(int(history_video.shape[2])):
            frame_video = history_video[:, :, frame_idx:frame_idx + 1]
            frame_latents = self._encode_video_latents(frame_video, tiled=tiled)
            self._validate_vae_latent_shape(frame_latents, video_frame_count=1)
            if frame_latents.shape[2] != 1:
                raise ValueError(
                    f"Expected single-frame VAE encode to produce latent T=1, got {frame_latents.shape[2]}"
                )
            latents.append(frame_latents[:, :, 0])
        if not latents:
            raise ValueError("`history_video` must contain at least one frame for VAE intent.")
        history_latents = torch.stack(latents, dim=2).contiguous()
        self._validate_vae_history_latent_shape(history_latents)
        return history_latents

    @torch.no_grad()
    def _encode_input_image_latents_tensor(
        self,
        input_image: torch.Tensor,
        tiled=False,
        tile_size=(30, 52),
        tile_stride=(15, 26),
    ):
        if input_image.ndim == 3:
            input_image = input_image.unsqueeze(0)
        if input_image.ndim != 4 or input_image.shape[0] != 1 or input_image.shape[1] != 3:
            raise ValueError(
                f"`input_image` must have shape [1,3,H,W] or [3,H,W], got {tuple(input_image.shape)}"
            )
        image = input_image.to(device=self.device)[0].unsqueeze(1)
        z = self.vae.encode([image], device=self.device, tiled=tiled, tile_size=tile_size, tile_stride=tile_stride)
        if isinstance(z, list):
            z = z[0].unsqueeze(0)
        return z

    @torch.no_grad()
    def _encode_video_dino(self, video_tensor: torch.Tensor) -> torch.Tensor:
        return self.dino_encoder.encode_video_to_latent(video_tensor)

    @torch.no_grad()
    def _encode_single_frame_dino(self, image: torch.Tensor) -> torch.Tensor:
        if image.ndim == 3:
            image = image.unsqueeze(0)
        return self._encode_video_dino(image.unsqueeze(2))

    def _validate_dino_latent_shape(self, latents: torch.Tensor) -> None:
        if latents.ndim != 5:
            raise ValueError(f"DINO latents must be 5D [B,D,T,H,W], got shape {tuple(latents.shape)}")
        expected_dim = int(getattr(self.dino_video_expert, "dino_dim", latents.shape[1]))
        if latents.shape[1] != expected_dim:
            raise ValueError(
                f"DINO latent channel mismatch: cache D={latents.shape[1]}, model D={expected_dim}."
            )
        if self.dino_encoder is None:
            return
        input_h, input_w = self.dino_encoder.input_resolution
        patch_size = int(self.dino_encoder.patch_size)
        pool_h, pool_w = self.dino_encoder.latent_spatial_pool
        expected_h = (input_h // patch_size) // pool_h
        expected_w = (input_w // patch_size) // pool_w
        if tuple(latents.shape[-2:]) != (expected_h, expected_w):
            raise ValueError(
                "DINO latent spatial shape mismatch: "
                f"cache={(latents.shape[-2], latents.shape[-1])}, expected={(expected_h, expected_w)}."
            )

    def build_inputs(self, sample: dict, tiled: bool = False) -> dict[str, Any]:
        if "context" not in sample or "context_mask" not in sample:
            raise ValueError("FastWAMVAEDinoMoT training requires `context` and `context_mask`.")
        if "action" not in sample:
            raise ValueError("FastWAMVAEDinoMoT training requires `action`.")

        video = sample.get("video", None)
        vae_latents = sample.get("vae_latents", None)
        if video is None and vae_latents is None:
            raise ValueError(
                "FastWAMVAEDinoMoT training requires either RGB `video` or cached `vae_latents`."
            )
        action = sample["action"]
        context = sample["context"]
        context_mask = sample["context_mask"]
        proprio = sample.get("proprio", None)
        action_is_pad = sample.get("action_is_pad", None)
        image_is_pad = sample.get("image_is_pad", None)
        history_dino_latents = sample.get("history_dino_latents", None)
        history_vae_latents = sample.get("history_vae_latents", None)
        history_video = sample.get("history_video", None)
        semantic_image = sample.get("semantic_image", sample.get("vlm_image", None))
        semantic_prompt = sample.get("semantic_prompt", sample.get("prompt", None))

        if self.intent_encoder is None and self.semantic_history_encoder is None and (
            history_dino_latents is not None
            or history_vae_latents is not None
            or history_video is not None
        ):
            raise ValueError(
                "FastWAMVAEDinoMoT does not consume short history tokens. "
                "Disable history inputs for this config, or enable "
                "`model.intent_config.enabled=true` or `model.semantic_history_config.enabled=true`."
            )

        if action.ndim != 3:
            raise ValueError(f"`action` must be 3D [B,T,A], got shape {tuple(action.shape)}")
        batch_size = action.shape[0]
        temporal_factor = int(self.vae.temporal_downsample_factor)
        if temporal_factor <= 0:
            raise ValueError(f"`vae.temporal_downsample_factor` must be positive, got {temporal_factor}.")

        input_video = None
        video_frame_count = None
        if video is not None:
            if video.ndim != 5 or video.shape[1] != 3:
                raise ValueError(f"`video` must be 5D [B,3,T,H,W], got shape {tuple(video.shape)}")
            if video.shape[0] != batch_size:
                raise ValueError(f"Batch mismatch between video and action: {video.shape[0]} vs {batch_size}")
            if video.shape[2] <= 1:
                raise ValueError(f"`video` must include f0 plus future frames, got T={video.shape[2]}")
            video_frame_count = int(video.shape[2])
            input_video = video.to(device=self.device, dtype=self.torch_dtype, non_blocking=True)

        if "dino_latents" in sample and sample["dino_latents"] is not None:
            input_dino_latents = sample["dino_latents"]
            if input_dino_latents.ndim != 5:
                raise ValueError(
                    f"`dino_latents` must be 5D [B,D,T,H,W], got shape {tuple(input_dino_latents.shape)}"
                )
            if input_dino_latents.shape[0] != batch_size:
                raise ValueError(
                    f"Batch mismatch between `dino_latents` and action: {input_dino_latents.shape[0]} vs {batch_size}"
                )
            self._validate_dino_latent_shape(input_dino_latents)
            input_dino_latents = input_dino_latents.to(
                device=self.device,
                dtype=self.torch_dtype,
                non_blocking=True,
            )
            if video_frame_count is None:
                video_frame_count = int(input_dino_latents.shape[2])
        else:
            if input_video is None:
                raise ValueError(
                    "Missing `dino_latents` while RGB `video` is omitted. "
                    "Latent-only training requires cached DINO latents."
                )
            if not bool(getattr(self.dino_encoder, "_loaded", False)):
                raise ValueError(
                    "Missing `dino_latents` while DINO backbone loading is disabled. "
                    "Set `data.train.dino_latent_cache_dir`/`data.val.dino_latent_cache_dir` "
                    "to a valid frame or frame_mmap cache, or set `model.dino_config.load_backbone=true`."
                )
            input_dino_latents = self._encode_video_dino(input_video)
            self._validate_dino_latent_shape(input_dino_latents)
            video_frame_count = int(input_dino_latents.shape[2])

        if video_frame_count is None:
            raise ValueError("Unable to infer selected video frame count from RGB video or DINO latents.")
        if video_frame_count <= 1:
            raise ValueError(f"`video` must include f0 plus future frames, got T={video_frame_count}")
        if (video_frame_count - 1) % temporal_factor != 0:
            raise ValueError(
                "Selected video frame count is incompatible with Wan VAE latent downsampling: "
                f"T={video_frame_count}, temporal_downsample_factor={temporal_factor}. "
                "Expected (T - 1) % temporal_downsample_factor == 0."
            )
        if action.shape[1] % (video_frame_count - 1) != 0:
            raise ValueError(
                "`action` temporal dimension must be divisible by video transitions: "
                f"action_T={action.shape[1]}, video_T={video_frame_count}."
            )

        if vae_latents is not None:
            input_vae_latents = vae_latents
            if input_vae_latents.ndim == 4:
                input_vae_latents = input_vae_latents.unsqueeze(0)
            if input_vae_latents.ndim != 5:
                raise ValueError(
                    f"`vae_latents` must be 5D [B,C,T,H,W], got shape {tuple(input_vae_latents.shape)}"
                )
            if input_vae_latents.shape[0] != batch_size:
                raise ValueError(
                    f"Batch mismatch between `vae_latents` and action: {input_vae_latents.shape[0]} vs {batch_size}"
                )
            self._validate_vae_latent_shape(input_vae_latents, video_frame_count=video_frame_count)
            input_vae_latents = input_vae_latents.to(
                device=self.device,
                dtype=self.torch_dtype,
                non_blocking=True,
            )
        else:
            if input_video is None:
                raise ValueError("Missing cached `vae_latents` while RGB `video` is omitted.")
            input_vae_latents = self._encode_video_latents(input_video, tiled=tiled)
            self._validate_vae_latent_shape(input_vae_latents, video_frame_count=video_frame_count)

        if input_dino_latents.shape[2] != video_frame_count:
            raise ValueError(
                "DINO latent temporal length must match selected video frame count for the three-branch mask, "
                f"got dino_T={input_dino_latents.shape[2]}, video_T={video_frame_count}."
            )

        if context.ndim != 3 or context_mask.ndim != 2:
            raise ValueError(
                f"`context/context_mask` must be [B,L,D]/[B,L], got {tuple(context.shape)} and {tuple(context_mask.shape)}"
            )
        context = context.to(device=self.device, dtype=self.torch_dtype, non_blocking=True)
        context_mask = context_mask.to(device=self.device, dtype=torch.bool, non_blocking=True)

        if self.proprio_encoder is not None:
            if proprio is None:
                raise ValueError("`proprio` is required when `proprio_dim` is enabled.")
            if proprio.ndim != 3:
                raise ValueError(f"`proprio` must be 3D [B,T,D], got shape {tuple(proprio.shape)}")
            proprio = proprio[:, 0, :]
            context, context_mask = self._append_proprio_to_context(
                context=context,
                context_mask=context_mask,
                proprio=proprio.to(device=self.device, dtype=self.torch_dtype),
            )

        context_video = context
        context_video_mask = context_mask
        context_dino = context
        context_dino_mask = context_mask
        context_action = context
        context_action_mask = context_mask

        if self.intent_encoder is not None:
            if self.intent_source == "vae":
                if history_dino_latents is not None:
                    raise ValueError(
                        "`history_dino_latents` was provided but `intent_config.source='vae'`. "
                        "Disable `data.train.load_history_dino_latents` for this config."
                    )
                if history_vae_latents is not None and history_video is not None:
                    raise ValueError("Provide only one of `history_vae_latents` or `history_video`.")
                if history_vae_latents is None:
                    if history_video is None:
                        raise ValueError(
                            "`history_video` or `history_vae_latents` is required when "
                            "`intent_config.enabled=true` and `intent_config.source='vae'`. "
                            "Enable `data.train.load_history_vae_video=true`."
                        )
                    if history_video.ndim != 5:
                        raise ValueError(
                            "`history_video` must be [B,3,T,H,W] when used for VAE intent training, "
                            f"got {tuple(history_video.shape)}"
                        )
                    if history_video.shape[0] != batch_size:
                        raise ValueError(
                            f"Batch mismatch between `history_video` and action batch: "
                            f"{history_video.shape[0]} vs {batch_size}"
                        )
                    history_vae_latents = self._encode_history_vae_video(history_video, tiled=tiled)
                elif history_vae_latents.ndim == 4:
                    history_vae_latents = history_vae_latents.unsqueeze(0)
                if history_vae_latents.ndim != 5:
                    raise ValueError(
                        f"`history_vae_latents` must be 5D [B,C,T,H,W], got "
                        f"{tuple(history_vae_latents.shape)}"
                    )
                if history_vae_latents.shape[0] != batch_size:
                    raise ValueError(
                        f"Batch mismatch between `history_vae_latents` and action batch: "
                        f"{history_vae_latents.shape[0]} vs {batch_size}"
                    )
                intent_tokens = self._encode_history_intent(history_vae_latents)
            else:
                if history_vae_latents is not None:
                    raise ValueError(
                        "`history_vae_latents` was provided but `intent_config.source='dino'`."
                    )
                if history_dino_latents is not None and history_video is not None:
                    raise ValueError("Provide only one of `history_dino_latents` or `history_video`.")
                if history_dino_latents is None:
                    if history_video is None:
                        raise ValueError(
                            "`history_dino_latents` or `history_video` is required when "
                            "`intent_config.enabled=true` and `intent_config.source='dino'` "
                            "for 3-MOT training. Enable `data.train.load_history_dino_latents=true` "
                            "for cached history DINO latents, or `data.train.load_history_dino_video=true` "
                            "for online history DINO encoding."
                        )
                    if history_video.ndim != 5:
                        raise ValueError(
                            "`history_video` must be [B,3,T,H,W] when used for training, "
                            f"got {tuple(history_video.shape)}"
                        )
                    if history_video.shape[0] != batch_size:
                        raise ValueError(
                            f"Batch mismatch between `history_video` and action batch: "
                            f"{history_video.shape[0]} vs {batch_size}"
                        )
                    if not bool(getattr(self.dino_encoder, "_loaded", False)):
                        raise ValueError(
                            "Online DINO intent history encoding requires "
                            "`model.dino_config.load_backbone=true`."
                        )
                    history_dino_latents = self._encode_video_dino(
                        history_video.to(device=self.device, dtype=self.torch_dtype, non_blocking=True)
                    )
                elif history_dino_latents.ndim == 4:
                    history_dino_latents = history_dino_latents.unsqueeze(0)
                if history_dino_latents.ndim != 5:
                    raise ValueError(
                        f"`history_dino_latents` must be 5D [B,D,T,H,W], got "
                        f"{tuple(history_dino_latents.shape)}"
                    )
                if history_dino_latents.shape[0] != batch_size:
                    raise ValueError(
                        f"Batch mismatch between `history_dino_latents` and action batch: "
                        f"{history_dino_latents.shape[0]} vs {batch_size}"
                    )
                intent_tokens = self._encode_history_intent(history_dino_latents)

            context, context_mask = self._append_intent_to_context(
                context=context,
                context_mask=context_mask,
                intent_tokens=intent_tokens,
            )
            context_video = context
            context_video_mask = context_mask
            context_dino = context
            context_dino_mask = context_mask
            context_action = context
            context_action_mask = context_mask

        if self.semantic_history_encoder is not None:
            if semantic_image is None:
                raise ValueError(
                    "`semantic_image` is required when `model.semantic_history_config.enabled=true`. "
                    "Set `data.train.load_semantic_image=true`."
                )
            semantic_tokens = self._encode_semantic_history_tokens(
                prompts=semantic_prompt,
                semantic_image=semantic_image,
                history_dino_latents=history_dino_latents,
                history_vae_latents=history_vae_latents,
                history_video=history_video,
                tiled=tiled,
            )
            context_action, context_action_mask = self._append_intent_to_context(
                context=context,
                context_mask=context_mask,
                intent_tokens=semantic_tokens,
            )

        action = action.to(device=self.device, dtype=self.torch_dtype, non_blocking=True)
        if action_is_pad is not None:
            action_is_pad = action_is_pad.to(device=self.device, dtype=torch.bool, non_blocking=True)
        if image_is_pad is not None:
            image_is_pad = image_is_pad.to(device=self.device, dtype=torch.bool, non_blocking=True)
            if image_is_pad.ndim != 2 or image_is_pad.shape[0] != batch_size:
                raise ValueError(
                    f"`image_is_pad` must be [B,T_video], got shape {tuple(image_is_pad.shape)}"
                )
            if image_is_pad.shape[1] != video_frame_count:
                raise ValueError(
                    f"`image_is_pad` temporal length must match selected video frame count: "
                    f"mask_T={image_is_pad.shape[1]}, video_T={video_frame_count}."
                )

        return {
            "context": context_action,
            "context_mask": context_action_mask,
            "context_video": context_video,
            "context_video_mask": context_video_mask,
            "context_dino": context_dino,
            "context_dino_mask": context_dino_mask,
            "context_action": context_action,
            "context_action_mask": context_action_mask,
            "input_vae_latents": input_vae_latents,
            "input_dino_latents": input_dino_latents,
            "first_frame_vae_latents": input_vae_latents[:, :, 0:1],
            "first_frame_dino_latents": input_dino_latents[:, :, 0:1],
            "fuse_vae_embedding_in_latents": bool(
                getattr(self.video_expert, "fuse_vae_embedding_in_latents", False)
            ),
            "action": action,
            "action_is_pad": action_is_pad,
            "image_is_pad": image_is_pad,
        }

    @torch.no_grad()
    def _build_tribranch_attention_mask(
        self,
        vae_seq_len: int,
        dino_seq_len: int,
        action_seq_len: int,
        vae_tokens_per_frame: int,
        dino_tokens_per_frame: int,
        device: torch.device,
    ) -> torch.Tensor:
        total_seq_len = vae_seq_len + dino_seq_len + action_seq_len
        mask = torch.zeros((total_seq_len, total_seq_len), dtype=torch.bool, device=device)

        vae_start = 0
        dino_start = vae_seq_len
        action_start = vae_seq_len + dino_seq_len
        vae_f0 = min(int(vae_tokens_per_frame), int(vae_seq_len))
        dino_f0 = min(int(dino_tokens_per_frame), int(dino_seq_len))

        mask[vae_start:dino_start, vae_start:dino_start] = self.video_expert.build_video_to_video_mask(
            video_seq_len=vae_seq_len,
            video_tokens_per_frame=vae_tokens_per_frame,
            device=device,
        )
        mask[dino_start:action_start, dino_start:action_start] = (
            self.dino_video_expert.build_video_to_video_mask(
                video_seq_len=dino_seq_len,
                video_tokens_per_frame=dino_tokens_per_frame,
                device=device,
            )
        )

        # Current/first-frame condition tokens see each other, but never action
        # or future tokens.  Future VAE/DINO tokens can mutually denoise.
        mask[vae_start : vae_start + vae_f0, dino_start : dino_start + dino_f0] = True
        mask[dino_start : dino_start + dino_f0, vae_start : vae_start + vae_f0] = True
        if vae_seq_len > vae_f0:
            mask[vae_start + vae_f0 : dino_start, dino_start:action_start] = True
        if dino_seq_len > dino_f0:
            mask[dino_start + dino_f0 : action_start, vae_start:dino_start] = True

        # Action reads only f0/current VAE+DINO condition tokens and action
        # tokens. It never reads noisy future VAE or noisy future DINO tokens.
        mask[action_start:, action_start:] = True
        mask[action_start:, vae_start : vae_start + vae_f0] = True
        mask[action_start:, dino_start : dino_start + dino_f0] = True
        return mask

    def _compute_vae_video_loss_per_sample(
        self,
        pred_video: torch.Tensor,
        target_video: torch.Tensor,
        image_is_pad: Optional[torch.Tensor],
        include_initial_video_step: bool,
    ) -> torch.Tensor:
        video_loss_token = F.mse_loss(pred_video.float(), target_video.float(), reduction="none").mean(dim=(1, 3, 4))
        if image_is_pad is None:
            return video_loss_token.mean(dim=1)

        temporal_factor = int(self.vae.temporal_downsample_factor)
        if (image_is_pad.shape[1] - 1) % temporal_factor != 0:
            raise ValueError(
                "Cannot align `image_is_pad` with VAE latent steps: "
                f"num_frames={image_is_pad.shape[1]}, temporal_downsample_factor={temporal_factor}."
            )
        tail_is_pad = image_is_pad[:, 1:]
        latent_tail_is_pad = tail_is_pad.view(image_is_pad.shape[0], -1, temporal_factor).all(dim=2)
        if include_initial_video_step:
            video_is_pad = torch.cat([image_is_pad[:, :1], latent_tail_is_pad], dim=1)
        else:
            video_is_pad = latent_tail_is_pad
        if video_is_pad.shape[1] != video_loss_token.shape[1]:
            raise ValueError(
                f"VAE video-loss mask mismatch: mask={video_is_pad.shape[1]}, loss={video_loss_token.shape[1]}."
            )
        valid = (~video_is_pad).to(device=video_loss_token.device, dtype=video_loss_token.dtype)
        return (video_loss_token * valid).sum(dim=1) / valid.sum(dim=1).clamp(min=1.0)

    @staticmethod
    def _compute_dino_video_loss_per_sample(
        pred_video: torch.Tensor,
        target_video: torch.Tensor,
        image_is_pad: Optional[torch.Tensor],
        include_initial_frame: bool,
    ) -> torch.Tensor:
        video_loss_token = F.mse_loss(pred_video.float(), target_video.float(), reduction="none").mean(dim=(1, 3, 4))
        if image_is_pad is None:
            return video_loss_token.mean(dim=1)
        video_is_pad = image_is_pad if include_initial_frame else image_is_pad[:, 1:]
        if video_is_pad.shape[1] != video_loss_token.shape[1]:
            raise ValueError(
                f"DINO video-loss mask mismatch: mask={video_is_pad.shape[1]}, loss={video_loss_token.shape[1]}."
            )
        valid = (~video_is_pad).to(device=video_loss_token.device, dtype=video_loss_token.dtype)
        return (video_loss_token * valid).sum(dim=1) / valid.sum(dim=1).clamp(min=1.0)

    def training_loss(self, sample: dict, tiled: bool = False) -> tuple[torch.Tensor, dict]:
        inputs = self.build_inputs(sample, tiled=tiled)
        context = inputs["context"]
        context_mask = inputs["context_mask"]
        context_video = inputs.get("context_video", context)
        context_video_mask = inputs.get("context_video_mask", context_mask)
        context_dino = inputs.get("context_dino", context)
        context_dino_mask = inputs.get("context_dino_mask", context_mask)
        context_action = inputs.get("context_action", context)
        context_action_mask = inputs.get("context_action_mask", context_mask)
        action = inputs["action"]
        action_is_pad = inputs["action_is_pad"]
        image_is_pad = inputs["image_is_pad"]
        batch_size = action.shape[0]

        clean_vae = inputs["input_vae_latents"]
        clean_dino = inputs["input_dino_latents"]
        timestep_video = self.train_video_scheduler.sample_training_t(
            batch_size=batch_size,
            device=self.device,
            dtype=clean_vae.dtype,
        )

        noise_vae = torch.randn_like(clean_vae)
        noisy_vae = self.train_video_scheduler.add_noise(clean_vae, noise_vae, timestep_video)
        target_vae = self.train_video_scheduler.training_target(clean_vae, noise_vae, timestep_video)
        noisy_vae[:, :, 0:1] = inputs["first_frame_vae_latents"]

        if self.dino_future_mode == "condition_only":
            # Match action inference: the DINO expert receives only f0 at t=0.
            noisy_dino = inputs["first_frame_dino_latents"]
            timestep_dino = torch.zeros_like(timestep_video)
            target_dino = None
        else:
            noise_dino = torch.randn_like(clean_dino)
            noisy_dino = self.train_video_scheduler.add_noise(clean_dino, noise_dino, timestep_video)
            target_dino = self.train_video_scheduler.training_target(clean_dino, noise_dino, timestep_video)
            noisy_dino[:, :, 0:1] = inputs["first_frame_dino_latents"]
            timestep_dino = timestep_video

        noise_action = torch.randn_like(action)
        timestep_action = self.train_action_scheduler.sample_training_t(
            batch_size=batch_size,
            device=self.device,
            dtype=action.dtype,
        )
        noisy_action = self.train_action_scheduler.add_noise(action, noise_action, timestep_action)
        target_action = self.train_action_scheduler.training_target(action, noise_action, timestep_action)

        vae_pre = self.video_expert.pre_dit(
            x=noisy_vae,
            timestep=timestep_video,
            context=context_video,
            context_mask=context_video_mask,
            action=action,
            fuse_vae_embedding_in_latents=inputs["fuse_vae_embedding_in_latents"],
        )
        dino_pre = self.dino_video_expert.pre_dit(
            x=noisy_dino,
            timestep=timestep_dino,
            context=context_dino,
            context_mask=context_dino_mask,
            fuse_vae_embedding_in_latents=True,
        )
        action_pre = self.action_expert.pre_dit(
            action_tokens=noisy_action,
            timestep=timestep_action,
            context=context_action,
            context_mask=context_action_mask,
        )

        attention_mask = self._build_tribranch_attention_mask(
            vae_seq_len=vae_pre["tokens"].shape[1],
            dino_seq_len=dino_pre["tokens"].shape[1],
            action_seq_len=action_pre["tokens"].shape[1],
            vae_tokens_per_frame=int(vae_pre["meta"]["tokens_per_frame"]),
            dino_tokens_per_frame=int(dino_pre["meta"]["tokens_per_frame"]),
            device=vae_pre["tokens"].device,
        )

        tokens_out = self.mot(
            embeds_all={
                "video": vae_pre["tokens"],
                "dino": dino_pre["tokens"],
                "action": action_pre["tokens"],
            },
            attention_mask=attention_mask,
            freqs_all={
                "video": vae_pre["freqs"],
                "dino": dino_pre["freqs"],
                "action": action_pre["freqs"],
            },
            context_all={
                "video": {"context": vae_pre["context"], "mask": vae_pre["context_mask"]},
                "dino": {"context": dino_pre["context"], "mask": dino_pre["context_mask"]},
                "action": {"context": action_pre["context"], "mask": action_pre["context_mask"]},
            },
            t_mod_all={
                "video": vae_pre["t_mod"],
                "dino": dino_pre["t_mod"],
                "action": action_pre["t_mod"],
            },
        )

        pred_vae = self.video_expert.post_dit(tokens_out["video"], vae_pre)
        pred_dino = self.dino_video_expert.post_dit(tokens_out["dino"], dino_pre)
        pred_action = self.action_expert.post_dit(tokens_out["action"], action_pre)

        pred_vae = pred_vae[:, :, 1:]
        target_vae = target_vae[:, :, 1:]
        loss_vae_per_sample = self._compute_vae_video_loss_per_sample(
            pred_video=pred_vae,
            target_video=target_vae,
            image_is_pad=image_is_pad,
            include_initial_video_step=False,
        )
        video_weight = self.train_video_scheduler.training_weight(timestep_video).to(
            loss_vae_per_sample.device,
            dtype=loss_vae_per_sample.dtype,
        )
        loss_video = (loss_vae_per_sample * video_weight).mean()

        if self.dino_future_mode == "condition_only":
            loss_dino = torch.zeros((), device=loss_video.device, dtype=loss_video.dtype)
        else:
            pred_dino = pred_dino[:, :, 1:]
            target_dino = self.dino_video_expert.target_to_output_space(target_dino)[:, :, 1:]
            loss_dino_per_sample = self._compute_dino_video_loss_per_sample(
                pred_video=pred_dino,
                target_video=target_dino,
                image_is_pad=image_is_pad,
                include_initial_frame=False,
            )
            dino_weight = self.train_video_scheduler.training_weight(timestep_video).to(
                loss_dino_per_sample.device,
                dtype=loss_dino_per_sample.dtype,
            )
            loss_dino = (loss_dino_per_sample * dino_weight).mean()

        action_loss_token = F.mse_loss(pred_action.float(), target_action.float(), reduction="none").mean(dim=2)
        if action_is_pad is not None:
            valid = (~action_is_pad).to(device=action_loss_token.device, dtype=action_loss_token.dtype)
            action_loss_per_sample = (action_loss_token * valid).sum(dim=1) / valid.sum(dim=1).clamp(min=1.0)
        else:
            action_loss_per_sample = action_loss_token.mean(dim=1)
        action_weight = self.train_action_scheduler.training_weight(timestep_action).to(
            action_loss_per_sample.device,
            dtype=action_loss_per_sample.dtype,
        )
        loss_action = (action_loss_per_sample * action_weight).mean()

        loss_total = (
            self.loss_lambda_video * loss_video
            + self.loss_lambda_dino * loss_dino
            + self.loss_lambda_action * loss_action
        )
        return loss_total, {
            "loss_video": self.loss_lambda_video * float(loss_video.detach().item()),
            "loss_dino": self.loss_lambda_dino * float(loss_dino.detach().item()),
            "loss_action": self.loss_lambda_action * float(loss_action.detach().item()),
        }

    @torch.no_grad()
    def _predict_action_noise(
        self,
        first_frame_vae_latents: torch.Tensor,
        first_frame_dino_latents: torch.Tensor,
        latents_action: torch.Tensor,
        timestep_action: torch.Tensor,
        context: torch.Tensor,
        context_mask: torch.Tensor,
        fuse_vae_embedding_in_latents: bool,
        context_video: Optional[torch.Tensor] = None,
        context_video_mask: Optional[torch.Tensor] = None,
        context_dino: Optional[torch.Tensor] = None,
        context_dino_mask: Optional[torch.Tensor] = None,
        context_action: Optional[torch.Tensor] = None,
        context_action_mask: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        if context_video is None:
            context_video = context
        if context_video_mask is None:
            context_video_mask = context_mask
        if context_dino is None:
            context_dino = context
        if context_dino_mask is None:
            context_dino_mask = context_mask
        if context_action is None:
            context_action = context
        if context_action_mask is None:
            context_action_mask = context_mask
        timestep_video = torch.zeros_like(timestep_action, dtype=first_frame_vae_latents.dtype, device=self.device)
        vae_pre = self.video_expert.pre_dit(
            x=first_frame_vae_latents,
            timestep=timestep_video,
            context=context_video,
            context_mask=context_video_mask,
            action=None,
            fuse_vae_embedding_in_latents=fuse_vae_embedding_in_latents,
        )
        dino_pre = self.dino_video_expert.pre_dit(
            x=first_frame_dino_latents,
            timestep=timestep_video,
            context=context_dino,
            context_mask=context_dino_mask,
            fuse_vae_embedding_in_latents=True,
        )
        action_pre = self.action_expert.pre_dit(
            action_tokens=latents_action,
            timestep=timestep_action,
            context=context_action,
            context_mask=context_action_mask,
        )
        attention_mask = self._build_tribranch_attention_mask(
            vae_seq_len=vae_pre["tokens"].shape[1],
            dino_seq_len=dino_pre["tokens"].shape[1],
            action_seq_len=action_pre["tokens"].shape[1],
            vae_tokens_per_frame=int(vae_pre["meta"]["tokens_per_frame"]),
            dino_tokens_per_frame=int(dino_pre["meta"]["tokens_per_frame"]),
            device=vae_pre["tokens"].device,
        )
        tokens_out = self.mot(
            embeds_all={
                "video": vae_pre["tokens"],
                "dino": dino_pre["tokens"],
                "action": action_pre["tokens"],
            },
            attention_mask=attention_mask,
            freqs_all={
                "video": vae_pre["freqs"],
                "dino": dino_pre["freqs"],
                "action": action_pre["freqs"],
            },
            context_all={
                "video": {"context": vae_pre["context"], "mask": vae_pre["context_mask"]},
                "dino": {"context": dino_pre["context"], "mask": dino_pre["context_mask"]},
                "action": {"context": action_pre["context"], "mask": action_pre["context_mask"]},
            },
            t_mod_all={
                "video": vae_pre["t_mod"],
                "dino": dino_pre["t_mod"],
                "action": action_pre["t_mod"],
            },
        )
        return self.action_expert.post_dit(tokens_out["action"], action_pre)

    @torch.no_grad()
    def _prefill_action_static_cache(
        self,
        first_frame_vae_latents: torch.Tensor,
        first_frame_dino_latents: torch.Tensor,
        latents_action: torch.Tensor,
        context_video: torch.Tensor,
        context_video_mask: torch.Tensor,
        context_dino: torch.Tensor,
        context_dino_mask: torch.Tensor,
        fuse_vae_embedding_in_latents: bool,
    ) -> tuple[list[dict[str, torch.Tensor]], torch.Tensor, int]:
        timestep_video = torch.zeros(
            (first_frame_vae_latents.shape[0],),
            dtype=first_frame_vae_latents.dtype,
            device=self.device,
        )
        vae_pre = self.video_expert.pre_dit(
            x=first_frame_vae_latents,
            timestep=timestep_video,
            context=context_video,
            context_mask=context_video_mask,
            action=None,
            fuse_vae_embedding_in_latents=fuse_vae_embedding_in_latents,
        )
        dino_pre = self.dino_video_expert.pre_dit(
            x=first_frame_dino_latents,
            timestep=timestep_video,
            context=context_dino,
            context_mask=context_dino_mask,
            fuse_vae_embedding_in_latents=True,
        )
        vae_seq_len = int(vae_pre["tokens"].shape[1])
        dino_seq_len = int(dino_pre["tokens"].shape[1])
        static_seq_len = vae_seq_len + dino_seq_len
        attention_mask = self._build_tribranch_attention_mask(
            vae_seq_len=vae_seq_len,
            dino_seq_len=dino_seq_len,
            action_seq_len=int(latents_action.shape[1]),
            vae_tokens_per_frame=int(vae_pre["meta"]["tokens_per_frame"]),
            dino_tokens_per_frame=int(dino_pre["meta"]["tokens_per_frame"]),
            device=vae_pre["tokens"].device,
        )
        static_kv_cache = self.mot.prefill_static_expert_cache(
            static_expert_names=("video", "dino"),
            embeds_all={"video": vae_pre["tokens"], "dino": dino_pre["tokens"]},
            freqs_all={"video": vae_pre["freqs"], "dino": dino_pre["freqs"]},
            t_mod_all={"video": vae_pre["t_mod"], "dino": dino_pre["t_mod"]},
            context_all={
                "video": {"context": vae_pre["context"], "mask": vae_pre["context_mask"]},
                "dino": {"context": dino_pre["context"], "mask": dino_pre["context_mask"]},
            },
            static_attention_mask=attention_mask[:static_seq_len, :static_seq_len],
        )
        return static_kv_cache, attention_mask, static_seq_len

    @torch.no_grad()
    def _predict_action_noise_with_static_cache(
        self,
        latents_action: torch.Tensor,
        timestep_action: torch.Tensor,
        context_action: torch.Tensor,
        context_action_mask: torch.Tensor,
        static_kv_cache: list[dict[str, torch.Tensor]],
        attention_mask: torch.Tensor,
        static_seq_len: int,
    ) -> torch.Tensor:
        action_pre = self.action_expert.pre_dit(
            action_tokens=latents_action,
            timestep=timestep_action,
            context=context_action,
            context_mask=context_action_mask,
        )
        action_tokens = self.mot.forward_action_with_static_cache(
            action_tokens=action_pre["tokens"],
            action_freqs=action_pre["freqs"],
            action_t_mod=action_pre["t_mod"],
            action_context_payload={
                "context": action_pre["context"],
                "mask": action_pre["context_mask"],
            },
            static_kv_cache=static_kv_cache,
            attention_mask=attention_mask,
            static_seq_len=static_seq_len,
        )
        return self.action_expert.post_dit(action_tokens, action_pre)

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
        history_vae_latents: Optional[torch.Tensor] = None,
        history_video: Optional[torch.Tensor] = None,
        semantic_image: Optional[torch.Tensor] = None,
        semantic_prompt: Optional[str] = None,
        use_static_kv_cache: bool = True,
    ) -> dict[str, Any]:
        self.eval()
        if self.intent_encoder is None and self.semantic_history_encoder is None and (
            history_dino_latents is not None
            or history_vae_latents is not None
            or history_video is not None
        ):
            raise ValueError(
                "FastWAMVAEDinoMoT does not consume short history tokens. "
                "Disable history inputs for this config, or enable "
                "`model.intent_config.enabled=true` or `model.semantic_history_config.enabled=true`."
            )
        if str(getattr(self.video_expert, "video_attention_mask_mode", "")) != "first_frame_causal":
            raise ValueError("`infer_action` requires VAE video_attention_mask_mode='first_frame_causal'.")
        if str(getattr(self.dino_video_expert, "video_attention_mask_mode", "")) != "first_frame_causal":
            raise ValueError("`infer_action` requires DINO video_attention_mask_mode='first_frame_causal'.")

        if input_image.ndim == 3:
            input_image = input_image.unsqueeze(0)
        if input_image.ndim != 4 or input_image.shape[0] != 1 or input_image.shape[1] != 3:
            raise ValueError(f"`input_image` must be [1,3,H,W] or [3,H,W], got {tuple(input_image.shape)}")
        if input_image.shape[-2] % 16 != 0 or input_image.shape[-1] % 16 != 0:
            raise ValueError(
                f"`input_image` must be resized before infer, got HxW={tuple(input_image.shape[-2:])}."
            )

        if proprio is not None:
            if self.proprio_dim is None:
                raise ValueError("`proprio` provided but `proprio_dim=None`.")
            if proprio.ndim == 1:
                proprio = proprio.unsqueeze(0)
            if proprio.shape[1] != self.proprio_dim:
                raise ValueError(f"`proprio` last dim must be {self.proprio_dim}, got {proprio.shape[1]}")
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
            context = context.to(device=self.device, dtype=self.torch_dtype, non_blocking=True)
            context_mask = context_mask.to(device=self.device, dtype=torch.bool, non_blocking=True)
        if proprio is not None:
            context, context_mask = self._append_proprio_to_context(context, context_mask, proprio)

        context_video = context
        context_video_mask = context_mask
        context_dino = context
        context_dino_mask = context_mask
        context_action = context
        context_action_mask = context_mask

        generator = None if seed is None else torch.Generator(device=rand_device).manual_seed(seed)
        latents_action = torch.randn(
            (1, action_horizon, self.action_expert.action_dim),
            generator=generator,
            device=rand_device,
            dtype=torch.float32,
        ).to(device=self.device, dtype=self.torch_dtype)

        input_image = input_image.to(device=self.device, dtype=self.torch_dtype)
        first_frame_vae_latents = self._encode_input_image_latents_tensor(input_image=input_image, tiled=tiled)
        first_frame_dino_latents = self._encode_single_frame_dino(input_image)
        fuse_flag = bool(getattr(self.video_expert, "fuse_vae_embedding_in_latents", False))

        if self.intent_encoder is not None:
            if history_video is not None:
                if history_video.ndim == 4:
                    if history_video.shape[0] == 3:
                        history_video = history_video.unsqueeze(0).contiguous()
                    elif history_video.shape[1] == 3:
                        history_video = history_video.permute(1, 0, 2, 3).unsqueeze(0).contiguous()
                    else:
                        raise ValueError(
                            f"`history_video` must be [3,T,H,W] or [T,3,H,W], "
                            f"got {tuple(history_video.shape)}"
                        )
                elif history_video.ndim == 5:
                    if history_video.shape[0] != 1:
                        raise ValueError(f"`history_video` batch must be 1, got {history_video.shape[0]}")
                    if history_video.shape[1] == 3:
                        pass
                    elif history_video.shape[2] == 3:
                        history_video = history_video.permute(0, 2, 1, 3, 4).contiguous()
                    else:
                        raise ValueError(
                            f"`history_video` must be [1,3,T,H,W] or [1,T,3,H,W], "
                            f"got {tuple(history_video.shape)}"
                        )
                else:
                    raise ValueError(
                        f"`history_video` must be 4D or 5D, got {tuple(history_video.shape)}"
                    )
                history_video = history_video.to(device=self.device, dtype=self.torch_dtype)

            if self.intent_source == "vae":
                if history_dino_latents is not None:
                    raise ValueError(
                        "`history_dino_latents` was provided but `intent_config.source='vae'`."
                    )
                if history_vae_latents is not None and history_video is not None:
                    raise ValueError("Provide only one of `history_vae_latents` or `history_video`.")
                if history_vae_latents is not None:
                    if history_vae_latents.ndim == 4:
                        history_vae_latents = history_vae_latents.unsqueeze(0)
                    history_vae_latents = history_vae_latents.to(
                        device=self.device,
                        dtype=self.torch_dtype,
                        non_blocking=True,
                    )
                elif history_video is not None:
                    history_vae_latents = self._encode_history_vae_video(history_video, tiled=tiled)
                elif bool(self.intent_config.get("infer_repeat_current_if_missing", False)):
                    history_count = len(
                        self.intent_config.get(
                            "history_offsets",
                            self.intent_config.get("history_dino_frame_offsets", [-8, -4, 0]),
                        )
                    )
                    history_vae_latents = first_frame_vae_latents.expand(
                        -1, -1, history_count, -1, -1
                    )
                else:
                    raise ValueError(
                        "`history_vae_latents` or `history_video` is required for inference when "
                        "`intent_config.enabled=true` and `intent_config.source='vae'`."
                    )
                intent_tokens = self._encode_history_intent(history_vae_latents)
            else:
                if history_vae_latents is not None:
                    raise ValueError(
                        "`history_vae_latents` was provided but `intent_config.source='dino'`."
                    )
                if history_dino_latents is not None and history_video is not None:
                    raise ValueError("Provide only one of `history_dino_latents` or `history_video`.")
                if history_dino_latents is not None:
                    if history_dino_latents.ndim == 4:
                        history_dino_latents = history_dino_latents.unsqueeze(0)
                    history_dino_latents = history_dino_latents.to(
                        device=self.device,
                        dtype=self.torch_dtype,
                        non_blocking=True,
                    )
                elif history_video is not None:
                    history_dino_latents = self._encode_video_dino(history_video)
                elif bool(self.intent_config.get("infer_repeat_current_if_missing", False)):
                    history_count = len(
                        self.intent_config.get(
                            "history_offsets",
                            self.intent_config.get("history_dino_frame_offsets", [-8, -4, 0]),
                        )
                    )
                    history_dino_latents = first_frame_dino_latents.expand(
                        -1, -1, history_count, -1, -1
                    )
                else:
                    raise ValueError(
                        "`history_dino_latents` or `history_video` is required for inference when "
                        "`intent_config.enabled=true`."
                    )
                intent_tokens = self._encode_history_intent(history_dino_latents)

            context, context_mask = self._append_intent_to_context(
                context=context,
                context_mask=context_mask,
                intent_tokens=intent_tokens,
            )
            context_video = context
            context_video_mask = context_mask
            context_dino = context
            context_dino_mask = context_mask
            context_action = context
            context_action_mask = context_mask

        if self.semantic_history_encoder is not None:
            if semantic_image is None:
                semantic_image = input_image
            if semantic_prompt is None:
                semantic_prompt = prompt
            semantic_tokens = self._encode_semantic_history_tokens(
                prompts=semantic_prompt,
                semantic_image=semantic_image,
                history_dino_latents=history_dino_latents,
                history_vae_latents=history_vae_latents,
                history_video=history_video,
                tiled=tiled,
            )
            context_action, context_action_mask = self._append_intent_to_context(
                context=context,
                context_mask=context_mask,
                intent_tokens=semantic_tokens,
            )

        static_kv_cache = None
        static_attention_mask = None
        static_seq_len = None
        if use_static_kv_cache:
            static_kv_cache, static_attention_mask, static_seq_len = self._prefill_action_static_cache(
                first_frame_vae_latents=first_frame_vae_latents,
                first_frame_dino_latents=first_frame_dino_latents,
                latents_action=latents_action,
                context_video=context_video,
                context_video_mask=context_video_mask,
                context_dino=context_dino,
                context_dino_mask=context_dino_mask,
                fuse_vae_embedding_in_latents=fuse_flag,
            )

        infer_timesteps_action, infer_deltas_action = self.infer_action_scheduler.build_inference_schedule(
            num_inference_steps=num_inference_steps,
            device=self.device,
            dtype=latents_action.dtype,
            shift_override=sigma_shift,
        )
        for step_t_action, step_delta_action in zip(infer_timesteps_action, infer_deltas_action):
            timestep_action = step_t_action.unsqueeze(0).to(dtype=latents_action.dtype, device=self.device)
            if use_static_kv_cache:
                pred_action = self._predict_action_noise_with_static_cache(
                    latents_action=latents_action,
                    timestep_action=timestep_action,
                    context_action=context_action,
                    context_action_mask=context_action_mask,
                    static_kv_cache=static_kv_cache,
                    attention_mask=static_attention_mask,
                    static_seq_len=static_seq_len,
                )
            else:
                pred_action = self._predict_action_noise(
                    first_frame_vae_latents=first_frame_vae_latents,
                    first_frame_dino_latents=first_frame_dino_latents,
                    latents_action=latents_action,
                    timestep_action=timestep_action,
                    context=context_action,
                    context_mask=context_action_mask,
                    fuse_vae_embedding_in_latents=fuse_flag,
                    context_video=context_video,
                    context_video_mask=context_video_mask,
                    context_dino=context_dino,
                    context_dino_mask=context_dino_mask,
                    context_action=context_action,
                    context_action_mask=context_action_mask,
                )
            latents_action = self.infer_action_scheduler.step(pred_action, step_delta_action, latents_action)

        return {"action": latents_action[0].detach().to(device="cpu", dtype=torch.float32)}

    @staticmethod
    def _clone_checkpoint_value_to_cpu(value):
        """Break shared tensor storage so each serialized record stays bounded."""
        if isinstance(value, torch.Tensor):
            return value.detach().to(device="cpu", copy=True).contiguous()
        if isinstance(value, dict):
            return {key: FastWAMVAEDinoMoT._clone_checkpoint_value_to_cpu(item) for key, item in value.items()}
        if isinstance(value, list):
            return [FastWAMVAEDinoMoT._clone_checkpoint_value_to_cpu(item) for item in value]
        if isinstance(value, tuple):
            return tuple(FastWAMVAEDinoMoT._clone_checkpoint_value_to_cpu(item) for item in value)
        return value

    @staticmethod
    def _atomic_torch_save(payload, path) -> None:
        checkpoint_path = Path(path)
        checkpoint_path.parent.mkdir(parents=True, exist_ok=True)
        temporary_path = checkpoint_path.with_name(f".{checkpoint_path.name}.tmp-{os.getpid()}")
        try:
            torch.save(payload, temporary_path)
            with open(temporary_path, "rb") as checkpoint_file:
                os.fsync(checkpoint_file.fileno())
            with zipfile.ZipFile(temporary_path, "r") as archive:
                if not archive.namelist():
                    raise RuntimeError(f"Checkpoint archive is empty: {temporary_path}")
            os.replace(temporary_path, checkpoint_path)
        except Exception:
            temporary_path.unlink(missing_ok=True)
            raise

    def save_checkpoint(self, path, optimizer=None, step=None, safe_serialization: bool = False):
        payload = {
            "mot": self.mot.state_dict(),
            "step": step,
            "torch_dtype": str(self.torch_dtype),
            "loss_config": {
                "lambda_video": self.loss_lambda_video,
                "lambda_dino": self.loss_lambda_dino,
                "lambda_action": self.loss_lambda_action,
            },
            "dino_future_mode": self.dino_future_mode,
        }
        if self.proprio_encoder is not None:
            payload["proprio_encoder"] = self.proprio_encoder.state_dict()
        if self.intent_encoder is not None:
            payload["intent_encoder"] = self.intent_encoder.state_dict()
            payload["intent_config"] = self.intent_config
        if self.semantic_history_encoder is not None:
            payload["semantic_history_encoder"] = self.semantic_history_encoder.adapter_state_dict()
            payload["semantic_history_config"] = self.semantic_history_config
        if optimizer is not None:
            payload["optimizer"] = optimizer.state_dict()
        if safe_serialization:
            logger.info("Materializing independent CPU tensors for safe checkpoint serialization: %s", path)
            payload = self._clone_checkpoint_value_to_cpu(payload)
            self._atomic_torch_save(payload, path)
        else:
            torch.save(payload, path)

    def _validate_checkpoint_intent_config(self, checkpoint: dict[str, Any]) -> None:
        checkpoint_has_intent = "intent_encoder" in checkpoint
        current_has_intent = self.intent_encoder is not None
        if checkpoint_has_intent and not current_has_intent:
            raise ValueError(
                "Checkpoint contains `intent_encoder` weights, but current FastWAMVAEDinoMoT "
                "was built with `intent_config.enabled=false` or without intent_config. "
                "Rebuild the model with the checkpoint's intent_config before evaluation/loading."
            )
        if not checkpoint_has_intent:
            return
        checkpoint_config = dict(checkpoint.get("intent_config") or {})
        checkpoint_mode = str(checkpoint_config.get("injection_mode", "context_after_proprio"))
        current_mode = str(self.intent_config.get("injection_mode", "context_after_proprio"))
        if checkpoint_mode != current_mode:
            raise ValueError(
                "Checkpoint intent injection_mode mismatch: "
                f"checkpoint={checkpoint_mode!r}, current={current_mode!r}."
            )
        for key in ("source", "history_offsets", "max_history_frames", "num_intent_tokens"):
            if key in checkpoint_config and key in self.intent_config:
                if checkpoint_config[key] != self.intent_config[key]:
                    raise ValueError(
                        f"Checkpoint intent_config.{key} mismatch: "
                        f"checkpoint={checkpoint_config[key]!r}, current={self.intent_config[key]!r}."
                    )

    def _validate_checkpoint_semantic_history_config(self, checkpoint: dict[str, Any]) -> None:
        checkpoint_has_semantic = "semantic_history_encoder" in checkpoint
        current_has_semantic = self.semantic_history_encoder is not None
        if checkpoint_has_semantic and not current_has_semantic:
            raise ValueError(
                "Checkpoint contains `semantic_history_encoder` weights, but current "
                "FastWAMVAEDinoMoT was built without semantic_history_config. Rebuild the "
                "model with the checkpoint's semantic_history_config before evaluation/loading."
            )
        if not checkpoint_has_semantic:
            return
        checkpoint_config = dict(checkpoint.get("semantic_history_config") or {})
        checkpoint_mode = str(checkpoint_config.get("injection_mode", "action_context_after_proprio"))
        current_mode = str(self.semantic_history_config.get("injection_mode", "action_context_after_proprio"))
        if checkpoint_mode != current_mode:
            raise ValueError(
                "Checkpoint semantic_history_config.injection_mode mismatch: "
                f"checkpoint={checkpoint_mode!r}, current={current_mode!r}."
            )
        for key in (
            "source",
            "use_history",
            "history_offsets",
            "max_history_frames",
            "num_output_tokens",
        ):
            if key in checkpoint_config and key in self.semantic_history_config:
                if checkpoint_config[key] != self.semantic_history_config[key]:
                    raise ValueError(
                        f"Checkpoint semantic_history_config.{key} mismatch: "
                        f"checkpoint={checkpoint_config[key]!r}, current={self.semantic_history_config[key]!r}."
                    )
        checkpoint_vlm_path = checkpoint_config.get("vlm_model_name_or_path")
        current_vlm_path = self.semantic_history_config.get("vlm_model_name_or_path")
        if (
            checkpoint_vlm_path is not None
            and current_vlm_path is not None
            and checkpoint_vlm_path != current_vlm_path
        ):
            if not bool(self.semantic_history_config.get("allow_vlm_path_relocation", False)):
                raise ValueError(
                    "Checkpoint semantic_history_config.vlm_model_name_or_path mismatch: "
                    f"checkpoint={checkpoint_vlm_path!r}, current={current_vlm_path!r}."
                )
            logger.warning(
                "Loading semantic VLM from a relocated path: checkpoint=%r current=%r. "
                "All architecture-affecting semantic history settings remain strictly validated.",
                checkpoint_vlm_path,
                current_vlm_path,
            )

    def load_checkpoint(self, path, optimizer=None):
        payload = torch.load(path, map_location="cpu")
        if "mot" not in payload:
            raise ValueError(f"Checkpoint missing `mot` keys for FastWAMVAEDinoMoT: {path}")
        checkpoint_dino_future_mode = str(payload.get("dino_future_mode", "predict")).strip().lower()
        if checkpoint_dino_future_mode != self.dino_future_mode:
            raise ValueError(
                "Checkpoint dino_future_mode mismatch: "
                f"checkpoint={checkpoint_dino_future_mode!r}, current={self.dino_future_mode!r}."
            )
        self._validate_checkpoint_intent_config(payload)
        self._validate_checkpoint_semantic_history_config(payload)
        missing, unexpected = self.mot.load_state_dict(payload["mot"], strict=False)
        if missing or unexpected:
            logger.warning(
                "FastWAMVAEDinoMoT checkpoint loaded with non-strict MoT keys: "
                "missing=%d unexpected=%d missing_head=%s unexpected_head=%s",
                len(missing),
                len(unexpected),
                missing[:10],
                unexpected[:10],
            )
        if self.proprio_encoder is not None and "proprio_encoder" in payload:
            self.proprio_encoder.load_state_dict(payload["proprio_encoder"], strict=True)
        if self.intent_encoder is not None:
            if "intent_encoder" in payload:
                self.intent_encoder.load_state_dict(payload["intent_encoder"], strict=True)
            else:
                logger.info(
                    "FastWAMVAEDinoMoT intent_encoder is enabled, but checkpoint has no "
                    "intent_encoder weights; keeping randomly initialized intent encoder."
                )
        if self.semantic_history_encoder is not None:
            if "semantic_history_encoder" in payload:
                self.semantic_history_encoder.load_adapter_state_dict(
                    payload["semantic_history_encoder"],
                    strict=True,
                )
            else:
                logger.info(
                    "FastWAMVAEDinoMoT semantic_history_encoder is enabled, but checkpoint has no "
                    "semantic_history_encoder weights; keeping randomly initialized adapter."
                )
        if optimizer is not None and "optimizer" in payload:
            optimizer.load_state_dict(payload["optimizer"])
        step = payload.get("step", None)
        logger.info("FastWAMVAEDinoMoT checkpoint loaded from %s, step=%s", path, step)
        return step

    def forward(self, *args, **kwargs):
        return self.training_loss(*args, **kwargs)
