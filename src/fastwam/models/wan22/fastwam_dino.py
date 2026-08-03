"""FastWAM-DINO: FastWAM variant that operates in DINO feature space.

This module replaces the VAE latent space with a frozen DINO encoder for visual
representation. The Video Expert predicts velocity in DINO feature space instead
of VAE latent space, while keeping the same MoT + Action Expert architecture
and KV-cache-based fast inference.

Key differences from original FastWAM:
- Uses frozen DINOv2/v3 encoder instead of frozen VAE for visual encoding
- Video Expert operates on DINO patch features (dim=1024) instead of VAE latents (dim=48)
- No video decoding capability (DINO has no decoder) — this is action-only at inference
- Training loss is flow matching MSE in DINO feature space
- Inference uses the same KV-cache trick: encode first frame with DINO, prefill video cache,
  then iteratively denoise action
"""

from typing import Any, Optional, Sequence, Union

import torch
import torch.nn as nn
import torch.nn.functional as F

from fastwam.utils.logging_config import get_logger

from .action_dit import ActionDiT
from .dino_encoder import DinoVideoEncoder
from .dino_intent import DINOHistoryIntentResampler
from .dino_video_dit import DinoVideoDiT
from .mot import MoT
from .schedulers.scheduler_continuous import WanContinuousFlowMatchScheduler

logger = get_logger(__name__)

INTENT_INJECTION_MODES = {"context_after_proprio", "video_prefix"}


class FastWAM_DINO(nn.Module):
    """FastWAM variant with DINO feature space for video representation.

    Architecture:
        - DinoVideoEncoder (frozen): video frames → DINO patch features
        - DinoVideoDiT (trainable): Video Expert in DINO space
        - ActionDiT (trainable): Action Expert (same as original FastWAM)
        - MoT (trainable): Mixed attention between video and action experts

    Training:
        - Encode GT video with frozen DINO → clean DINO features
        - Add noise via flow matching → noisy DINO features
        - MoT forward: video + action tokens joint attention
        - Loss = MSE(predicted_velocity, target_velocity) in DINO space + action space

    Inference (action-only, fast):
        - Encode first frame with DINO → clean first-frame features
        - Video Expert prefill → cache KV
        - Action denoising loop (20 steps) reusing cached video KV
    """

    def __init__(
        self,
        video_expert: DinoVideoDiT,
        action_expert: ActionDiT,
        mot: MoT,
        dino_encoder: DinoVideoEncoder,
        intent_encoder: Optional[DINOHistoryIntentResampler] = None,
        intent_config: Optional[dict[str, Any]] = None,
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
        loss_lambda_action: float = 1.0,
    ):
        super().__init__()
        self.video_expert = video_expert
        self.action_expert = action_expert
        self.mot = mot
        # Trainer compatibility: optimizer and freeze logic use `model.dit`
        self.dit = self.mot

        self.dino_encoder = dino_encoder
        self.intent_encoder = intent_encoder
        self.intent_config = dict(intent_config or {})
        self.intent_injection_mode = str(
            self.intent_config.get("injection_mode", "context_after_proprio")
        )
        if self.intent_injection_mode not in INTENT_INJECTION_MODES:
            raise ValueError(
                f"Unsupported `intent_config.injection_mode={self.intent_injection_mode!r}`. "
                f"Expected one of {sorted(INTENT_INJECTION_MODES)}."
            )
        if self.intent_encoder is not None:
            self.intent_config["injection_mode"] = self.intent_injection_mode
        history_offsets = self.intent_config.get("history_offsets", None)
        self.intent_history_frame_count = (
            len(history_offsets)
            if self.intent_encoder is not None and history_offsets is not None
            else None
        )
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
        self.loss_lambda_action = float(loss_lambda_action)

        self.to(self.device)

    @classmethod
    def from_config(
        cls,
        dino_config: dict[str, Any],
        video_dit_config: dict[str, Any],
        action_dit_config: dict[str, Any],
        device: str = "cuda",
        torch_dtype: torch.dtype = torch.bfloat16,
        text_encoder=None,
        tokenizer=None,
        text_dim: int = 4096,
        proprio_dim: Optional[int] = None,
        action_dit_pretrained_path: Optional[str] = None,
        skip_dit_load_from_pretrain: bool = False,
        mot_checkpoint_mixed_attn: bool = True,
        load_text_encoder: bool = False,
        tokenizer_model_id: str = "Wan-AI/Wan2.1-T2V-1.3B",
        tokenizer_max_len: int = 512,
        video_train_shift: float = 5.0,
        video_infer_shift: float = 5.0,
        video_num_train_timesteps: int = 1000,
        action_train_shift: float = 5.0,
        action_infer_shift: float = 5.0,
        action_num_train_timesteps: int = 1000,
        loss_lambda_video: float = 1.0,
        loss_lambda_action: float = 1.0,
        video_dit_init_from_wan: bool = False,
        video_dit_pretrained_path: Optional[str] = None,
        wan_model_id: str = "Wan-AI/Wan2.2-TI2V-5B",
        intent_config: Optional[dict[str, Any]] = None,
    ):
        """Create FastWAM_DINO from configuration dicts."""
        if video_dit_init_from_wan and video_dit_pretrained_path:
            raise ValueError(
                "Use only one DinoVideoDiT initialization path: "
                "`video_dit_init_from_wan=true` for same-shape Wan loading, or "
                "`video_dit_pretrained_path=...` for preprocessed smallvideo payloads."
            )

        # Build DINO encoder (frozen)
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
                "DinoVideoEncoder backbone loading disabled by dino_config.load_backbone=false. "
                "Training must provide precomputed `dino_latents`; inference will require enabling the backbone."
            )

        # Build DINO Video Expert
        video_expert = DinoVideoDiT(
            hidden_dim=video_dit_config["hidden_dim"],
            dino_dim=dino_config.get("feature_dim", 1024),
            ffn_dim=video_dit_config["ffn_dim"],
            text_dim=video_dit_config.get("text_dim", text_dim),
            freq_dim=video_dit_config.get("freq_dim", 256),
            eps=video_dit_config.get("eps", 1e-6),
            num_heads=video_dit_config["num_heads"],
            attn_head_dim=video_dit_config["attn_head_dim"],
            num_layers=video_dit_config["num_layers"],
            video_attention_mask_mode=video_dit_config.get(
                "video_attention_mask_mode", "first_frame_causal"
            ),
            use_gradient_checkpointing=video_dit_config.get("use_gradient_checkpointing", True),
            latent_patch_size=tuple(video_dit_config.get("latent_patch_size", [1, 1, 1])),
            latent_patch_mode=video_dit_config.get("latent_patch_mode", "flat"),
            latent_num_views=video_dit_config.get("latent_num_views", 1),
            output_patch_space=video_dit_config.get("output_patch_space", "dense"),
        ).to(device=device, dtype=torch_dtype)

        # Optionally initialize DinoVideoDiT from Wan2.2 Video DiT weights
        if video_dit_pretrained_path:
            video_expert.load_preprocessed_backbone(str(video_dit_pretrained_path))
        elif video_dit_init_from_wan:
            from .helpers.loader import load_wan22_ti2v_5b_components

            wan5b_shape = {
                "hidden_dim": 3072,
                "ffn_dim": 14336,
                "num_heads": 24,
                "attn_head_dim": 128,
                "num_layers": 30,
            }
            incompatible = {
                key: (int(video_dit_config[key]), expected)
                for key, expected in wan5b_shape.items()
                if int(video_dit_config.get(key, -1)) != expected
            }
            if incompatible:
                raise ValueError(
                    "`video_dit_init_from_wan=true` currently requires the "
                    "DinoVideoDiT backbone to keep the Wan2.2-5B Transformer shape. "
                    f"Incompatible video_dit_config entries: {incompatible}. "
                    "Use the big DINO video config, or keep this flag false for "
                    "smallvideo until an explicit adapter path is implemented."
                )

            wan_dit_config = {
                "has_image_input": False,
                "patch_size": [1, 2, 2],
                "in_dim": 48,
                "hidden_dim": video_dit_config["hidden_dim"],
                "ffn_dim": video_dit_config["ffn_dim"],
                "freq_dim": video_dit_config.get("freq_dim", 256),
                "text_dim": video_dit_config.get("text_dim", text_dim),
                "out_dim": 48,
                "num_heads": video_dit_config["num_heads"],
                "attn_head_dim": video_dit_config["attn_head_dim"],
                "num_layers": video_dit_config["num_layers"],
                "eps": video_dit_config.get("eps", 1e-6),
                "seperated_timestep": True,
                "require_clip_embedding": False,
                "require_vae_embedding": False,
                "fuse_vae_embedding_in_latents": True,
                "video_attention_mask_mode": "first_frame_causal",
                "action_conditioned": False,
            }
            logger.info(f"Loading Wan2.2 DiT from '{wan_model_id}' to initialize DinoVideoDiT...")
            wan_components = load_wan22_ti2v_5b_components(
                device=device,
                torch_dtype=torch_dtype,
                model_id=wan_model_id,
                tokenizer_model_id=tokenizer_model_id,
                tokenizer_max_len=tokenizer_max_len,
                redirect_common_files=True,
                dit_config=wan_dit_config,
                skip_dit_load_from_pretrain=False,
                load_text_encoder=False,
                load_vae=False,
            )
            if wan_components.dit is None:
                raise RuntimeError("Wan DiT initialization requested but loader returned no DiT.")
            video_expert.init_from_wan_dit(wan_components.dit.state_dict())
            del wan_components  # free memory

        # Build Action Expert
        action_expert = ActionDiT.from_pretrained(
            action_dit_config=action_dit_config,
            action_dit_pretrained_path=action_dit_pretrained_path,
            skip_dit_load_from_pretrain=skip_dit_load_from_pretrain,
            device=device,
            torch_dtype=torch_dtype,
        )

        # Validate MoT compatibility
        if int(action_expert.num_heads) != int(video_expert.num_heads):
            raise ValueError(
                "ActionDiT `num_heads` must match DinoVideoDiT for MoT mixed attention."
            )
        if int(action_expert.attn_head_dim) != int(video_expert.attn_head_dim):
            raise ValueError(
                "ActionDiT `attn_head_dim` must match DinoVideoDiT for MoT mixed attention."
            )
        if len(action_expert.blocks) != len(video_expert.blocks):
            raise ValueError("ActionDiT `num_layers` must match DinoVideoDiT.")

        # Build MoT
        mot = MoT(
            mixtures={"video": video_expert, "action": action_expert},
            mot_checkpoint_mixed_attn=mot_checkpoint_mixed_attn,
        )

        intent_config = dict(intent_config or {})
        intent_encoder = None
        if bool(intent_config.get("enabled", False)):
            injection_mode = str(intent_config.get("injection_mode", "context_after_proprio"))
            if injection_mode not in INTENT_INJECTION_MODES:
                raise ValueError(
                    f"Unsupported `intent_config.injection_mode={injection_mode!r}`. "
                    f"Expected one of {sorted(INTENT_INJECTION_MODES)}."
                )
            intent_config["injection_mode"] = injection_mode
            history_offsets = intent_config.get(
                "history_offsets",
                intent_config.get("history_dino_frame_offsets", [-8, -4, 0]),
            )
            history_offsets = [int(offset) for offset in history_offsets]
            if len(history_offsets) == 0:
                raise ValueError("`intent_config.history_offsets` must contain at least one frame offset.")
            intent_config["history_offsets"] = history_offsets
            max_history_frames = int(
                intent_config.get("max_history_frames", max(1, len(history_offsets)))
            )
            if max_history_frames < len(history_offsets):
                raise ValueError(
                    "`intent_config.max_history_frames` must be >= len(history_offsets), "
                    f"got {max_history_frames} and {len(history_offsets)}."
                )
            intent_encoder = DINOHistoryIntentResampler(
                dino_dim=int(dino_config.get("feature_dim", 1024)),
                text_dim=int(video_dit_config.get("text_dim", text_dim)),
                num_intent_tokens=int(intent_config.get("num_intent_tokens", 8)),
                resampler_dim=int(intent_config.get("resampler_dim", 1024)),
                num_layers=int(intent_config.get("num_resampler_layers", 2)),
                num_heads=int(intent_config.get("num_heads", 8)),
                max_history_frames=max_history_frames,
                mlp_ratio=float(intent_config.get("mlp_ratio", 4.0)),
                dropout=float(intent_config.get("dropout", 0.0)),
            ).to(device=device, dtype=torch_dtype)


        # Optionally load text encoder
        loaded_text_encoder = text_encoder
        loaded_tokenizer = tokenizer
        if load_text_encoder and loaded_text_encoder is None:
            # Load only text encoder + tokenizer from Wan2.2 components.
            # We must provide a valid dit_config to satisfy the loader, but
            # skip_dit_load_from_pretrain=True avoids downloading/loading the
            # actual video DiT weights (they are unused in DINO mode).
            dummy_dit_config = {
                "has_image_input": False,
                "patch_size": [1, 2, 2],
                "in_dim": 48,
                "hidden_dim": 3072,
                "ffn_dim": 14336,
                "freq_dim": 256,
                "text_dim": text_dim,
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
            from .helpers.loader import load_wan22_ti2v_5b_components
            components = load_wan22_ti2v_5b_components(
                device=device,
                torch_dtype=torch_dtype,
                model_id="Wan-AI/Wan2.2-TI2V-5B",
                tokenizer_model_id=tokenizer_model_id,
                tokenizer_max_len=tokenizer_max_len,
                redirect_common_files=True,
                dit_config=dummy_dit_config,
                skip_dit_load_from_pretrain=True,
                load_text_encoder=True,
                load_dit=False,
                load_vae=False,
            )
            loaded_text_encoder = components.text_encoder
            loaded_tokenizer = components.tokenizer


        model = cls(
            video_expert=video_expert,
            action_expert=action_expert,
            mot=mot,
            dino_encoder=dino_encoder,
            intent_encoder=intent_encoder,
            intent_config=intent_config,
            text_encoder=loaded_text_encoder,
            tokenizer=loaded_tokenizer,
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
            loss_lambda_action=loss_lambda_action,
        )
        return model

    def to(self, *args, **kwargs):
        super().to(*args, **kwargs)
        self.mot.to(*args, **kwargs)
        if self.text_encoder is not None:
            self.text_encoder.to(*args, **kwargs)
        # dino_encoder is frozen, but still needs to be on the correct device
        if self.dino_encoder is not None and self.dino_encoder._loaded:
            self.dino_encoder.backbone.to(*args, **kwargs)
        return self

    @torch.no_grad()
    def encode_prompt(self, prompt: Union[str, Sequence[str]]):
        """Encode text prompt using text encoder."""
        if self.text_encoder is None or self.tokenizer is None:
            raise ValueError(
                "Prompt encoding requires loaded text encoder/tokenizer."
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
        """Append proprioception token to context sequence."""
        if self.proprio_encoder is None or proprio is None:
            return context, context_mask
        if proprio.ndim != 2:
            raise ValueError(f"`proprio` must be 2D [B, D], got shape {tuple(proprio.shape)}")
        proprio_token = self.proprio_encoder(
            proprio.to(device=self.device, dtype=context.dtype).unsqueeze(1)
        ).to(dtype=context.dtype)
        proprio_mask = torch.ones(
            (context_mask.shape[0], 1), dtype=torch.bool, device=context_mask.device
        )
        return (
            torch.cat([context, proprio_token], dim=1),
            torch.cat([context_mask, proprio_mask], dim=1),
        )

    def _append_intent_to_context(
        self,
        context: torch.Tensor,
        context_mask: torch.Tensor,
        intent_tokens: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Append Short-DINO-Intent tokens after text/proprio context."""
        if intent_tokens.ndim != 3:
            raise ValueError(
                f"`intent_tokens` must be 3D [B,K,D], got shape {tuple(intent_tokens.shape)}"
            )
        if intent_tokens.shape[0] != context.shape[0]:
            raise ValueError(
                f"Batch mismatch between context and intent tokens: "
                f"{context.shape[0]} vs {intent_tokens.shape[0]}"
            )
        if intent_tokens.shape[2] != self.text_dim:
            raise ValueError(
                f"Intent token dim must match text_dim={self.text_dim}, "
                f"got {intent_tokens.shape[2]}"
            )
        intent_mask = torch.ones(
            (context_mask.shape[0], intent_tokens.shape[1]),
            dtype=torch.bool,
            device=context_mask.device,
        )
        return (
            torch.cat([context, intent_tokens.to(dtype=context.dtype)], dim=1),
            torch.cat([context_mask, intent_mask], dim=1),
        )

    def _apply_video_prefix_intent(
        self,
        video_pre: dict[str, Any],
        intent_tokens: Optional[torch.Tensor],
    ) -> dict[str, Any]:
        if intent_tokens is None:
            return video_pre
        return self.video_expert.prepend_video_prefix_tokens(video_pre, intent_tokens)

    def _encode_history_intent(self, history_dino_latents: torch.Tensor) -> torch.Tensor:
        if self.intent_encoder is None:
            raise RuntimeError("Intent encoder is disabled.")
        if history_dino_latents.ndim != 5:
            raise ValueError(
                f"`history_dino_latents` must be [B,D,T,H,W], got "
                f"{tuple(history_dino_latents.shape)}"
            )
        if (
            self.intent_history_frame_count is not None
            and history_dino_latents.shape[2] != self.intent_history_frame_count
        ):
            raise ValueError(
                "History DINO frame count must match intent history offsets: "
                f"got T={history_dino_latents.shape[2]}, "
                f"expected {self.intent_history_frame_count}."
            )
        self._validate_dino_latent_shape(history_dino_latents)
        history_dino_latents = history_dino_latents.to(
            device=self.device,
            dtype=self.torch_dtype,
            non_blocking=True,
        )
        return self.intent_encoder(history_dino_latents)

    @torch.no_grad()
    def _encode_video_dino(self, video_tensor: torch.Tensor) -> torch.Tensor:
        """Encode video frames to DINO features.

        Args:
            video_tensor: [B, 3, T, H, W] RGB video, values in [-1, 1].

        Returns:
            [B, D_dino, T, H_grid, W_grid] DINO feature latents.
        """
        return self.dino_encoder.encode_video_to_latent(video_tensor)

    @torch.no_grad()
    def _encode_single_frame_dino(self, image: torch.Tensor) -> torch.Tensor:
        """Encode a single image to DINO features.

        Args:
            image: [1, 3, H, W] or [3, H, W] single RGB image.

        Returns:
            [1, D_dino, 1, H_grid, W_grid] DINO features for one frame.
        """
        if image.ndim == 3:
            image = image.unsqueeze(0)
        # Add temporal dimension: [B, 3, H, W] → [B, 3, 1, H, W]
        video = image.unsqueeze(2)
        return self._encode_video_dino(video)

    def _validate_dino_latent_shape(self, latents: torch.Tensor) -> None:
        if latents.ndim != 5:
            raise ValueError(f"DINO latents must be 5D [B,D,T,H,W], got shape {tuple(latents.shape)}")
        expected_dim = int(getattr(self.video_expert, "dino_dim", latents.shape[1]))
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
                f"cache={(latents.shape[-2], latents.shape[-1])}, "
                f"expected={(expected_h, expected_w)} from input_resolution="
                f"{self.dino_encoder.input_resolution}, patch_size={patch_size}, "
                f"latent_spatial_pool={self.dino_encoder.latent_spatial_pool}."
            )

    @torch.no_grad()
    def _build_mot_attention_mask(
        self,
        video_seq_len: int,
        action_seq_len: int,
        video_tokens_per_frame: int,
        device: torch.device,
        video_prefix_len: int = 0,
    ) -> torch.Tensor:
        """Build attention mask for MoT (video + action)."""
        total_seq_len = video_seq_len + action_seq_len
        mask = torch.zeros((total_seq_len, total_seq_len), dtype=torch.bool, device=device)
        video_prefix_len = int(video_prefix_len)

        # video → video
        mask[:video_seq_len, :video_seq_len] = self.video_expert.build_video_to_video_mask(
            video_seq_len=video_seq_len,
            video_tokens_per_frame=video_tokens_per_frame,
            device=device,
            video_prefix_len=video_prefix_len,
        )
        # action → action (full visibility)
        mask[video_seq_len:, video_seq_len:] = True
        # action → video prefix plus original first-frame video.  The prefix is
        # part of the video branch sequence, not extra ActionDiT cross-attn
        # context, so this keeps the injection mechanism in mixed attention.
        if video_prefix_len:
            mask[video_seq_len:, :video_prefix_len] = True
        original_video_seq_len = video_seq_len - video_prefix_len
        first_frame_tokens = min(video_tokens_per_frame, original_video_seq_len)
        first_frame_start = video_prefix_len
        first_frame_end = first_frame_start + first_frame_tokens
        mask[video_seq_len:, first_frame_start:first_frame_end] = True
        return mask

    def _compute_video_loss_per_sample(
        self,
        pred_video: torch.Tensor,
        target_video: torch.Tensor,
        image_is_pad: Optional[torch.Tensor],
        include_initial_frame: bool,
    ) -> torch.Tensor:
        """Compute per-sample video loss in DINO feature space.

        Args:
            pred_video: [B, D, T, H, W] predicted velocity.
            target_video: [B, D, T, H, W] target velocity.
            image_is_pad: [B, T_original] padding mask for frames.
            include_initial_frame: Whether loss includes the first frame.

        Returns:
            [B] per-sample loss values.
        """
        # MSE over (D, H, W), keep temporal dimension for masking
        video_loss_token = F.mse_loss(
            pred_video.float(), target_video.float(), reduction="none"
        ).mean(dim=(1, 3, 4))  # [B, T]

        if image_is_pad is None:
            return video_loss_token.mean(dim=1)

        # For DINO, temporal_downsample_factor = 1 (no temporal compression)
        if include_initial_frame:
            video_is_pad = image_is_pad
        else:
            video_is_pad = image_is_pad[:, 1:]

        if video_is_pad.shape[1] != video_loss_token.shape[1]:
            raise ValueError(
                f"Video loss mask shape mismatch: mask={video_is_pad.shape[1]}, "
                f"loss={video_loss_token.shape[1]}."
            )

        valid = (~video_is_pad).to(device=video_loss_token.device, dtype=video_loss_token.dtype)
        valid_sum = valid.sum(dim=1).clamp(min=1.0)
        return (video_loss_token * valid).sum(dim=1) / valid_sum

    def build_inputs(self, sample: dict, tiled: bool = False) -> dict:
        """Process training sample: encode video with DINO, validate inputs.

        Args:
            sample: Dict with keys 'video' [B, 3, T, H, W], 'action' [B, T_a, a_dim],
                    'context' [B, L, D], 'context_mask' [B, L], optionally 'proprio',
                    optionally 'history_dino_latents' when intent_config.enabled=true,
                    'action_is_pad', 'image_is_pad'.

        Returns:
            Dict with processed tensors ready for training_loss.
        """
        if "context" not in sample or "context_mask" not in sample:
            raise ValueError("FastWAM_DINO training requires `context` and `context_mask`.")
        context = sample["context"]
        context_mask = sample["context_mask"]
        proprio = sample.get("proprio", None)
        history_dino_latents = sample.get("history_dino_latents", None)
        video_prefix_tokens = None

        if "action" not in sample:
            raise ValueError("`action` is required for training.")
        action = sample["action"]
        if action.ndim != 3:
            raise ValueError(f"`action` must be 3D [B, T, a_dim], got shape {tuple(action.shape)}")
        batch_size = action.shape[0]

        action_is_pad = sample.get("action_is_pad", None)
        image_is_pad = sample.get("image_is_pad", None)

        # Prefer precomputed DINO latents when present. This keeps the training
        # graph identical after the frozen encoder, but removes the expensive
        # DINO forward pass from every optimizer microstep.
        if "dino_latents" in sample and sample["dino_latents"] is not None:
            input_latents = sample["dino_latents"]
            if input_latents.ndim != 5:
                raise ValueError(
                    f"`dino_latents` must be 5D [B,D,T,H,W], got shape {tuple(input_latents.shape)}"
                )
            if input_latents.shape[0] != batch_size:
                raise ValueError(
                    f"Batch mismatch between `dino_latents` and `action`: "
                    f"{input_latents.shape[0]} vs {batch_size}"
                )
            self._validate_dino_latent_shape(input_latents)
            input_latents = input_latents.to(device=self.device, dtype=self.torch_dtype, non_blocking=True)
        else:
            if "video" not in sample:
                raise ValueError("FastWAM_DINO training requires either `dino_latents` or `video`.")
            video = sample["video"]
            if video.ndim != 5:
                raise ValueError(f"`video` must be 5D [B, 3, T, H, W], got shape {tuple(video.shape)}")
            if video.shape[0] != batch_size:
                raise ValueError(
                    f"Batch mismatch between `video` and `action`: {video.shape[0]} vs {batch_size}"
                )
            input_video = video.to(device=self.device, dtype=self.torch_dtype, non_blocking=True)
            input_latents = self._encode_video_dino(input_video)  # [B, D, T, H_g, W_g]
            self._validate_dino_latent_shape(input_latents)

        # First frame for conditioning
        first_frame_latents = input_latents[:, :, 0:1]

        # Context
        if context.ndim != 3 or context_mask.ndim != 2:
            raise ValueError(
                f"`context/context_mask` must be [B,L,D]/[B,L], got shapes "
                f"{tuple(context.shape)} and {tuple(context_mask.shape)}"
            )
        context = context.to(device=self.device, dtype=self.torch_dtype, non_blocking=True)
        context_mask = context_mask.to(device=self.device, dtype=torch.bool, non_blocking=True)

        # Proprio
        if self.proprio_encoder is not None:
            if proprio is None:
                raise ValueError("`proprio` is required when `proprio_dim` is enabled.")
            if proprio.ndim != 3:
                raise ValueError(f"`proprio` must be 3D [B, T, d], got shape {tuple(proprio.shape)}")
            proprio = proprio[:, 0, :]  # [B, D]
            context, context_mask = self._append_proprio_to_context(
                context=context,
                context_mask=context_mask,
                proprio=proprio.to(device=self.device, dtype=self.torch_dtype),
            )

        if self.intent_encoder is not None:
            if history_dino_latents is None:
                raise ValueError(
                    "`history_dino_latents` is required when `intent_config.enabled=true`."
                )
            if history_dino_latents.ndim != 5:
                raise ValueError(
                    f"`history_dino_latents` must be 5D [B,D,T,H,W], got "
                    f"{tuple(history_dino_latents.shape)}"
                )
            if history_dino_latents.shape[0] != batch_size:
                raise ValueError(
                    f"Batch mismatch between `history_dino_latents` and `action`: "
                    f"{history_dino_latents.shape[0]} vs {batch_size}"
                )
            intent_tokens = self._encode_history_intent(history_dino_latents)
            if self.intent_injection_mode == "context_after_proprio":
                context, context_mask = self._append_intent_to_context(
                    context=context,
                    context_mask=context_mask,
                    intent_tokens=intent_tokens,
                )
            elif self.intent_injection_mode == "video_prefix":
                video_prefix_tokens = intent_tokens
            else:
                raise RuntimeError(f"Unhandled intent injection mode: {self.intent_injection_mode}")

        action = action.to(device=self.device, dtype=self.torch_dtype, non_blocking=True)
        if action_is_pad is not None:
            action_is_pad = action_is_pad.to(device=self.device, dtype=torch.bool, non_blocking=True)
        if image_is_pad is not None:
            image_is_pad = image_is_pad.to(device=self.device, dtype=torch.bool, non_blocking=True)

        return {
            "context": context,
            "context_mask": context_mask,
            "input_latents": input_latents,
            "first_frame_latents": first_frame_latents,
            "action": action,
            "action_is_pad": action_is_pad,
            "image_is_pad": image_is_pad,
            "video_prefix_tokens": video_prefix_tokens,
        }

    def training_loss(self, sample: dict, tiled: bool = False) -> tuple[torch.Tensor, dict]:
        """Compute training loss for FastWAM-DINO.

        Flow matching in DINO feature space:
        - target = noise - clean_latent (velocity field)
        - Loss = MSE(predicted_velocity, target_velocity)

        Returns:
            (loss_total, loss_dict) where loss_dict has 'loss_video' and 'loss_action'.
        """
        inputs = self.build_inputs(sample, tiled=tiled)
        input_latents = inputs["input_latents"]  # [B, D, T, H_g, W_g]
        batch_size = input_latents.shape[0]
        context = inputs["context"]
        context_mask = inputs["context_mask"]
        action = inputs["action"]
        action_is_pad = inputs["action_is_pad"]
        image_is_pad = inputs["image_is_pad"]
        video_prefix_tokens = inputs["video_prefix_tokens"]

        # === Video flow matching ===
        noise_video = torch.randn_like(input_latents)
        timestep_video = self.train_video_scheduler.sample_training_t(
            batch_size=batch_size,
            device=self.device,
            dtype=input_latents.dtype,
        )
        latents = self.train_video_scheduler.add_noise(input_latents, noise_video, timestep_video)
        target_video = self.train_video_scheduler.training_target(input_latents, noise_video, timestep_video)

        # Fuse first frame (clean, no noise)
        first_frame_latents = inputs["first_frame_latents"]
        latents[:, :, 0:1] = first_frame_latents

        # === Action flow matching ===
        noise_action = torch.randn_like(action)
        timestep_action = self.train_action_scheduler.sample_training_t(
            batch_size=batch_size,
            device=self.device,
            dtype=action.dtype,
        )
        noisy_action = self.train_action_scheduler.add_noise(action, noise_action, timestep_action)
        target_action = self.train_action_scheduler.training_target(action, noise_action, timestep_action)

        # === Pre-DiT for both experts ===
        video_pre = self.video_expert.pre_dit(
            x=latents,
            timestep=timestep_video,
            context=context,
            context_mask=context_mask,
            fuse_vae_embedding_in_latents=True,  # First frame is clean
        )
        video_pre = self._apply_video_prefix_intent(video_pre, video_prefix_tokens)

        action_pre = self.action_expert.pre_dit(
            action_tokens=noisy_action,
            timestep=timestep_action,
            context=context,
            context_mask=context_mask,
        )

        # === MoT forward ===
        video_tokens = video_pre["tokens"]
        action_tokens = action_pre["tokens"]

        attention_mask = self._build_mot_attention_mask(
            video_seq_len=video_tokens.shape[1],
            action_seq_len=action_tokens.shape[1],
            video_tokens_per_frame=int(video_pre["meta"]["tokens_per_frame"]),
            device=video_tokens.device,
            video_prefix_len=int(video_pre["meta"].get("video_prefix_len", 0)),
        )

        tokens_out = self.mot(
            embeds_all={
                "video": video_tokens,
                "action": action_tokens,
            },
            attention_mask=attention_mask,
            freqs_all={
                "video": video_pre["freqs"],
                "action": action_pre["freqs"],
            },
            context_all={
                "video": {
                    "context": video_pre["context"],
                    "mask": video_pre["context_mask"],
                },
                "action": {
                    "context": action_pre["context"],
                    "mask": action_pre["context_mask"],
                },
            },
            t_mod_all={
                "video": video_pre["t_mod"],
                "action": action_pre["t_mod"],
            },
        )

        # === Post-DiT ===
        pred_video = self.video_expert.post_dit(tokens_out["video"], video_pre)
        pred_action = self.action_expert.post_dit(tokens_out["action"], action_pre)

        # === Compute losses ===
        # Skip first frame (it's clean conditioning, not predicted)
        pred_video = pred_video[:, :, 1:]
        target_video = self.video_expert.target_to_output_space(target_video)
        target_video = target_video[:, :, 1:]

        loss_video_per_sample = self._compute_video_loss_per_sample(
            pred_video=pred_video,
            target_video=target_video,
            image_is_pad=image_is_pad,
            include_initial_frame=False,
        )
        video_weight = self.train_video_scheduler.training_weight(timestep_video).to(
            loss_video_per_sample.device, dtype=loss_video_per_sample.dtype
        )
        loss_video = (loss_video_per_sample * video_weight).mean()

        # Action loss
        action_loss_token = F.mse_loss(
            pred_action.float(), target_action.float(), reduction="none"
        ).mean(dim=2)  # [B, T]
        if action_is_pad is not None:
            valid = (~action_is_pad).to(device=action_loss_token.device, dtype=action_loss_token.dtype)
            valid_sum = valid.sum(dim=1).clamp(min=1.0)
            action_loss_per_sample = (action_loss_token * valid).sum(dim=1) / valid_sum
        else:
            action_loss_per_sample = action_loss_token.mean(dim=1)

        action_weight = self.train_action_scheduler.training_weight(timestep_action).to(
            action_loss_per_sample.device, dtype=action_loss_per_sample.dtype
        )
        loss_action = (action_loss_per_sample * action_weight).mean()

        loss_total = self.loss_lambda_video * loss_video + self.loss_lambda_action * loss_action
        loss_dict = {
            "loss_video": self.loss_lambda_video * float(loss_video.detach().item()),
            "loss_action": self.loss_lambda_action * float(loss_action.detach().item()),
        }
        return loss_total, loss_dict

    @torch.no_grad()
    def _predict_action_noise_with_cache(
        self,
        latents_action: torch.Tensor,
        timestep_action: torch.Tensor,
        context: torch.Tensor,
        context_mask: torch.Tensor,
        video_kv_cache: list[dict[str, torch.Tensor]],
        attention_mask: torch.Tensor,
        video_seq_len: int,
    ) -> torch.Tensor:
        """Predict action velocity using cached video KV."""
        action_pre = self.action_expert.pre_dit(
            action_tokens=latents_action,
            timestep=timestep_action,
            context=context,
            context_mask=context_mask,
        )
        action_tokens = self.mot.forward_action_with_video_cache(
            action_tokens=action_pre["tokens"],
            action_freqs=action_pre["freqs"],
            action_t_mod=action_pre["t_mod"],
            action_context_payload={
                "context": action_pre["context"],
                "mask": action_pre["context_mask"],
            },
            video_kv_cache=video_kv_cache,
            attention_mask=attention_mask,
            video_seq_len=video_seq_len,
        )
        return self.action_expert.post_dit(action_tokens, action_pre)

    @torch.no_grad()
    def infer_action(
        self,
        prompt: Optional[str] = None,
        input_image: Optional[torch.Tensor] = None,
        history_dino_latents: Optional[torch.Tensor] = None,
        history_video: Optional[torch.Tensor] = None,
        action_horizon: int = 16,
        proprio: Optional[torch.Tensor] = None,
        context: Optional[torch.Tensor] = None,
        context_mask: Optional[torch.Tensor] = None,
        num_inference_steps: int = 20,
        sigma_shift: Optional[float] = None,
        seed: Optional[int] = None,
        rand_device: str = "cpu",
        tiled: bool = False,
    ) -> dict[str, Any]:
        """Infer action using DINO-encoded first frame and KV-cache.

        This is the primary inference method. It:
        1. Encodes the input image with frozen DINO
        2. Runs video expert once to build KV cache
        3. Iteratively denoises action using cached video features

        Args:
            prompt: Text instruction (mutually exclusive with context/context_mask).
            input_image: [1, 3, H, W] or [3, H, W] current observation image.
            history_dino_latents: Optional [1,D,T_h,H_g,W_g] or [D,T_h,H_g,W_g] history DINO cache.
            history_video: Optional history RGB frames, [T_h,3,H,W], [1,T_h,3,H,W], or [1,3,T_h,H,W].
            action_horizon: Number of action timesteps to predict.
            proprio: [D] or [1, D] proprioception state.
            context: [1, L, D] pre-computed text embeddings.
            context_mask: [1, L] text mask.
            num_inference_steps: Number of denoising steps.
            sigma_shift: Optional override for inference scheduler shift.
            seed: Random seed for reproducibility.
            rand_device: Device for random number generation.
            tiled: Unused (kept for API compatibility with original FastWAM).

        Returns:
            Dict with key 'action': [T_a, a_dim] predicted action chunk.
        """
        self.eval()
        if str(getattr(self.video_expert, "video_attention_mask_mode", "")) != "first_frame_causal":
            raise ValueError(
                "`infer_action` requires `video_attention_mask_mode='first_frame_causal'."
            )

        if input_image is None:
            raise ValueError("`input_image` is required for inference.")
        if input_image.ndim == 3:
            input_image = input_image.unsqueeze(0)
        if input_image.ndim != 4 or input_image.shape[0] != 1 or input_image.shape[1] != 3:
            raise ValueError(
                f"`input_image` must be [1, 3, H, W] or [3, H, W], got {tuple(input_image.shape)}"
            )

        if proprio is not None:
            if self.proprio_dim is None:
                raise ValueError("`proprio` provided but `proprio_dim=None`.")
            if proprio.ndim == 1:
                proprio = proprio.unsqueeze(0)
            if proprio.shape[1] != self.proprio_dim:
                raise ValueError(
                    f"`proprio` last dim must be {self.proprio_dim}, got {proprio.shape[1]}"
                )
            proprio = proprio.to(device=self.device, dtype=self.torch_dtype)

        # Initialize noisy action
        generator = None if seed is None else torch.Generator(device=rand_device).manual_seed(seed)
        latents_action = torch.randn(
            (1, action_horizon, self.action_expert.action_dim),
            generator=generator,
            device=rand_device,
            dtype=torch.float32,
        ).to(device=self.device, dtype=self.torch_dtype)

        # Encode input image with DINO
        input_image = input_image.to(device=self.device, dtype=self.torch_dtype)
        first_frame_latents = self._encode_single_frame_dino(input_image)  # [1, D, 1, H_g, W_g]

        # Prepare context
        video_prefix_tokens = None
        use_prompt = prompt is not None
        use_context = context is not None or context_mask is not None
        if use_prompt and use_context:
            raise ValueError("`prompt` and `context/context_mask` are mutually exclusive.")
        if not use_prompt and not use_context:
            raise ValueError("Either `prompt` or `context/context_mask` must be provided.")

        if use_prompt:
            context, context_mask = self.encode_prompt(prompt)
        else:
            if context is None or context_mask is None:
                raise ValueError("`context` and `context_mask` must both be provided.")
            if context.ndim == 2:
                context = context.unsqueeze(0)
            if context_mask.ndim == 1:
                context_mask = context_mask.unsqueeze(0)
            context = context.to(device=self.device, dtype=self.torch_dtype, non_blocking=True)
            context_mask = context_mask.to(device=self.device, dtype=torch.bool, non_blocking=True)

        if proprio is not None:
            context, context_mask = self._append_proprio_to_context(
                context=context, context_mask=context_mask, proprio=proprio
            )

        if self.intent_encoder is not None:
            if history_dino_latents is not None and history_video is not None:
                raise ValueError("Provide only one of `history_dino_latents` or `history_video`.")
            if history_dino_latents is not None:
                if history_dino_latents.ndim == 4:
                    history_dino_latents = history_dino_latents.unsqueeze(0)
                history_dino_latents = history_dino_latents.to(
                    device=self.device, dtype=self.torch_dtype, non_blocking=True
                )
            elif history_video is not None:
                if history_video.ndim == 4:
                    # [T,3,H,W] -> [1,3,T,H,W]
                    if history_video.shape[1] != 3:
                        raise ValueError(
                            f"`history_video` as [T,3,H,W] must have channel dim=3, "
                            f"got {tuple(history_video.shape)}"
                        )
                    history_video = history_video.permute(1, 0, 2, 3).unsqueeze(0)
                elif history_video.ndim == 5:
                    if history_video.shape[0] != 1:
                        raise ValueError(
                            f"`history_video` inference batch size must be 1, got "
                            f"{tuple(history_video.shape)}"
                        )
                    if history_video.shape[1] == 3:
                        pass  # [1,3,T,H,W]
                    elif history_video.shape[2] == 3:
                        history_video = history_video.permute(0, 2, 1, 3, 4)
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
                history_dino_latents = self._encode_video_dino(history_video)
            elif bool(self.intent_config.get("infer_repeat_current_if_missing", False)):
                history_count = len(
                    self.intent_config.get(
                        "history_offsets",
                        self.intent_config.get("history_dino_frame_offsets", [-8, -4, 0]),
                    )
                )
                history_dino_latents = first_frame_latents.expand(-1, -1, history_count, -1, -1)
            else:
                raise ValueError(
                    "`history_dino_latents` or `history_video` is required for inference when "
                    "`intent_config.enabled=true`."
                )

            intent_tokens = self._encode_history_intent(history_dino_latents)
            if self.intent_injection_mode == "context_after_proprio":
                context, context_mask = self._append_intent_to_context(
                    context=context,
                    context_mask=context_mask,
                    intent_tokens=intent_tokens,
                )
            elif self.intent_injection_mode == "video_prefix":
                video_prefix_tokens = intent_tokens
            else:
                raise RuntimeError(f"Unhandled intent injection mode: {self.intent_injection_mode}")

        # Video prefill: encode first frame, cache KV
        timestep_video = torch.zeros(
            (first_frame_latents.shape[0],),
            dtype=first_frame_latents.dtype,
            device=self.device,
        )
        video_pre = self.video_expert.pre_dit(
            x=first_frame_latents,
            timestep=timestep_video,
            context=context,
            context_mask=context_mask,
            fuse_vae_embedding_in_latents=True,
        )
        video_pre = self._apply_video_prefix_intent(video_pre, video_prefix_tokens)

        video_seq_len = int(video_pre["tokens"].shape[1])
        attention_mask = self._build_mot_attention_mask(
            video_seq_len=video_seq_len,
            action_seq_len=latents_action.shape[1],
            video_tokens_per_frame=int(video_pre["meta"]["tokens_per_frame"]),
            device=video_pre["tokens"].device,
            video_prefix_len=int(video_pre["meta"].get("video_prefix_len", 0)),
        )

        video_kv_cache = self.mot.prefill_video_cache(
            video_tokens=video_pre["tokens"],
            video_freqs=video_pre["freqs"],
            video_t_mod=video_pre["t_mod"],
            video_context_payload={
                "context": video_pre["context"],
                "mask": video_pre["context_mask"],
            },
            video_attention_mask=attention_mask[:video_seq_len, :video_seq_len],
        )

        # Action denoising loop
        infer_timesteps_action, infer_deltas_action = self.infer_action_scheduler.build_inference_schedule(
            num_inference_steps=num_inference_steps,
            device=self.device,
            dtype=latents_action.dtype,
            shift_override=sigma_shift,
        )

        for step_t_action, step_delta_action in zip(infer_timesteps_action, infer_deltas_action):
            timestep_action = step_t_action.unsqueeze(0).to(
                dtype=latents_action.dtype, device=self.device
            )

            pred_action = self._predict_action_noise_with_cache(
                latents_action=latents_action,
                timestep_action=timestep_action,
                context=context,
                context_mask=context_mask,
                video_kv_cache=video_kv_cache,
                attention_mask=attention_mask,
                video_seq_len=video_seq_len,
            )

            latents_action = self.infer_action_scheduler.step(
                pred_action, step_delta_action, latents_action
            )

        return {
            "action": latents_action[0].detach().to(device="cpu", dtype=torch.float32),
        }

    def save_checkpoint(self, path, optimizer=None, step=None):
        """Save model checkpoint (only trainable components)."""
        payload = {
            "mot": self.mot.state_dict(),
            "step": step,
            "torch_dtype": str(self.torch_dtype),
        }
        if self.proprio_encoder is not None:
            payload["proprio_encoder"] = self.proprio_encoder.state_dict()
        if self.intent_encoder is not None:
            payload["intent_encoder"] = self.intent_encoder.state_dict()
            payload["intent_config"] = self.intent_config
        if optimizer is not None:
            payload["optimizer"] = optimizer.state_dict()
        torch.save(payload, path)

    def _validate_checkpoint_intent_config(self, checkpoint: dict[str, Any]) -> None:
        if "intent_encoder" not in checkpoint:
            return
        if self.intent_encoder is None:
            raise ValueError(
                "Checkpoint contains `intent_encoder`, but this model was built with "
                "`intent_config.enabled=false` or without intent_config. Rebuild the model "
                "with the checkpoint's intent_config before evaluation/loading."
            )
        checkpoint_config = dict(checkpoint.get("intent_config") or {})
        checkpoint_mode = str(checkpoint_config.get("injection_mode", "context_after_proprio"))
        if checkpoint_mode != self.intent_injection_mode:
            raise ValueError(
                "Intent injection mode mismatch between checkpoint and current model: "
                f"checkpoint={checkpoint_mode!r}, current={self.intent_injection_mode!r}."
            )
        for key in (
            "history_offsets",
            "max_history_frames",
            "num_intent_tokens",
            "resampler_dim",
            "num_resampler_layers",
            "num_heads",
            "mlp_ratio",
        ):
            if key in checkpoint_config and key in self.intent_config:
                if checkpoint_config[key] != self.intent_config[key]:
                    raise ValueError(
                        f"Intent config mismatch for `{key}`: "
                        f"checkpoint={checkpoint_config[key]!r}, "
                        f"current={self.intent_config[key]!r}."
                    )

    def load_checkpoint(self, path, optimizer=None):
        """Load model checkpoint."""
        checkpoint = torch.load(path, map_location="cpu")
        self._validate_checkpoint_intent_config(checkpoint)
        missing, unexpected = self.mot.load_state_dict(checkpoint["mot"], strict=False)
        if missing or unexpected:
            logger.warning(
                "FastWAM_DINO checkpoint loaded with non-strict MoT keys: "
                "missing=%d unexpected=%d missing_head=%s unexpected_head=%s",
                len(missing),
                len(unexpected),
                missing[:10],
                unexpected[:10],
            )
        if self.proprio_encoder is not None and "proprio_encoder" in checkpoint:
            self.proprio_encoder.load_state_dict(checkpoint["proprio_encoder"])
        if self.intent_encoder is not None:
            if "intent_encoder" in checkpoint:
                self.intent_encoder.load_state_dict(checkpoint["intent_encoder"])
            else:
                logger.info(
                    "Checkpoint has no `intent_encoder`; keeping current intent encoder initialization."
                )
        if optimizer is not None and "optimizer" in checkpoint:
            optimizer.load_state_dict(checkpoint["optimizer"])
        step = checkpoint.get("step", None)
        logger.info(f"FastWAM_DINO checkpoint loaded from {path}, step={step}")
        return step
