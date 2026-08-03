"""Qwen-VL current semantic tokens with optional short visual-history fusion."""

from __future__ import annotations

from typing import Any, Iterable, Optional, Sequence

import torch
import torch.nn as nn
from PIL import Image


def _as_list_of_prompts(prompts: str | Sequence[str], batch_size: int) -> list[str]:
    if isinstance(prompts, str):
        prompts = [prompts]
    else:
        prompts = list(prompts)
    if len(prompts) != batch_size:
        raise ValueError(f"Prompt batch mismatch: got {len(prompts)} prompts for batch={batch_size}.")
    return [str(prompt) for prompt in prompts]


def _normalized_tensor_to_pil(image: torch.Tensor) -> Image.Image:
    if image.ndim != 3 or image.shape[0] != 3:
        raise ValueError(f"Semantic image must be [3,H,W], got {tuple(image.shape)}")
    x = image.detach().to(device="cpu", dtype=torch.float32).clamp(-1.0, 1.0)
    x = ((x + 1.0) * 127.5).round().clamp(0, 255).to(dtype=torch.uint8)
    arr = x.permute(1, 2, 0).contiguous().numpy()
    return Image.fromarray(arr, mode="RGB")


def _move_vlm_inputs_to_device(inputs: dict[str, Any], device: torch.device, dtype: torch.dtype) -> dict[str, Any]:
    moved: dict[str, Any] = {}
    for key, value in inputs.items():
        if torch.is_tensor(value):
            if torch.is_floating_point(value):
                moved[key] = value.to(device=device, dtype=dtype, non_blocking=True)
            else:
                moved[key] = value.to(device=device, non_blocking=True)
        else:
            moved[key] = value
    return moved


class QwenDINOHistoryActionAdapter(nn.Module):
    """Frozen Qwen-VL current semantics + frozen visual history -> action tokens.

    The VLM itself is frozen and excluded from FastWAM checkpoints. Only the
    small resampler/fusion adapter below is trainable.
    """

    def __init__(
        self,
        *,
        vlm_model_name_or_path: str,
        dino_dim: int,
        text_dim: int,
        history_offsets: Sequence[int],
        num_output_tokens: int = 8,
        resampler_dim: int = 1024,
        num_layers: int = 2,
        num_heads: int = 8,
        dropout: float = 0.0,
        max_history_frames: Optional[int] = None,
        use_history: bool = True,
        vlm_family: str = "qwen3_vl",
        trust_remote_code: bool = True,
        freeze_vlm: bool = True,
        processor_min_pixels: Optional[int] = None,
        processor_max_pixels: Optional[int] = None,
        attn_implementation: Optional[str] = None,
        torch_dtype: torch.dtype = torch.bfloat16,
        history_source: str = "dino",
    ):
        super().__init__()
        self.history_source = str(history_source).strip().lower()
        if self.history_source not in {"dino", "vae"}:
            raise ValueError(
                f"Unsupported semantic history source {self.history_source!r}; expected 'dino' or 'vae'."
            )
        self.use_history = bool(use_history)
        if self.use_history and not history_offsets:
            raise ValueError("`semantic_history_config.history_offsets` must contain at least one frame offset.")
        self.vlm_model_name_or_path = str(vlm_model_name_or_path)
        self.vlm_family = str(vlm_family).strip().lower()
        self.history_offsets = [int(offset) for offset in history_offsets] if self.use_history else []
        self.max_history_frames = int(
            max_history_frames or max(1, len(self.history_offsets))
        )
        if self.use_history and self.max_history_frames < len(self.history_offsets):
            raise ValueError(
                "`semantic_history_config.max_history_frames` must be >= len(history_offsets), "
                f"got {self.max_history_frames} vs {len(self.history_offsets)}."
            )
        self.num_output_tokens = int(num_output_tokens)
        self.resampler_dim = int(resampler_dim)
        self.text_dim = int(text_dim)
        self.dino_dim = int(dino_dim)
        self.freeze_vlm = bool(freeze_vlm)
        self.vlm_dtype = torch_dtype

        self.processor = self._load_processor(
            processor_min_pixels=processor_min_pixels,
            processor_max_pixels=processor_max_pixels,
            trust_remote_code=trust_remote_code,
        )
        self.vlm_model = self._load_vlm_model(
            trust_remote_code=trust_remote_code,
            attn_implementation=attn_implementation,
            torch_dtype=torch_dtype,
        )
        self.vlm_hidden_dim = self._infer_vlm_hidden_dim(self.vlm_model)
        if self.freeze_vlm:
            self.vlm_model.eval()
            self.vlm_model.requires_grad_(False)

        self.qwen_in = nn.Linear(self.vlm_hidden_dim, self.resampler_dim)
        self.qwen_queries = nn.Parameter(torch.randn(self.num_output_tokens, self.resampler_dim) * 0.02)
        self.qwen_attn = nn.MultiheadAttention(
            self.resampler_dim,
            num_heads=int(num_heads),
            dropout=float(dropout),
            batch_first=True,
        )
        self.history_in = nn.Linear(self.dino_dim, self.resampler_dim)
        self.history_time_embed = nn.Embedding(self.max_history_frames, self.resampler_dim)
        self.fusion_layers = nn.ModuleList(
            [
                nn.ModuleDict(
                    {
                        "q_norm": nn.LayerNorm(self.resampler_dim),
                        "h_norm": nn.LayerNorm(self.resampler_dim),
                        "attn": nn.MultiheadAttention(
                            self.resampler_dim,
                            num_heads=int(num_heads),
                            dropout=float(dropout),
                            batch_first=True,
                        ),
                        "ffn": nn.Sequential(
                            nn.LayerNorm(self.resampler_dim),
                            nn.Linear(self.resampler_dim, self.resampler_dim * 4),
                            nn.SiLU(),
                            nn.Linear(self.resampler_dim * 4, self.resampler_dim),
                        ),
                    }
                )
                for _ in range(int(num_layers))
            ]
        )
        self.out_norm = nn.LayerNorm(self.resampler_dim)
        self.out = nn.Linear(self.resampler_dim, self.text_dim)

    def _load_processor(self, *, processor_min_pixels, processor_max_pixels, trust_remote_code: bool):
        from transformers import AutoProcessor

        kwargs: dict[str, Any] = {"trust_remote_code": bool(trust_remote_code)}
        if processor_min_pixels is not None:
            kwargs["min_pixels"] = int(processor_min_pixels)
        if processor_max_pixels is not None:
            kwargs["max_pixels"] = int(processor_max_pixels)
        return AutoProcessor.from_pretrained(self.vlm_model_name_or_path, **kwargs)

    def _select_vlm_model_class(self):
        import transformers

        family = self.vlm_family
        if family in {"qwen3", "qwen3_vl", "qwen3vl"}:
            cls = getattr(transformers, "Qwen3VLForConditionalGeneration", None)
            if cls is not None:
                return cls
            auto_cls = getattr(transformers, "AutoModelForImageTextToText", None)
            if auto_cls is not None:
                return auto_cls
            auto_cls = getattr(transformers, "AutoModelForVision2Seq", None)
            if auto_cls is not None:
                return auto_cls
            raise ImportError(
                "Qwen3-VL is requested, but this transformers install does not expose "
                "Qwen3VLForConditionalGeneration or a compatible AutoModel class. "
                "Upgrade transformers on the training machine, or set "
                "semantic_history_config.vlm_family=qwen2_5_vl."
            )
        if family in {"qwen2_5", "qwen2_5_vl", "qwen25vl"}:
            cls = getattr(transformers, "Qwen2_5_VLForConditionalGeneration", None)
            if cls is not None:
                return cls
            return getattr(transformers, "AutoModelForVision2Seq")
        if family == "auto":
            return getattr(transformers, "AutoModelForImageTextToText")
        raise ValueError(
            f"Unsupported semantic_history_config.vlm_family={self.vlm_family!r}; "
            "expected qwen3_vl, qwen2_5_vl, or auto."
        )

    def _load_vlm_model(self, *, trust_remote_code: bool, attn_implementation: Optional[str], torch_dtype: torch.dtype):
        cls = self._select_vlm_model_class()
        kwargs: dict[str, Any] = {
            "torch_dtype": torch_dtype,
            "trust_remote_code": bool(trust_remote_code),
        }
        if attn_implementation:
            kwargs["attn_implementation"] = str(attn_implementation)
        try:
            return cls.from_pretrained(self.vlm_model_name_or_path, **kwargs)
        except TypeError:
            kwargs.pop("attn_implementation", None)
            return cls.from_pretrained(self.vlm_model_name_or_path, **kwargs)

    @staticmethod
    def _infer_vlm_hidden_dim(model) -> int:
        config = getattr(model, "config", None)
        candidates = [getattr(config, "text_config", None), config]
        for candidate in candidates:
            if candidate is None:
                continue
            for attr in ("hidden_size", "dim", "embed_dim"):
                value = getattr(candidate, attr, None)
                if value is not None:
                    return int(value)
        embeddings = model.get_input_embeddings()
        return int(embeddings.embedding_dim)

    def trainable_parameters(self) -> Iterable[nn.Parameter]:
        for name, parameter in self.named_parameters():
            if name.startswith("vlm_model."):
                continue
            yield parameter

    def set_adapter_train_mode(self) -> None:
        self.eval()
        self.requires_grad_(False)
        self.vlm_model.eval()
        for module in (
            self.qwen_in,
            self.qwen_attn,
            self.history_in,
            self.history_time_embed,
            self.fusion_layers,
            self.out_norm,
            self.out,
        ):
            module.train()
            module.requires_grad_(True)
        self.qwen_queries.requires_grad_(True)

    def adapter_state_dict(self) -> dict[str, torch.Tensor]:
        return {
            key: value
            for key, value in self.state_dict().items()
            if not key.startswith("vlm_model.")
        }

    def load_adapter_state_dict(self, state_dict: dict[str, torch.Tensor], strict: bool = True):
        own_keys = set(self.adapter_state_dict().keys())
        incoming_keys = set(state_dict.keys())
        missing = sorted(own_keys - incoming_keys)
        unexpected = sorted(incoming_keys - own_keys)
        if strict and (missing or unexpected):
            raise RuntimeError(
                "Semantic history adapter state mismatch: "
                f"missing={missing[:10]} unexpected={unexpected[:10]}"
            )
        own_state = self.state_dict()
        own_state.update({key: value for key, value in state_dict.items() if key in own_keys})
        return self.load_state_dict(own_state, strict=False)

    def _preprocess_vlm_inputs(self, prompts: list[str], semantic_image: torch.Tensor) -> dict[str, Any]:
        pil_images = [_normalized_tensor_to_pil(image) for image in semantic_image]
        texts = []
        for prompt, pil_image in zip(prompts, pil_images):
            messages = [
                {
                    "role": "user",
                    "content": [
                        {"type": "image", "image": pil_image},
                        {"type": "text", "text": prompt},
                    ],
                }
            ]
            if hasattr(self.processor, "apply_chat_template"):
                text = self.processor.apply_chat_template(
                    messages,
                    tokenize=False,
                    add_generation_prompt=True,
                )
            else:
                text = prompt
            texts.append(text)

        inputs = self.processor(
            text=texts,
            images=pil_images,
            padding=True,
            return_tensors="pt",
        )
        return dict(inputs)

    @torch.no_grad()
    def encode_qwen_hidden(
        self,
        prompts: str | Sequence[str],
        semantic_image: torch.Tensor,
    ) -> tuple[torch.Tensor, Optional[torch.Tensor]]:
        if semantic_image.ndim == 3:
            semantic_image = semantic_image.unsqueeze(0)
        if semantic_image.ndim != 4 or semantic_image.shape[1] != 3:
            raise ValueError(
                f"`semantic_image` must be [B,3,H,W] or [3,H,W], got {tuple(semantic_image.shape)}"
            )
        batch_size = int(semantic_image.shape[0])
        prompts = _as_list_of_prompts(prompts, batch_size)
        device = next(self.vlm_model.parameters()).device
        inputs = self._preprocess_vlm_inputs(prompts, semantic_image)
        inputs = _move_vlm_inputs_to_device(inputs, device=device, dtype=self.vlm_dtype)
        outputs = self.vlm_model(
            **inputs,
            output_hidden_states=True,
            return_dict=True,
            use_cache=False,
        )
        hidden_states = getattr(outputs, "hidden_states", None)
        if hidden_states is not None:
            hidden = hidden_states[-1]
        else:
            hidden = getattr(outputs, "last_hidden_state", None)
        if hidden is None:
            raise RuntimeError("VLM forward returned neither hidden_states nor last_hidden_state.")
        attention_mask = inputs.get("attention_mask", None)
        return hidden.detach(), attention_mask

    def _resample_qwen(self, qwen_hidden: torch.Tensor, attention_mask: Optional[torch.Tensor]) -> torch.Tensor:
        dtype = self.qwen_in.weight.dtype
        qwen_hidden = self.qwen_in(qwen_hidden.to(device=self.qwen_in.weight.device, dtype=dtype))
        queries = self.qwen_queries.to(device=qwen_hidden.device, dtype=dtype).unsqueeze(0).expand(
            qwen_hidden.shape[0], -1, -1
        )
        key_padding_mask = None
        if attention_mask is not None:
            key_padding_mask = ~attention_mask.to(device=qwen_hidden.device, dtype=torch.bool)
        tokens, _ = self.qwen_attn(
            query=queries,
            key=qwen_hidden,
            value=qwen_hidden,
            key_padding_mask=key_padding_mask,
            need_weights=False,
        )
        return tokens

    def _history_tokens(self, history_latents: torch.Tensor) -> torch.Tensor:
        if history_latents.ndim != 5:
            raise ValueError(
                f"History latents must be [B,D,T,H,W], got {tuple(history_latents.shape)}"
            )
        batch_size, dino_dim, history_t, hist_h, hist_w = history_latents.shape
        if dino_dim != self.dino_dim:
            raise ValueError(
                f"History {self.history_source.upper()} dim mismatch: "
                f"got {dino_dim}, expected {self.dino_dim}."
            )
        if history_t != len(self.history_offsets):
            raise ValueError(
                "History temporal length does not match semantic_history_config.history_offsets: "
                f"got T={history_t}, expected {len(self.history_offsets)}."
            )
        if history_t > self.max_history_frames:
            raise ValueError(f"History T={history_t} exceeds max_history_frames={self.max_history_frames}.")
        dtype = self.history_in.weight.dtype
        history = history_latents.to(device=self.history_in.weight.device, dtype=dtype)
        history = history.permute(0, 2, 3, 4, 1).reshape(batch_size, history_t * hist_h * hist_w, dino_dim)
        history = self.history_in(history)
        time_ids = torch.arange(history_t, device=history.device).view(1, history_t, 1, 1)
        time_ids = time_ids.expand(batch_size, history_t, hist_h, hist_w).reshape(batch_size, -1)
        history = history + self.history_time_embed(time_ids)
        return history

    def forward(
        self,
        *,
        prompts: str | Sequence[str],
        semantic_image: torch.Tensor,
        history_dino_latents: Optional[torch.Tensor] = None,
        history_vae_latents: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        qwen_hidden, attention_mask = self.encode_qwen_hidden(prompts, semantic_image)
        qwen_tokens = self._resample_qwen(qwen_hidden, attention_mask)
        if not self.use_history:
            if history_dino_latents is not None or history_vae_latents is not None:
                raise ValueError(
                    "History latents were provided although this semantic adapter has use_history=false."
                )
            return self.out(self.out_norm(qwen_tokens))
        if history_dino_latents is not None and history_vae_latents is not None:
            raise ValueError("Provide only one of `history_dino_latents` or `history_vae_latents`.")
        if self.history_source == "dino":
            if history_vae_latents is not None:
                raise ValueError("DINO semantic history cannot consume `history_vae_latents`.")
            history_latents = history_dino_latents
        else:
            if history_dino_latents is not None:
                raise ValueError("VAE semantic history cannot consume `history_dino_latents`.")
            history_latents = history_vae_latents
        if history_latents is None:
            raise ValueError(
                f"History {self.history_source.upper()} latents are required when use_history=true."
            )
        history_tokens = self._history_tokens(history_latents)
        x = qwen_tokens
        for layer in self.fusion_layers:
            q = layer["q_norm"](x)
            h = layer["h_norm"](history_tokens)
            y, _ = layer["attn"](query=q, key=h, value=h, need_weights=False)
            x = x + y
            x = x + layer["ffn"](x)
        return self.out(self.out_norm(x))
