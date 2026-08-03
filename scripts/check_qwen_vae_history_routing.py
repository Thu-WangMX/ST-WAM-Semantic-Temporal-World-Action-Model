"""Lightweight routing checks for Qwen VAE-history without loading large backbones."""

from __future__ import annotations

from unittest.mock import patch

import torch
import torch.nn as nn

from fastwam.models.wan22.fastwam_vae_dino_mot import FastWAMVAEDinoMoT
from fastwam.models.wan22.semantic_history import QwenDINOHistoryActionAdapter


class _DummyVLM(nn.Module):
    def __init__(self, hidden_dim: int = 32):
        super().__init__()
        self.embedding = nn.Embedding(4, hidden_dim)
        self.config = type("Config", (), {"hidden_size": hidden_dim})()


class _CaptureAdapter(nn.Module):
    def __init__(self):
        super().__init__()
        self.kwargs = None

    def forward(self, **kwargs):
        self.kwargs = kwargs
        return torch.zeros(kwargs["semantic_image"].shape[0], 8, 16)


def check_adapter_contract() -> None:
    with (
        patch.object(QwenDINOHistoryActionAdapter, "_load_processor", return_value=None),
        patch.object(QwenDINOHistoryActionAdapter, "_load_vlm_model", return_value=_DummyVLM()),
    ):
        adapter = QwenDINOHistoryActionAdapter(
            vlm_model_name_or_path="unused",
            dino_dim=48,
            text_dim=16,
            history_offsets=[-24, -16, -8, -1],
            num_output_tokens=8,
            resampler_dim=32,
            num_layers=1,
            num_heads=4,
            history_source="vae",
            torch_dtype=torch.float32,
        )
    adapter.encode_qwen_hidden = lambda prompts, semantic_image: (
        torch.randn(semantic_image.shape[0], 6, 32),
        torch.ones(semantic_image.shape[0], 6, dtype=torch.long),
    )
    output = adapter(
        prompts=["task", "task"],
        semantic_image=torch.zeros(2, 3, 16, 16),
        history_vae_latents=torch.randn(2, 48, 4, 2, 3),
    )
    assert output.shape == (2, 8, 16)
    try:
        adapter(
            prompts=["task", "task"],
            semantic_image=torch.zeros(2, 3, 16, 16),
            history_dino_latents=torch.randn(2, 48, 4, 2, 3),
        )
    except ValueError as error:
        assert "VAE semantic history" in str(error)
    else:
        raise AssertionError("VAE semantic adapter accepted DINO history latents.")


def check_existing_adapter_modes() -> None:
    with (
        patch.object(QwenDINOHistoryActionAdapter, "_load_processor", return_value=None),
        patch.object(QwenDINOHistoryActionAdapter, "_load_vlm_model", return_value=_DummyVLM()),
    ):
        dino_adapter = QwenDINOHistoryActionAdapter(
            vlm_model_name_or_path="unused",
            dino_dim=384,
            text_dim=16,
            history_offsets=[-24, -16, -8, -1],
            num_output_tokens=8,
            resampler_dim=32,
            num_layers=1,
            num_heads=4,
            history_source="dino",
            torch_dtype=torch.float32,
        )
        current_adapter = QwenDINOHistoryActionAdapter(
            vlm_model_name_or_path="unused",
            dino_dim=384,
            text_dim=16,
            history_offsets=[],
            use_history=False,
            num_output_tokens=8,
            resampler_dim=32,
            num_layers=1,
            num_heads=4,
            torch_dtype=torch.float32,
        )
    for adapter in (dino_adapter, current_adapter):
        adapter.encode_qwen_hidden = lambda prompts, semantic_image: (
            torch.randn(semantic_image.shape[0], 6, 32),
            torch.ones(semantic_image.shape[0], 6, dtype=torch.long),
        )
    dino_output = dino_adapter(
        prompts=["task", "task"],
        semantic_image=torch.zeros(2, 3, 16, 16),
        history_dino_latents=torch.randn(2, 384, 4, 2, 3),
    )
    current_output = current_adapter(
        prompts=["task", "task"],
        semantic_image=torch.zeros(2, 3, 16, 16),
    )
    assert dino_output.shape == current_output.shape == (2, 8, 16)


def check_model_routing() -> None:
    model = FastWAMVAEDinoMoT.__new__(FastWAMVAEDinoMoT)
    nn.Module.__init__(model)
    capture = _CaptureAdapter()
    model.semantic_history_encoder = capture
    model.semantic_history_uses_history = True
    model.semantic_history_source = "vae"
    model.semantic_history_frame_count = 4
    model.device = torch.device("cpu")
    model.torch_dtype = torch.float32
    model._encode_history_vae_video = lambda history_video, tiled=False: torch.ones(
        history_video.shape[0], 48, history_video.shape[2], 2, 3
    )
    model._validate_vae_history_latent_shape = lambda latents: None

    output = model._encode_semantic_history_tokens(
        prompts=["task", "task"],
        semantic_image=torch.zeros(2, 3, 16, 16),
        history_video=torch.zeros(2, 3, 4, 16, 16),
    )
    assert output.shape == (2, 8, 16)
    assert capture.kwargs["history_dino_latents"] is None
    assert capture.kwargs["history_vae_latents"].shape == (2, 48, 4, 2, 3)

    output = model._encode_semantic_history_tokens(
        prompts="task",
        semantic_image=torch.zeros(3, 16, 16),
        history_video=torch.zeros(4, 3, 16, 16),
    )
    assert output.shape == (1, 8, 16)
    assert capture.kwargs["history_dino_latents"] is None
    assert capture.kwargs["history_vae_latents"].shape == (1, 48, 4, 2, 3)


def check_checkpoint_source_guard() -> None:
    model = FastWAMVAEDinoMoT.__new__(FastWAMVAEDinoMoT)
    nn.Module.__init__(model)
    model.semantic_history_encoder = _CaptureAdapter()
    model.semantic_history_config = {
        "source": "vae",
        "use_history": True,
        "history_offsets": [-24, -16, -8, -1],
    }
    try:
        model._validate_checkpoint_semantic_history_config(
            {
                "semantic_history_encoder": {},
                "semantic_history_config": {
                    "source": "dino",
                    "use_history": True,
                    "history_offsets": [-24, -16, -8, -1],
                },
            }
        )
    except ValueError as error:
        assert "source mismatch" in str(error)
    else:
        raise AssertionError("Checkpoint source mismatch was not rejected.")


if __name__ == "__main__":
    check_adapter_contract()
    check_existing_adapter_modes()
    check_model_routing()
    check_checkpoint_source_guard()
    print("Qwen VAE-history routing checks passed.")
