#!/usr/bin/env python3
"""Lightweight regression checks for DINO merged-token loss.

This script intentionally uses tiny CPU-only DinoVideoDiT instances.  It checks
the shape and target-space contracts that matter for the real [1,2,2] LIBERO
run without loading the 1B model or launching training.
"""

from __future__ import annotations

import inspect
import sys
from pathlib import Path

import torch
from einops import rearrange
from hydra import compose, initialize_config_dir

REPO_ROOT = Path(__file__).resolve().parents[1]
SRC_DIR = REPO_ROOT / "src"
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

from fastwam.models.wan22.action_dit import ActionDiT
from fastwam.models.wan22.dino_video_dit import DinoVideoDiT
from fastwam.models.wan22.fastwam_dino import FastWAM_DINO
from fastwam.models.wan22.mot import MoT
from fastwam.trainer import Wan22Trainer


def _make_tiny_dit(
    *,
    latent_patch_size: tuple[int, int, int] = (1, 1, 1),
    latent_patch_mode: str = "flat",
    latent_num_views: int = 1,
    output_patch_space: str = "dense",
    dino_dim: int = 8,
) -> DinoVideoDiT:
    return DinoVideoDiT(
        hidden_dim=32,
        dino_dim=dino_dim,
        ffn_dim=64,
        text_dim=16,
        freq_dim=16,
        num_heads=2,
        attn_head_dim=8,
        num_layers=1,
        video_attention_mask_mode="first_frame_causal",
        use_gradient_checkpointing=False,
        latent_patch_size=latent_patch_size,
        latent_patch_mode=latent_patch_mode,
        latent_num_views=latent_num_views,
        output_patch_space=output_patch_space,
    ).eval()


def _run_pre_post(model: DinoVideoDiT, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    b = x.shape[0]
    context = torch.randn(b, 3, 16)
    context_mask = torch.ones(b, 3, dtype=torch.bool)
    timestep = torch.rand(b)

    pre = model.pre_dit(
        x=x,
        timestep=timestep,
        context=context,
        context_mask=context_mask,
        fuse_vae_embedding_in_latents=True,
    )
    pred = model.post_dit(torch.randn_like(pre["tokens"]), pre)
    return pre["tokens"], pred


def check_old_no_pool_dense_default() -> None:
    model = _make_tiny_dit()
    x = torch.randn(2, 8, 3, 4, 8)
    tokens, pred = _run_pre_post(model, x)

    assert model.output_patch_space == "dense"
    assert tokens.shape == (2, 3 * 4 * 8, 32)
    assert pred.shape == x.shape
    assert model.target_to_output_space(x) is x
    print("[OK] old no-pool default stays dense:", tuple(pred.shape))


def check_old_viewpatch_dense_default() -> None:
    model = _make_tiny_dit(
        latent_patch_size=(1, 2, 2),
        latent_patch_mode="view",
        latent_num_views=2,
    )
    x = torch.randn(2, 8, 3, 4, 8)
    tokens, pred = _run_pre_post(model, x)

    assert model.output_patch_space == "dense"
    assert tokens.shape == (2, 3 * 2 * 2 * 2, 32)
    assert pred.shape == x.shape
    assert model.target_to_output_space(x) is x
    print("[OK] old [1,2,2] viewpatch still unpatchifies dense:", tuple(pred.shape))


def check_viewpatch_merged_output_and_target() -> None:
    model = _make_tiny_dit(
        latent_patch_size=(1, 2, 2),
        latent_patch_mode="view",
        latent_num_views=2,
        output_patch_space="merged",
    )
    x = torch.arange(2 * 8 * 3 * 4 * 8, dtype=torch.float32).reshape(2, 8, 3, 4, 8)
    tokens, pred = _run_pre_post(model, x)
    target = model.target_to_output_space(x)
    manual_target = rearrange(
        x,
        "b d t (h ph) (v w pw) -> b d t h (v w) (ph pw)",
        v=2,
        ph=2,
        pw=2,
    ).mean(dim=-1)

    assert tokens.shape == (2, 3 * 2 * 2 * 2, 32)
    assert pred.shape == (2, 8, 3, 2, 4)
    assert target.shape == pred.shape
    assert torch.equal(target, manual_target)
    print("[OK] merged [1,2,2] predicts and supervises compact grid:", tuple(pred.shape))


def check_loss_mask_alignment() -> None:
    model = _make_tiny_dit(
        latent_patch_size=(1, 2, 2),
        latent_patch_mode="view",
        latent_num_views=2,
        output_patch_space="merged",
    )
    dense_target = torch.randn(2, 8, 3, 4, 8)
    pred = torch.randn(2, 8, 3, 2, 4)
    pred_tail = pred[:, :, 1:]
    target_tail = model.target_to_output_space(dense_target)[:, :, 1:]
    image_is_pad = torch.tensor([[False, False, True], [False, False, False]])

    per_sample = FastWAM_DINO._compute_video_loss_per_sample(
        object(),
        pred_video=pred_tail,
        target_video=target_tail,
        image_is_pad=image_is_pad,
        include_initial_frame=False,
    )

    assert pred_tail.shape == target_tail.shape == (2, 8, 2, 2, 4)
    assert per_sample.shape == (2,)
    assert torch.isfinite(per_sample).all()
    print("[OK] training-loss tail and image mask align:", tuple(pred_tail.shape))


def _make_tiny_fastwam(output_patch_space: str) -> FastWAM_DINO:
    video = _make_tiny_dit(
        latent_patch_size=(1, 2, 2),
        latent_patch_mode="view",
        latent_num_views=2,
        output_patch_space=output_patch_space,
    )
    action = ActionDiT(
        hidden_dim=32,
        action_dim=6,
        ffn_dim=64,
        text_dim=16,
        freq_dim=16,
        eps=1e-6,
        num_heads=2,
        attn_head_dim=8,
        num_layers=1,
        use_gradient_checkpointing=False,
    )
    mot = MoT(
        mixtures={"video": video, "action": action},
        mot_checkpoint_mixed_attn=False,
    )
    return FastWAM_DINO(
        video_expert=video,
        action_expert=action,
        mot=mot,
        dino_encoder=None,
        text_dim=16,
        device="cpu",
        torch_dtype=torch.float32,
        loss_lambda_video=0.05,
        loss_lambda_action=5.0,
    )


def check_full_tiny_training_loss() -> None:
    sample = {
        "dino_latents": torch.randn(2, 8, 3, 4, 8),
        "action": torch.randn(2, 4, 6),
        "context": torch.randn(2, 3, 16),
        "context_mask": torch.ones(2, 3, dtype=torch.bool),
        "action_is_pad": torch.tensor([[False, False, False, True], [False, False, False, False]]),
        "image_is_pad": torch.tensor([[False, False, True], [False, False, False]]),
    }

    for output_patch_space in ("dense", "merged"):
        model = _make_tiny_fastwam(output_patch_space)
        model.train()
        with torch.enable_grad():
            loss, loss_dict = model.training_loss(sample)
            assert loss.ndim == 0
            assert torch.isfinite(loss)
            assert set(loss_dict) == {"loss_video", "loss_action"}
            loss.backward()

        grad_params = [
            p.grad
            for p in model.dit.parameters()
            if p.requires_grad and p.grad is not None
        ]
        assert grad_params, f"No gradients reached MoT parameters for {output_patch_space}"
        assert all(torch.isfinite(g).all() for g in grad_params[:10])
        print(f"[OK] full tiny FastWAM_DINO training_loss/backward works in {output_patch_space} mode")


def check_hydra_configs() -> None:
    cfg_dir = str(REPO_ROOT / "configs")
    with initialize_config_dir(config_dir=cfg_dir, version_base="1.3"):
        no_pool = compose(
            config_name="train",
            overrides=["task=libero_dino_s_smallvideo_2cam_224_1e-4"],
        )
        dense_view = compose(
            config_name="train",
            overrides=["task=libero_dino_s_smallvideo_2cam_224_viewpatch_1x2x2_1e-4"],
        )
        merged = compose(
            config_name="train",
            overrides=[
                "task=libero_dino_s_smallvideo_2cam_224_viewpatch_1x2x2_mergedloss_1e-4",
                "model.loss.lambda_video=0.05",
            ],
        )
        merged_eval = compose(
            config_name="sim_libero",
            overrides=[
                "task=libero_dino_s_smallvideo_2cam_224_viewpatch_1x2x2_mergedloss_1e-4",
                "model.dino_config.load_backbone=true",
            ],
        )

    assert no_pool.model.video_dit_config.get("output_patch_space", "dense") == "dense"
    assert dense_view.model.video_dit_config.get("output_patch_space", "dense") == "dense"
    assert merged.model.video_dit_config.output_patch_space == "merged"
    assert list(merged.model.video_dit_config.latent_patch_size) == [1, 2, 2]
    assert merged.model.video_dit_config.latent_patch_mode == "view"
    assert merged.model.video_dit_config.latent_num_views == 2
    assert float(merged.model.loss.lambda_video) == 0.05

    assert merged_eval.model.video_dit_config.output_patch_space == "merged"
    assert bool(merged_eval.model.load_text_encoder) is True
    assert bool(merged_eval.model.skip_dit_load_from_pretrain) is True
    assert bool(merged_eval.model.dino_config.load_backbone) is True
    print("[OK] Hydra train/eval configs compose with expected old/new behavior")


def check_source_level_eval_and_infer_contracts() -> None:
    trainer_eval_src = inspect.getsource(Wan22Trainer.evaluate)
    infer_src = inspect.getsource(FastWAM_DINO.infer_action)

    assert "model.training_loss(sample)" in trainer_eval_src
    assert "model.infer_action" in trainer_eval_src
    assert "self.video_expert.pre_dit" in infer_src
    assert "self.mot.prefill_video_cache" in infer_src
    assert "self.video_expert.post_dit" not in infer_src
    print("[OK] train-time eval uses training_loss; rollout infer_action does not use video head")


def main() -> None:
    torch.set_grad_enabled(False)
    check_old_no_pool_dense_default()
    check_old_viewpatch_dense_default()
    check_viewpatch_merged_output_and_target()
    check_loss_mask_alignment()
    check_full_tiny_training_loss()
    check_hydra_configs()
    check_source_level_eval_and_infer_contracts()
    print("[OK] all DINO merged-loss regression checks passed")


if __name__ == "__main__":
    main()
