"""Load one real LIBERO sample and verify the Qwen VAE-history task contract."""

from __future__ import annotations

import argparse
from pathlib import Path

import torch
from hydra import compose, initialize_config_dir
from hydra.utils import instantiate
from omegaconf import OmegaConf

from fastwam.utils.config_resolvers import register_default_resolvers


TASK = "libero_wan5b_dino_s_aux_mot_short_qwen3vl_vae_hist4_vae_mmap_2cam_224_1e-4"


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--stats", type=Path, required=True)
    parser.add_argument("--index", type=int, default=1000)
    args = parser.parse_args()

    register_default_resolvers()
    config_dir = Path(__file__).resolve().parents[1] / "configs"
    with initialize_config_dir(config_dir=str(config_dir), version_base="1.3"):
        cfg = compose(config_name="train", overrides=[f"task={TASK}"])
    OmegaConf.set_struct(cfg, False)
    cfg.data.train.pretrained_norm_stats = str(args.stats.resolve())
    dataset = instantiate(cfg.data.train)
    sample = dataset[int(args.index)]

    semantic_image = sample["semantic_image"]
    history_video = sample["history_video"]
    vae_latents = sample["vae_latents"]
    dino_latents = sample["dino_latents"]
    assert semantic_image.shape == (3, 224, 448)
    assert history_video.shape == (3, 4, 224, 448)
    assert vae_latents.ndim == 4 and vae_latents.shape[0] == 48
    assert dino_latents.ndim == 4 and dino_latents.shape[0] == 384
    assert dataset.history_vae_frame_offsets == [-24, -16, -8, -1]
    assert not dataset.load_history_dino_latents
    assert dataset.load_history_vae_video
    for name, tensor in (("semantic_image", semantic_image), ("history_video", history_video)):
        assert torch.isfinite(tensor).all(), f"{name} contains non-finite values"
        assert float(tensor.min()) >= -1.001 and float(tensor.max()) <= 1.001

    print(
        {
            "dataset_idx": int(sample["dataset_idx"]),
            "history_offsets": dataset.history_vae_frame_offsets,
            "semantic_image": tuple(semantic_image.shape),
            "history_video": tuple(history_video.shape),
            "vae_latents": tuple(vae_latents.shape),
            "dino_latents": tuple(dino_latents.shape),
        }
    )


if __name__ == "__main__":
    main()
