import math
import os
import sys
from pathlib import Path
from typing import Iterable

import hydra
import numpy as np
import torch
from hydra.utils import instantiate
from omegaconf import DictConfig, ListConfig, OmegaConf
from PIL import Image, ImageDraw

REPO_ROOT = Path(__file__).resolve().parents[1]
SRC_ROOT = REPO_ROOT / "src"
for path in (str(SRC_ROOT), str(REPO_ROOT)):
    if path not in sys.path:
        sys.path.insert(0, path)

from fastwam.runtime import _mixed_precision_to_model_dtype, _normalize_mixed_precision  # noqa: E402
from fastwam.utils.config_resolvers import register_default_resolvers  # noqa: E402


register_default_resolvers()


def _as_list(value) -> list[int]:
    if value is None:
        return []
    if isinstance(value, (list, tuple, ListConfig)):
        return [int(v) for v in value]
    if isinstance(value, str):
        value = value.strip()
        if not value:
            return []
        return [int(v.strip()) for v in value.split(",") if v.strip()]
    return [int(value)]


def _tensor_to_rgb_image(video: torch.Tensor, frame_idx: int, size: tuple[int, int]) -> Image.Image:
    frame = video[:, frame_idx].detach().cpu().float()
    frame = ((frame.clamp(-1, 1) + 1.0) * 127.5).byte()
    arr = frame.permute(1, 2, 0).numpy()
    return Image.fromarray(arr, mode="RGB").resize(size, Image.BILINEAR)


def _fit_pca_rgb(gt: torch.Tensor, pred: torch.Tensor) -> tuple[np.ndarray, np.ndarray]:
    """Fit one PCA basis on GT+pred tokens and return RGB maps.

    Args:
        gt/pred: [D,T,H,W] tensors.

    Returns:
        Pair of uint8 arrays [T,H,W,3].
    """
    if gt.shape != pred.shape:
        raise ValueError(f"GT/pred shape mismatch: {tuple(gt.shape)} vs {tuple(pred.shape)}")
    d, t, h, w = gt.shape
    gt_tokens = gt.detach().cpu().float().permute(1, 2, 3, 0).reshape(-1, d)
    pred_tokens = pred.detach().cpu().float().permute(1, 2, 3, 0).reshape(-1, d)
    x = torch.cat([gt_tokens, pred_tokens], dim=0)
    x = x - x.mean(dim=0, keepdim=True)
    # Full SVD is fine here: DINO-S pooled LIBERO windows are only 2 * 9 * 14 * 14 tokens.
    _, _, vh = torch.linalg.svd(x, full_matrices=False)
    comps = vh[:3].T
    rgb = x @ comps
    lo = torch.quantile(rgb, 0.01, dim=0)
    hi = torch.quantile(rgb, 0.99, dim=0)
    rgb = (rgb - lo) / (hi - lo).clamp(min=1e-6)
    rgb = rgb.clamp(0, 1)
    gt_rgb, pred_rgb = rgb[: gt_tokens.shape[0]], rgb[gt_tokens.shape[0] :]
    gt_rgb = (gt_rgb.reshape(t, h, w, 3).numpy() * 255.0).astype(np.uint8)
    pred_rgb = (pred_rgb.reshape(t, h, w, 3).numpy() * 255.0).astype(np.uint8)
    return gt_rgb, pred_rgb


def _pca_rgb_single_map(features_hwd: torch.Tensor) -> np.ndarray:
    """LDA-style PCA visualization for one feature map.

    This intentionally fits PCA independently for each image, matching
    the legacy visualization implementation. Colors are therefore not comparable
    between GT and pred, but spatial structure is much easier to inspect.
    """
    if features_hwd.ndim != 3:
        raise ValueError(f"Expected [H,W,D] feature map, got {tuple(features_hwd.shape)}")
    h, w, d = features_hwd.shape
    x = features_hwd.detach().cpu().float().reshape(-1, d)
    x = x - x.mean(dim=0, keepdim=True)
    # torch.pca_lowrank is fast here and avoids requiring sklearn in the env.
    _, _, v = torch.pca_lowrank(x, q=3, center=False)
    rgb = x @ v[:, :3]
    rgb = rgb - rgb.min()
    rgb = rgb / rgb.max().clamp(min=1e-6)
    return (rgb.reshape(h, w, 3).numpy().clip(0, 1) * 255.0).astype(np.uint8)


def _fit_lda_style_pca_rgb(gt: torch.Tensor, pred: torch.Tensor) -> tuple[np.ndarray, np.ndarray]:
    """Return independently PCA-colored GT/pred maps, [T,H,W,3]."""
    if gt.shape != pred.shape:
        raise ValueError(f"GT/pred shape mismatch: {tuple(gt.shape)} vs {tuple(pred.shape)}")
    t = gt.shape[1]
    gt_maps = []
    pred_maps = []
    for frame_idx in range(t):
        gt_hwd = gt[:, frame_idx].permute(1, 2, 0)
        pred_hwd = pred[:, frame_idx].permute(1, 2, 0)
        gt_maps.append(_pca_rgb_single_map(gt_hwd))
        pred_maps.append(_pca_rgb_single_map(pred_hwd))
    return np.stack(gt_maps, axis=0), np.stack(pred_maps, axis=0)


def _error_heatmap(gt: torch.Tensor, pred: torch.Tensor) -> np.ndarray:
    err = (gt.detach().cpu().float() - pred.detach().cpu().float()).pow(2).mean(dim=0).sqrt()
    # [T,H,W]
    lo = torch.quantile(err, 0.01)
    hi = torch.quantile(err, 0.99)
    err = ((err - lo) / (hi - lo).clamp(min=1e-6)).clamp(0, 1).numpy()
    heat = np.zeros((*err.shape, 3), dtype=np.uint8)
    heat[..., 0] = (255 * err).astype(np.uint8)
    heat[..., 1] = (255 * (1.0 - np.abs(err - 0.5) * 2.0).clip(0, 1)).astype(np.uint8)
    heat[..., 2] = (255 * (1.0 - err)).astype(np.uint8)
    return heat


def _resize_token_map(
    arr: np.ndarray,
    size: tuple[int, int],
    *,
    resample: int = Image.NEAREST,
) -> Image.Image:
    return Image.fromarray(arr, mode="RGB").resize(size, resample)


def _label(img: Image.Image, text: str) -> Image.Image:
    pad = 24
    out = Image.new("RGB", (img.width, img.height + pad), "white")
    out.paste(img, (0, pad))
    draw = ImageDraw.Draw(out)
    draw.text((4, 4), text, fill=(0, 0, 0))
    return out


def _make_frame_grid(
    video: torch.Tensor,
    gt_rgb: np.ndarray,
    pred_rgb: np.ndarray,
    heat_rgb: np.ndarray,
    frame_indices: Iterable[int],
    out_path: Path,
) -> None:
    display_size = (448, 224)
    rows = []
    for frame_idx in frame_indices:
        obs = _label(_tensor_to_rgb_image(video, frame_idx, display_size), f"RGB frame {frame_idx}")
        gt = _label(_resize_token_map(gt_rgb[frame_idx], display_size), f"GT DINO PCA {frame_idx}")
        pred = _label(_resize_token_map(pred_rgb[frame_idx], display_size), f"Pred DINO PCA {frame_idx}")
        heat = _label(_resize_token_map(heat_rgb[frame_idx], display_size), f"RMSE heat {frame_idx}")
        row = Image.new("RGB", (obs.width * 4, obs.height), "white")
        x = 0
        for im in (obs, gt, pred, heat):
            row.paste(im, (x, 0))
            x += im.width
        rows.append(row)

    grid = Image.new("RGB", (rows[0].width, rows[0].height * len(rows)), "white")
    y = 0
    for row in rows:
        grid.paste(row, (0, y))
        y += row.height
    grid.save(out_path)


def _make_lda_style_grid(
    video: torch.Tensor,
    gt_rgb: np.ndarray,
    pred_rgb: np.ndarray,
    frame_indices: Iterable[int],
    out_path: Path,
) -> None:
    """Save LDA-style PCA grid: RGB | GT independent PCA | Pred independent PCA."""
    display_size = (448, 224)
    rows = []
    for frame_idx in frame_indices:
        obs = _label(_tensor_to_rgb_image(video, frame_idx, display_size), f"RGB frame {frame_idx}")
        gt = _label(
            _resize_token_map(gt_rgb[frame_idx], display_size, resample=Image.BILINEAR),
            f"GT LDA-style PCA {frame_idx}",
        )
        pred = _label(
            _resize_token_map(pred_rgb[frame_idx], display_size, resample=Image.BILINEAR),
            f"Pred LDA-style PCA {frame_idx}",
        )
        row = Image.new("RGB", (obs.width * 3, obs.height), "white")
        x = 0
        for im in (obs, gt, pred):
            row.paste(im, (x, 0))
            x += im.width
        rows.append(row)

    grid = Image.new("RGB", (rows[0].width, rows[0].height * len(rows)), "white")
    y = 0
    for row in rows:
        grid.paste(row, (0, y))
        y += row.height
    grid.save(out_path)


@torch.no_grad()
def _predict_dino_rollout(
    model,
    clean_latents: torch.Tensor,
    context: torch.Tensor,
    context_mask: torch.Tensor,
    proprio: torch.Tensor | None,
    num_inference_steps: int,
    seed: int,
    rand_device: str,
) -> torch.Tensor:
    """Denoise a DINO future clip from noise, conditioning on the clean first frame."""
    model.eval()
    clean_latents = clean_latents.to(model.device, dtype=model.torch_dtype)
    context = context.to(model.device, dtype=model.torch_dtype)
    context_mask = context_mask.to(model.device, dtype=torch.bool)

    if proprio is not None and getattr(model, "proprio_encoder", None) is not None:
        if proprio.ndim == 3:
            proprio = proprio[:, 0, :]
        proprio = proprio.to(model.device, dtype=model.torch_dtype)
        context, context_mask = model._append_proprio_to_context(context, context_mask, proprio)

    generator = torch.Generator(device=rand_device).manual_seed(int(seed))
    latents = torch.randn(
        clean_latents.shape,
        generator=generator,
        device=rand_device,
        dtype=torch.float32,
    ).to(device=model.device, dtype=model.torch_dtype)
    latents[:, :, 0:1] = clean_latents[:, :, 0:1]

    timesteps, deltas = model.infer_video_scheduler.build_inference_schedule(
        num_inference_steps=int(num_inference_steps),
        device=model.device,
        dtype=latents.dtype,
    )

    for step_t, step_delta in zip(timesteps, deltas):
        timestep = step_t.reshape(1).expand(latents.shape[0]).to(model.device, dtype=latents.dtype)
        pred_velocity = model.video_expert(
            x=latents,
            timestep=timestep,
            context=context,
            context_mask=context_mask,
            fuse_vae_embedding_in_latents=True,
        )
        latents = model.infer_video_scheduler.step(pred_velocity, step_delta, latents)
        latents[:, :, 0:1] = clean_latents[:, :, 0:1]
    return latents.detach().cpu().float()


def _prepare_sample(sample: dict) -> dict:
    out = {}
    for key, value in sample.items():
        if torch.is_tensor(value):
            out[key] = value.unsqueeze(0)
        else:
            out[key] = value
    return out


@hydra.main(config_path="../configs", config_name="train", version_base="1.3")
def main(cfg: DictConfig) -> None:
    if not cfg.get("ckpt"):
        raise ValueError("Pass ckpt=/path/to/step_xxxxxx.pt")
    out_dir = Path(str(cfg.get("pca_output_dir", "outputs/dino_pca")))
    out_dir.mkdir(parents=True, exist_ok=True)

    mixed_precision = _normalize_mixed_precision(str(cfg.get("mixed_precision", "bf16")))
    model_dtype = _mixed_precision_to_model_dtype(mixed_precision)
    device = str(cfg.get("device", "cuda:0" if torch.cuda.is_available() else "cpu"))

    # Visualization uses cached DINO latents as GT, so avoid loading the DINO backbone by default.
    if "model" in cfg and "dino_config" in cfg.model:
        cfg.model.dino_config.load_backbone = bool(cfg.get("load_dino_backbone", False))

    model = instantiate(cfg.model, model_dtype=model_dtype, device=device)
    model.load_checkpoint(str(cfg.ckpt))
    model.eval()

    dataset = instantiate(cfg.data.train)
    sample_indices = _as_list(cfg.get("sample_indices"))
    if not sample_indices:
        num_samples = int(cfg.get("num_samples", 4))
        start = int(cfg.get("sample_start", 0))
        stride = int(cfg.get("sample_stride", max(1, len(dataset) // max(num_samples, 1))))
        sample_indices = [min(len(dataset) - 1, start + i * stride) for i in range(num_samples)]

    num_inference_steps = int(cfg.get("dino_pca_num_inference_steps", cfg.get("eval_num_inference_steps", 10)))
    seed = int(cfg.get("seed", 42))
    rand_device = str(cfg.get("rand_device", "cpu"))
    frame_indices = _as_list(cfg.get("pca_frame_indices"))

    manifest = {
        "ckpt": str(cfg.ckpt),
        "task": str(cfg.get("task", "")),
        "num_inference_steps": num_inference_steps,
        "sample_indices": sample_indices,
        "outputs": [],
    }

    for order, sample_idx in enumerate(sample_indices):
        raw = dataset[int(sample_idx)]
        if "dino_latents" not in raw:
            raise ValueError(
                "Sample has no `dino_latents`. Pass data.train.dino_latent_cache_dir=... "
                "and data.train.dino_latent_cache_mode=frame/window."
            )
        sample = _prepare_sample(raw)
        clean = sample["dino_latents"].float()
        video = sample["video"][0]
        context = sample["context"]
        context_mask = sample["context_mask"]
        proprio = sample.get("proprio")

        pred = _predict_dino_rollout(
            model=model,
            clean_latents=clean,
            context=context,
            context_mask=context_mask,
            proprio=proprio,
            num_inference_steps=num_inference_steps,
            seed=seed + order,
            rand_device=rand_device,
        )[0]
        gt = clean[0].cpu().float()

        gt_rgb, pred_rgb = _fit_pca_rgb(gt=gt, pred=pred)
        gt_lda_rgb, pred_lda_rgb = _fit_lda_style_pca_rgb(gt=gt, pred=pred)
        heat_rgb = _error_heatmap(gt=gt, pred=pred)

        if frame_indices:
            frames = [i for i in frame_indices if 0 <= i < gt.shape[1]]
        else:
            t = gt.shape[1]
            frames = sorted(set([0, max(0, t // 4), max(0, t // 2), max(0, (3 * t) // 4), t - 1]))

        prefix = out_dir / f"sample{int(sample_idx):08d}"
        grid_path = prefix.with_suffix(".pca_grid.png")
        lda_grid_path = prefix.with_suffix(".lda_style_pca_grid.png")
        _make_frame_grid(
            video=video,
            gt_rgb=gt_rgb,
            pred_rgb=pred_rgb,
            heat_rgb=heat_rgb,
            frame_indices=frames,
            out_path=grid_path,
        )
        _make_lda_style_grid(
            video=video,
            gt_rgb=gt_lda_rgb,
            pred_rgb=pred_lda_rgb,
            frame_indices=frames,
            out_path=lda_grid_path,
        )

        mse_per_frame = (gt - pred).pow(2).mean(dim=(0, 2, 3)).numpy().tolist()
        np.savez_compressed(
            prefix.with_suffix(".npz"),
            gt=gt.numpy(),
            pred=pred.numpy(),
            mse_per_frame=np.asarray(mse_per_frame, dtype=np.float32),
        )
        manifest["outputs"].append(
            {
                "dataset_idx": int(raw.get("dataset_idx", sample_idx)),
                "requested_idx": int(sample_idx),
                "prompt": raw.get("prompt", ""),
                "grid": str(grid_path),
                "lda_style_grid": str(lda_grid_path),
                "npz": str(prefix.with_suffix(".npz")),
                "mse_per_frame": mse_per_frame,
            }
        )
        print(f"[dino-pca] wrote {grid_path}")
        print(f"[dino-pca] wrote {lda_grid_path}")

    manifest_path = out_dir / "manifest.yaml"
    OmegaConf.save(OmegaConf.create(manifest), manifest_path)
    print(f"[dino-pca] manifest: {manifest_path}")


if __name__ == "__main__":
    main()
