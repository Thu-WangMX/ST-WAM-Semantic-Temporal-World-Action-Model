#!/usr/bin/env python3
"""Benchmark FastWAM action-chunk latency with synchronized CUDA timing."""

from __future__ import annotations

import argparse
import inspect
import json
import statistics
import sys
import time
from pathlib import Path

import torch
from hydra import compose, initialize_config_dir
from hydra.core.global_hydra import GlobalHydra
from hydra.utils import instantiate
from omegaconf import OmegaConf


PROJECT_ROOT = Path(__file__).resolve().parents[1]
SRC_ROOT = PROJECT_ROOT / "src"
for path in (PROJECT_ROOT, SRC_ROOT):
    if str(path) not in sys.path:
        sys.path.insert(0, str(path))


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--task", required=True, help="Hydra task config name without .yaml")
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--qwen-path", default=None)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--warmup", type=int, default=3)
    parser.add_argument("--repeats", type=int, default=20)
    parser.add_argument("--num-inference-steps", type=int, default=10)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--compare-cache", action="store_true")
    parser.add_argument("--output", default=None)
    return parser.parse_args()


def compose_config(task: str):
    if GlobalHydra.instance().is_initialized():
        GlobalHydra.instance().clear()
    with initialize_config_dir(version_base="1.3", config_dir=str(PROJECT_ROOT / "configs")):
        return compose(config_name="sim_robotwin.yaml", overrides=[f"task={task}"])


def build_model(cfg, args: argparse.Namespace):
    model_cfg = OmegaConf.create(OmegaConf.to_container(cfg.model, resolve=True))
    model_cfg.load_text_encoder = False
    model_cfg.skip_dit_load_from_pretrain = True
    if model_cfg.get("action_dit_pretrained_path") is not None:
        model_cfg.action_dit_pretrained_path = None
    if model_cfg.get("vae_video_dit_pretrained_path") is not None:
        model_cfg.vae_video_dit_pretrained_path = None
    if model_cfg.get("dino_video_dit_pretrained_path") is not None:
        model_cfg.dino_video_dit_pretrained_path = None
    if model_cfg.get("dino_config") is not None:
        model_cfg.dino_config.load_backbone = True
    if args.qwen_path and model_cfg.get("semantic_history_config") is not None:
        model_cfg.semantic_history_config.vlm_model_name_or_path = args.qwen_path

    model = instantiate(model_cfg, model_dtype=torch.bfloat16, device=args.device)
    model.load_checkpoint(args.checkpoint)
    return model.to(args.device).eval()


def build_inputs(model, cfg, args: argparse.Namespace) -> dict:
    height, width = [int(value) for value in cfg.data.train.video_size]
    cpu_generator = torch.Generator(device="cpu").manual_seed(20260721)
    image = torch.rand((1, 3, height, width), generator=cpu_generator, dtype=torch.float32)
    image = image.mul(2.0).sub(1.0).to(device=args.device, dtype=model.torch_dtype)

    context_len = int(cfg.model.get("tokenizer_max_len", 128))
    context = torch.randn(
        (1, context_len, int(model.text_dim)),
        generator=cpu_generator,
        dtype=torch.float32,
    ).to(device=args.device, dtype=model.torch_dtype)
    context_mask = torch.ones((1, context_len), device=args.device, dtype=torch.bool)
    proprio = None
    if model.proprio_dim is not None:
        proprio = torch.zeros((1, int(model.proprio_dim)), device=args.device, dtype=model.torch_dtype)

    kwargs = {
        "prompt": None,
        "input_image": image,
        "action_horizon": int(cfg.data.train.num_frames) - 1,
        "proprio": proprio,
        "context": context,
        "context_mask": context_mask,
        "text_cfg_scale": 1.0,
        "num_inference_steps": int(args.num_inference_steps),
        "seed": int(args.seed),
        "rand_device": "cpu",
        "tiled": False,
    }

    infer_params = inspect.signature(model.infer_action).parameters
    semantic_cfg = getattr(model, "semantic_history_config", {}) or {}
    if "history_video" in infer_params and getattr(model, "semantic_history_encoder", None) is not None:
        history_offsets = semantic_cfg.get("history_offsets", [-24, -16, -8, -1])
        kwargs["history_video"] = image.unsqueeze(2).repeat(1, 1, len(history_offsets), 1, 1)
    if "semantic_image" in infer_params and getattr(model, "semantic_history_encoder", None) is not None:
        kwargs["semantic_image"] = image
    if "semantic_prompt" in infer_params and getattr(model, "semantic_history_encoder", None) is not None:
        kwargs["semantic_prompt"] = "Adjust the bottle."
    return kwargs


def synchronized_infer(model, kwargs: dict) -> tuple[torch.Tensor, float]:
    torch.cuda.synchronize()
    start = time.perf_counter()
    output = model.infer_action(**kwargs)
    torch.cuda.synchronize()
    elapsed_ms = (time.perf_counter() - start) * 1000.0
    return output["action"], elapsed_ms


def compare_cache_paths(model, infer_kwargs: dict) -> dict:
    if "use_static_kv_cache" not in inspect.signature(model.infer_action).parameters:
        raise ValueError("This model does not expose `use_static_kv_cache`.")
    uncached_kwargs = dict(infer_kwargs, use_static_kv_cache=False)
    cached_kwargs = dict(infer_kwargs, use_static_kv_cache=True)
    uncached_action, uncached_ms = synchronized_infer(model, uncached_kwargs)
    cached_action, cached_ms = synchronized_infer(model, cached_kwargs)
    diff = (uncached_action - cached_action).abs()
    result = {
        "uncached_ms": uncached_ms,
        "cached_ms": cached_ms,
        "max_abs_error": float(diff.max().item()),
        "mean_abs_error": float(diff.mean().item()),
        "allclose_atol_1e-2_rtol_1e-2": bool(
            torch.allclose(uncached_action, cached_action, atol=1e-2, rtol=1e-2)
        ),
    }
    if not result["allclose_atol_1e-2_rtol_1e-2"]:
        raise RuntimeError(f"Cached inference failed numerical equivalence: {result}")
    return result


def main() -> None:
    args = parse_args()
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is required for this benchmark.")
    if args.warmup < 0 or args.repeats <= 0:
        raise ValueError("`warmup` must be non-negative and `repeats` must be positive.")

    cfg = compose_config(args.task)
    model = build_model(cfg, args)
    infer_kwargs = build_inputs(model, cfg, args)

    cache_comparison = None
    if args.compare_cache:
        cache_comparison = compare_cache_paths(model, infer_kwargs)

    if "use_static_kv_cache" in inspect.signature(model.infer_action).parameters:
        infer_kwargs["use_static_kv_cache"] = True
    for _ in range(args.warmup):
        synchronized_infer(model, infer_kwargs)

    latencies_ms = []
    for _ in range(args.repeats):
        _, elapsed_ms = synchronized_infer(model, infer_kwargs)
        latencies_ms.append(elapsed_ms)

    result = {
        "task": args.task,
        "checkpoint": str(Path(args.checkpoint).resolve()),
        "device": torch.cuda.get_device_name(torch.device(args.device)),
        "batch_size": 1,
        "action_horizon": int(infer_kwargs["action_horizon"]),
        "num_inference_steps": int(args.num_inference_steps),
        "warmup": int(args.warmup),
        "repeats": int(args.repeats),
        "mean_ms_per_action_chunk": statistics.mean(latencies_ms),
        "median_ms_per_action_chunk": statistics.median(latencies_ms),
        "cache_comparison": cache_comparison,
        "latencies_ms": latencies_ms,
    }
    rendered = json.dumps(result, indent=2)
    print(rendered)
    if args.output:
        output_path = Path(args.output)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        output_path.write_text(rendered + "\n", encoding="utf-8")


if __name__ == "__main__":
    main()
