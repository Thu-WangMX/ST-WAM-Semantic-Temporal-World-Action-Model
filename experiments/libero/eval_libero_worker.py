import json
import logging
import os
import sys
import time
from pathlib import Path
from typing import Any

import hydra
from accelerate import PartialState
from hydra.utils import instantiate
from omegaconf import DictConfig, OmegaConf

os.environ.setdefault("MUJOCO_GL", "egl")
os.environ.setdefault("PYOPENGL_PLATFORM", "egl")
os.environ.setdefault("MPLCONFIGDIR", "/tmp/matplotlib-cache")
os.environ.setdefault("TORCH_FORCE_NO_WEIGHTS_ONLY_LOAD", "1")
os.environ["TOKENIZERS_PARALLELISM"] = "false"

project_root = Path(__file__).resolve().parents[2]
project_src = project_root / "src"
libero_root = Path(os.environ.get("LIBERO_ROOT", project_root.parent / "LIBERO"))
for path in (libero_root, project_src, project_root):
    path_str = str(path)
    if path.exists() and path_str not in sys.path:
        sys.path.insert(0, path_str)

from experiments.libero.eval_libero_single import (  # noqa: E402
    NumpyEncoder,
    _load_model_checkpoint,
    _mixed_precision_to_model_dtype,
    _resolve_dataset_stats_path,
    _resolve_eval_device,
    _validate_visualize_future_video_cfg,
    run_single_task,
)
from fastwam.datasets.lerobot.processors.fastwam_processor import FastWAMProcessor  # noqa: E402
from fastwam.datasets.lerobot.utils.normalizer import load_dataset_stats_from_json  # noqa: E402
from fastwam.utils.pytorch_utils import set_global_seed  # noqa: E402
from libero.libero import benchmark  # noqa: E402

OmegaConf.register_new_resolver("eval", eval, replace=True)
OmegaConf.register_new_resolver("max", lambda x: max(x), replace=True)
OmegaConf.register_new_resolver("split", lambda s, idx: s.split("/")[int(idx)], replace=True)


def _read_manifest(path: str) -> list[dict[str, Any]]:
    manifest_path = Path(os.path.expanduser(os.path.expandvars(path)))
    with manifest_path.open("r", encoding="utf-8") as f:
        tasks = json.load(f)
    if not isinstance(tasks, list):
        raise ValueError(f"Task manifest must be a JSON list, got: {type(tasks)}")
    for item in tasks:
        if not isinstance(item, dict) or "suite" not in item or "task_id" not in item:
            raise ValueError(f"Invalid task manifest entry: {item}")
    return tasks


def _ensure_initial_states(initial_states: list, num_trials: int) -> list:
    if len(initial_states) == 0:
        raise ValueError("Task has no initial states.")
    while len(initial_states) < num_trials:
        initial_states.extend(initial_states[: num_trials - len(initial_states)])
    return initial_states


def _result_file(output_dir: Path, suite: str, gpu_id: str, task_id: int) -> Path:
    return output_dir / suite / f"gpu{gpu_id}_task{task_id}_results.json"


@hydra.main(version_base="1.3", config_path="../../configs", config_name="sim_libero.yaml")
def eval_worker_process(cfg: DictConfig) -> None:
    partial_state = PartialState()
    partial_state.config = cfg

    if cfg.get("seed") is not None:
        set_global_seed(int(cfg.seed), get_worker_init_fn=False)
    if cfg.ckpt is None:
        raise ValueError("cfg.ckpt must not be None.")
    _validate_visualize_future_video_cfg(cfg)

    env_num = int(cfg.EVALUATION.get("env_num", 1))
    if env_num != 1:
        raise ValueError("Persistent LIBERO worker currently supports only EVALUATION.env_num=1.")

    task_manifest = cfg.EVALUATION.get("task_manifest", None)
    if task_manifest is None:
        raise ValueError("Set +EVALUATION.task_manifest=/path/to/tasks.json for eval_libero_worker.py.")
    tasks = _read_manifest(str(task_manifest))

    model_device = _resolve_eval_device(cfg)
    model_dtype = _mixed_precision_to_model_dtype(cfg.get("mixed_precision", "bf16"))
    model = instantiate(cfg.model, model_dtype=model_dtype, device=model_device)
    _load_model_checkpoint(model, str(cfg.ckpt))
    model = model.to(model_device).eval()

    dataset_stats_path = _resolve_dataset_stats_path(cfg)
    dataset_stats = load_dataset_stats_from_json(str(dataset_stats_path))
    processor: FastWAMProcessor = instantiate(cfg.data.train.processor).eval()
    processor.set_normalizer_from_stats(dataset_stats)
    logging.info("Using dataset stats: %s", dataset_stats_path)

    action_horizon_cfg = cfg.EVALUATION.get("action_horizon", None)
    if action_horizon_cfg is None:
        action_horizon = int(cfg.data.train.num_frames) - 1
    else:
        action_horizon = int(action_horizon_cfg)
    if action_horizon <= 0:
        raise ValueError(f"EVALUATION.action_horizon must be positive, got {action_horizon}")

    video_size = cfg.data.train.get("video_size", [224, 224])
    if len(video_size) != 2:
        raise ValueError(f"data.train.video_size must be [H, W], got {video_size}")
    input_h = int(video_size[0])
    input_w = int(video_size[1])

    output_root = Path(cfg.EVALUATION.output_dir)
    output_root.mkdir(parents=True, exist_ok=True)
    gpu_id = str(cfg.gpu_id)
    skip_existing = bool(cfg.EVALUATION.get("skip_existing", True))
    num_trials = int(cfg.EVALUATION.num_trials)

    benchmark_dict = benchmark.get_benchmark_dict()
    suites: dict[str, Any] = {}
    failed: list[dict[str, Any]] = []

    print(
        f"Persistent worker gpu={gpu_id} loaded ckpt={cfg.ckpt}; "
        f"tasks={len(tasks)}; stats={dataset_stats_path}",
        flush=True,
    )

    for index, item in enumerate(tasks, start=1):
        suite_name = str(item["suite"])
        task_id = int(item["task_id"])
        output_file = _result_file(output_root, suite_name, gpu_id, task_id)
        if skip_existing and output_file.exists() and output_file.stat().st_size > 0:
            print(f"[{index}/{len(tasks)}] skip existing {suite_name} task={task_id}", flush=True)
            continue

        task_start_time = time.time()
        output_file.parent.mkdir(parents=True, exist_ok=True)
        video_dir = output_root / suite_name / "videos"
        video_dir.mkdir(parents=True, exist_ok=True)
        predicted_video_dir = output_root / suite_name / "predicted_videos"
        if bool(cfg.EVALUATION.get("visualize_future_video", False)):
            predicted_video_dir.mkdir(parents=True, exist_ok=True)

        cfg.EVALUATION.task_suite_name = suite_name
        cfg.EVALUATION.task_id = task_id

        try:
            if suite_name not in suites:
                suites[suite_name] = benchmark_dict[suite_name]()
            task_suite = suites[suite_name]
            task = task_suite.get_task(task_id)
            initial_states = _ensure_initial_states(task_suite.get_task_init_states(task_id), num_trials)

            results = {
                "task_suite": suite_name,
                "task_id": task_id,
                "task_description": None,
                "successes": 0,
                "total_episodes": num_trials,
                "gpu_id": int(gpu_id),
                "success_episodes": [],
                "failure_episodes": [],
                "start_time": time.strftime("%Y-%m-%d %H:%M:%S"),
                "duration": 0,
            }
            task_results = run_single_task(
                task=task,
                initial_states=initial_states,
                model=model,
                processor=processor,
                cfg=cfg,
                video_dir=video_dir,
                predicted_video_dir=predicted_video_dir,
                action_horizon=action_horizon,
                input_w=input_w,
                input_h=input_h,
                model_device=model_device,
            )
            results.update(task_results)
            results["duration"] = time.time() - task_start_time

            with output_file.open("w", encoding="utf-8") as f:
                json.dump(results, f, indent=4, cls=NumpyEncoder)
            print(
                f"[{index}/{len(tasks)}] done {suite_name} task={task_id}: "
                f"{results['successes']}/{num_trials} in {results['duration']:.2f}s",
                flush=True,
            )
        except Exception as exc:
            failed.append({"suite": suite_name, "task_id": task_id, "error": repr(exc)})
            fail_file = output_root / f"worker_gpu{gpu_id}_failed_tasks.jsonl"
            with fail_file.open("a", encoding="utf-8") as f:
                f.write(json.dumps(failed[-1], ensure_ascii=False) + "\n")
            print(f"[{index}/{len(tasks)}] FAILED {suite_name} task={task_id}: {exc!r}", flush=True)
            if not bool(cfg.EVALUATION.get("continue_on_error", False)):
                raise

    if failed:
        raise RuntimeError(f"Persistent worker gpu={gpu_id} finished with {len(failed)} failed tasks.")


if __name__ == "__main__":
    eval_worker_process()
