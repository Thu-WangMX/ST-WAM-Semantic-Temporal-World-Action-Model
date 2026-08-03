#!/usr/bin/env python3
"""Wait until enough GPUs are idle and print the selected GPU ids.

This helper intentionally depends only on the Python standard library and
`nvidia-smi`, so shell entrypoints can use it before activating heavier code.
"""

from __future__ import annotations

import argparse
import os
import subprocess
import sys
import time
from dataclasses import dataclass


@dataclass(frozen=True)
class GpuInfo:
    index: int
    memory_used_mb: int
    memory_total_mb: int
    utilization: int


def _env_int(name: str, default: int) -> int:
    raw = os.environ.get(name)
    if raw is None or raw == "":
        return default
    try:
        return int(raw)
    except ValueError as exc:
        raise SystemExit(f"{name} must be an integer, got {raw!r}") from exc


def _parse_visible(visible: str | None) -> list[int] | None:
    if visible is None or visible.strip() == "":
        return None
    ids: list[int] = []
    for item in visible.split(","):
        item = item.strip()
        if item == "":
            continue
        if not item.isdigit():
            raise SystemExit(
                "wait_for_gpus.py currently expects numeric CUDA_VISIBLE_DEVICES ids; "
                f"got {visible!r}"
            )
        ids.append(int(item))
    return ids


def query_gpus() -> list[GpuInfo]:
    cmd = [
        "nvidia-smi",
        "--query-gpu=index,memory.used,memory.total,utilization.gpu",
        "--format=csv,noheader,nounits",
    ]
    try:
        proc = subprocess.run(cmd, check=True, text=True, capture_output=True)
    except FileNotFoundError as exc:
        raise SystemExit("nvidia-smi was not found; cannot wait for GPUs.") from exc
    except subprocess.CalledProcessError as exc:
        raise SystemExit(f"nvidia-smi failed:\n{exc.stderr}") from exc

    gpus: list[GpuInfo] = []
    for line in proc.stdout.splitlines():
        if not line.strip():
            continue
        parts = [part.strip() for part in line.split(",")]
        if len(parts) != 4:
            raise SystemExit(f"Unexpected nvidia-smi output line: {line!r}")
        index, mem_used, mem_total, util = (int(part) for part in parts)
        gpus.append(
            GpuInfo(
                index=index,
                memory_used_mb=mem_used,
                memory_total_mb=mem_total,
                utilization=util,
            )
        )
    return gpus


def select_idle_gpus(
    *,
    count: int,
    visible: list[int] | None,
    max_used_mb: int,
    max_util: int,
) -> tuple[list[int], list[GpuInfo]]:
    visible_set = None if visible is None else set(visible)
    candidates = [
        gpu
        for gpu in query_gpus()
        if visible_set is None or gpu.index in visible_set
    ]
    idle = [
        gpu
        for gpu in candidates
        if gpu.memory_used_mb <= max_used_mb and gpu.utilization <= max_util
    ]
    idle.sort(key=lambda gpu: (gpu.memory_used_mb, gpu.utilization, gpu.index))
    return [gpu.index for gpu in idle[:count]], candidates


def summarize(candidates: list[GpuInfo]) -> str:
    if not candidates:
        return "no visible GPUs"
    chunks = []
    for gpu in sorted(candidates, key=lambda item: item.index):
        chunks.append(
            f"{gpu.index}:mem={gpu.memory_used_mb}/{gpu.memory_total_mb}MiB,util={gpu.utilization}%"
        )
    return "; ".join(chunks)


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--count", type=int, required=True)
    parser.add_argument("--visible", default=os.environ.get("CUDA_VISIBLE_DEVICES", ""))
    parser.add_argument("--max-used-mb", type=int, default=_env_int("WAIT_GPU_MAX_USED_MB", 1024))
    parser.add_argument("--max-util", type=int, default=_env_int("WAIT_GPU_MAX_UTIL", 10))
    parser.add_argument("--interval", type=int, default=_env_int("WAIT_GPU_INTERVAL", 30))
    parser.add_argument("--stable-checks", type=int, default=_env_int("WAIT_GPU_STABLE_CHECKS", 2))
    parser.add_argument(
        "--timeout",
        type=int,
        default=_env_int("WAIT_GPU_TIMEOUT", 0),
        help="Seconds before failing. 0 means wait forever.",
    )
    args = parser.parse_args()

    if args.count <= 0:
        raise SystemExit(f"--count must be positive, got {args.count}")
    if args.interval <= 0:
        raise SystemExit(f"--interval must be positive, got {args.interval}")
    if args.stable_checks <= 0:
        raise SystemExit(f"--stable-checks must be positive, got {args.stable_checks}")

    visible = _parse_visible(args.visible)
    start = time.monotonic()
    stable = 0
    last_selection: list[int] | None = None

    while True:
        selected, candidates = select_idle_gpus(
            count=args.count,
            visible=visible,
            max_used_mb=args.max_used_mb,
            max_util=args.max_util,
        )
        enough = len(selected) >= args.count
        if enough and selected == last_selection:
            stable += 1
        elif enough:
            stable = 1
            last_selection = selected
        else:
            stable = 0
            last_selection = None

        if enough and stable >= args.stable_checks:
            print(",".join(str(gpu_id) for gpu_id in selected[: args.count]))
            return 0

        elapsed = int(time.monotonic() - start)
        print(
            "[wait_for_gpus] waiting: "
            f"need={args.count}, idle={len(selected)}, stable={stable}/{args.stable_checks}, "
            f"thresholds=max_used_mb:{args.max_used_mb},max_util:{args.max_util}, "
            f"elapsed={elapsed}s, visible={args.visible or 'all'}, {summarize(candidates)}",
            file=sys.stderr,
            flush=True,
        )

        if args.timeout > 0 and elapsed >= args.timeout:
            print(
                f"[wait_for_gpus] timeout after {elapsed}s while waiting for {args.count} GPUs.",
                file=sys.stderr,
            )
            return 2
        time.sleep(args.interval)


if __name__ == "__main__":
    raise SystemExit(main())
