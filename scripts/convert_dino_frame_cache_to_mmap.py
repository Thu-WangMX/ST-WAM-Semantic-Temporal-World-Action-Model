#!/usr/bin/env python
"""Convert per-frame DINO .pt cache files into one mmap-friendly payload.

The original ``frame`` cache stores one ``frames/00000000.pt`` file per global
frame.  That is space-efficient but punishes DataLoader workers with thousands
of tiny ``torch.load`` calls per step.  This converter keeps the same one-frame
storage semantics while packing all frames into a contiguous binary file that
workers can read with ``np.memmap``.
"""

import argparse
from concurrent.futures import ProcessPoolExecutor
import json
import os
import sys
import uuid
from pathlib import Path
from typing import Any

import numpy as np
import torch
from tqdm import tqdm


_WORKER_FRAME_DIR: Path | None = None
_WORKER_OUTPUT_FD: int | None = None
_WORKER_LATENT_SHAPE: tuple[int, ...] | None = None
_WORKER_LATENT_DTYPE: torch.dtype | None = None
_WORKER_SAVE_DTYPE: str | None = None
_WORKER_FRAME_NBYTES: int | None = None


def _atomic_json_save(payload: dict[str, Any], output_path: Path):
    output_path.parent.mkdir(parents=True, exist_ok=True)
    tmp_path = output_path.parent / f".{output_path.name}.tmp.{uuid.uuid4().hex}"
    with tmp_path.open("w", encoding="utf-8") as f:
        json.dump(payload, f, indent=2, sort_keys=True)
    os.replace(tmp_path, output_path)


def _load_frame(path: Path) -> torch.Tensor:
    payload = torch.load(path, map_location="cpu")
    latent = payload.get("dino_latent", payload.get("latent")) if isinstance(payload, dict) else payload
    if not torch.is_tensor(latent):
        raise TypeError(f"Expected tensor DINO latent in {path}, got {type(latent)}")
    if latent.ndim != 3:
        raise ValueError(f"Expected frame latent [D,H,W] in {path}, got {tuple(latent.shape)}")
    return latent.contiguous()


def _dtype_info(dtype: torch.dtype) -> tuple[str, np.dtype, str]:
    if dtype is torch.bfloat16:
        return "bf16", np.dtype("uint16"), "uint16"
    if dtype is torch.float16:
        return "fp16", np.dtype("float16"), "float16"
    if dtype is torch.float32:
        return "fp32", np.dtype("float32"), "float32"
    raise ValueError(f"Unsupported DINO frame dtype for mmap conversion: {dtype}")


def _tensor_to_storage_numpy(tensor: torch.Tensor, save_dtype: str) -> np.ndarray:
    if save_dtype == "bf16":
        return tensor.view(torch.uint16).numpy()
    return tensor.numpy()


def _init_pack_worker(
    frame_dir: str,
    output_path: str,
    latent_shape: tuple[int, ...],
    latent_dtype: torch.dtype,
    save_dtype: str,
    frame_nbytes: int,
) -> None:
    global _WORKER_FRAME_DIR
    global _WORKER_OUTPUT_FD
    global _WORKER_LATENT_SHAPE
    global _WORKER_LATENT_DTYPE
    global _WORKER_SAVE_DTYPE
    global _WORKER_FRAME_NBYTES

    torch.set_num_threads(1)
    _WORKER_FRAME_DIR = Path(frame_dir)
    _WORKER_OUTPUT_FD = os.open(output_path, os.O_WRONLY)
    _WORKER_LATENT_SHAPE = latent_shape
    _WORKER_LATENT_DTYPE = latent_dtype
    _WORKER_SAVE_DTYPE = save_dtype
    _WORKER_FRAME_NBYTES = frame_nbytes


def _pack_frame_range(frame_range: tuple[int, int]) -> int:
    if (
        _WORKER_FRAME_DIR is None
        or _WORKER_OUTPUT_FD is None
        or _WORKER_LATENT_SHAPE is None
        or _WORKER_LATENT_DTYPE is None
        or _WORKER_SAVE_DTYPE is None
        or _WORKER_FRAME_NBYTES is None
    ):
        raise RuntimeError("DINO mmap pack worker was not initialized")

    start, end = frame_range
    for idx in range(start, end):
        frame_path = _WORKER_FRAME_DIR / f"{idx:08d}.pt"
        latent = _load_frame(frame_path)
        if tuple(latent.shape) != _WORKER_LATENT_SHAPE:
            raise ValueError(
                f"Shape mismatch in {frame_path}: expected {_WORKER_LATENT_SHAPE}, "
                f"got {tuple(latent.shape)}"
            )
        if latent.dtype != _WORKER_LATENT_DTYPE:
            raise ValueError(
                f"Dtype mismatch in {frame_path}: expected {_WORKER_LATENT_DTYPE}, got {latent.dtype}"
            )

        storage = _tensor_to_storage_numpy(latent, _WORKER_SAVE_DTYPE)
        payload = memoryview(storage).cast("B")
        file_offset = idx * _WORKER_FRAME_NBYTES
        while payload:
            written = os.pwrite(_WORKER_OUTPUT_FD, payload, file_offset)
            if written <= 0:
                raise OSError(f"Short pwrite while packing {frame_path}: wrote {written} bytes")
            payload = payload[written:]
            file_offset += written
    return end - start


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--src", required=True, type=Path, help="Existing frame cache directory.")
    parser.add_argument("--dst", required=True, type=Path, help="Output frame_mmap cache directory.")
    parser.add_argument("--overwrite", action="store_true", help="Replace an existing output payload.")
    parser.add_argument(
        "--num-frames",
        type=int,
        default=None,
        help="Optional frame count override. Defaults to metadata total_samples or files in src/frames.",
    )
    parser.add_argument(
        "--flush-every",
        type=int,
        default=65536,
        help="Flush mmap writes every N frames.",
    )
    parser.add_argument(
        "--read-workers",
        type=int,
        default=16,
        help="Number of worker processes used to load and pack frame files concurrently.",
    )
    parser.add_argument(
        "--read-batch-size",
        type=int,
        default=512,
        help="Number of consecutive frames assigned to each worker task.",
    )
    return parser.parse_args()


def main():
    args = parse_args()
    if args.read_workers < 1:
        raise ValueError(f"--read-workers must be at least 1, got {args.read_workers}")
    if args.read_batch_size < 1:
        raise ValueError(f"--read-batch-size must be at least 1, got {args.read_batch_size}")

    # Workers load one tensor at a time and write directly to non-overlapping
    # file offsets, avoiding multi-GB tensor transfers back to the parent.
    torch.set_num_threads(1)
    src = args.src.expanduser().resolve()
    dst = args.dst.expanduser().resolve()
    frame_dir = src / "frames"
    if not frame_dir.is_dir():
        raise FileNotFoundError(f"Missing source frame cache directory: {frame_dir}")

    src_meta_path = src / "metadata.json"
    src_meta: dict[str, Any] = {}
    if src_meta_path.exists():
        with src_meta_path.open("r", encoding="utf-8") as f:
            src_meta = json.load(f)

    if args.num_frames is not None:
        total_frames = int(args.num_frames)
    elif "total_samples" in src_meta:
        total_frames = int(src_meta["total_samples"])
    else:
        total_frames = len(list(frame_dir.glob("*.pt")))
    if total_frames <= 0:
        raise ValueError(f"Invalid total frame count: {total_frames}")

    first_path = frame_dir / "00000000.pt"
    if not first_path.exists():
        first_paths = sorted(frame_dir.glob("*.pt"))
        if not first_paths:
            raise FileNotFoundError(f"No frame cache files found in {frame_dir}")
        first_path = first_paths[0]
    first_latent = _load_frame(first_path)
    save_dtype, np_dtype, storage_dtype = _dtype_info(first_latent.dtype)
    latent_shape = tuple(int(v) for v in first_latent.shape)

    dst.mkdir(parents=True, exist_ok=True)
    mmap_name = f"frames.{save_dtype}.bin"
    mmap_path = dst / mmap_name
    tmp_mmap_path = dst / f".{mmap_name}.tmp"
    metadata_path = dst / "metadata.json"
    if (mmap_path.exists() or metadata_path.exists() or tmp_mmap_path.exists()) and not args.overwrite:
        raise FileExistsError(
            f"Output already exists in {dst}. Pass --overwrite to replace {mmap_path.name}/metadata.json."
        )
    for path in (mmap_path, metadata_path, tmp_mmap_path):
        if path.exists():
            path.unlink()

    frame_nbytes = int(np.prod(latent_shape, dtype=np.int64)) * np_dtype.itemsize
    total_nbytes = total_frames * frame_nbytes
    output_fd = os.open(tmp_mmap_path, os.O_RDWR | os.O_CREAT | os.O_TRUNC, 0o644)
    os.ftruncate(output_fd, total_nbytes)

    print(
        f"Packing with worker_processes={args.read_workers}, "
        f"frames_per_task={args.read_batch_size}, flush_every={args.flush_every}"
    )
    next_flush = args.flush_every if args.flush_every > 0 else None
    progress = tqdm(total=total_frames, desc="Packing DINO frame cache", unit="frame", dynamic_ncols=True)

    frame_ranges = (
        (start, min(start + args.read_batch_size, total_frames))
        for start in range(0, total_frames, args.read_batch_size)
    )
    packed_frames = 0
    with ProcessPoolExecutor(
        max_workers=args.read_workers,
        initializer=_init_pack_worker,
        initargs=(
            str(frame_dir),
            str(tmp_mmap_path),
            latent_shape,
            first_latent.dtype,
            save_dtype,
            frame_nbytes,
        ),
    ) as executor:
        for count in executor.map(_pack_frame_range, frame_ranges, chunksize=1):
            packed_frames += count
            progress.update(count)
            if next_flush is not None and packed_frames >= next_flush:
                os.fsync(output_fd)
                while next_flush <= packed_frames:
                    next_flush += args.flush_every

    progress.close()
    os.fsync(output_fd)
    os.close(output_fd)
    os.replace(tmp_mmap_path, mmap_path)

    metadata = dict(src_meta)
    metadata.update(
        {
            "cache_mode": "frame_mmap",
            "source_cache_mode": src_meta.get("cache_mode", "frame"),
            "source_cache_dir": str(src),
            "mmap_file": mmap_name,
            "total_frames": total_frames,
            "latent_shape": list(latent_shape),
            "save_dtype": save_dtype,
            "storage_dtype": storage_dtype,
            "format_version": 1,
        }
    )
    _atomic_json_save(metadata, metadata_path)
    size_gib = mmap_path.stat().st_size / (1024**3)
    print(
        f"frame_mmap cache written: {dst}\n"
        f"  frames={total_frames} shape={latent_shape} save_dtype={save_dtype} "
        f"storage={storage_dtype} size={size_gib:.2f} GiB"
    )


if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        sys.exit(130)
