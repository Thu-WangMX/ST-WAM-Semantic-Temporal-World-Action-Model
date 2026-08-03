#!/usr/bin/env bash
set -Eeuo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "${ROOT_DIR}"

PYTHON="${PYTHON:-python}"
TASK="${TASK:-robotwin_wan5b_dino_s_aux_mot_3cam_384x320_1e-4}"
VERIFY_TASK="${VERIFY_TASK:-robotwin_wan5b_dino_s_aux_mot_short_intent_3cam_384x320_1e-4}"
FRAME_CACHE_DIR="${FRAME_CACHE_DIR:-./data/dino_latents_cache/robotwin_dino_s_3cam384x320_pool1x1_frame}"
MMAP_CACHE_DIR="${MMAP_CACHE_DIR:-./data/dino_latents_cache/robotwin_dino_s_3cam384x320_pool1x1_frame_mmap}"
NPROC_PER_NODE="${NPROC_PER_NODE:-8}"
DINO_PRECOMPUTE_BATCH_SIZE="${DINO_PRECOMPUTE_BATCH_SIZE:-8}"
DINO_PRECOMPUTE_NUM_WORKERS="${DINO_PRECOMPUTE_NUM_WORKERS:-4}"
DINO_ENCODE_MICROBATCH_SIZE="${DINO_ENCODE_MICROBATCH_SIZE:-8}"
DINO_LATENT_CACHE_DTYPE="${DINO_LATENT_CACHE_DTYPE:-bf16}"
DINO_MMAP_READ_WORKERS="${DINO_MMAP_READ_WORKERS:-16}"
DINO_MMAP_READ_BATCH_SIZE="${DINO_MMAP_READ_BATCH_SIZE:-512}"
DINO_MMAP_FLUSH_EVERY="${DINO_MMAP_FLUSH_EVERY:-0}"
STAGE="${STAGE:-all}"  # all, frames, mmap, verify
OVERWRITE="${OVERWRITE:-false}"
LOG_DIR="${LOG_DIR:-./logs/robotwin_dino_cache}"
STAMP="$(date +%Y%m%d_%H%M%S)"
LOG_FILE="${LOG_FILE:-${LOG_DIR}/robotwin_dino_cache_${STAMP}.log}"

mkdir -p "${LOG_DIR}"

log() {
  echo "[$(date '+%Y-%m-%d %H:%M:%S')] $*" | tee -a "${LOG_FILE}"
}

bool_arg() {
  case "${1}" in
    1|true|TRUE|yes|YES|y|Y) echo "true" ;;
    0|false|FALSE|no|NO|n|N) echo "false" ;;
    *) echo "ERROR: cannot parse boolean: ${1}" >&2; exit 2 ;;
  esac
}

require_file() {
  local path="${1:?path required}"
  local label="${2:-${path}}"
  if [[ ! -s "${path}" ]]; then
    log "ERROR: missing ${label}: ${path}"
    exit 2
  fi
}

require_dir() {
  local path="${1:?path required}"
  local label="${2:-${path}}"
  if [[ ! -d "${path}" ]]; then
    log "ERROR: missing ${label}: ${path}"
    exit 2
  fi
}

ensure_robotwin_default_path() {
  if [[ -e ./data/robotwin2.0 ]]; then
    return 0
  fi
  if [[ -d ./data/robotwin2.0-fastwam ]]; then
    ln -s robotwin2.0-fastwam ./data/robotwin2.0
    log "Created symlink: data/robotwin2.0 -> robotwin2.0-fastwam"
    return 0
  fi
}

run_frames() {
  local overwrite_bool
  overwrite_bool="$(bool_arg "${OVERWRITE}")"
  mkdir -p "${FRAME_CACHE_DIR}"

  log "Stage=frames task=${TASK}"
  log "Frame cache dir: ${FRAME_CACHE_DIR}"
  log "nproc=${NPROC_PER_NODE}, batch=${DINO_PRECOMPUTE_BATCH_SIZE}, workers=${DINO_PRECOMPUTE_NUM_WORKERS}, encode_microbatch=${DINO_ENCODE_MICROBATCH_SIZE}, dtype=${DINO_LATENT_CACHE_DTYPE}, overwrite=${overwrite_bool}"

  "${PYTHON}" -m torch.distributed.run \
    --standalone \
    --nproc_per_node="${NPROC_PER_NODE}" \
    scripts/precompute_dino_latents.py \
    task="${TASK}" \
    dino_latent_cache_dir="${FRAME_CACHE_DIR}" \
    dino_latent_cache_mode=frame \
    dino_precompute_batch_size="${DINO_PRECOMPUTE_BATCH_SIZE}" \
    dino_precompute_num_workers="${DINO_PRECOMPUTE_NUM_WORKERS}" \
    dino_latent_cache_dtype="${DINO_LATENT_CACHE_DTYPE}" \
    overwrite="${overwrite_bool}" \
    model.dino_config.load_backbone=true \
    model.dino_config.encode_microbatch_size="${DINO_ENCODE_MICROBATCH_SIZE}" \
    2>&1 | tee -a "${LOG_FILE}"
}

run_mmap() {
  local overwrite_flag=()
  if [[ "$(bool_arg "${OVERWRITE}")" == "true" ]]; then
    overwrite_flag=(--overwrite)
  fi

  log "Stage=mmap src=${FRAME_CACHE_DIR} dst=${MMAP_CACHE_DIR}"
  log "mmap read_workers=${DINO_MMAP_READ_WORKERS}, read_batch_size=${DINO_MMAP_READ_BATCH_SIZE}, flush_every=${DINO_MMAP_FLUSH_EVERY}"
  require_dir "${FRAME_CACHE_DIR}/frames" "Robotwin frame cache directory"

  if [[ "$(bool_arg "${OVERWRITE}")" == "false" && -s "${MMAP_CACHE_DIR}/metadata.json" ]]; then
    if compgen -G "${MMAP_CACHE_DIR}/frames.*.bin" >/dev/null; then
      log "Existing mmap cache found; skipping pack. Set OVERWRITE=true to rebuild."
      return 0
    fi
  fi

  "${PYTHON}" scripts/convert_dino_frame_cache_to_mmap.py \
    --src "${FRAME_CACHE_DIR}" \
    --dst "${MMAP_CACHE_DIR}" \
    --read-workers "${DINO_MMAP_READ_WORKERS}" \
    --read-batch-size "${DINO_MMAP_READ_BATCH_SIZE}" \
    --flush-every "${DINO_MMAP_FLUSH_EVERY}" \
    "${overwrite_flag[@]}" \
    2>&1 | tee -a "${LOG_FILE}"
}

run_verify() {
  log "Stage=verify mmap cache=${MMAP_CACHE_DIR}"
  require_file "${MMAP_CACHE_DIR}/metadata.json" "Robotwin mmap cache metadata"

  VERIFY_TASK="${VERIFY_TASK}" MMAP_CACHE_DIR="${MMAP_CACHE_DIR}" "${PYTHON}" - <<'PY' 2>&1 | tee -a "${LOG_FILE}"
import os
from pathlib import Path

from hydra import compose, initialize_config_dir
from hydra.utils import instantiate
from omegaconf import OmegaConf

root = Path.cwd()
config_dir = str(root / "configs")
task = os.environ["VERIFY_TASK"]
mmap_cache_dir = os.environ["MMAP_CACHE_DIR"]
print("verify_task", task)
with initialize_config_dir(config_dir=config_dir, version_base="1.3"):
    cfg = compose(
        config_name="train",
        overrides=[
            f"task={task}",
            f"data.train.dino_latent_cache_dir={mmap_cache_dir}",
            "data.train.load_text_context=false",
            "data.train.val_set_proportion=0.0",
            "data.val=null",
        ],
    )
data_cfg = OmegaConf.create(OmegaConf.to_container(cfg.data.train, resolve=True))
dataset = instantiate(data_cfg)
sample = dataset[0]
print("dataset_len", len(dataset))
print("video", tuple(sample["video"].shape))
print("dino_latents", tuple(sample["dino_latents"].shape), sample["dino_latents"].dtype)
print("history_dino_latents", tuple(sample["history_dino_latents"].shape), sample["history_dino_latents"].dtype)
PY
}

case "${STAGE}" in
  all|frames|mmap|verify) ;;
  *)
    echo "Usage: STAGE=all|frames|mmap|verify bash scripts/precompute_robotwin_dino_cache.sh" >&2
    exit 2
    ;;
esac

ensure_robotwin_default_path
require_file "${PYTHON}" "Python executable"
require_dir "./data/robotwin2.0/robotwin2.0" "Robotwin LeRobot dataset"
require_file "./data/robotwin2.0/dataset_stats.json" "Robotwin dataset stats"
require_file "./checkpoints/dinov3_weights/dinov3_vits16_timm_lvd1689m.safetensors" "DINO-S weights"

log "Log file: ${LOG_FILE}"
log "Root: ${ROOT_DIR}"

if [[ "${STAGE}" == "all" || "${STAGE}" == "frames" ]]; then
  run_frames
fi
if [[ "${STAGE}" == "all" || "${STAGE}" == "mmap" ]]; then
  run_mmap
fi
if [[ "${STAGE}" == "all" || "${STAGE}" == "verify" ]]; then
  run_verify
fi

log "Done."
