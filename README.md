# ST-WAM

Official implementation of **ST-WAM: Semantic-Temporal World Action Model for
Robust Robot Manipulation**.

ST-WAM extends [FastWAM](https://github.com/yuantianyuan01/FastWAM) with two
complementary components:

- a semantic future expert that predicts frozen DINOv3 features alongside the
  VAE video and action experts; and
- Current-Anchored Intent Readout (CAIR), which uses frozen Qwen3-VL current
  semantics to retrieve short-horizon evidence from frozen DINO history and
  conditions only the action expert.

The repository includes the full LIBERO and RoboTwin training/evaluation paths,
the paper ablations, latent-cache tools, and persistent multi-GPU evaluators.

## Installation

The reference environment uses Python 3.10, CUDA 12.8, PyTorch 2.7.1, and
bf16 training.

```bash
conda create -n st-wam python=3.10 -y
conda activate st-wam
pip install -U pip
pip install torch==2.7.1+cu128 torchvision==0.22.1+cu128 \
  --extra-index-url https://download.pytorch.org/whl/cu128
pip install -e .
```

Install LIBERO and RoboTwin from their official repositories. The RoboTwin
evaluation wrapper expects its checkout at `third_party/RoboTwin` by default.

## Required Models

The main configuration uses:

- `Wan-AI/Wan2.2-TI2V-5B`
- `Wan-AI/Wan2.1-T2V-1.3B` tokenizer assets
- `facebook/dinov3-vits16-pretrain-lvd1689m`
- `Qwen/Qwen3-VL-4B-Instruct`

Put locally generated initialization weights under `checkpoints/`:

```text
checkpoints/
  ActionDiT_linear_interp_Wan22_alphascale_1024hdim.pt
  DinoVideoDiT_smallvideo_from_Wan22_alphascale_1024hdim.pt
  dinov3_weights/
    dinov3_vits16_timm_lvd1689m.safetensors
```

Generate the two DiT initializations with:

```bash
python scripts/preprocess_action_dit_backbone.py \
  --model-config configs/model/fastwam_wan5b_dino_s_aux_mot_short_qwen3vl_hist4.yaml \
  --output checkpoints/ActionDiT_linear_interp_Wan22_alphascale_1024hdim.pt

python scripts/preprocess_dino_video_dit_backbone.py \
  --model-config configs/model/fastwam_wan5b_dino_s_aux_mot_short_qwen3vl_hist4.yaml \
  --output checkpoints/DinoVideoDiT_smallvideo_from_Wan22_alphascale_1024hdim.pt
```

Released checkpoints are hosted at
[ModelScope](https://modelscope.cn/models/THU4Spiderman/Semantic_Temporal_World_Action_Model).

## Data And Caches

Dataset locations are relative in `configs/data/` and can be overridden with
Hydra. The LIBERO main task expects precomputed Wan VAE and DINO frame caches;
RoboTwin can encode DINO history online.

```bash
TASK=libero_wan5b_dino_s_aux_mot_short_qwen3vl_hist4_vae_mmap_2cam_224_1e-4

torchrun --standalone --nproc_per_node=8 \
  scripts/precompute_text_embeds.py task=${TASK}
torchrun --standalone --nproc_per_node=8 \
  scripts/precompute_vae_latents.py task=${TASK}
torchrun --standalone --nproc_per_node=8 \
  scripts/precompute_dino_latents.py task=${TASK}
```

Cache directories and cache modes are defined in the selected task YAML. Use
Hydra overrides when storing caches elsewhere.

## Training

Main LIBERO training uses 8 GPUs, micro-batch 8 per GPU, gradient accumulation
2, and global batch 128:

```bash
bash scripts/train_zero1.sh 8 \
  task=libero_wan5b_dino_s_aux_mot_short_qwen3vl_hist4_vae_mmap_2cam_224_1e-4 \
  output_dir=./runs/st_wam_libero \
  model.semantic_history_config.vlm_model_name_or_path=/path/to/Qwen3-VL-4B-Instruct \
  wandb.enabled=false
```

The main RoboTwin configuration is
`robotwin_wan5b_dino_s_aux_mot_short_qwen3vl_hist4_3cam_384x320_1e-4`.
The clean-only 15-task setup and its normalization statistics are also included.
Distributed launch details are cluster dependent; keep the paper's global batch
1024 when changing GPU count or per-device batch size.

## Evaluation

LIBERO evaluates all four suites with 30 trials per task and replans every 10
environment steps:

```bash
export LIBERO_ROOT=/path/to/LIBERO
python experiments/libero/run_libero_manager.py \
  task=libero_wan5b_dino_s_aux_mot_short_qwen3vl_hist4_vae_mmap_2cam_224_1e-4 \
  ckpt=/path/to/checkpoint.pt \
  EVALUATION.num_trials=30 \
  EVALUATION.output_dir=./evaluate_results/libero/st_wam \
  MULTIRUN.num_gpus=8 MULTIRUN.max_tasks_per_gpu=2
```

`experiments/libero/run_libero_persistent_pool.py` keeps one model instance per
worker, skips completed tasks on resume, supports worker restarts, and accepts a
CUDA-to-EGL device map for headless rendering. Run it with `--help` for all
options.

RoboTwin replans every 24 steps. The manager evaluates every task by default;
set either one task or a validated task list:

```bash
python experiments/robotwin/run_robotwin_manager.py \
  task=robotwin_wan5b_dino_s_aux_mot_short_qwen3vl_hist4_3cam_384x320_1e-4 \
  ckpt=/path/to/checkpoint.pt \
  EVALUATION.task_names='[adjust_bottle,click_bell]' \
  EVALUATION.eval_video_log=false \
  model.semantic_history_config.vlm_model_name_or_path=/path/to/Qwen3-VL-4B-Instruct \
  model.semantic_history_config.allow_vlm_path_relocation=true
```

## Ablation Configurations

| Paper variant | Model configuration |
| --- | --- |
| DINO Future Only | `fastwam_dino_s_smallvideo` |
| Dual-Space w/o CAIR | `fastwam_wan5b_dino_s_aux_mot` |
| w/o Semantic Future Expert | `fastwam_qwen3vl_dino_history` |
| Semantic Expert w/o Future Obj. | `fastwam_wan5b_dino_s_aux_mot_condition_only` |
| Naive History Retrieval | `fastwam_wan5b_dino_s_aux_mot_short_intent_hist4` |
| Qwen Current Only | `fastwam_wan5b_dino_s_aux_mot_short_qwen3vl_current` |
| CAIR with VAE History | `fastwam_wan5b_dino_s_aux_mot_short_qwen3vl_vae_hist4` |
| ST-WAM | `fastwam_wan5b_dino_s_aux_mot_short_qwen3vl_hist4` |

Matching benchmark task YAMLs are in `configs/task/`. Static contract checks for
the semantic variants live in `scripts/check_semantic_ablation_variants.py` and
`scripts/check_qwen_vae_history_*.py`.

## Acknowledgements

This project is built on FastWAM and uses components from Wan, DINOv3,
Qwen3-VL, LIBERO, and RoboTwin. Their licenses and usage terms continue to
apply. RoboTwin-derived source is kept under `third_party/RoboTwin` with its
upstream notices.

## License

See [LICENSE](LICENSE).
