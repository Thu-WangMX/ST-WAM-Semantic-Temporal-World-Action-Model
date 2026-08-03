"""Static contract checks for the LIBERO semantic-ablation variants."""

from __future__ import annotations

import argparse
import ast
from pathlib import Path

import yaml


def load_yaml(root: Path, relative_path: str):
    with (root / relative_path).open("r", encoding="utf-8") as handle:
        return yaml.safe_load(handle)


def require(condition: bool, message: str) -> None:
    if not condition:
        raise AssertionError(message)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", type=Path, default=Path(__file__).resolve().parents[1])
    args = parser.parse_args()
    root = args.root.resolve()

    qwen_current_model = load_yaml(
        root, "configs/model/fastwam_wan5b_dino_s_aux_mot_short_qwen3vl_current.yaml"
    )
    qwen_current_task = load_yaml(
        root, "configs/task/libero_wan5b_dino_s_aux_mot_short_qwen3vl_current_vae_mmap_2cam_224_1e-4.yaml"
    )
    qwen_current_semantic = qwen_current_model["semantic_history_config"]
    require(qwen_current_semantic["enabled"], "Qwen-current must enable its semantic adapter.")
    require(not qwen_current_semantic["use_history"], "Qwen-current must disable history memory.")
    require(qwen_current_semantic["history_offsets"] == [], "Qwen-current must have no history offsets.")
    require(
        not qwen_current_task.get("data", {}).get("train", {}).get("load_history_dino_latents", False),
        "Qwen-current task must not load DINO history.",
    )

    qwen_vae_model = load_yaml(
        root, "configs/model/fastwam_wan5b_dino_s_aux_mot_short_qwen3vl_vae_hist4.yaml"
    )
    qwen_vae_task = load_yaml(
        root,
        "configs/task/libero_wan5b_dino_s_aux_mot_short_qwen3vl_vae_hist4_vae_mmap_2cam_224_1e-4.yaml",
    )
    qwen_vae_semantic = qwen_vae_model["semantic_history_config"]
    qwen_vae_data = qwen_vae_task["data"]["train"]
    require(qwen_vae_semantic["source"] == "vae", "Qwen VAE-history must select VAE history.")
    require(qwen_vae_semantic["use_history"], "Qwen VAE-history must enable history memory.")
    require(qwen_vae_data["load_history_vae_video"], "Qwen VAE-history must load RGB history.")
    require(not qwen_vae_data["load_history_dino_latents"], "Qwen VAE-history must not load DINO history.")
    require(
        qwen_vae_semantic["history_offsets"] == qwen_vae_data["history_vae_frame_offsets"],
        "Qwen VAE-history train/infer offsets must match.",
    )

    qwen_2mot_model = load_yaml(root, "configs/model/fastwam_qwen3vl_dino_history.yaml")
    qwen_2mot_task = load_yaml(
        root, "configs/task/libero_fastwam_qwen3vl_dino_history_2cam_224_1e-4.yaml"
    )
    require(
        qwen_2mot_model["_target_"] == "fastwam.runtime.create_fastwam_semantic_history",
        "Qwen-history 2-MoT must use the isolated two-expert factory.",
    )
    require(qwen_2mot_model["semantic_history_config"]["use_history"], "2-MoT readout needs history.")
    require(
        qwen_2mot_task["data"]["train"]["load_history_dino_latents"],
        "2-MoT readout task must load cached DINO history.",
    )
    require(
        qwen_2mot_task["data"]["train"]["load_semantic_image"],
        "2-MoT readout task must load the current semantic image.",
    )

    condition_model = load_yaml(root, "configs/model/fastwam_wan5b_dino_s_aux_mot_condition_only.yaml")
    condition_task = load_yaml(
        root, "configs/task/libero_wan5b_dino_s_aux_mot_condition_only_2cam_224_1e-4.yaml"
    )
    require(condition_model["dino_future_mode"] == "condition_only", "Condition-only mode is not selected.")
    require(
        float(condition_model["loss"]["lambda_dino"]) == 0.0,
        "Condition-only configuration must not weight a DINO future loss.",
    )
    require(
        condition_task["defaults"][1]["override /model"] == "fastwam_wan5b_dino_s_aux_mot_condition_only",
        "Condition-only task must select the condition-only model.",
    )

    mot_source = (root / "src/fastwam/models/wan22/fastwam_vae_dino_mot.py").read_text(encoding="utf-8")
    semantic_source = (root / "src/fastwam/models/wan22/semantic_history.py").read_text(encoding="utf-8")
    two_mot_source = (root / "src/fastwam/models/wan22/fastwam_semantic_history.py").read_text(
        encoding="utf-8"
    )
    trainer_source = (root / "src/fastwam/trainer.py").read_text(encoding="utf-8")
    runtime_source = (root / "src/fastwam/runtime.py").read_text(encoding="utf-8")
    libero_eval_source = (root / "experiments/libero/eval_libero_single.py").read_text(
        encoding="utf-8"
    )
    for source, path in (
        (mot_source, "fastwam_vae_dino_mot.py"),
        (semantic_source, "semantic_history.py"),
        (two_mot_source, "fastwam_semantic_history.py"),
        (trainer_source, "trainer.py"),
        (runtime_source, "runtime.py"),
        (libero_eval_source, "eval_libero_single.py"),
    ):
        ast.parse(source, filename=path)
    require("dino_future_mode" in mot_source, "3-MoT implementation lacks the future-mode switch.")
    require("inputs[\"first_frame_dino_latents\"]" in mot_source, "Condition-only path must use DINO f0.")
    require("timestep_dino = torch.zeros_like(timestep_video)" in mot_source, "DINO f0 must use t=0.")
    require("if not self.use_history" in semantic_source, "Semantic adapter lacks the no-history path.")
    require("history_source" in semantic_source, "Semantic adapter lacks the history-source switch.")
    require("self.semantic_history_source == \"vae\"" in mot_source, "3-MoT lacks VAE semantic-history routing.")
    require("class FastWAMSemanticHistory" in two_mot_source, "Two-expert semantic model is missing.")
    require("context_action" in two_mot_source, "Two-expert adapter is not action-context-only.")
    require("def infer_action" in two_mot_source, "Two-expert model lacks policy inference.")
    require(
        "action_only_inference = True" in two_mot_source,
        "Two-expert model must route trainer evaluation through infer_action.",
    )
    require(
        "Qwen current-semantic adapter verified without history memory." in trainer_source,
        "Trainer lacks the Qwen-current no-history validation path.",
    )
    require(
        "action_only_inference or not hasattr(model, \"infer\")" in trainer_source,
        "Trainer does not honor the action-only inference contract.",
    )
    require(
        "def create_fastwam_semantic_history" in runtime_source,
        "Runtime factory for two-expert semantic model is missing.",
    )
    require(
        'semantic_config.get("use_history", True)' in libero_eval_source,
        "LIBERO evaluation must not construct history for Qwen-current.",
    )
    print("Semantic ablation variant contracts passed.")


if __name__ == "__main__":
    main()
