import argparse
import csv
import json
import os
from collections import defaultdict
from pathlib import Path


CATEGORY_ORDER = [
    ("Camera", "Camera Viewpoints"),
    ("Robot", "Robot Initial States"),
    ("Language", "Language Instructions"),
    ("Light", "Light Conditions"),
    ("Background", "Background Textures"),
    ("Noise", "Sensor Noise"),
    ("Layout", "Objects Layout"),
]


def _find_classification_path(explicit_path: str | None) -> Path:
    candidates = []
    if explicit_path:
        candidates.append(Path(explicit_path))

    libero_plus_root = os.environ.get("LIBERO_PLUS_ROOT") or os.environ.get("LIBERO_ROOT")
    if libero_plus_root:
        candidates.append(Path(libero_plus_root) / "libero" / "libero" / "benchmark" / "task_classification.json")

    for candidate in candidates:
        candidate = candidate.expanduser().resolve()
        if candidate.is_file():
            return candidate

    raise FileNotFoundError(
        "Failed to locate LIBERO-plus task_classification.json. "
        "Pass --classification_path or set LIBERO_PLUS_ROOT/LIBERO_ROOT."
    )


def _load_classification(path: Path) -> dict[tuple[str, int], dict]:
    raw = json.loads(path.read_text(encoding="utf-8"))
    mapping: dict[tuple[str, int], dict] = {}

    for suite, entries in raw.items():
        for entry in entries:
            # LIBERO-plus classification ids are 1-based; eval result filenames use 0-based task ids.
            task_id = int(entry["id"]) - 1
            mapping[(suite, task_id)] = {
                "plus_id": int(entry["id"]),
                "name": entry.get("name", ""),
                "category": entry.get("category", "Unknown"),
                "difficulty_level": entry.get("difficulty_level"),
            }
    return mapping


def _iter_result_files(output_dir: Path):
    for suite_dir in sorted(output_dir.iterdir()):
        if not suite_dir.is_dir() or not suite_dir.name.startswith("libero_"):
            continue
        suite = suite_dir.name
        for result_file in sorted(suite_dir.glob("gpu*_task*_results.json")):
            stem = result_file.stem
            task_part = stem.split("_")[1]
            task_id = int(task_part.replace("task", ""))
            yield suite, task_id, result_file


def _new_bucket():
    return {
        "tasks": 0,
        "episodes": 0,
        "successes": 0,
        "duration": 0.0,
    }


def _add(bucket: dict, result: dict) -> None:
    bucket["tasks"] += 1
    bucket["episodes"] += int(result.get("total_episodes", 0))
    bucket["successes"] += int(result.get("successes", 0))
    bucket["duration"] += float(result.get("duration", 0.0))


def _rate(bucket: dict) -> float:
    episodes = bucket["episodes"]
    return 100.0 * bucket["successes"] / episodes if episodes else 0.0


def _row(label: str, bucket: dict) -> dict:
    avg_time = bucket["duration"] / bucket["tasks"] if bucket["tasks"] else 0.0
    return {
        "Group": label,
        "Success Rate (%)": f"{_rate(bucket):.2f}",
        "Tasks": str(bucket["tasks"]),
        "Episodes": str(bucket["episodes"]),
        "Successes": str(bucket["successes"]),
        "Average Time (s)": f"{avg_time:.2f}",
    }


def summarize_libero_plus(output_dir: Path, classification_path: Path) -> None:
    classification = _load_classification(classification_path)

    by_category = defaultdict(_new_bucket)
    by_suite = defaultdict(_new_bucket)
    by_difficulty = defaultdict(_new_bucket)
    by_suite_category = defaultdict(_new_bucket)
    total = _new_bucket()
    task_rows = []
    missing_classification = []

    for suite, task_id, result_file in _iter_result_files(output_dir):
        result = json.loads(result_file.read_text(encoding="utf-8"))
        meta = classification.get((suite, task_id))
        if meta is None:
            missing_classification.append(f"{suite}:{task_id}")
            meta = {
                "plus_id": task_id + 1,
                "name": "",
                "category": "Unknown",
                "difficulty_level": None,
            }

        category = meta["category"]
        difficulty = meta["difficulty_level"]
        success_rate = _rate(
            {
                "tasks": 1,
                "episodes": int(result.get("total_episodes", 0)),
                "successes": int(result.get("successes", 0)),
                "duration": float(result.get("duration", 0.0)),
            }
        )

        _add(total, result)
        _add(by_category[category], result)
        _add(by_suite[suite], result)
        _add(by_difficulty[str(difficulty)], result)
        _add(by_suite_category[(suite, category)], result)

        task_rows.append(
            {
                "Suite": suite,
                "Task ID": str(task_id),
                "LIBERO-plus ID": str(meta["plus_id"]),
                "Category": category,
                "Difficulty": str(difficulty),
                "Success Rate (%)": f"{success_rate:.2f}",
                "Episodes": str(result.get("total_episodes", 0)),
                "Successes": str(result.get("successes", 0)),
                "Name": meta["name"],
                "Description": result.get("task_description", ""),
                "Result File": str(result_file),
            }
        )

    leaderboard_rows = []
    for short_name, full_name in CATEGORY_ORDER:
        leaderboard_rows.append(_row(short_name, by_category[full_name]))
    leaderboard_rows.append(_row("Total", total))

    category_rows = [_row(full_name, by_category[full_name]) for _, full_name in CATEGORY_ORDER]
    category_rows.append(_row("Total", total))
    suite_rows = [_row(suite, by_suite[suite]) for suite in sorted(by_suite)]
    difficulty_rows = [_row(diff, by_difficulty[diff]) for diff in sorted(by_difficulty, key=lambda x: (x == "None", x))]
    suite_category_rows = []
    for suite in sorted(by_suite):
        for _, full_name in CATEGORY_ORDER:
            bucket = by_suite_category[(suite, full_name)]
            if bucket["tasks"]:
                row = _row(full_name, bucket)
                row["Suite"] = suite
                suite_category_rows.append(row)

    output_dir.mkdir(parents=True, exist_ok=True)
    _write_csv(output_dir / "plus_leaderboard_summary.csv", leaderboard_rows)
    _write_csv(output_dir / "plus_category_summary.csv", category_rows)
    _write_csv(output_dir / "plus_suite_summary.csv", suite_rows)
    _write_csv(output_dir / "plus_difficulty_summary.csv", difficulty_rows)
    _write_csv(output_dir / "plus_suite_category_summary.csv", suite_category_rows)
    _write_csv(output_dir / "plus_task_success_rates.csv", task_rows)

    summary = {
        "run_id": output_dir.name,
        "ckpt": os.environ.get("CKPT", ""),
        "config": os.environ.get("CONFIG", ""),
        "classification_path": str(classification_path),
        "expected_tasks_from_classification": len(classification),
        "completed_tasks": total["tasks"],
        "missing_classification": missing_classification,
        "leaderboard": {row["Group"]: row for row in leaderboard_rows},
        "categories": {row["Group"]: row for row in category_rows},
        "suites": {row["Group"]: row for row in suite_rows},
        "difficulties": {row["Group"]: row for row in difficulty_rows},
    }
    (output_dir / "plus_summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")

    print("\n=== LIBERO-plus Leaderboard Summary ===")
    for row in leaderboard_rows:
        print(
            f"{row['Group']}: {row['Success Rate (%)']}% "
            f"({row['Successes']}/{row['Episodes']} over {row['Tasks']} tasks)"
        )
    print(f"LIBERO-plus summary: {output_dir / 'plus_summary.json'}")
    print(f"Leaderboard CSV: {output_dir / 'plus_leaderboard_summary.csv'}")
    if missing_classification:
        print(f"WARNING: {len(missing_classification)} result files lacked classification metadata.")


def _write_csv(path: Path, rows: list[dict]) -> None:
    if not rows:
        path.write_text("", encoding="utf-8")
        return
    fieldnames = list(rows[0].keys())
    for row in rows[1:]:
        for key in row:
            if key not in fieldnames:
                fieldnames.append(key)

    with path.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output_dir", required=True, type=Path)
    parser.add_argument("--classification_path", default=None)
    args = parser.parse_args()

    classification_path = _find_classification_path(args.classification_path)
    summarize_libero_plus(args.output_dir, classification_path)


if __name__ == "__main__":
    main()
