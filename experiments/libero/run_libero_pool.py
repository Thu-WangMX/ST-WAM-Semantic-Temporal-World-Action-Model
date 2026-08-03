import argparse
import os
import queue
import subprocess
import sys
import time
from dataclasses import dataclass
from pathlib import Path


project_root = Path(__file__).resolve().parents[2]
project_src = project_root / "src"
libero_root = Path(os.environ.get("LIBERO_ROOT", project_root.parent / "LIBERO"))
for path in (libero_root, project_src, project_root):
    path_str = str(path)
    if path.exists() and path_str not in sys.path:
        sys.path.insert(0, path_str)

from libero.libero import benchmark


@dataclass(frozen=True)
class Task:
    suite: str
    task_id: int


@dataclass
class RunningTask:
    task: Task
    gpu_id: str
    proc: subprocess.Popen
    log_file: Path
    result_file: Path


def parse_csv(value: str) -> list[str]:
    return [x.strip() for x in value.split(",") if x.strip()]


def build_task_list(suites: list[str]) -> list[Task]:
    benchmark_dict = benchmark.get_benchmark_dict()
    tasks: list[Task] = []
    for suite_name in suites:
        task_suite = benchmark_dict[suite_name]()
        n_tasks = int(task_suite.n_tasks)
        print(f"{suite_name}: {n_tasks} tasks", flush=True)
        for task_id in range(n_tasks):
            tasks.append(Task(suite=suite_name, task_id=task_id))
    print(f"Total tasks: {len(tasks)}", flush=True)
    return tasks


def result_file_for(output_dir: Path, gpu_id: str, task: Task) -> Path:
    return output_dir / task.suite / f"gpu{gpu_id}_task{task.task_id}_results.json"


def any_result_exists(output_dir: Path, task: Task) -> bool:
    suite_dir = output_dir / task.suite
    return any(suite_dir.glob(f"gpu*_task{task.task_id}_results.json"))


def launch_task(
    *,
    task: Task,
    gpu_id: str,
    args: argparse.Namespace,
    extra_overrides: list[str],
    env_base: dict[str, str],
) -> RunningTask:
    output_dir = Path(args.output_dir)
    log_dir = output_dir / "task_logs"
    log_dir.mkdir(parents=True, exist_ok=True)
    (output_dir / task.suite).mkdir(parents=True, exist_ok=True)

    log_file = log_dir / f"{task.suite}_task{task.task_id}_gpu{gpu_id}.log"
    result_file = result_file_for(output_dir, gpu_id, task)
    if result_file.exists():
        result_file.unlink()

    env = env_base.copy()
    env["CUDA_VISIBLE_DEVICES"] = str(gpu_id)
    env["EXP_NAME"] = args.exp_name

    cmd = [
        args.python_bin,
        "experiments/libero/eval_libero_single.py",
        f"task={args.config}",
        f"ckpt={args.ckpt}",
        f"EVALUATION.task_suite_name={task.suite}",
        f"EVALUATION.task_id={task.task_id}",
        f"gpu_id={gpu_id}",
        f"EVALUATION.num_trials={args.num_trials}",
        f"EVALUATION.output_dir={args.output_dir}",
        *extra_overrides,
    ]
    with log_file.open("w", encoding="utf-8") as log:
        proc = subprocess.Popen(
            cmd,
            cwd=args.project_dir,
            env=env,
            stdout=log,
            stderr=subprocess.STDOUT,
            text=True,
        )
    print(
        f"[{time.strftime('%F %T')}] launch {task.suite} task={task.task_id} "
        f"gpu={gpu_id} pid={proc.pid}",
        flush=True,
    )
    return RunningTask(
        task=task,
        gpu_id=gpu_id,
        proc=proc,
        log_file=log_file,
        result_file=result_file,
    )


def summarize(args: argparse.Namespace, env_base: dict[str, str]) -> None:
    output_dir = Path(args.output_dir)
    subprocess.run(
        [args.python_bin, "experiments/libero/summarize_results.py", f"--output_dir={output_dir}"],
        cwd=args.project_dir,
        env=env_base,
        check=True,
        text=True,
    )
    if args.plus_summary:
        cmd = [
            args.python_bin,
            "experiments/libero/summarize_libero_plus_results.py",
            "--output_dir",
            str(output_dir),
        ]
        if args.classification_path:
            cmd.extend(["--classification_path", args.classification_path])
        subprocess.run(cmd, cwd=args.project_dir, env=env_base, check=True, text=True)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--project-dir", default=str(project_root))
    parser.add_argument("--python-bin", default=sys.executable)
    parser.add_argument("--config", required=True)
    parser.add_argument("--ckpt", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--suites", default="libero_spatial,libero_object,libero_goal,libero_10")
    parser.add_argument("--expected-num-tasks", type=int, default=None)
    parser.add_argument("--num-trials", type=int, default=1)
    parser.add_argument("--gpu-ids", required=True)
    parser.add_argument("--max-tasks-per-gpu", type=int, default=1)
    parser.add_argument("--extra-override", action="append", default=[])
    parser.add_argument("--classification-path", default=None)
    parser.add_argument("--plus-summary", action="store_true")
    parser.add_argument("--skip-existing", action="store_true")
    parser.add_argument("--exp-name", default="")
    parser.add_argument("--status-interval", type=int, default=60)
    args = parser.parse_args()

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    gpu_ids = parse_csv(args.gpu_ids)
    if not gpu_ids:
        raise ValueError("--gpu-ids cannot be empty")

    suites = parse_csv(args.suites)
    tasks = build_task_list(suites)
    if args.expected_num_tasks is not None and len(tasks) != args.expected_num_tasks:
        raise ValueError(
            f"Task list has {len(tasks)} tasks, expected {args.expected_num_tasks}. "
            "Check LIBERO_ROOT/PYTHONPATH."
        )

    if args.skip_existing:
        before = len(tasks)
        tasks = [task for task in tasks if not any_result_exists(output_dir, task)]
        print(f"Skipping {before - len(tasks)} completed tasks; {len(tasks)} remain.", flush=True)

    task_queue: queue.Queue[Task] = queue.Queue()
    for task in tasks:
        task_queue.put(task)

    env_base = os.environ.copy()
    pythonpath_parts = [str(libero_root), str(project_root / "src"), str(project_root)]
    if env_base.get("PYTHONPATH"):
        pythonpath_parts.append(env_base["PYTHONPATH"])
    env_base["PYTHONPATH"] = os.pathsep.join(pythonpath_parts)
    env_base.setdefault("MUJOCO_GL", "egl")
    env_base.setdefault("PYOPENGL_PLATFORM", "egl")
    env_base.setdefault("MPLCONFIGDIR", "/tmp/matplotlib-cache")
    env_base.setdefault("TOKENIZERS_PARALLELISM", "false")
    env_base.setdefault("TORCH_FORCE_NO_WEIGHTS_ONLY_LOAD", "1")
    for key, value in {
        "OMP_NUM_THREADS": "2",
        "MKL_NUM_THREADS": "1",
        "OPENBLAS_NUM_THREADS": "1",
        "NUMEXPR_NUM_THREADS": "1",
    }.items():
        env_base.setdefault(key, value)

    capacity = {gpu_id: args.max_tasks_per_gpu for gpu_id in gpu_ids}
    running: list[RunningTask] = []
    completed = 0
    failed: list[RunningTask] = []
    last_status = 0.0

    while not task_queue.empty() or running:
        for gpu_id in gpu_ids:
            while capacity[gpu_id] > 0 and not task_queue.empty() and not failed:
                task = task_queue.get()
                running_task = launch_task(
                    task=task,
                    gpu_id=gpu_id,
                    args=args,
                    extra_overrides=args.extra_override,
                    env_base=env_base,
                )
                running.append(running_task)
                capacity[gpu_id] -= 1

        still_running: list[RunningTask] = []
        for item in running:
            rc = item.proc.poll()
            if rc is None:
                still_running.append(item)
                continue
            capacity[item.gpu_id] += 1
            if rc == 0 and item.result_file.exists() and item.result_file.stat().st_size > 0:
                completed += 1
            else:
                failed.append(item)
                with (output_dir / "failed_tasks.txt").open("a", encoding="utf-8") as f:
                    f.write(
                        f"{item.task.suite},{item.task.task_id},gpu={item.gpu_id},"
                        f"rc={rc},log={item.log_file}\n"
                    )
                print(
                    f"[{time.strftime('%F %T')}] FAILED {item.task.suite} "
                    f"task={item.task.task_id} gpu={item.gpu_id} rc={rc} log={item.log_file}",
                    flush=True,
                )
        running = still_running

        now = time.time()
        if now - last_status >= args.status_interval:
            pending = task_queue.qsize()
            print(
                f"[{time.strftime('%F %T')}] status completed={completed} "
                f"running={len(running)} pending={pending} failed={len(failed)}",
                flush=True,
            )
            last_status = now

        if failed:
            for item in running:
                item.proc.terminate()
            time.sleep(5)
            for item in running:
                if item.proc.poll() is None:
                    item.proc.kill()
            raise SystemExit(1)

        time.sleep(2)

    summarize(args, env_base)
    print(f"[{time.strftime('%F %T')}] all tasks completed successfully", flush=True)


if __name__ == "__main__":
    main()
