import argparse
import csv
from dataclasses import asdict, dataclass
import json
import multiprocessing as mp
import os
from pathlib import Path
import queue
import random
import time
import traceback

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.ticker import FuncFormatter
import numpy as np
import torch

from modern_npi.graph.executor import execute_dijkstra_batch
from modern_npi.graph.experiment import (
    Trainer,
    make_data,
    oracle_distances,
    valid_parent_tree,
)
from modern_npi.graph.model import GraphNPI
from modern_npi.graph.problems import GraphProblem, generate_problem


DEFAULT_OUTPUT = Path("artifacts/graph_generalization_sweep")
DEFAULT_FAMILIES = ("path", "star", "sparse", "dense", "disconnected", "directed")
COLORS = ("#00798C", "#D1495B", "#2A9D6F", "#8D6A9F", "#EDAE49", "#5B5F97")


@dataclass(frozen=True)
class CandidateResult:
    nodes: int
    required_tests: int
    correct_distances: int
    valid_parent_trees: int
    passed: bool
    average_model_steps: float
    seconds: float
    first_failure: str | None


@dataclass(frozen=True)
class CapacityResult:
    maximum_train_nodes: int
    seed: int
    training_step: int
    maximum_generalization_nodes: int
    first_failed_nodes: int | None
    reached_search_limit: bool
    candidates: tuple[CandidateResult, ...]


def parse_int_list(value: str) -> list[int]:
    values = [int(item.strip()) for item in value.split(",") if item.strip()]
    if not values:
        raise argparse.ArgumentTypeError("expected a comma-separated integer list")
    return values


def parse_str_list(value: str) -> list[str]:
    values = [item.strip() for item in value.split(",") if item.strip()]
    if not values:
        raise argparse.ArgumentTypeError("expected a comma-separated list")
    return values


def atomic_torch_save(payload: dict, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    torch.save(payload, temporary)
    os.replace(temporary, path)


def atomic_json_write(payload: dict, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(payload, indent=2) + "\n")
    os.replace(temporary, path)


def checkpoint_path(run_directory: Path, step: int) -> Path:
    return run_directory / "checkpoints" / f"step_{step:08d}.pt"


def evaluation_path(run_directory: Path, step: int) -> Path:
    return run_directory / "evaluations" / f"step_{step:08d}.json"


def make_evaluation_problems(
    nodes: int,
    tests: int,
    seed: int,
    families: tuple[str, ...],
    maximum_weight: int,
) -> list[GraphProblem]:
    family_order = [families[index % len(families)] for index in range(tests)]
    random.Random(seed + nodes * 1_000_003).shuffle(family_order)
    return [
        generate_problem(
            nodes,
            random.Random(seed + nodes * 1_000_003 + index * 97_409),
            family,
            maximum_weight,
        )
        for index, family in enumerate(family_order)
    ]


class CapacityEvaluator:
    def __init__(
        self,
        node_candidates: tuple[int, ...],
        tests_per_size: int,
        evaluation_seed: int,
        families: tuple[str, ...],
        maximum_weight: int,
        execution_batch_size: int,
    ):
        self.node_candidates = node_candidates
        self.tests_per_size = tests_per_size
        self.evaluation_seed = evaluation_seed
        self.families = families
        self.maximum_weight = maximum_weight
        self.execution_batch_size = execution_batch_size
        self.cache: dict[int, tuple[list[GraphProblem], list[list[int | None]]]] = {}

    def problems(self, nodes: int):
        if nodes not in self.cache:
            problems = make_evaluation_problems(
                nodes,
                self.tests_per_size,
                self.evaluation_seed,
                self.families,
                self.maximum_weight,
            )
            self.cache[nodes] = (
                problems,
                [oracle_distances(problem) for problem in problems],
            )
        return self.cache[nodes]

    def evaluate_candidate(self, model, nodes: int, device) -> CandidateResult:
        problems, expected_distances = self.problems(nodes)
        correct = 0
        valid_parents = 0
        model_steps = 0
        completed = 0
        first_failure = None
        started = time.time()

        for start in range(0, len(problems), self.execution_batch_size):
            problem_batch = problems[start : start + self.execution_batch_size]
            expected_batch = expected_distances[
                start : start + self.execution_batch_size
            ]
            outcomes = execute_dijkstra_batch(model, problem_batch, device)
            for problem, expected, outcome in zip(
                problem_batch, expected_batch, outcomes, strict=True
            ):
                completed += 1
                if outcome.failure is not None:
                    if first_failure is None:
                        first_failure = outcome.failure
                    continue
                result = outcome.result
                model_steps += result.model_steps
                if result.distances != expected:
                    if first_failure is None:
                        mismatches = [
                            index
                            for index, (actual, target) in enumerate(
                                zip(result.distances, expected, strict=True)
                            )
                            if actual != target
                        ]
                        first_failure = (
                            f"distance mismatch at nodes {mismatches[:8]}"
                        )
                    continue
                correct += 1
                if valid_parent_tree(problem, expected, result.parents):
                    valid_parents += 1
                elif first_failure is None:
                    first_failure = "distance correct but parent tree invalid"

        passed = correct == self.tests_per_size and valid_parents == self.tests_per_size
        return CandidateResult(
            nodes=nodes,
            required_tests=self.tests_per_size,
            correct_distances=correct,
            valid_parent_trees=valid_parents,
            passed=passed,
            average_model_steps=model_steps / max(completed, 1),
            seconds=time.time() - started,
            first_failure=first_failure,
        )

    def evaluate_capacity(
        self,
        model,
        maximum_train_nodes: int,
        seed: int,
        training_step: int,
        device,
    ) -> CapacityResult:
        candidates = []
        capacity = 0
        first_failed_nodes = None
        for nodes in self.node_candidates:
            result = self.evaluate_candidate(model, nodes, device)
            candidates.append(result)
            failure = (
                f" first_failure={result.first_failure}"
                if result.first_failure is not None
                else ""
            )
            print(
                f"[gpu {device.index}] train_max={maximum_train_nodes} "
                f"seed={seed} step={training_step} nodes={nodes}: "
                f"{result.correct_distances}/{result.required_tests} "
                f"seconds={result.seconds:.1f}{failure}",
                flush=True,
            )
            if not result.passed:
                first_failed_nodes = nodes
                break
            capacity = nodes
        return CapacityResult(
            maximum_train_nodes=maximum_train_nodes,
            seed=seed,
            training_step=training_step,
            maximum_generalization_nodes=capacity,
            first_failed_nodes=first_failed_nodes,
            reached_search_limit=first_failed_nodes is None,
            candidates=tuple(candidates),
        )


def save_training_checkpoint(
    model,
    trainer,
    path: Path,
    *,
    step: int,
    epoch: int,
    next_batch_index: int,
    maximum_train_nodes: int,
    seed: int,
) -> None:
    metadata = {
        "optimizer_step": step,
        "epoch": epoch,
        "next_batch_index": next_batch_index,
        "maximum_train_nodes": maximum_train_nodes,
        "seed": seed,
    }
    atomic_torch_save(
        {"model_state": model.state_dict(), **metadata},
        path,
    )
    atomic_torch_save(
        {
            "model_state": model.state_dict(),
            "optimizer_state": trainer.optimizer.state_dict(),
            "scheduler_state": trainer.scheduler.state_dict(),
            **metadata,
        },
        path.parent / "resume.pt",
    )


def train_checkpoints(
    maximum_train_nodes: int,
    seed: int,
    run_directory: Path,
    args,
    device,
) -> list[Path]:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    training_data, _ = make_data(
        2,
        maximum_train_nodes,
        args.training_examples_per_size,
        seed,
    )
    model = GraphNPI().to(device)
    trainer = Trainer(model, args.learning_rate, device)
    target_steps = tuple(sorted(set(args.checkpoint_steps)))
    target_set = set(target_steps)
    existing = [
        checkpoint_path(run_directory, step)
        for step in target_steps
        if checkpoint_path(run_directory, step).exists()
    ]

    optimizer_step = 0
    epoch = 1
    next_batch_index = 0
    resume_path = run_directory / "checkpoints" / "resume.pt"
    if args.resume and existing and resume_path.exists():
        payload = torch.load(resume_path, map_location=device, weights_only=False)
        resume_step = payload["optimizer_step"]
        if resume_step not in target_set or not checkpoint_path(
            run_directory, resume_step
        ).exists():
            raise RuntimeError("resume state has no matching model checkpoint")
        model.load_state_dict(payload["model_state"])
        trainer.optimizer.load_state_dict(payload["optimizer_state"])
        trainer.scheduler.load_state_dict(payload["scheduler_state"])
        optimizer_step = payload["optimizer_step"]
        epoch = payload["epoch"]
        next_batch_index = payload["next_batch_index"]
        print(
            f"[gpu {device.index}] resumed train_max={maximum_train_nodes} "
            f"seed={seed} at step={optimizer_step}",
            flush=True,
        )

    maximum_step = target_steps[-1]
    started = time.time()
    while optimizer_step < maximum_step:
        batches = training_data.batches(
            args.batch_size,
            shuffle=True,
            seed=seed + epoch,
        )
        consumed_any = False
        for batch_index, batch in enumerate(batches):
            if batch_index < next_batch_index:
                continue
            consumed_any = True
            model.train()
            trainer.train_batch(batch)
            optimizer_step += 1
            next_batch_index = batch_index + 1
            if optimizer_step in target_set:
                path = checkpoint_path(run_directory, optimizer_step)
                save_training_checkpoint(
                    model,
                    trainer,
                    path,
                    step=optimizer_step,
                    epoch=epoch,
                    next_batch_index=next_batch_index,
                    maximum_train_nodes=maximum_train_nodes,
                    seed=seed,
                )
                print(
                    f"[gpu {device.index}] train_max={maximum_train_nodes} "
                    f"seed={seed}: saved step={optimizer_step}",
                    flush=True,
                )
            if optimizer_step >= maximum_step:
                break
        if optimizer_step >= maximum_step:
            break
        if not consumed_any and next_batch_index == 0:
            raise RuntimeError("training data produced no batches")
        epoch += 1
        next_batch_index = 0

    print(
        f"[gpu {device.index}] train_max={maximum_train_nodes} seed={seed}: "
        f"training completed in {time.time() - started:.1f}s",
        flush=True,
    )
    del trainer, model, training_data
    torch.cuda.empty_cache()
    return [checkpoint_path(run_directory, step) for step in target_steps]


def run_job(maximum_train_nodes: int, seed: int, args, device) -> list[dict]:
    run_directory = (
        args.output / f"train_max_{maximum_train_nodes:04d}" / f"seed_{seed:04d}"
    )
    checkpoints = train_checkpoints(
        maximum_train_nodes,
        seed,
        run_directory,
        args,
        device,
    )
    evaluator = CapacityEvaluator(
        tuple(args.generalization_nodes),
        args.tests_per_size,
        args.evaluation_seed,
        tuple(args.families),
        args.evaluation_maximum_weight,
        args.execution_batch_size,
    )
    model = GraphNPI().to(device)
    rows = []
    for step, checkpoint in zip(args.checkpoint_steps, checkpoints, strict=True):
        result_path = evaluation_path(run_directory, step)
        if args.resume and result_path.exists():
            payload = json.loads(result_path.read_text())
            rows.append(payload)
            print(
                f"[gpu {device.index}] reused {result_path}",
                flush=True,
            )
            continue
        payload = torch.load(checkpoint, map_location=device, weights_only=False)
        model.load_state_dict(payload["model_state"])
        model.eval()
        result = evaluator.evaluate_capacity(
            model,
            maximum_train_nodes,
            seed,
            step,
            device,
        )
        serialized = asdict(result)
        atomic_json_write(serialized, result_path)
        rows.append(serialized)
    return rows


def gpu_worker(gpu_id: int, job_queue, result_queue, args_dict: dict) -> None:
    try:
        args = argparse.Namespace(**args_dict)
        torch.cuda.set_device(gpu_id)
        torch.set_num_threads(args.cpu_threads_per_worker)
        torch.set_float32_matmul_precision("high")
        device = torch.device("cuda", gpu_id)
        while True:
            job = job_queue.get()
            if job is None:
                break
            maximum_train_nodes, seed = job
            try:
                rows = run_job(maximum_train_nodes, seed, args, device)
                result_queue.put(("result", rows))
            except Exception:
                result_queue.put(
                    (
                        "error",
                        {
                            "gpu": gpu_id,
                            "maximum_train_nodes": maximum_train_nodes,
                            "seed": seed,
                            "traceback": traceback.format_exc(),
                        },
                    )
                )
    except Exception:
        result_queue.put(("worker_error", {"gpu": gpu_id, "traceback": traceback.format_exc()}))


def collect_existing_results(output: Path) -> list[dict]:
    rows = []
    for path in sorted(output.glob("train_max_*/seed_*/evaluations/step_*.json")):
        rows.append(json.loads(path.read_text()))
    return rows


def write_summary(rows: list[dict], output: Path, args) -> None:
    rows.sort(
        key=lambda row: (
            row["maximum_train_nodes"],
            row["seed"],
            row["training_step"],
        )
    )
    metadata = {
        "criterion": f"{args.tests_per_size}/{args.tests_per_size} exact closed-loop tests",
        "maximum_train_nodes": args.maximum_train_nodes,
        "checkpoint_steps": args.checkpoint_steps,
        "generalization_nodes": args.generalization_nodes,
        "families": args.families,
        "evaluation_seed": args.evaluation_seed,
        "evaluation_maximum_weight": args.evaluation_maximum_weight,
        "results": rows,
    }
    atomic_json_write(metadata, output / "sweep_results.json")
    csv_path = output / "sweep_results.csv"
    with csv_path.open("w", newline="") as handle:
        writer = csv.DictWriter(
            handle,
            fieldnames=(
                "maximum_train_nodes",
                "seed",
                "training_step",
                "maximum_generalization_nodes",
                "first_failed_nodes",
                "reached_search_limit",
            ),
        )
        writer.writeheader()
        for row in rows:
            writer.writerow({field: row[field] for field in writer.fieldnames})


def plot_results(rows: list[dict], output: Path, tests_per_size: int) -> None:
    grouped: dict[int, dict[int, list[dict]]] = {}
    for row in rows:
        grouped.setdefault(row["maximum_train_nodes"], {}).setdefault(
            row["training_step"], []
        ).append(row)

    plt.rcParams.update(
        {
            "font.family": "DejaVu Sans",
            "font.size": 10,
            "axes.edgecolor": "#343A40",
            "axes.linewidth": 0.8,
            "text.color": "#202428",
        }
    )
    figure, axis = plt.subplots(figsize=(10.5, 6.2))
    figure.patch.set_facecolor("white")
    for series_index, maximum_train_nodes in enumerate(sorted(grouped)):
        color = COLORS[series_index % len(COLORS)]
        by_step = grouped[maximum_train_nodes]
        steps = sorted(by_step)
        medians = []
        lows = []
        highs = []
        for step in steps:
            values = [
                row["maximum_generalization_nodes"] for row in by_step[step]
            ]
            medians.append(float(np.median(values)))
            lows.append(min(values))
            highs.append(max(values))
        axis.plot(
            steps,
            medians,
            color=color,
            marker="o",
            linewidth=2,
            label=f"train max {maximum_train_nodes}",
        )
        if any(low != high for low, high in zip(lows, highs, strict=True)):
            axis.fill_between(steps, lows, highs, color=color, alpha=0.14)

    figure.suptitle(
        "Maximum Verified Graph-Size Generalization",
        x=0.10,
        y=0.965,
        ha="left",
        fontsize=15,
        fontweight="bold",
    )
    figure.text(
        0.10,
        0.91,
        f"A size passes only with {tests_per_size}/{tests_per_size} exact closed-loop tests; "
        "lines show seed median and bands show range.",
        color="#5B6268",
        fontsize=9.5,
    )
    axis.set_xlabel("Optimizer step")
    axis.set_ylabel("Maximum generalization nodes")
    all_steps = [row["training_step"] for row in rows]
    all_bounds = [
        max(
            row["maximum_generalization_nodes"],
            row["first_failed_nodes"] or 0,
        )
        for row in rows
    ]
    maximum_step = max(all_steps)
    maximum_bound = max(max(all_bounds), 1)
    if min(all_steps) == maximum_step:
        axis.set_xlim(0, maximum_step * 1.15)
    axis.set_ylim(-0.03 * maximum_bound, 1.12 * maximum_bound)
    axis.xaxis.set_major_formatter(FuncFormatter(lambda value, _: f"{int(value):,}"))
    axis.yaxis.set_major_formatter(FuncFormatter(lambda value, _: f"{int(value):,}"))
    axis.grid(color="#DDE2E5", linewidth=0.8)
    axis.spines[["top", "right"]].set_visible(False)
    axis.legend(frameon=False, ncol=2)
    figure.subplots_adjust(left=0.10, right=0.97, top=0.84, bottom=0.12)
    figure.savefig(output / "generalization_vs_steps.png", dpi=220, facecolor="white")
    figure.savefig(output / "generalization_vs_steps.pdf", facecolor="white")
    plt.close(figure)


def validate_args(args, available_gpus: int) -> None:
    args.maximum_train_nodes = sorted(set(args.maximum_train_nodes))
    args.seeds = sorted(set(args.seeds))
    args.checkpoint_steps = sorted(set(args.checkpoint_steps))
    args.generalization_nodes = sorted(set(args.generalization_nodes))
    if min(args.maximum_train_nodes) < 2:
        raise ValueError("maximum training nodes must be at least 2")
    if min(args.checkpoint_steps) < 1:
        raise ValueError("checkpoint steps must be positive")
    if min(args.generalization_nodes) < 2:
        raise ValueError("generalization node candidates must be at least 2")
    if args.tests_per_size != 100:
        raise ValueError("--tests-per-size must be 100 for the requested 100%/100-test criterion")
    if args.gpu_count < 1:
        raise ValueError("--gpu-count must be at least 1")
    if args.gpu_count > available_gpus:
        raise ValueError(
            f"requested {args.gpu_count} GPUs, but PyTorch sees {available_gpus}"
        )
    unknown = set(args.families) - set(DEFAULT_FAMILIES)
    if unknown:
        raise ValueError(f"unknown graph families: {sorted(unknown)}")


def experiment_configuration(args) -> dict:
    excluded = {
        "output",
        "gpu_count",
        "cpu_threads_per_worker",
        "execution_batch_size",
        "resume",
    }
    return {
        key: value
        for key, value in vars(args).items()
        if key not in excluded
    }


def parser() -> argparse.ArgumentParser:
    result = argparse.ArgumentParser(
        description=(
            "Sweep graph NPI training size and optimizer steps, requiring 100/100 "
            "closed-loop tests at every reported generalization size."
        )
    )
    result.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    result.add_argument(
        "--maximum-train-nodes",
        type=parse_int_list,
        default=[5, 10, 15, 20],
        help="comma-separated training-size maxima; one run is scheduled per value and seed",
    )
    result.add_argument(
        "--seeds", type=parse_int_list, default=[1], help="comma-separated training seeds"
    )
    result.add_argument(
        "--checkpoint-steps",
        type=parse_int_list,
        default=[
            1_000,
            2_000,
            4_000,
            6_000,
            8_000,
            10_000,
            12_000,
            16_000,
            20_000,
            30_000,
            40_000,
            50_000,
            60_000,
        ],
    )
    result.add_argument(
        "--generalization-nodes",
        type=parse_int_list,
        default=[10, 20, 30, 40, 50, 75, 100, 125, 150, 200],
        help="ascending capacity candidates; evaluation stops at the first failed size",
    )
    result.add_argument("--tests-per-size", type=int, default=100)
    result.add_argument("--families", type=parse_str_list, default=list(DEFAULT_FAMILIES))
    result.add_argument("--evaluation-seed", type=int, default=48_271)
    result.add_argument("--evaluation-maximum-weight", type=int, default=100)
    result.add_argument("--training-examples-per-size", type=int, default=32)
    result.add_argument("--batch-size", type=int, default=256)
    result.add_argument(
        "--execution-batch-size",
        type=int,
        default=100,
        help="closed-loop graph interpreters advanced in each GPU batch",
    )
    result.add_argument("--learning-rate", type=float, default=3e-4)
    result.add_argument("--gpu-count", type=int, default=1)
    result.add_argument("--cpu-threads-per-worker", type=int, default=2)
    result.add_argument("--resume", action=argparse.BooleanOptionalAction, default=True)
    return result


def main() -> None:
    args = parser().parse_args()
    available_gpus = torch.cuda.device_count()
    validate_args(args, available_gpus)
    args.output.mkdir(parents=True, exist_ok=True)
    configuration_path = args.output / "configuration.json"
    configuration = experiment_configuration(args)
    if configuration_path.exists():
        previous_configuration = json.loads(configuration_path.read_text())
        if previous_configuration != configuration:
            changed = {
                key
                for key in set(previous_configuration) | set(configuration)
                if previous_configuration.get(key) != configuration.get(key)
            }
            extends_checkpoints = (
                changed == {"checkpoint_steps"}
                and set(previous_configuration["checkpoint_steps"]).issubset(
                    configuration["checkpoint_steps"]
                )
            )
            if extends_checkpoints:
                atomic_json_write(configuration, configuration_path)
            else:
                raise ValueError(
                    f"{configuration_path} describes a different sweep; use a new "
                    "--output directory to avoid mixing results"
                )
    else:
        atomic_json_write(configuration, configuration_path)

    jobs = [
        (maximum_train_nodes, seed)
        for maximum_train_nodes in args.maximum_train_nodes
        for seed in args.seeds
    ]
    worker_count = min(args.gpu_count, len(jobs))
    context = mp.get_context("spawn")
    job_queue = context.Queue()
    result_queue = context.Queue()
    for job in jobs:
        job_queue.put(job)
    for _ in range(worker_count):
        job_queue.put(None)

    args_dict = vars(args).copy()
    workers = [
        context.Process(
            target=gpu_worker,
            args=(gpu_id, job_queue, result_queue, args_dict),
        )
        for gpu_id in range(worker_count)
    ]
    for worker in workers:
        worker.start()

    errors = []
    completed = 0
    while completed < len(jobs):
        try:
            kind, payload = result_queue.get(timeout=5)
        except queue.Empty:
            if not any(worker.is_alive() for worker in workers):
                break
            continue
        if kind == "result":
            completed += 1
        else:
            errors.append(payload)
            if kind == "error":
                completed += 1
    for worker in workers:
        worker.join()

    if errors:
        for error in errors:
            print(error["traceback"], flush=True)
        raise RuntimeError(f"{len(errors)} sweep worker(s) failed")
    if completed != len(jobs):
        raise RuntimeError(f"only {completed}/{len(jobs)} sweep jobs completed")

    rows = collect_existing_results(args.output)
    write_summary(rows, args.output, args)
    plot_results(rows, args.output, args.tests_per_size)
    print(f"wrote {args.output / 'generalization_vs_steps.png'}")


if __name__ == "__main__":
    main()
