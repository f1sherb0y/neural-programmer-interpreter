import argparse
import csv
import json
import multiprocessing as mp
import queue
import random
import time
import traceback
from dataclasses import asdict, dataclass
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import tensorflow as tf

from npi.core.experiment import set_seed
from npi.core.hardware import configure_worker_gpu
from npi.core.model import NeuralProgrammerInterpreter
from npi.core.runtime import execute_batch
from npi.core.trainer import Trainer
from npi.tasks.graph.codec import CODEC
from npi.tasks.graph.data import make_dataset
from npi.tasks.graph.environment import GraphEnvironment
from npi.tasks.graph.experiment import FAMILIES, oracle_distances, valid_parent_tree
from npi.tasks.graph.problems import generate_problem
from npi.tasks.graph.spec import SPEC

DEFAULT_OUTPUT = Path("artifacts/graph_generalization_sweep_tf")
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


def parse_int_list(value):
    values = [int(item) for item in value.split(",") if item]
    if not values:
        raise argparse.ArgumentTypeError("Expected comma-separated integers")
    return values


def parse_str_list(value):
    return [item for item in value.split(",") if item]


def run_directory(output, maximum_train_nodes, seed):
    return output / f"train_max_{maximum_train_nodes:04d}" / f"seed_{seed:04d}"


def model_path(directory, step):
    return directory / "checkpoints" / f"step_{step:08d}.weights.h5"


def evaluation_path(directory, step):
    return directory / "evaluations" / f"step_{step:08d}.json"


def write_json(path, payload):
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(payload, indent=2) + "\n")
    temporary.replace(path)


def make_problems(nodes, tests, seed, families, maximum_weight):
    order = [families[index % len(families)] for index in range(tests)]
    random.Random(seed + nodes * 1_000_003).shuffle(order)
    return [
        generate_problem(
            nodes,
            random.Random(seed + nodes * 1_000_003 + index * 97_409),
            family,
            maximum_weight,
        )
        for index, family in enumerate(order)
    ]


class CapacityEvaluator:
    def __init__(self, args):
        self.args = args
        self.cache = {}

    def problems(self, nodes):
        if nodes not in self.cache:
            problems = make_problems(
                nodes,
                self.args.tests_per_size,
                self.args.evaluation_seed,
                tuple(self.args.families),
                self.args.evaluation_maximum_weight,
            )
            self.cache[nodes] = (
                problems,
                [oracle_distances(problem) for problem in problems],
            )
        return self.cache[nodes]

    def candidate(self, model, nodes):
        problems, expected = self.problems(nodes)
        correct = parents = steps = completed = 0
        first_failure = None
        started = time.time()
        for start in range(0, len(problems), self.args.execution_batch_size):
            batch = problems[start : start + self.args.execution_batch_size]
            environments = [
                GraphEnvironment(p.node_count, list(p.edges), p.source) for p in batch
            ]
            limits = [
                200 * (p.node_count * p.node_count + len(p.edges) + 1) for p in batch
            ]
            outcomes = execute_batch(
                model, SPEC, CODEC, environments, limits, use_xla=not self.args.no_xla
            )
            for offset, (problem, outcome) in enumerate(
                zip(batch, outcomes, strict=True)
            ):
                completed += 1
                target = expected[start + offset]
                if outcome.failure:
                    if first_failure is None:
                        first_failure = outcome.failure
                    continue
                steps += outcome.stats.model_steps
                distances, parent_values = outcome.result
                if distances != target:
                    if first_failure is None:
                        mismatch = [
                            index
                            for index, (actual, wanted) in enumerate(
                                zip(distances, target, strict=True)
                            )
                            if actual != wanted
                        ]
                        first_failure = f"distance mismatch at nodes {mismatch[:8]}"
                    continue
                correct += 1
                if valid_parent_tree(problem, target, parent_values):
                    parents += 1
                elif first_failure is None:
                    first_failure = "distance correct but parent tree invalid"
        passed = (
            correct == self.args.tests_per_size and parents == self.args.tests_per_size
        )
        return CandidateResult(
            nodes,
            self.args.tests_per_size,
            correct,
            parents,
            passed,
            steps / max(completed, 1),
            time.time() - started,
            first_failure,
        )

    def capacity(self, model, maximum_train_nodes, seed, step, gpu_id):
        results = []
        capacity = 0
        failure_nodes = None
        for nodes in self.args.generalization_nodes:
            result = self.candidate(model, nodes)
            results.append(result)
            suffix = (
                f" first_failure={result.first_failure}" if result.first_failure else ""
            )
            print(
                f"[gpu {gpu_id}] train_max={maximum_train_nodes} seed={seed} "
                f"step={step} nodes={nodes}: {result.correct_distances}/"
                f"{result.required_tests} seconds={result.seconds:.1f}{suffix}",
                flush=True,
            )
            if not result.passed:
                if failure_nodes is None:
                    failure_nodes = nodes
                if not self.args.continue_after_failure:
                    break
            elif failure_nodes is None:
                capacity = nodes
        return CapacityResult(
            maximum_train_nodes,
            seed,
            step,
            capacity,
            failure_nodes,
            failure_nodes is None,
            tuple(results),
        )


def train_checkpoints(maximum_train_nodes, seed, directory, args, gpu_id):
    set_seed(seed)
    data, _ = make_dataset(
        2, maximum_train_nodes, args.training_examples_per_size, seed
    )
    model = NeuralProgrammerInterpreter(SPEC)
    model.build_for_task()
    trainer = Trainer(
        model,
        args.learning_rate,
        use_xla=not args.no_xla,
        weight_decay=args.weight_decay,
        l1_regularization=args.l1_regularization,
        l2_regularization=args.l2_regularization,
    )
    checkpoint = tf.train.Checkpoint(model=model, optimizer=trainer.optimizer)
    manager = tf.train.CheckpointManager(
        checkpoint, str(directory / "resume"), max_to_keep=1
    )
    metadata_path = directory / "resume.json"
    optimizer_step = 0
    epoch = 1
    next_batch = 0
    if args.resume and manager.latest_checkpoint and metadata_path.exists():
        checkpoint.restore(manager.latest_checkpoint).expect_partial()
        metadata = json.loads(metadata_path.read_text())
        optimizer_step = metadata["optimizer_step"]
        epoch = metadata["epoch"]
        next_batch = metadata["next_batch"]
        print(
            f"[gpu {gpu_id}] resumed train_max={maximum_train_nodes} seed={seed} "
            f"at step={optimizer_step}",
            flush=True,
        )
    targets = sorted(set(args.checkpoint_steps))
    target_set = set(targets)
    while optimizer_step < targets[-1]:
        consumed = False
        for batch_index, batch in enumerate(
            data.batches(args.batch_size, shuffle=True, seed=seed + epoch)
        ):
            if batch_index < next_batch:
                continue
            consumed = True
            trainer.train_batch(batch)
            optimizer_step += 1
            next_batch = batch_index + 1
            if optimizer_step in target_set:
                path = model_path(directory, optimizer_step)
                path.parent.mkdir(parents=True, exist_ok=True)
                model.save_weights(path)
                manager.save(checkpoint_number=optimizer_step)
                write_json(
                    metadata_path,
                    {
                        "optimizer_step": optimizer_step,
                        "epoch": epoch,
                        "next_batch": next_batch,
                    },
                )
                print(
                    f"[gpu {gpu_id}] train_max={maximum_train_nodes} seed={seed}: "
                    f"saved step={optimizer_step}",
                    flush=True,
                )
            if optimizer_step >= targets[-1]:
                break
        if optimizer_step >= targets[-1]:
            break
        if not consumed and next_batch == 0:
            raise RuntimeError("Training dataset produced no batches")
        epoch += 1
        next_batch = 0
    return [model_path(directory, step) for step in targets]


def run_job(maximum_train_nodes, seed, args, gpu_id):
    directory = run_directory(args.output, maximum_train_nodes, seed)
    checkpoints = train_checkpoints(maximum_train_nodes, seed, directory, args, gpu_id)
    evaluator = CapacityEvaluator(args)
    model = NeuralProgrammerInterpreter(SPEC)
    model.build_for_task()
    for step, path in zip(args.checkpoint_steps, checkpoints, strict=True):
        if step not in args.evaluation_steps:
            continue
        output = evaluation_path(directory, step)
        if args.resume and output.exists():
            print(f"[gpu {gpu_id}] reused {output}", flush=True)
            continue
        model.load_weights(path)
        result = evaluator.capacity(model, maximum_train_nodes, seed, step, gpu_id)
        write_json(output, asdict(result))


def worker(gpu_id, jobs, results, args_dict):
    try:
        args = argparse.Namespace(**args_dict)
        configure_worker_gpu(gpu_id)
        while True:
            job = jobs.get()
            if job is None:
                return
            maximum_train_nodes, seed = job
            try:
                run_job(maximum_train_nodes, seed, args, gpu_id)
                results.put(("ok", job))
            except Exception:  # noqa: BLE001 - process boundary must report any worker failure
                results.put(
                    ("error", {"job": job, "traceback": traceback.format_exc()})
                )
    except Exception:  # noqa: BLE001 - process boundary must report initialization failures
        results.put(("error", {"job": None, "traceback": traceback.format_exc()}))


def collect(output):
    return [
        json.loads(path.read_text())
        for path in sorted(output.glob("train_max_*/seed_*/evaluations/step_*.json"))
    ]


def summarize(rows, output, args):
    rows.sort(
        key=lambda row: (row["maximum_train_nodes"], row["seed"], row["training_step"])
    )
    write_json(
        output / "sweep_results.json",
        {
            "framework": "tensorflow",
            "tensorflow": tf.__version__,
            "xla": not args.no_xla,
            "criterion": f"{args.tests_per_size}/{args.tests_per_size} exact tests",
            "results": rows,
        },
    )
    with (output / "sweep_results.csv").open("w", newline="") as handle:
        fields = (
            "maximum_train_nodes",
            "seed",
            "training_step",
            "maximum_generalization_nodes",
            "first_failed_nodes",
            "reached_search_limit",
        )
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        for row in rows:
            writer.writerow({field: row[field] for field in fields})


def plot(rows, output, tests):
    grouped = {}
    for row in rows:
        grouped.setdefault(row["maximum_train_nodes"], {}).setdefault(
            row["training_step"], []
        ).append(row["maximum_generalization_nodes"])
    figure, axis = plt.subplots(figsize=(10.5, 6.2))
    for index, maximum in enumerate(sorted(grouped)):
        steps = sorted(grouped[maximum])
        medians = [np.median(grouped[maximum][step]) for step in steps]
        lows = [min(grouped[maximum][step]) for step in steps]
        highs = [max(grouped[maximum][step]) for step in steps]
        color = COLORS[index % len(COLORS)]
        axis.plot(
            steps,
            medians,
            marker="o",
            linewidth=2,
            color=color,
            label=f"train max {maximum}",
        )
        axis.fill_between(steps, lows, highs, color=color, alpha=0.14)
    axis.set_title(
        "TensorFlow/XLA NPI Generalization", loc="left", fontweight="bold", fontsize=15
    )
    axis.set_xlabel("Optimizer step")
    axis.set_ylabel("Maximum verified nodes")
    axis.grid(color="#DDE2E5")
    axis.spines[["top", "right"]].set_visible(False)
    axis.legend(frameon=False, ncol=2)
    figure.text(
        0.1,
        0.02,
        f"Every point requires {tests}/{tests} exact closed-loop tests.",
        color="#5B6268",
    )
    figure.subplots_adjust(left=0.1, right=0.97, top=0.9, bottom=0.13)
    figure.savefig(output / "generalization_vs_steps.png", dpi=220)
    figure.savefig(output / "generalization_vs_steps.pdf")
    plt.close(figure)


def configuration(args):
    excluded = {"output", "gpu_count", "resume", "execution_batch_size"}
    return {key: value for key, value in vars(args).items() if key not in excluded}


def parser():
    result = argparse.ArgumentParser(
        description="Multi-GPU TensorFlow/XLA graph NPI sweep"
    )
    result.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    result.add_argument(
        "--maximum-train-nodes", type=parse_int_list, default=[5, 10, 15, 20]
    )
    result.add_argument("--seeds", type=parse_int_list, default=[1, 2])
    result.add_argument(
        "--checkpoint-steps",
        type=parse_int_list,
        default=[1000, 2000, 4000, 8000, 12000, 20000, 30000, 40000, 50000, 60000],
    )
    result.add_argument(
        "--evaluation-steps",
        type=parse_int_list,
        default=None,
        help="checkpoint steps to evaluate; defaults to every checkpoint",
    )
    result.add_argument(
        "--generalization-nodes",
        type=parse_int_list,
        default=[10, 20, 30, 40, 50, 75, 100, 125, 150, 200],
    )
    result.add_argument("--tests-per-size", type=int, default=100)
    result.add_argument(
        "--continue-after-failure",
        action="store_true",
        help="evaluate every requested node count even after an earlier failure",
    )
    result.add_argument("--families", type=parse_str_list, default=list(FAMILIES))
    result.add_argument("--evaluation-seed", type=int, default=48271)
    result.add_argument("--evaluation-maximum-weight", type=int, default=100)
    result.add_argument("--training-examples-per-size", type=int, default=32)
    result.add_argument("--batch-size", type=int, default=256)
    result.add_argument("--execution-batch-size", type=int, default=100)
    result.add_argument("--learning-rate", type=float, default=3e-4)
    result.add_argument("--weight-decay", type=float, default=0.0)
    result.add_argument("--l1-regularization", type=float, default=1e-10)
    result.add_argument("--l2-regularization", type=float, default=0.0)
    result.add_argument("--gpu-count", type=int, default=1)
    result.add_argument("--no-xla", action="store_true")
    result.add_argument("--resume", action=argparse.BooleanOptionalAction, default=True)
    return result


def main():
    args = parser().parse_args()
    args.maximum_train_nodes = sorted(set(args.maximum_train_nodes))
    args.seeds = sorted(set(args.seeds))
    args.checkpoint_steps = sorted(set(args.checkpoint_steps))
    if args.evaluation_steps is None:
        args.evaluation_steps = list(args.checkpoint_steps)
    else:
        args.evaluation_steps = sorted(set(args.evaluation_steps))
    if not set(args.evaluation_steps).issubset(args.checkpoint_steps):
        raise ValueError("Evaluation steps must also be checkpoint steps")
    args.generalization_nodes = sorted(set(args.generalization_nodes))
    if args.tests_per_size != 100:
        raise ValueError("The capacity criterion requires exactly 100 tests")
    available = len(tf.config.list_physical_devices("GPU"))
    if not 1 <= args.gpu_count <= available:
        raise ValueError(
            f"Requested {args.gpu_count} GPUs; TensorFlow sees {available}"
        )
    args.output.mkdir(parents=True, exist_ok=True)
    config_path = args.output / "configuration.json"
    current = configuration(args)
    if config_path.exists() and json.loads(config_path.read_text()) != current:
        raise ValueError("Output directory contains a different sweep configuration")
    write_json(config_path, current)

    job_values = [
        (maximum, seed) for maximum in args.maximum_train_nodes for seed in args.seeds
    ]
    worker_count = min(args.gpu_count, len(job_values))
    context = mp.get_context("spawn")
    jobs = context.Queue()
    results = context.Queue()
    for job in job_values:
        jobs.put(job)
    for _ in range(worker_count):
        jobs.put(None)
    processes = [
        context.Process(target=worker, args=(gpu_id, jobs, results, vars(args).copy()))
        for gpu_id in range(worker_count)
    ]
    for process in processes:
        process.start()
    errors = []
    completed = 0
    while completed < len(job_values):
        try:
            kind, payload = results.get(timeout=5)
        except queue.Empty:
            if not any(process.is_alive() for process in processes):
                break
            continue
        completed += 1
        if kind == "error":
            errors.append(payload)
    for process in processes:
        process.join()
    if errors:
        for error in errors:
            print(error["traceback"])
        raise RuntimeError(f"{len(errors)} sweep workers failed")
    if completed != len(job_values):
        raise RuntimeError(f"Only {completed}/{len(job_values)} jobs completed")
    rows = collect(args.output)
    summarize(rows, args.output, args)
    plot(rows, args.output, args.tests_per_size)


if __name__ == "__main__":
    main()
