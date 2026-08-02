import argparse
import gc
import json
import math
import time
from dataclasses import asdict, dataclass
from pathlib import Path

import tensorflow as tf

from npi.core.checkpoint import (
    create_training_checkpoint,
    restore_training_checkpoint,
)
from npi.core.experiment import set_seed
from npi.core.hardware import configure_tensorflow
from npi.core.model import NeuralProgrammerInterpreter, NPIConfig
from npi.core.runtime import execute_batch
from npi.core.trainer import Trainer
from npi.tasks.graph.experiment import oracle_distances, valid_parent_tree
from npi.tasks.graph.problems import generate_problem
from npi.tasks.graph.weight_video import WeightVideoWriter, parameter_metadata
from npi.tasks.graph_ram.codec import CODEC
from npi.tasks.graph_ram.data import make_sampled_dataset
from npi.tasks.graph_ram.environment import RamGraphEnvironment
from npi.tasks.graph_ram.spec import SPEC

DEFAULT_OUTPUT = Path("artifacts/graph_ram_dijkstra")


@dataclass(frozen=True)
class Evaluation:
    nodes: int
    examples: int
    correct_distances: int
    valid_parent_trees: int
    average_model_steps: float
    seconds: float
    first_failure: str | None


def write_json(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, indent=2) + "\n")


def evaluate(model, nodes, examples, seed, maximum_weight, batch_size, use_xla):
    import random

    problems = [
        generate_problem(
            nodes,
            random.Random(seed + nodes * 1_000_003 + index * 97_409),
            "random_connected",
            maximum_weight,
        )
        for index in range(examples)
    ]
    expected = [oracle_distances(problem) for problem in problems]
    correct = parents = steps = 0
    first_failure = None
    started = time.time()
    for start in range(0, examples, batch_size):
        batch = problems[start : start + batch_size]
        environments = [
            RamGraphEnvironment(problem.node_count, problem.edges, problem.source)
            for problem in batch
        ]
        limits = [
            1_000
            * (problem.node_count + len(problem.edges) + 1)
            * max(1, problem.node_count.bit_length())
            for problem in batch
        ]
        outcomes = execute_batch(
            model, SPEC, CODEC, environments, limits, use_xla=use_xla
        )
        for offset, (problem, outcome) in enumerate(zip(batch, outcomes, strict=True)):
            target = expected[start + offset]
            if outcome.failure:
                first_failure = first_failure or outcome.failure
                continue
            steps += outcome.stats.model_steps
            distances, parent_values = outcome.result
            if distances != target:
                first_failure = first_failure or "distance mismatch"
                continue
            correct += 1
            if valid_parent_tree(problem, target, parent_values):
                parents += 1
            else:
                first_failure = first_failure or "invalid parent tree"
    return Evaluation(
        nodes,
        examples,
        correct,
        parents,
        steps / examples,
        time.time() - started,
        first_failure,
    )


def parser():
    result = argparse.ArgumentParser(
        description="Train elementary-RAM hierarchical Dijkstra"
    )
    result.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    result.add_argument("--maximum-train-nodes", type=int, default=100)
    result.add_argument("--mean-train-nodes", type=float, default=30.0)
    result.add_argument("--training-examples", type=int, default=396)
    result.add_argument("--validation-examples", type=int, default=99)
    result.add_argument("--steps", type=int, default=120_000)
    result.add_argument("--checkpoint-interval", type=int, default=10_000)
    result.add_argument("--log-interval", type=int, default=1_000)
    result.add_argument("--batch-size", type=int, default=256)
    result.add_argument("--learning-rate", type=float, default=3e-4)
    result.add_argument("--weight-decay", type=float, default=0.0)
    result.add_argument("--l1-regularization", type=float, default=1e-10)
    result.add_argument("--l2-regularization", type=float, default=0.0)
    result.add_argument("--state-size", type=int, default=128)
    result.add_argument("--program-size", type=int, default=64)
    result.add_argument("--key-size", type=int, default=32)
    result.add_argument("--hidden-size", type=int, default=256)
    result.add_argument("--layers", type=int, default=2)
    result.add_argument("--seed", type=int, default=1)
    result.add_argument("--evaluation-nodes", type=int, default=30)
    result.add_argument("--evaluation-examples", type=int, default=100)
    result.add_argument("--evaluation-maximum-weight", type=int, default=100)
    result.add_argument("--execution-batch-size", type=int, default=100)
    result.add_argument("--weight-video", type=Path)
    result.add_argument("--video-frame-interval", type=int, default=50)
    result.add_argument("--video-frame-rate", type=int, default=30)
    result.add_argument("--video-magnitude-maximum", type=float, default=4.0)
    result.add_argument("--video-colormap", default="magma")
    result.add_argument("--cpu", action="store_true")
    result.add_argument("--no-xla", action="store_true")
    result.add_argument("--resume", action=argparse.BooleanOptionalAction, default=True)
    return result


def main():
    args = parser().parse_args()
    if args.video_frame_interval < 1 or args.video_frame_rate < 1:
        raise ValueError("Video frame interval and frame rate must be positive")
    if args.log_interval < 1 or args.checkpoint_interval < 1:
        raise ValueError("Log and checkpoint intervals must be positive")
    if args.video_magnitude_maximum <= 0:
        raise ValueError("Video magnitude maximum must be positive")
    configure_tensorflow(device="cpu" if args.cpu else "auto")
    set_seed(args.seed)
    training, training_problems = make_sampled_dataset(
        2,
        args.maximum_train_nodes,
        args.mean_train_nodes,
        args.training_examples,
        args.seed,
    )
    validation, validation_problems = make_sampled_dataset(
        2,
        args.maximum_train_nodes,
        args.mean_train_nodes,
        args.validation_examples,
        args.seed + 1,
    )
    model_config = NPIConfig(
        state_size=args.state_size,
        program_size=args.program_size,
        key_size=args.key_size,
        hidden_size=args.hidden_size,
        layers=args.layers,
    )
    model = NeuralProgrammerInterpreter(SPEC, model_config)
    model.build_for_task()
    trainer = Trainer(
        model,
        args.learning_rate,
        use_xla=not args.no_xla,
        weight_decay=args.weight_decay,
        l1_regularization=args.l1_regularization,
        l2_regularization=args.l2_regularization,
    )
    checkpoint, manager = create_training_checkpoint(
        model,
        trainer.optimizer,
        args.output / "resume",
        max_to_keep=1,
    )
    state_path = args.output / "resume.json"
    step = 0
    epoch = 1
    next_batch = 0
    if args.resume and manager.latest_checkpoint and state_path.exists():
        state = json.loads(state_path.read_text())
        step, epoch, next_batch = (
            state["optimizer_step"],
            state["epoch"],
            state["next_batch"],
        )
        restore_training_checkpoint(
            trainer.optimizer,
            checkpoint,
            manager,
            step,
        )
        print(f"resumed step={step} epoch={epoch} batch={next_batch}", flush=True)

    args.output.mkdir(parents=True, exist_ok=True)
    started = time.time()
    video_start_step = step
    video_writer = None
    if args.weight_video is not None:
        if step % args.video_frame_interval or args.steps % args.video_frame_interval:
            raise ValueError(
                "Video start and end steps must be divisible by the frame interval"
            )
        parameter_count = sum(
            math.prod(variable.shape) for variable in model.trainable_variables
        )
        video_writer = WeightVideoWriter(
            args.weight_video,
            parameter_count,
            frame_rate=args.video_frame_rate,
            magnitude_maximum=args.video_magnitude_maximum,
            colormap=args.video_colormap,
        )
        video_writer.add(model)
    interval_loss = 0.0
    interval_batches = 0
    try:
        while step < args.steps:
            for batch_index, batch in enumerate(
                training.batches(args.batch_size, shuffle=True, seed=args.seed + epoch)
            ):
                if batch_index < next_batch:
                    continue
                metrics = trainer.train_batch(batch)
                step += 1
                next_batch = batch_index + 1
                interval_loss += metrics.loss
                interval_batches += 1
                if video_writer is not None and step % args.video_frame_interval == 0:
                    video_writer.add(model)
                if step % args.checkpoint_interval == 0 or step == args.steps:
                    path = args.output / "checkpoints" / f"step_{step:08d}.weights.h5"
                    path.parent.mkdir(parents=True, exist_ok=True)
                    model.save_weights(path)
                    manager.save(checkpoint_number=step)
                    write_json(
                        state_path,
                        {
                            "optimizer_step": step,
                            "epoch": epoch,
                            "next_batch": next_batch,
                        },
                    )
                if step % args.log_interval == 0 or step == args.steps:
                    frames = (
                        f" frames={video_writer.frame_count}"
                        if video_writer is not None
                        else ""
                    )
                    mean_loss = interval_loss / max(interval_batches, 1)
                    print(
                        f"step={step} epoch={epoch} loss={mean_loss:.6f}{frames} "
                        f"seconds={time.time() - started:.1f}",
                        flush=True,
                    )
                    interval_loss = 0.0
                    interval_batches = 0
                if step >= args.steps:
                    break
            if step >= args.steps:
                break
            epoch += 1
            next_batch = 0
    finally:
        if video_writer is not None:
            video_writer.close()

    training_accuracy = trainer.accuracy(training, args.batch_size)
    validation_accuracy = trainer.accuracy(validation, args.batch_size)
    final_path = args.output / "final.weights.h5"
    model.save_weights(final_path)
    video_metadata = None
    if video_writer is not None:
        video_metadata = {
            "file": str(args.weight_video),
            "starting_optimizer_step": video_start_step,
            "ending_optimizer_step": step,
            "frame_interval_steps": args.video_frame_interval,
            "step_for_frame": (
                "starting_optimizer_step + frame_index * frame_interval_steps"
            ),
            "frame_rate": args.video_frame_rate,
            "frame_count": video_writer.frame_count,
            "duration_seconds": video_writer.frame_count / args.video_frame_rate,
            "width": video_writer.width,
            "height": video_writer.height,
            "parameter_count": sum(
                math.prod(variable.shape) for variable in model.trainable_variables
            ),
            "variables": parameter_metadata(model, video_writer.width),
            "color": {
                "value": "absolute parameter magnitude",
                "colormap": args.video_colormap,
                "normalization": ("log1p(abs(weight) / 1e-4) / log1p(maximum / 1e-4)"),
                "maximum": args.video_magnitude_maximum,
                "observed_maximum": video_writer.observed_maximum,
                "clipped_values_across_all_frames": video_writer.clipped_values,
            },
        }
        write_json(args.weight_video.with_suffix(".json"), video_metadata)
    metadata = {
        "framework": "tensorflow",
        "tensorflow": tf.__version__,
        "task": SPEC.name,
        "optimizer": type(trainer.optimizer).__name__,
        "learning_rate": args.learning_rate,
        "weight_decay": args.weight_decay,
        "l1_regularization": args.l1_regularization,
        "l2_regularization": args.l2_regularization,
        "model_config": asdict(model_config),
        "parameter_count": sum(
            math.prod(variable.shape) for variable in model.trainable_variables
        ),
        "minimum_train_nodes": 2,
        "maximum_train_nodes": args.maximum_train_nodes,
        "node_count_distribution": "balanced_bounded_maximum_entropy",
        "requested_mean_train_nodes": args.mean_train_nodes,
        "training_examples": args.training_examples,
        "validation_examples": args.validation_examples,
        "observed_training_mean_nodes": sum(
            problem.node_count for problem in training_problems
        )
        / len(training_problems),
        "observed_validation_mean_nodes": sum(
            problem.node_count for problem in validation_problems
        )
        / len(validation_problems),
        "training_node_count_histogram": {
            str(nodes): sum(
                problem.node_count == nodes for problem in training_problems
            )
            for nodes in range(2, args.maximum_train_nodes + 1)
        },
        "validation_node_count_histogram": {
            str(nodes): sum(
                problem.node_count == nodes for problem in validation_problems
            )
            for nodes in range(2, args.maximum_train_nodes + 1)
        },
        "training_invocations": training.size,
        "training_decisions": training.decisions,
        "validation_invocations": validation.size,
        "validation_decisions": validation.decisions,
        "optimizer_steps": step,
        "training_accuracy": training_accuracy,
        "validation_accuracy": validation_accuracy,
        "training_seconds": time.time() - started,
        "weight_video": video_metadata,
    }
    write_json(args.output / "training.json", metadata)
    print(
        f"training_accuracy={training_accuracy:.9f} "
        f"validation_accuracy={validation_accuracy:.9f}",
        flush=True,
    )
    del training, validation
    gc.collect()
    result = evaluate(
        model,
        args.evaluation_nodes,
        args.evaluation_examples,
        48271,
        args.evaluation_maximum_weight,
        args.execution_batch_size,
        not args.no_xla,
    )
    write_json(args.output / "evaluation.json", asdict(result))
    print(result, flush=True)


if __name__ == "__main__":
    main()
