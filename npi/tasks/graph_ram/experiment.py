import argparse
import gc
import json
import time
from dataclasses import asdict, dataclass
from pathlib import Path

import tensorflow as tf

from npi.core.experiment import set_seed
from npi.core.hardware import configure_tensorflow
from npi.core.model import NeuralProgrammerInterpreter
from npi.core.runtime import execute_batch
from npi.core.trainer import Trainer
from npi.tasks.graph.experiment import oracle_distances, valid_parent_tree
from npi.tasks.graph.problems import generate_problem
from npi.tasks.graph_ram.codec import CODEC
from npi.tasks.graph_ram.data import make_dataset
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
    result.add_argument("--maximum-train-nodes", type=int, default=30)
    result.add_argument("--training-examples-per-size", type=int, default=4)
    result.add_argument("--validation-examples-per-size", type=int, default=1)
    result.add_argument("--steps", type=int, default=120_000)
    result.add_argument("--checkpoint-interval", type=int, default=10_000)
    result.add_argument("--batch-size", type=int, default=256)
    result.add_argument("--learning-rate", type=float, default=3e-4)
    result.add_argument("--weight-decay", type=float, default=0.0)
    result.add_argument("--l1-regularization", type=float, default=1e-10)
    result.add_argument("--seed", type=int, default=1)
    result.add_argument("--evaluation-nodes", type=int, default=30)
    result.add_argument("--evaluation-examples", type=int, default=100)
    result.add_argument("--evaluation-maximum-weight", type=int, default=100)
    result.add_argument("--execution-batch-size", type=int, default=100)
    result.add_argument("--cpu", action="store_true")
    result.add_argument("--no-xla", action="store_true")
    result.add_argument("--resume", action=argparse.BooleanOptionalAction, default=True)
    return result


def main():
    args = parser().parse_args()
    configure_tensorflow(device="cpu" if args.cpu else "auto")
    set_seed(args.seed)
    training, _ = make_dataset(
        2,
        args.maximum_train_nodes,
        args.training_examples_per_size,
        args.seed,
    )
    validation, _ = make_dataset(
        2,
        args.maximum_train_nodes,
        args.validation_examples_per_size,
        args.seed + 1,
    )
    model = NeuralProgrammerInterpreter(SPEC)
    model.build_for_task()
    trainer = Trainer(
        model,
        args.learning_rate,
        use_xla=not args.no_xla,
        weight_decay=args.weight_decay,
        l1_regularization=args.l1_regularization,
    )
    checkpoint = tf.train.Checkpoint(model=model, optimizer=trainer.optimizer)
    manager = tf.train.CheckpointManager(
        checkpoint, str(args.output / "resume"), max_to_keep=1
    )
    state_path = args.output / "resume.json"
    step = 0
    epoch = 1
    next_batch = 0
    if args.resume and manager.latest_checkpoint and state_path.exists():
        checkpoint.restore(manager.latest_checkpoint).expect_partial()
        state = json.loads(state_path.read_text())
        step, epoch, next_batch = (
            state["optimizer_step"],
            state["epoch"],
            state["next_batch"],
        )
        print(f"resumed step={step} epoch={epoch} batch={next_batch}", flush=True)

    args.output.mkdir(parents=True, exist_ok=True)
    started = time.time()
    while step < args.steps:
        for batch_index, batch in enumerate(
            training.batches(args.batch_size, shuffle=True, seed=args.seed + epoch)
        ):
            if batch_index < next_batch:
                continue
            trainer.train_batch(batch)
            step += 1
            next_batch = batch_index + 1
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
                print(
                    f"step={step} epoch={epoch} seconds={time.time() - started:.1f}",
                    flush=True,
                )
            if step >= args.steps:
                break
        if step >= args.steps:
            break
        epoch += 1
        next_batch = 0

    training_accuracy = trainer.accuracy(training, args.batch_size)
    validation_accuracy = trainer.accuracy(validation, args.batch_size)
    final_path = args.output / "final.weights.h5"
    model.save_weights(final_path)
    metadata = {
        "framework": "tensorflow",
        "tensorflow": tf.__version__,
        "task": SPEC.name,
        "optimizer": type(trainer.optimizer).__name__,
        "learning_rate": args.learning_rate,
        "weight_decay": args.weight_decay,
        "l1_regularization": args.l1_regularization,
        "maximum_train_nodes": args.maximum_train_nodes,
        "training_examples_per_size": args.training_examples_per_size,
        "training_invocations": training.size,
        "training_decisions": training.decisions,
        "validation_invocations": validation.size,
        "validation_decisions": validation.decisions,
        "optimizer_steps": step,
        "training_accuracy": training_accuracy,
        "validation_accuracy": validation_accuracy,
        "training_seconds": time.time() - started,
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
