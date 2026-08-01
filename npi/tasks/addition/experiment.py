import argparse
import json
import random
import time
from dataclasses import asdict, dataclass
from pathlib import Path

import tensorflow as tf

from npi.core.checkpoint import load_model
from npi.core.experiment import train_epochs
from npi.core.hardware import configure_tensorflow
from npi.core.runtime import ExecutionFailure
from npi.tasks.addition.data import make_dataset
from npi.tasks.addition.environment import decimal_add
from npi.tasks.addition.runtime import execute_addition
from npi.tasks.addition.spec import SPEC
from npi.tasks.addition.traces import random_decimal

DEFAULT_CHECKPOINT = Path("artifacts/addition.weights.h5")
DEFAULT_RESULTS = Path("artifacts/addition_results.json")


@dataclass(frozen=True)
class LengthResult:
    length: int
    examples: int
    correct: int
    accuracy: float
    average_model_steps: float
    seconds: float
    first_failure: str | None


def parse_int_list(value):
    return [int(item) for item in value.split(",")]


def evaluate(model, lengths, examples, seed, use_xla):
    rng = random.Random(seed)
    results = []
    for length in lengths:
        count = examples if length <= 100 else min(examples, 2)
        correct = 0
        steps = 0
        failure = None
        started = time.time()
        for _ in range(count):
            first = random_decimal(length, rng)
            second = random_decimal(length, rng)
            expected = decimal_add(first, second)
            try:
                result = execute_addition(model, first, second, use_xla=use_xla)
                steps += result.model_steps
                if result.value == expected:
                    correct += 1
                elif failure is None:
                    failure = f"expected {expected[:80]}, got {result.value[:80]}"
            except ExecutionFailure as error:
                if failure is None:
                    failure = str(error)
        result = LengthResult(
            length,
            count,
            correct,
            correct / count,
            steps / count,
            time.time() - started,
            failure,
        )
        results.append(result)
        print(f"length={length}: {correct}/{count} accuracy={result.accuracy:.3f}")
    return results


def parser():
    result = argparse.ArgumentParser(description="TensorFlow/XLA addition NPI")
    result.add_argument("command", choices=("train", "evaluate", "reproduce"))
    result.add_argument("--checkpoint", type=Path, default=DEFAULT_CHECKPOINT)
    result.add_argument("--results", type=Path, default=DEFAULT_RESULTS)
    result.add_argument("--examples-per-length", type=int, default=32)
    result.add_argument("--maximum-train-length", type=int, default=20)
    result.add_argument("--epochs", type=int, default=80)
    result.add_argument("--batch-size", type=int, default=256)
    result.add_argument("--learning-rate", type=float, default=3e-4)
    result.add_argument("--weight-decay", type=float, default=1e-4)
    result.add_argument("--seed", type=int, default=1)
    result.add_argument("--device", default="auto")
    result.add_argument("--no-xla", action="store_true")
    result.add_argument(
        "--lengths", type=parse_int_list, default=[1, 5, 10, 20, 50, 100, 500, 1000]
    )
    result.add_argument("--evaluation-examples", type=int, default=16)
    return result


def main():
    args = parser().parse_args()
    selected = configure_tensorflow(args.device)
    print(f"tensorflow={tf.__version__} device={selected} xla={not args.no_xla}")
    if args.command in ("train", "reproduce"):
        train_data = make_dataset(
            args.examples_per_length, args.maximum_train_length, args.seed
        )
        validation_data = make_dataset(8, args.maximum_train_length, args.seed + 1)
        model, _ = train_epochs(
            SPEC,
            train_data,
            validation_data,
            args.checkpoint,
            epochs=args.epochs,
            batch_size=args.batch_size,
            learning_rate=args.learning_rate,
            weight_decay=args.weight_decay,
            seed=args.seed,
            use_xla=not args.no_xla,
        )
    else:
        model = load_model(SPEC, args.checkpoint)
    if args.command in ("evaluate", "reproduce"):
        results = evaluate(
            model,
            args.lengths,
            args.evaluation_examples,
            args.seed + 2,
            not args.no_xla,
        )
        payload = {
            "framework": "tensorflow",
            "tensorflow": tf.__version__,
            "xla": not args.no_xla,
            "seed": args.seed,
            "maximum_training_length": args.maximum_train_length,
            "results": [asdict(result) for result in results],
        }
        args.results.parent.mkdir(parents=True, exist_ok=True)
        args.results.write_text(json.dumps(payload, indent=2) + "\n")


if __name__ == "__main__":
    main()
