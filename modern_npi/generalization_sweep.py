import argparse
from dataclasses import asdict
import json
from pathlib import Path

import torch

from modern_npi.experiment import (
    choose_device,
    evaluate_lengths,
    make_data,
    teacher_forced_accuracy,
    train,
)


TRAIN_MAXIMUMS = (1, 2, 5, 10, 20)
EVALUATION_LENGTHS = (1, 2, 5, 10, 20, 50, 100, 500, 1000, 3000, 10000)
TOTAL_TRAINING_EXAMPLES = 640
TOTAL_VALIDATION_EXAMPLES = 160


def examples_for_length(length: int) -> int:
    if length <= 20:
        return 100
    if length <= 100:
        return 32
    return 2


def training_arguments(
    maximum_length: int,
    output_directory: Path,
    args: argparse.Namespace,
) -> argparse.Namespace:
    return argparse.Namespace(
        seed=args.seed,
        examples_per_length=TOTAL_TRAINING_EXAMPLES // maximum_length,
        validation_examples_per_length=TOTAL_VALIDATION_EXAMPLES // maximum_length,
        maximum_train_length=maximum_length,
        epochs=args.epochs,
        batch_size=args.batch_size,
        learning_rate=args.learning_rate,
        checkpoint=output_directory / f"train_max_{maximum_length}.pt",
    )


def measured_frontier(evaluations: list[dict]) -> tuple[int | None, int | None]:
    frontier = None
    first_failure = None
    for evaluation in evaluations:
        if evaluation["accuracy"] < 1.0:
            first_failure = evaluation["length"]
            break
        frontier = evaluation["length"]
    return frontier, first_failure


def run(args: argparse.Namespace) -> None:
    device = choose_device(args.device)
    args.output.mkdir(parents=True, exist_ok=True)
    models = []

    for maximum_length in TRAIN_MAXIMUMS:
        print(f"\n=== maximum training length: {maximum_length} ===")
        train_args = training_arguments(maximum_length, args.output, args)
        model = train(train_args, device)
        training_data = make_data(
            train_args.examples_per_length, maximum_length, args.seed
        )
        validation_data = make_data(
            train_args.validation_examples_per_length, maximum_length, args.seed + 1
        )
        training_accuracy = teacher_forced_accuracy(
            model, training_data, args.batch_size, device
        )
        validation_accuracy = teacher_forced_accuracy(
            model, validation_data, args.batch_size, device
        )

        evaluations = []
        for length in EVALUATION_LENGTHS:
            result = evaluate_lengths(
                model,
                [length],
                examples_for_length(length),
                args.seed + 1000 * maximum_length + length,
                device,
            )[0]
            evaluations.append(asdict(result))

        frontier, first_failure = measured_frontier(evaluations)
        models.append(
            {
                "maximum_training_digits": maximum_length,
                "training_examples": TOTAL_TRAINING_EXAMPLES,
                "training_examples_per_length": train_args.examples_per_length,
                "training_invocations": training_data.size,
                "training_decision_accuracy": training_accuracy,
                "validation_decision_accuracy": validation_accuracy,
                "maximum_tested_generalization_digits": frontier,
                "first_failing_test_digits": first_failure,
                "checkpoint": str(train_args.checkpoint),
                "evaluations": evaluations,
            }
        )
        del model
        if device.type == "cuda":
            torch.cuda.empty_cache()

    payload = {
        "torch": torch.__version__,
        "device": str(device),
        "seed": args.seed,
        "training_maximums": list(TRAIN_MAXIMUMS),
        "evaluation_lengths": list(EVALUATION_LENGTHS),
        "total_training_examples_per_model": TOTAL_TRAINING_EXAMPLES,
        "training_examples_per_length": {
            str(length): TOTAL_TRAINING_EXAMPLES // length
            for length in TRAIN_MAXIMUMS
        },
        "evaluation_examples": {
            "1_to_20_digits": 100,
            "50_and_100_digits": 32,
            "500_to_10000_digits": 2,
        },
        "frontier_definition": (
            "Largest tested length with 100% whole-sequence accuracy before the first tested "
            "length containing any failure. A value of 10000 is a lower bound, not a proven maximum."
        ),
        "boundary_feature": True,
        "models": models,
    }
    report = args.output / "generalization_sweep.json"
    report.write_text(json.dumps(payload, indent=2) + "\n")
    print(f"\nreport: {report}")


def parser() -> argparse.ArgumentParser:
    result = argparse.ArgumentParser(
        description="Sweep maximum training length against measured addition generalization."
    )
    result.add_argument(
        "--output", type=Path, default=Path("artifacts/generalization_sweep_fixed")
    )
    result.add_argument("--epochs", type=int, default=60)
    result.add_argument("--batch-size", type=int, default=256)
    result.add_argument("--learning-rate", type=float, default=3e-4)
    result.add_argument("--seed", type=int, default=1)
    result.add_argument("--device", default="auto")
    return result


def main() -> None:
    run(parser().parse_args())


if __name__ == "__main__":
    main()
