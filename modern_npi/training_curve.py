import argparse
import csv
import json
from pathlib import Path
import random
import time

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch
from matplotlib.ticker import FuncFormatter

from modern_npi.experiment import (
    Trainer,
    choose_device,
    make_data,
    teacher_forced_accuracy,
)
from modern_npi.model import NeuralProgrammerInterpreter


def stable_lr_multiplier(
    step: int,
    *,
    warmup_steps: int = 500,
    decay_start: int = 8_000,
    half_life: int = 5_000,
    minimum_ratio: float = 1.0 / 300.0,
) -> float:
    if step < warmup_steps:
        return max((step + 1) / warmup_steps, 1.0 / warmup_steps)
    if step <= decay_start:
        return 1.0
    decayed = 0.5 ** ((step - decay_start) / half_life)
    return max(decayed, minimum_ratio)


def write_artifacts(history: list[dict], output: Path, metadata: dict) -> None:
    output.mkdir(parents=True, exist_ok=True)
    json_path = output / "training_curve.json"
    csv_path = output / "training_curve.csv"
    json_path.write_text(json.dumps({**metadata, "history": history}, indent=2) + "\n")
    with csv_path.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=history[0].keys())
        writer.writeheader()
        writer.writerows(history)


def plot_curve(history: list[dict], output: Path) -> None:
    steps = [entry["optimizer_step"] for entry in history]
    train_accuracy = [entry["train_decision_accuracy"] for entry in history]
    validation_accuracy = [entry["validation_decision_accuracy"] for entry in history]
    best_index = max(range(len(history)), key=lambda index: validation_accuracy[index])

    plt.rcParams.update(
        {
            "font.family": "DejaVu Sans",
            "font.size": 10,
            "axes.edgecolor": "#343A40",
            "axes.linewidth": 0.8,
            "text.color": "#202428",
        }
    )
    figure, axis = plt.subplots(figsize=(9.5, 5.5))
    figure.patch.set_facecolor("white")
    axis.plot(
        steps,
        train_accuracy,
        color="#00798C",
        linewidth=2,
        marker="o",
        markersize=3.5,
        label="Training traces",
    )
    axis.plot(
        steps,
        validation_accuracy,
        color="#D1495B",
        linewidth=2,
        marker="o",
        markersize=3.5,
        label="Held-out traces",
    )
    axis.scatter(
        steps[best_index],
        validation_accuracy[best_index],
        s=75,
        color="#2A9D6F",
        edgecolor="white",
        linewidth=1,
        zorder=5,
    )
    axis.annotate(
        f"best held-out: {validation_accuracy[best_index]:.2%}\nstep {steps[best_index]:,}",
        (steps[best_index], validation_accuracy[best_index]),
        xytext=(-12, -34),
        textcoords="offset points",
        ha="right",
        va="top",
        fontsize=9,
        color="#237A58",
    )
    figure.suptitle(
        "20-Digit NPI Training Curve",
        x=0.10,
        y=0.965,
        ha="left",
        fontsize=15,
        fontweight="bold",
    )
    figure.text(
        0.10,
        0.91,
        "640 training additions | 32 per length from 1 to 20 | seed 1 | batch size 256",
        ha="left",
        fontsize=9.5,
        color="#5B6268",
    )
    axis.set_xlabel("Optimizer step")
    axis.set_ylabel("Exact program-decision accuracy")
    axis.set_xlim(0, max(steps) * 1.02)
    axis.set_ylim(0.35, 1.015)
    axis.yaxis.set_major_formatter(FuncFormatter(lambda value, _: f"{value:.0%}"))
    axis.xaxis.set_major_formatter(FuncFormatter(lambda value, _: f"{int(value):,}"))
    axis.grid(color="#DDE2E5", linewidth=0.8)
    axis.spines[["top", "right"]].set_visible(False)
    handles, labels = axis.get_legend_handles_labels()
    learning_rates = [entry.get("learning_rate") for entry in history]
    if all(rate is not None for rate in learning_rates) and len(set(learning_rates)) > 1:
        learning_rate_axis = axis.twinx()
        learning_rate_axis.plot(
            steps,
            learning_rates,
            color="#7A7F83",
            linewidth=1.4,
            linestyle="--",
            alpha=0.8,
            label="Learning rate",
        )
        learning_rate_axis.set_yscale("log")
        learning_rate_axis.set_ylabel("Learning rate", color="#5B6268")
        learning_rate_axis.tick_params(axis="y", colors="#5B6268")
        learning_rate_axis.spines["top"].set_visible(False)
        learning_rate_axis.spines["right"].set_color("#7A7F83")
        lr_handles, lr_labels = learning_rate_axis.get_legend_handles_labels()
        handles += lr_handles
        labels += lr_labels
    axis.legend(handles, labels, frameon=False, loc="lower right")
    figure.text(
        0.08,
        0.02,
        "Accuracy is evaluated with frozen epoch-end weights on every supervised decision. "
        "It is not closed-loop whole-addition accuracy.",
        fontsize=8.5,
        color="#5B6268",
    )
    figure.subplots_adjust(left=0.10, right=0.97, top=0.84, bottom=0.16)
    figure.savefig(output / "training_curve.png", dpi=220, facecolor="white")
    figure.savefig(output / "training_curve.pdf", facecolor="white")
    plt.close(figure)


def run(args: argparse.Namespace) -> None:
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    device = choose_device(args.device)
    if device.type == "cuda":
        torch.cuda.manual_seed_all(args.seed)

    training_data = make_data(32, 20, args.seed)
    validation_data = make_data(8, 20, args.seed + 1)
    model = NeuralProgrammerInterpreter().to(device)
    trainer = Trainer(model, args.learning_rate, device)
    scheduler = None
    if args.scheduler == "stable":
        scheduler = torch.optim.lr_scheduler.LambdaLR(
            trainer.optimizer,
            lr_lambda=stable_lr_multiplier,
        )
    history = []
    optimizer_step = 0
    best_validation = -1.0
    started = time.time()

    print(f"device: {device}")
    print(
        f"training invocations: {training_data.size}; "
        f"validation invocations: {validation_data.size}"
    )
    for epoch in range(1, args.epochs + 1):
        model.train()
        loss_sum = 0.0
        batch_count = 0
        for batch in training_data.batches(
            args.batch_size,
            shuffle=True,
            seed=args.seed + epoch,
        ):
            loss, _, _ = trainer.train_batch(batch)
            if scheduler is not None:
                scheduler.step()
            loss_sum += loss
            batch_count += 1
            optimizer_step += 1

        training_accuracy = teacher_forced_accuracy(
            model, training_data, args.batch_size, device
        )
        validation_accuracy = teacher_forced_accuracy(
            model, validation_data, args.batch_size, device
        )
        entry = {
            "epoch": epoch,
            "optimizer_step": optimizer_step,
            "mean_training_loss": loss_sum / batch_count,
            "learning_rate": trainer.optimizer.param_groups[0]["lr"],
            "train_decision_accuracy": training_accuracy,
            "validation_decision_accuracy": validation_accuracy,
            "elapsed_seconds": time.time() - started,
        }
        history.append(entry)
        print(
            f"epoch {epoch:02d} step {optimizer_step:5d}: "
            f"loss={entry['mean_training_loss']:.4f} "
            f"train={training_accuracy:.6f} validation={validation_accuracy:.6f}"
        )
        metadata = {
            "torch": torch.__version__,
            "device": str(device),
            "seed": args.seed,
            "maximum_training_digits": 20,
            "training_examples": 640,
            "training_invocations": training_data.size,
            "validation_invocations": validation_data.size,
            "batch_size": args.batch_size,
            "base_learning_rate": args.learning_rate,
            "scheduler": args.scheduler,
            "scheduler_parameters": {
                "warmup_steps": 500,
                "decay_start": 8000,
                "half_life": 5000,
                "minimum_learning_rate": args.learning_rate / 300,
            }
            if args.scheduler == "stable"
            else None,
        }
        write_artifacts(history, args.output, metadata)
        if validation_accuracy >= best_validation:
            best_validation = validation_accuracy
            torch.save({"model_state": model.state_dict(), **metadata}, args.output / "best.pt")

    plot_curve(history, args.output)
    print(f"artifacts: {args.output}")


def parser() -> argparse.ArgumentParser:
    result = argparse.ArgumentParser(description="Train and plot the 20-digit NPI learning curve.")
    result.add_argument("--output", type=Path, default=Path("artifacts/training_curve_20"))
    result.add_argument("--epochs", type=int, default=60)
    result.add_argument("--batch-size", type=int, default=256)
    result.add_argument("--learning-rate", type=float, default=3e-4)
    result.add_argument("--scheduler", choices=("constant", "stable"), default="constant")
    result.add_argument("--seed", type=int, default=1)
    result.add_argument("--device", default="auto")
    return result


def main() -> None:
    run(parser().parse_args())


if __name__ == "__main__":
    main()
