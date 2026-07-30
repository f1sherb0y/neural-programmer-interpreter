import argparse
import csv
import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.ticker import FuncFormatter, LogLocator


COLORS = {
    1: "#5B5F97",
    2: "#D1495B",
    5: "#EDAE49",
    10: "#00798C",
    20: "#2A9D6F",
}


def thousands(value, _position=None):
    return f"{int(value):,}"


def load_report(path: Path) -> dict:
    return json.loads(path.read_text())


def write_summary_csv(report: dict, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="") as handle:
        writer = csv.writer(handle)
        writer.writerow(
            [
                "maximum_training_digits",
                "training_examples",
                "training_decision_accuracy",
                "validation_decision_accuracy",
                "strict_generalization_frontier_digits",
                "first_failing_test_digits",
            ]
        )
        for model in report["models"]:
            writer.writerow(
                [
                    model["maximum_training_digits"],
                    model["training_examples"],
                    model["training_decision_accuracy"],
                    model["validation_decision_accuracy"],
                    model["maximum_tested_generalization_digits"] or 0,
                    model["first_failing_test_digits"] or "not observed",
                ]
            )


def plot(report: dict, output: Path) -> None:
    models = report["models"]
    training_digits = [model["maximum_training_digits"] for model in models]
    frontiers = [model["maximum_tested_generalization_digits"] or 0 for model in models]

    plt.rcParams.update(
        {
            "font.family": "DejaVu Sans",
            "font.size": 10,
            "axes.titlesize": 12,
            "axes.labelsize": 10,
            "axes.edgecolor": "#343A40",
            "axes.linewidth": 0.8,
            "xtick.color": "#343A40",
            "ytick.color": "#343A40",
            "text.color": "#202428",
        }
    )
    figure, (frontier_axis, accuracy_axis) = plt.subplots(
        1,
        2,
        figsize=(12, 5.6),
        gridspec_kw={"width_ratios": (0.9, 1.35)},
    )
    figure.patch.set_facecolor("#FFFFFF")

    frontier_axis.plot(
        training_digits,
        frontiers,
        color="#6C757D",
        linewidth=1.5,
        zorder=1,
    )
    for model, frontier in zip(models, frontiers, strict=True):
        maximum = model["maximum_training_digits"]
        color = COLORS[maximum]
        if frontier == 0:
            frontier_axis.scatter(
                maximum,
                0,
                marker="X",
                s=90,
                color=color,
                edgecolor="white",
                linewidth=0.8,
                zorder=3,
            )
        else:
            frontier_axis.scatter(
                maximum,
                frontier,
                marker="o",
                s=100,
                color=color,
                edgecolor="white",
                linewidth=1.0,
                zorder=3,
            )
            frontier_axis.annotate(
                f">= {frontier:,}",
                (maximum, frontier),
                xytext=(-8, -20),
                textcoords="offset points",
                ha="right",
                va="top",
                fontsize=10,
                fontweight="bold",
                color=color,
            )

    frontier_axis.set_title("Strict generalization frontier", loc="left", fontweight="bold")
    frontier_axis.set_xlabel("Maximum training length (digits)")
    frontier_axis.set_ylabel("Largest tested length at 100% accuracy")
    frontier_axis.set_xticks(training_digits)
    frontier_axis.set_ylim(-650, 10800)
    frontier_axis.yaxis.set_major_formatter(FuncFormatter(thousands))
    frontier_axis.grid(axis="y", color="#DDE2E5", linewidth=0.8)
    frontier_axis.spines[["top", "right"]].set_visible(False)
    frontier_axis.text(
        0.01,
        0.98,
        "X = failed the 1-digit test\naccuracy details are shown at right",
        transform=frontier_axis.transAxes,
        ha="left",
        va="top",
        fontsize=8,
        color="#5B6268",
    )

    for model in models:
        maximum = model["maximum_training_digits"]
        evaluations = model["evaluations"]
        accuracy_axis.plot(
            [entry["length"] for entry in evaluations],
            [entry["accuracy"] for entry in evaluations],
            marker="o",
            markersize=4.5,
            linewidth=1.8,
            color=COLORS[maximum],
            label=f"train max = {maximum}",
        )

    accuracy_axis.set_title("Closed-loop sequence accuracy", loc="left", fontweight="bold")
    accuracy_axis.set_xlabel("Evaluation length (digits, log scale)")
    accuracy_axis.set_ylabel("Exact sequence accuracy")
    accuracy_axis.set_xscale("log")
    accuracy_axis.set_xlim(0.8, 13000)
    accuracy_axis.set_ylim(-0.04, 1.04)
    accuracy_axis.xaxis.set_major_locator(LogLocator(base=10))
    accuracy_axis.xaxis.set_major_formatter(FuncFormatter(thousands))
    accuracy_axis.yaxis.set_major_formatter(FuncFormatter(lambda value, _: f"{value:.0%}"))
    accuracy_axis.grid(color="#DDE2E5", linewidth=0.8)
    accuracy_axis.spines[["top", "right"]].set_visible(False)
    accuracy_axis.legend(frameon=False, loc="center left", bbox_to_anchor=(1.0, 0.5))

    figure.suptitle(
        "NPI Addition: Maximum Training Length vs. Generalization",
        x=0.07,
        ha="left",
        fontsize=16,
        fontweight="bold",
    )
    figure.text(
        0.07,
        0.925,
        "640 training additions per model | seed 1 | exact recursive execution | tested through 10,000 digits",
        ha="left",
        fontsize=9.5,
        color="#5B6268",
    )
    figure.text(
        0.07,
        0.015,
        "Evaluation samples: 100 per length through 20 digits; 32 at 50 and 100; 2 at 500, 1,000, 3,000, and 10,000. "
        "The 10,000-digit point is a tested lower bound, not a proven maximum.",
        ha="left",
        fontsize=8,
        color="#5B6268",
    )
    figure.subplots_adjust(left=0.07, right=0.86, top=0.84, bottom=0.17, wspace=0.34)

    output.parent.mkdir(parents=True, exist_ok=True)
    figure.savefig(output, dpi=220, facecolor=figure.get_facecolor())
    figure.savefig(output.with_suffix(".pdf"), facecolor=figure.get_facecolor())
    plt.close(figure)


def parser() -> argparse.ArgumentParser:
    result = argparse.ArgumentParser(description="Plot the NPI addition generalization sweep.")
    result.add_argument(
        "--report",
        type=Path,
        default=Path("artifacts/generalization_sweep_fixed/generalization_sweep.json"),
    )
    result.add_argument(
        "--output",
        type=Path,
        default=Path("artifacts/generalization_sweep_fixed/train_vs_generalization.png"),
    )
    return result


def main() -> None:
    args = parser().parse_args()
    report = load_report(args.report)
    plot(report, args.output)
    write_summary_csv(report, args.output.with_suffix(".csv"))
    print(args.output)
    print(args.output.with_suffix(".pdf"))
    print(args.output.with_suffix(".csv"))


if __name__ == "__main__":
    main()
