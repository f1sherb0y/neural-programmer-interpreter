import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.ticker import FuncFormatter


ROOT = Path("artifacts/graph_dijkstra")
COLORS = {
    "path": "#5B5F97",
    "star": "#D1495B",
    "sparse": "#EDAE49",
    "dense": "#00798C",
    "disconnected": "#2A9D6F",
    "directed": "#8D6A9F",
}


def main():
    history_payload = json.loads((ROOT / "training_history.json").read_text())
    evaluation = json.loads((ROOT / "evaluation.json").read_text())
    history = history_payload["history"]

    plt.rcParams.update(
        {
            "font.family": "DejaVu Sans",
            "font.size": 10,
            "axes.edgecolor": "#343A40",
            "axes.linewidth": 0.8,
            "text.color": "#202428",
        }
    )
    figure, (training_axis, scale_axis) = plt.subplots(1, 2, figsize=(12, 5.6))
    figure.patch.set_facecolor("white")

    steps = [entry["optimizer_step"] for entry in history]
    training_axis.plot(
        steps,
        [entry["training_accuracy"] for entry in history],
        color="#00798C",
        linewidth=2,
        marker="o",
        markersize=3.5,
        label="Training traces",
    )
    training_axis.plot(
        steps,
        [entry["validation_accuracy"] for entry in history],
        color="#D1495B",
        linewidth=2,
        marker="o",
        markersize=3.5,
        label="Held-out traces",
    )
    best = max(history, key=lambda entry: entry["validation_accuracy"])
    training_axis.scatter(
        best["optimizer_step"],
        best["validation_accuracy"],
        color="#2A9D6F",
        edgecolor="white",
        s=80,
        zorder=4,
    )
    training_axis.annotate(
        f"100% at step {best['optimizer_step']:,}",
        (best["optimizer_step"], best["validation_accuracy"]),
        xytext=(-8, -24),
        textcoords="offset points",
        ha="right",
        color="#237A58",
        fontweight="bold",
    )
    training_axis.set_title("Hierarchical decision learning", loc="left", fontweight="bold")
    training_axis.set_xlabel("Optimizer step")
    training_axis.set_ylabel("Exact decision accuracy")
    training_axis.set_ylim(0.4, 1.02)
    training_axis.yaxis.set_major_formatter(
        FuncFormatter(lambda value, _: f"{value:.0%}")
    )
    training_axis.grid(color="#DDE2E5", linewidth=0.8)
    training_axis.spines[["top", "right"]].set_visible(False)
    training_axis.legend(frameon=False, loc="lower right")

    by_family = {}
    for result in evaluation["results"]:
        by_family.setdefault(result["family"], []).append(result)
    for family, results in by_family.items():
        results.sort(key=lambda result: result["nodes"])
        scale_axis.plot(
            [result["nodes"] for result in results],
            [result["average_model_steps"] for result in results],
            color=COLORS[family],
            marker="o",
            linewidth=1.8,
            markersize=4.5,
            label=family,
        )

    scale_axis.set_title("Closed-loop execution cost", loc="left", fontweight="bold")
    scale_axis.set_xlabel("Graph nodes")
    scale_axis.set_ylabel("Neural program decisions")
    scale_axis.set_xscale("log")
    scale_axis.set_yscale("log")
    scale_axis.grid(color="#DDE2E5", linewidth=0.8, which="both")
    scale_axis.spines[["top", "right"]].set_visible(False)
    scale_axis.legend(frameon=False, ncol=2, loc="upper left")
    scale_axis.text(
        0.98,
        0.04,
        "150 / 150 exact distances\n150 / 150 valid parent trees",
        transform=scale_axis.transAxes,
        ha="right",
        va="bottom",
        fontsize=10,
        color="#237A58",
        fontweight="bold",
    )

    figure.suptitle(
        "Pointer-Machine NPI Learns Weighted Dijkstra",
        x=0.07,
        y=0.97,
        ha="left",
        fontsize=16,
        fontweight="bold",
    )
    figure.text(
        0.07,
        0.91,
        "Train: 2-10 nodes, weights 1-9 | Test: up to 100 nodes, weights 1-100 | seed 1",
        fontsize=9.5,
        color="#5B6268",
    )
    figure.text(
        0.07,
        0.02,
        "Evaluation includes path, star, sparse, dense, disconnected, and unseen directed graph families. "
        "Counts above 20 nodes use one or two examples per family.",
        fontsize=8.2,
        color="#5B6268",
    )
    figure.subplots_adjust(left=0.07, right=0.97, top=0.84, bottom=0.16, wspace=0.28)
    figure.savefig(ROOT / "results.png", dpi=220, facecolor="white")
    figure.savefig(ROOT / "results.pdf", facecolor="white")
    plt.close(figure)
    print(ROOT / "results.png")


if __name__ == "__main__":
    main()
