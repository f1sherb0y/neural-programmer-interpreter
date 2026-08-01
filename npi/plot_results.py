import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

ROOT = Path("artifacts")


def main():
    addition = json.loads((ROOT / "addition.weights.json").read_text())
    graph = json.loads((ROOT / "graph_dijkstra" / "best.weights.json").read_text())
    evaluation = json.loads((ROOT / "graph_dijkstra" / "evaluation.json").read_text())
    plt.rcParams.update({"font.family": "DejaVu Sans", "font.size": 10})
    figure, axes = plt.subplots(1, 2, figsize=(12, 5.5))
    colors = (("#00798C", "#D1495B"), ("#2A9D6F", "#8D6A9F"))
    for axis, payload, title, palette in zip(
        axes,
        (addition, graph),
        ("Decimal addition", "Weighted Dijkstra"),
        colors,
        strict=True,
    ):
        history = payload["history"]
        steps = [entry["optimizer_step"] for entry in history]
        axis.plot(
            steps,
            [entry["training_accuracy"] for entry in history],
            color=palette[0],
            linewidth=2,
            label="Training",
        )
        axis.plot(
            steps,
            [entry["validation_accuracy"] for entry in history],
            color=palette[1],
            linewidth=2,
            label="Validation",
        )
        axis.set_title(title, loc="left", fontweight="bold")
        axis.set_xlabel("Optimizer step")
        axis.set_ylabel("Exact decision accuracy")
        axis.set_ylim(0.3, 1.02)
        axis.grid(color="#DDE2E5")
        axis.spines[["top", "right"]].set_visible(False)
        axis.legend(frameon=False, loc="lower right")
    total = sum(result["examples"] for result in evaluation["results"])
    correct = sum(result["correct_distances"] for result in evaluation["results"])
    axes[1].text(
        0.02,
        0.04,
        f"Closed loop: {correct}/{total} exact graph tests",
        transform=axes[1].transAxes,
        ha="left",
        color="#237A58",
        fontweight="bold",
    )
    figure.suptitle(
        "TensorFlow/XLA Neural Programmer-Interpreter",
        x=0.07,
        y=0.97,
        ha="left",
        fontsize=16,
        fontweight="bold",
    )
    figure.text(
        0.07,
        0.91,
        "Shared task-parameterized core | TensorFlow 2.20 | XLA-compiled training and inference",
        color="#5B6268",
    )
    figure.subplots_adjust(left=0.07, right=0.97, top=0.83, bottom=0.13, wspace=0.25)
    figure.savefig(ROOT / "tensorflow_xla_results.png", dpi=220, facecolor="white")
    figure.savefig(ROOT / "tensorflow_xla_results.pdf", facecolor="white")
    plt.close(figure)


if __name__ == "__main__":
    main()
