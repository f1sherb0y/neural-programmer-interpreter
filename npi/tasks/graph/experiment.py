import argparse
import json
import random
import time
from dataclasses import asdict, dataclass
from pathlib import Path

import networkx as nx
import tensorflow as tf

from npi.core.checkpoint import load_model
from npi.core.experiment import train_epochs
from npi.core.hardware import configure_tensorflow
from npi.core.runtime import execute_batch
from npi.tasks.graph.codec import CODEC
from npi.tasks.graph.data import make_dataset
from npi.tasks.graph.environment import GraphEnvironment
from npi.tasks.graph.problems import generate_problem
from npi.tasks.graph.spec import SPEC

DEFAULT_OUTPUT = Path("artifacts/graph_dijkstra_tf")
FAMILIES = (
    "path",
    "star",
    "sparse",
    "dense",
    "random_connected",
    "disconnected",
    "directed",
)


def oracle_distances(problem):
    graph = nx.MultiDiGraph()
    graph.add_nodes_from(range(problem.node_count))
    graph.add_weighted_edges_from(problem.edges)
    values = nx.single_source_dijkstra_path_length(graph, problem.source)
    return [values.get(node) for node in range(problem.node_count)]


def valid_parent_tree(problem, distances, parents):
    weights = {}
    for start, end, weight in problem.edges:
        weights[(start, end)] = min(weight, weights.get((start, end), weight))
    for node, distance in enumerate(distances):
        if node == problem.source or distance is None:
            continue
        parent = parents[node]
        if parent is None or (parent, node) not in weights:
            return False
        if (
            distances[parent] is None
            or distances[parent] + weights[parent, node] != distance
        ):
            return False
    return True


@dataclass(frozen=True)
class EvaluationResult:
    nodes: int
    family: str
    examples: int
    correct_distances: int
    valid_parent_trees: int
    seconds: float
    first_failure: str | None


def evaluate_group(model, problems, use_xla, batch_size):
    correct = parents = 0
    failure = None
    started = time.time()
    for start in range(0, len(problems), batch_size):
        batch = problems[start : start + batch_size]
        environments = [
            GraphEnvironment(p.node_count, list(p.edges), p.source) for p in batch
        ]
        limits = [200 * (p.node_count * p.node_count + len(p.edges) + 1) for p in batch]
        outcomes = execute_batch(
            model, SPEC, CODEC, environments, limits, use_xla=use_xla
        )
        for problem, outcome in zip(batch, outcomes, strict=True):
            if outcome.failure:
                if failure is None:
                    failure = outcome.failure
                continue
            distances, parent_values = outcome.result
            expected = oracle_distances(problem)
            if distances == expected:
                correct += 1
                if valid_parent_tree(problem, expected, parent_values):
                    parents += 1
            elif failure is None:
                mismatch = [
                    i
                    for i, (a, b) in enumerate(zip(distances, expected, strict=True))
                    if a != b
                ]
                failure = f"distance mismatch at nodes {mismatch[:8]}"
    return correct, parents, time.time() - started, failure


def parse_int_list(value):
    return [int(item) for item in value.split(",")]


def parser():
    result = argparse.ArgumentParser(description="TensorFlow/XLA weighted Dijkstra NPI")
    result.add_argument("command", choices=("train", "evaluate", "reproduce"))
    result.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    result.add_argument("--maximum-train-nodes", type=int, default=10)
    result.add_argument("--epochs", type=int, default=100)
    result.add_argument("--batch-size", type=int, default=256)
    result.add_argument("--execution-batch-size", type=int, default=100)
    result.add_argument("--learning-rate", type=float, default=3e-4)
    result.add_argument("--weight-decay", type=float, default=0.0)
    result.add_argument("--l1-regularization", type=float, default=1e-10)
    result.add_argument("--l2-regularization", type=float, default=0.0)
    result.add_argument("--seed", type=int, default=1)
    result.add_argument("--device", default="auto")
    result.add_argument("--no-xla", action="store_true")
    result.add_argument(
        "--evaluation-nodes", type=parse_int_list, default=[5, 10, 20, 50]
    )
    result.add_argument("--evaluation-examples", type=int, default=5)
    result.add_argument("--evaluation-maximum-weight", type=int, default=50)
    return result


def main():
    args = parser().parse_args()
    selected = configure_tensorflow(args.device)
    use_xla = not args.no_xla
    checkpoint = args.output / "best.weights.h5"
    print(f"tensorflow={tf.__version__} device={selected} xla={use_xla}")
    if args.command in ("train", "reproduce"):
        training, _ = make_dataset(2, args.maximum_train_nodes, 32, args.seed)
        validation, _ = make_dataset(2, args.maximum_train_nodes, 8, args.seed + 1)
        model, _history = train_epochs(
            SPEC,
            training,
            validation,
            checkpoint,
            epochs=args.epochs,
            batch_size=args.batch_size,
            learning_rate=args.learning_rate,
            weight_decay=args.weight_decay,
            l1_regularization=args.l1_regularization,
            l2_regularization=args.l2_regularization,
            seed=args.seed,
            use_xla=use_xla,
        )
    else:
        model = load_model(SPEC, checkpoint)
    if args.command in ("evaluate", "reproduce"):
        rng = random.Random(args.seed + 100)
        results = []
        for nodes in args.evaluation_nodes:
            for family in FAMILIES:
                problems = [
                    generate_problem(nodes, rng, family, args.evaluation_maximum_weight)
                    for _ in range(args.evaluation_examples)
                ]
                correct, parents, seconds, failure = evaluate_group(
                    model, problems, use_xla, args.execution_batch_size
                )
                result = EvaluationResult(
                    nodes, family, len(problems), correct, parents, seconds, failure
                )
                results.append(result)
                print(
                    f"nodes={nodes} family={family}: {correct}/{len(problems)} seconds={seconds:.1f}"
                )
        payload = {
            "framework": "tensorflow",
            "tensorflow": tf.__version__,
            "xla": use_xla,
            "maximum_train_nodes": args.maximum_train_nodes,
            "results": [asdict(result) for result in results],
        }
        args.output.mkdir(parents=True, exist_ok=True)
        (args.output / "evaluation.json").write_text(
            json.dumps(payload, indent=2) + "\n"
        )


if __name__ == "__main__":
    main()
