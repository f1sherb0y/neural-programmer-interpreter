import argparse
from dataclasses import asdict, dataclass
import json
from pathlib import Path
import random
import time

import networkx as nx
import numpy as np
import torch
from torch.nn import functional as F

from modern_npi.experiment import choose_device
from modern_npi.graph.constants import ARGUMENT_DEPTHS
from modern_npi.graph.data import EpisodeBatches
from modern_npi.graph.executor import ExecutionFailure, execute_dijkstra
from modern_npi.graph.model import GraphNPI
from modern_npi.graph.problems import GraphProblem, generate_problem, generate_problems
from modern_npi.graph.traces import DijkstraTrace
from modern_npi.training_curve import stable_lr_multiplier


DEFAULT_OUTPUT = Path("artifacts/graph_dijkstra")


def make_data(
    minimum_nodes: int,
    maximum_nodes: int,
    examples_per_size: int,
    seed: int,
) -> tuple[EpisodeBatches, list[GraphProblem]]:
    problems = generate_problems(
        minimum_nodes,
        maximum_nodes,
        examples_per_size,
        seed,
    )
    episodes = []
    for problem in problems:
        trace = DijkstraTrace(
            problem.node_count,
            list(problem.edges),
            problem.source,
        )
        episodes.extend(trace.episodes)
    return EpisodeBatches(episodes), problems


def to_tensors(batch, device):
    tensors = []
    for index, value in enumerate(batch):
        tensor = torch.from_numpy(value)
        if index in (0, len(batch) - 1):
            tensor = tensor.float()
        else:
            tensor = tensor.long()
        tensors.append(tensor.to(device, non_blocking=True))
    return tuple(tensors)


def masked_cross_entropy(logits, target, mask):
    losses = F.cross_entropy(
        logits.reshape(-1, logits.shape[-1]),
        target.reshape(-1),
        reduction="none",
    ).reshape_as(mask)
    return (losses * mask).sum() / mask.sum().clamp_min(1.0)


def decision_accuracy(
    end_logits,
    program_logits,
    argument_logits,
    target_end,
    target_program,
    target_arguments,
    child_mask,
):
    correct = end_logits.argmax(dim=-1).eq(target_end)
    child_correct = program_logits.argmax(dim=-1).eq(target_program)
    for logits, target in zip(argument_logits, target_arguments, strict=True):
        child_correct.logical_and_(logits.argmax(dim=-1).eq(target))
    correct.logical_and_(child_mask.eq(0).logical_or(child_correct))
    return int(correct.sum().item()), correct.numel()


class Trainer:
    def __init__(self, model, learning_rate, device):
        self.model = model
        self.device = device
        self.optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)
        self.scheduler = torch.optim.lr_scheduler.LambdaLR(
            self.optimizer,
            lr_lambda=stable_lr_multiplier,
        )

    def train_batch(self, batch):
        tensors = to_tensors(batch, self.device)
        features, programs, target_end, target_program = tensors[:4]
        target_arguments = tensors[4 : 4 + len(ARGUMENT_DEPTHS)]
        child_mask = tensors[-1]

        self.optimizer.zero_grad(set_to_none=True)
        end_logits, program_logits, argument_logits = self.model(features, programs)
        loss = F.cross_entropy(end_logits.flatten(0, 1), target_end.flatten())
        loss = loss + masked_cross_entropy(
            program_logits, target_program, child_mask
        )
        loss = loss + sum(
            masked_cross_entropy(logits, target, child_mask)
            for logits, target in zip(
                argument_logits, target_arguments, strict=True
            )
        )
        loss.backward()
        torch.nn.utils.clip_grad_norm_(self.model.parameters(), 1.0)
        self.optimizer.step()
        self.scheduler.step()
        correct, count = decision_accuracy(
            end_logits,
            program_logits,
            argument_logits,
            target_end,
            target_program,
            target_arguments,
            child_mask,
        )
        return float(loss.detach()), correct, count


@torch.inference_mode()
def teacher_forced_accuracy(model, data, batch_size, device):
    model.eval()
    correct = 0
    count = 0
    for batch in data.batches(batch_size, shuffle=False, seed=0):
        tensors = to_tensors(batch, device)
        features, programs, target_end, target_program = tensors[:4]
        target_arguments = tensors[4 : 4 + len(ARGUMENT_DEPTHS)]
        child_mask = tensors[-1]
        outputs = model(features, programs)
        batch_correct, batch_count = decision_accuracy(
            *outputs,
            target_end,
            target_program,
            target_arguments,
            child_mask,
        )
        correct += batch_correct
        count += batch_count
    return correct / count


def save_checkpoint(model, path, metadata):
    path.parent.mkdir(parents=True, exist_ok=True)
    torch.save({"model_state": model.state_dict(), **metadata}, path)


def load_model(path, device):
    payload = torch.load(path, map_location=device, weights_only=True)
    model = GraphNPI().to(device)
    model.load_state_dict(payload["model_state"])
    model.eval()
    return model


def train(args, device):
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    if device.type == "cuda":
        torch.cuda.manual_seed_all(args.seed)

    training_data, _ = make_data(2, args.maximum_train_nodes, 32, args.seed)
    validation_data, _ = make_data(2, args.maximum_train_nodes, 8, args.seed + 1)
    model = GraphNPI().to(device)
    trainer = Trainer(model, args.learning_rate, device)
    history = []
    best_validation = -1.0
    optimizer_step = 0
    started = time.time()

    print(f"device: {device}")
    print(
        f"training: {training_data.size:,} invocations / "
        f"{training_data.decisions:,} decisions"
    )
    for epoch in range(1, args.epochs + 1):
        model.train()
        total_loss = 0.0
        online_correct = 0
        online_count = 0
        batches = 0
        for batch in training_data.batches(
            args.batch_size,
            shuffle=True,
            seed=args.seed + epoch,
        ):
            loss, correct, count = trainer.train_batch(batch)
            total_loss += loss
            online_correct += correct
            online_count += count
            batches += 1
            optimizer_step += 1

        validation_accuracy = teacher_forced_accuracy(
            model, validation_data, args.batch_size, device
        )
        training_accuracy = teacher_forced_accuracy(
            model, training_data, args.batch_size, device
        )
        entry = {
            "epoch": epoch,
            "optimizer_step": optimizer_step,
            "loss": total_loss / batches,
            "online_accuracy": online_correct / online_count,
            "training_accuracy": training_accuracy,
            "validation_accuracy": validation_accuracy,
            "learning_rate": trainer.optimizer.param_groups[0]["lr"],
        }
        history.append(entry)
        print(
            f"epoch {epoch:03d} step {optimizer_step:6d}: "
            f"loss={entry['loss']:.4f} train={training_accuracy:.6f} "
            f"validation={validation_accuracy:.6f} "
            f"lr={entry['learning_rate']:.2e}"
        )
        metadata = {
            "seed": args.seed,
            "maximum_train_nodes": args.maximum_train_nodes,
            "training_invocations": training_data.size,
            "training_decisions": training_data.decisions,
            "history": history,
        }
        args.output.mkdir(parents=True, exist_ok=True)
        (args.output / "training_history.json").write_text(
            json.dumps(metadata, indent=2) + "\n"
        )
        if validation_accuracy >= best_validation:
            best_validation = validation_accuracy
            save_checkpoint(model, args.output / "best.pt", metadata)
        if training_accuracy == 1.0 and validation_accuracy == 1.0:
            break

    print(f"training_seconds: {time.time() - started:.1f}")
    return load_model(args.output / "best.pt", device)


def oracle_distances(problem: GraphProblem) -> list[int | None]:
    graph = nx.MultiDiGraph()
    graph.add_nodes_from(range(problem.node_count))
    graph.add_weighted_edges_from(problem.edges)
    distances = nx.single_source_dijkstra_path_length(graph, problem.source)
    return [distances.get(node) for node in range(problem.node_count)]


def valid_parent_tree(problem, distances, parents):
    edge_weights = {}
    for start, end, weight in problem.edges:
        key = (start, end)
        edge_weights[key] = min(weight, edge_weights.get(key, weight))
    for node, distance in enumerate(distances):
        if node == problem.source or distance is None:
            continue
        parent = parents[node]
        if parent is None or (parent, node) not in edge_weights:
            return False
        if distances[parent] is None:
            return False
        if distances[parent] + edge_weights[parent, node] != distance:
            return False
    return True


@dataclass(frozen=True)
class EvaluationResult:
    nodes: int
    family: str
    examples: int
    correct_distances: int
    valid_parent_trees: int
    accuracy: float
    average_model_steps: float
    seconds: float
    first_failure: str | None


def evaluate_group(model, problems, device):
    correct = 0
    valid_parents = 0
    steps = 0
    first_failure = None
    started = time.time()
    for problem in problems:
        expected = oracle_distances(problem)
        try:
            result = execute_dijkstra(
                model,
                problem.node_count,
                problem.edges,
                problem.source,
                device,
            )
            steps += result.model_steps
            if result.distances == expected:
                correct += 1
                if valid_parent_tree(problem, expected, result.parents):
                    valid_parents += 1
                elif first_failure is None:
                    first_failure = "distance correct but parent tree invalid"
            elif first_failure is None:
                first_failure = f"expected {expected}, got {result.distances}"
        except ExecutionFailure as error:
            if first_failure is None:
                first_failure = str(error)
    elapsed = time.time() - started
    return EvaluationResult(
        problems[0].node_count,
        problems[0].family,
        len(problems),
        correct,
        valid_parents,
        correct / len(problems),
        steps / len(problems),
        elapsed,
        first_failure,
    )


def evaluate(args, model, device):
    rng = random.Random(args.seed + 100)
    results = []
    families = ("path", "star", "sparse", "dense", "disconnected", "directed")
    for nodes in args.evaluation_nodes:
        examples = args.evaluation_examples if nodes <= 20 else min(2, args.evaluation_examples)
        for family in families:
            problems = [
                generate_problem(
                    nodes,
                    rng,
                    family,
                    maximum_weight=args.evaluation_maximum_weight,
                )
                for _ in range(examples)
            ]
            result = evaluate_group(model, problems, device)
            results.append(result)
            print(
                f"nodes={nodes:3d} family={family:12s}: "
                f"{result.correct_distances}/{result.examples} "
                f"distance_acc={result.accuracy:.3f} "
                f"steps={result.average_model_steps:.1f} seconds={result.seconds:.1f}"
            )
            if result.first_failure:
                print(f"  first failure: {result.first_failure}")
    payload = {
        "torch": torch.__version__,
        "device": str(device),
        "maximum_train_nodes": args.maximum_train_nodes,
        "training_weight_range": [1, 9],
        "evaluation_weight_range": [1, args.evaluation_maximum_weight],
        "results": [asdict(result) for result in results],
    }
    args.output.mkdir(parents=True, exist_ok=True)
    (args.output / "evaluation.json").write_text(json.dumps(payload, indent=2) + "\n")
    return results


def parse_nodes(value):
    return [int(item) for item in value.split(",")]


def parser():
    result = argparse.ArgumentParser(description="Train and evaluate pointer-machine Dijkstra NPI.")
    result.add_argument("command", choices=("train", "evaluate", "reproduce"))
    result.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    result.add_argument("--maximum-train-nodes", type=int, default=10)
    result.add_argument("--epochs", type=int, default=120)
    result.add_argument("--batch-size", type=int, default=256)
    result.add_argument("--learning-rate", type=float, default=3e-4)
    result.add_argument("--seed", type=int, default=1)
    result.add_argument("--device", default="auto")
    result.add_argument("--evaluation-nodes", type=parse_nodes, default=[5, 10, 15, 20])
    result.add_argument("--evaluation-examples", type=int, default=5)
    result.add_argument("--evaluation-maximum-weight", type=int, default=20)
    return result


def main():
    args = parser().parse_args()
    device = choose_device(args.device)
    if args.command in ("train", "reproduce"):
        model = train(args, device)
    else:
        model = load_model(args.output / "best.pt", device)
    if args.command in ("evaluate", "reproduce"):
        evaluate(args, model, device)


if __name__ == "__main__":
    main()
