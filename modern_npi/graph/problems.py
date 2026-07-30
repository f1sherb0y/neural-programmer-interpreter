from dataclasses import dataclass
import random


@dataclass(frozen=True)
class GraphProblem:
    node_count: int
    edges: tuple[tuple[int, int, int], ...]
    source: int
    family: str


def _add_undirected(
    edges: list[tuple[int, int, int]],
    first: int,
    second: int,
    rng: random.Random,
    maximum_weight: int,
) -> None:
    weight = rng.randint(1, maximum_weight)
    edges.append((first, second, weight))
    edges.append((second, first, weight))


def generate_problem(
    node_count: int,
    rng: random.Random,
    family: str,
    maximum_weight: int = 9,
) -> GraphProblem:
    if node_count < 2:
        raise ValueError("Graph problems require at least two nodes")
    edges: list[tuple[int, int, int]] = []
    permutation = list(range(node_count))
    rng.shuffle(permutation)

    if family == "path":
        for index in range(node_count - 1):
            _add_undirected(
                edges,
                permutation[index],
                permutation[index + 1],
                rng,
                maximum_weight,
            )

    elif family == "star":
        center = permutation[0]
        for node in permutation[1:]:
            _add_undirected(edges, center, node, rng, maximum_weight)

    elif family in ("sparse", "dense"):
        probability = min(0.35, 2.5 / node_count) if family == "sparse" else 0.65
        for index in range(node_count - 1):
            _add_undirected(
                edges,
                permutation[index],
                permutation[index + 1],
                rng,
                maximum_weight,
            )
        existing = {(start, end) for start, end, _ in edges}
        for first in range(node_count):
            for second in range(first + 1, node_count):
                if (first, second) not in existing and rng.random() < probability:
                    _add_undirected(edges, first, second, rng, maximum_weight)

    elif family == "directed":
        for index in range(node_count - 1):
            edges.append(
                (
                    permutation[index],
                    permutation[index + 1],
                    rng.randint(1, maximum_weight),
                )
            )
        probability = min(0.4, 3.0 / node_count)
        existing = {(start, end) for start, end, _ in edges}
        for first in range(node_count):
            for second in range(node_count):
                if first != second and (first, second) not in existing and rng.random() < probability:
                    edges.append((first, second, rng.randint(1, maximum_weight)))

    elif family == "disconnected":
        split = max(1, node_count // 2)
        groups = (permutation[:split], permutation[split:])
        for group in groups:
            for index in range(len(group) - 1):
                _add_undirected(
                    edges,
                    group[index],
                    group[index + 1],
                    rng,
                    maximum_weight,
                )
        for group in groups:
            for first_index, first in enumerate(group):
                for second in group[first_index + 1 :]:
                    if rng.random() < 0.25:
                        _add_undirected(edges, first, second, rng, maximum_weight)

    else:
        raise ValueError(f"Unknown graph family: {family}")

    rng.shuffle(edges)
    return GraphProblem(node_count, tuple(edges), rng.randrange(node_count), family)


def generate_problems(
    minimum_nodes: int,
    maximum_nodes: int,
    examples_per_size: int,
    seed: int,
    maximum_weight: int = 9,
) -> list[GraphProblem]:
    rng = random.Random(seed)
    families = ("path", "star", "sparse", "dense", "disconnected")
    return [
        generate_problem(
            node_count,
            rng,
            families[example % len(families)],
            maximum_weight,
        )
        for node_count in range(minimum_nodes, maximum_nodes + 1)
        for example in range(examples_per_size)
    ]
