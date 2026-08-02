import math
import random

from npi.core.data import EpisodeDataset
from npi.tasks.graph.problems import generate_problem, generate_problems
from npi.tasks.graph_ram.codec import CODEC
from npi.tasks.graph_ram.traces import RamDijkstraTrace


def _bounded_exponential_weights(minimum_nodes, maximum_nodes, mean_nodes):
    values = range(minimum_nodes, maximum_nodes + 1)

    def expected(rate):
        weights = [math.exp(rate * (value - minimum_nodes)) for value in values]
        return sum(value * weight for value, weight in zip(values, weights)) / sum(
            weights
        )

    lower, upper = -1.0, 1.0
    for _ in range(80):
        middle = (lower + upper) / 2.0
        if expected(middle) < mean_nodes:
            lower = middle
        else:
            upper = middle
    rate = (lower + upper) / 2.0
    return [math.exp(rate * (value - minimum_nodes)) for value in values]


def sample_node_counts(
    minimum_nodes,
    maximum_nodes,
    mean_nodes,
    example_count,
    seed,
):
    if not minimum_nodes <= mean_nodes <= maximum_nodes:
        raise ValueError("Mean nodes must be within the node-count bounds")
    if example_count < 2:
        raise ValueError("At least two examples are required for boundary coverage")
    target_sum = round(mean_nodes * example_count)
    minimum_sum = maximum_nodes + minimum_nodes * (example_count - 1)
    maximum_sum = minimum_nodes + maximum_nodes * (example_count - 1)
    if not minimum_sum <= target_sum <= maximum_sum:
        raise ValueError("Requested mean is incompatible with boundary coverage")
    rng = random.Random(seed)
    values = list(range(minimum_nodes, maximum_nodes + 1))
    weights = _bounded_exponential_weights(minimum_nodes, maximum_nodes, mean_nodes)
    counts = rng.choices(values, weights=weights, k=example_count)
    counts[0], counts[1] = minimum_nodes, maximum_nodes

    difference = target_sum - sum(counts)
    while difference:
        if difference > 0:
            candidates = [
                index
                for index, value in enumerate(counts[2:], start=2)
                if value < maximum_nodes
            ]
            counts[rng.choice(candidates)] += 1
            difference -= 1
        else:
            candidates = [
                index
                for index, value in enumerate(counts[2:], start=2)
                if value > minimum_nodes
            ]
            counts[rng.choice(candidates)] -= 1
            difference += 1
    rng.shuffle(counts)
    return counts


def make_sampled_dataset(
    minimum_nodes,
    maximum_nodes,
    mean_nodes,
    example_count,
    seed,
    maximum_weight=9,
):
    rng = random.Random(seed)
    node_counts = sample_node_counts(
        minimum_nodes,
        maximum_nodes,
        mean_nodes,
        example_count,
        seed,
    )
    problems = [
        generate_problem(
            node_count,
            rng,
            "random_connected",
            maximum_weight,
        )
        for node_count in node_counts
    ]
    episodes = []
    for problem in problems:
        episodes.extend(
            RamDijkstraTrace(
                problem.node_count, list(problem.edges), problem.source
            ).episodes
        )
    return EpisodeDataset(episodes, CODEC), problems


def make_dataset(minimum_nodes, maximum_nodes, examples_per_size, seed):
    problems = generate_problems(minimum_nodes, maximum_nodes, examples_per_size, seed)
    episodes = []
    for problem in problems:
        episodes.extend(
            RamDijkstraTrace(
                problem.node_count, list(problem.edges), problem.source
            ).episodes
        )
    return EpisodeDataset(episodes, CODEC), problems
