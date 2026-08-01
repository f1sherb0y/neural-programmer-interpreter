from npi.core.data import EpisodeDataset
from npi.tasks.graph.codec import CODEC
from npi.tasks.graph.problems import generate_problems
from npi.tasks.graph.traces import DijkstraTrace


def make_dataset(minimum_nodes, maximum_nodes, examples_per_size, seed):
    problems = generate_problems(minimum_nodes, maximum_nodes, examples_per_size, seed)
    episodes = []
    for problem in problems:
        episodes.extend(
            DijkstraTrace(
                problem.node_count, list(problem.edges), problem.source
            ).episodes
        )
    return EpisodeDataset(episodes, CODEC), problems
