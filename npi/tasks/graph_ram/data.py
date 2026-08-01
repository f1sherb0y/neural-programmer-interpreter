from npi.core.data import EpisodeDataset
from npi.tasks.graph.problems import generate_problems
from npi.tasks.graph_ram.codec import CODEC
from npi.tasks.graph_ram.traces import RamDijkstraTrace


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
