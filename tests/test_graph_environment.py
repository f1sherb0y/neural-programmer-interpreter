import heapq
import random
import unittest

import networkx as nx

from npi.tasks.graph.problems import generate_problem, generate_problems
from npi.tasks.graph.spec import Program
from npi.tasks.graph.traces import DijkstraTrace


def oracle_distances(node_count, edges, source):
    adjacency = [[] for _ in range(node_count)]
    for start, end, weight in edges:
        adjacency[start].append((end, weight))
    distances = [None] * node_count
    distances[source] = 0
    queue = [(0, source)]
    while queue:
        distance, node = heapq.heappop(queue)
        if distances[node] != distance:
            continue
        for neighbor, weight in adjacency[node]:
            candidate = distance + weight
            if distances[neighbor] is None or candidate < distances[neighbor]:
                distances[neighbor] = candidate
                heapq.heappush(queue, (candidate, neighbor))
    return distances


class GraphEnvironmentTest(unittest.TestCase):
    def test_reference_trace_computes_weighted_shortest_paths(self):
        edges = [(0, 1, 7), (0, 2, 2), (2, 1, 1), (1, 3, 2), (2, 3, 9)]
        trace = DijkstraTrace(5, edges, 0)
        self.assertEqual(trace.environment.distances(), [0, 3, 2, 5, None])
        self.assertEqual(trace.environment.distances(), oracle_distances(5, edges, 0))

    def test_trace_contains_hierarchical_programs(self):
        trace = DijkstraTrace(3, [(0, 1, 2), (1, 2, 4)], 0)
        programs = [episode.program for episode in trace.episodes]
        self.assertIn(Program.FIND_MIN, programs)
        self.assertIn(Program.RELAX, programs)
        self.assertGreater(programs.count(Program.ACT), 10)

    def test_negative_weights_are_rejected(self):
        with self.assertRaises(ValueError):
            DijkstraTrace(2, [(0, 1, -1)], 0)

    def test_random_connected_problem_is_weighted_erdos_renyi_graph(self):
        problem = generate_problem(
            100, random.Random(17), "random_connected", maximum_weight=100
        )
        graph = nx.Graph()
        graph.add_nodes_from(range(problem.node_count))
        graph.add_weighted_edges_from(problem.edges)
        self.assertTrue(nx.is_connected(graph))
        self.assertGreater(graph.number_of_edges(), problem.node_count - 1)
        self.assertTrue(all(1 <= weight <= 100 for _, _, weight in problem.edges))
        weights = {(start, end): weight for start, end, weight in problem.edges}
        self.assertTrue(
            all(
                weights[end, start] == weight
                for (start, end), weight in weights.items()
            )
        )

    def test_training_problems_are_random_connected_graphs(self):
        problems = generate_problems(5, 8, 3, seed=23, maximum_weight=100)
        self.assertEqual(len(problems), 12)
        self.assertEqual({problem.family for problem in problems}, {"random_connected"})
        for problem in problems:
            graph = nx.Graph()
            graph.add_nodes_from(range(problem.node_count))
            graph.add_weighted_edges_from(problem.edges)
            self.assertTrue(nx.is_connected(graph))


if __name__ == "__main__":
    unittest.main()
