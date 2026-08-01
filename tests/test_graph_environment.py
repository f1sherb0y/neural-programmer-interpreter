import heapq
import unittest

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


if __name__ == "__main__":
    unittest.main()
