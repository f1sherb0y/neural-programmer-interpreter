import random
import unittest

from npi.tasks.graph.experiment import oracle_distances, valid_parent_tree
from npi.tasks.graph.problems import generate_problem
from npi.tasks.graph_ram.data import make_sampled_dataset, sample_node_counts
from npi.tasks.graph_ram.spec import Opcode
from npi.tasks.graph_ram.traces import RamDijkstraTrace


class RamGraphTest(unittest.TestCase):
    def test_action_space_is_elementary_register_machine(self):
        self.assertEqual(
            {opcode.name for opcode in Opcode},
            {"MOV", "LOAD", "STORE", "ADD", "SUB", "SHL1", "SHR1", "CMP"},
        )

    def test_random_node_counts_have_bounded_exact_mean(self):
        counts = sample_node_counts(2, 100, 30, 396, 731)
        for example_count, seed in ((396, 1), (99, 2), (396, 731)):
            sample = sample_node_counts(2, 100, 30, example_count, seed)
            self.assertEqual(len(sample), example_count)
            self.assertEqual(min(sample), 2)
            self.assertEqual(max(sample), 100)
            self.assertEqual(sum(sample) / len(sample), 30)
        self.assertEqual(counts, sample_node_counts(2, 100, 30, 396, 731))
        self.assertNotEqual(counts, sample_node_counts(2, 100, 30, 396, 732))
        with self.assertRaises(ValueError):
            sample_node_counts(2, 100, 101, 396, 731)
        with self.assertRaises(ValueError):
            sample_node_counts(2, 100, 2, 2, 731)

        dataset, problems = make_sampled_dataset(2, 5, 3.5, 4, 731)
        self.assertGreater(dataset.decisions, 0)
        self.assertEqual(len(problems), 4)
        self.assertEqual(sum(problem.node_count for problem in problems), 14)
        self.assertEqual({problem.family for problem in problems}, {"random_connected"})

    def test_reference_trace_computes_random_weighted_shortest_paths(self):
        for nodes in (2, 5, 10, 20):
            problem = generate_problem(
                nodes,
                random.Random(700 + nodes),
                "random_connected",
                maximum_weight=100,
            )
            trace = RamDijkstraTrace(
                problem.node_count, list(problem.edges), problem.source
            )
            expected = oracle_distances(problem)
            self.assertEqual(trace.environment.distances(), expected)
            self.assertTrue(
                valid_parent_tree(problem, expected, trace.environment.parents())
            )


if __name__ == "__main__":
    unittest.main()
