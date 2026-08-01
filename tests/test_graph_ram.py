import random
import unittest

from npi.tasks.graph.experiment import oracle_distances, valid_parent_tree
from npi.tasks.graph.problems import generate_problem
from npi.tasks.graph_ram.spec import Opcode
from npi.tasks.graph_ram.traces import RamDijkstraTrace


class RamGraphTest(unittest.TestCase):
    def test_action_space_is_elementary_register_machine(self):
        self.assertEqual(
            {opcode.name for opcode in Opcode},
            {"MOV", "LOAD", "STORE", "ADD", "SUB", "SHL1", "SHR1", "CMP"},
        )

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
