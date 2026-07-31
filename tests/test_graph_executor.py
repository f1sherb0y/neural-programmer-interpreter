import unittest
from unittest.mock import patch

import torch

from modern_npi.graph.constants import ARGUMENT_DEPTHS, NUM_PROGRAMS, Symbol
from modern_npi.graph.environment import GraphEnvironment
from modern_npi.graph.executor import (
    DijkstraExecutor,
    ExecutionFailure,
    execute_dijkstra_batch,
)
from modern_npi.graph.model import GraphNPI
from modern_npi.graph.problems import GraphProblem


class ImmediateReturnModel:
    layers_count = 1
    hidden_size = 1

    def eval(self):
        return self

    def initial_state(self, batch_size, device):
        zeros = torch.zeros(batch_size, self.hidden_size, device=device)
        return [(zeros, zeros.clone())]

    def inference_step(self, features, program_ids, states):
        batch_size = features.shape[0]
        device = features.device
        end_logits = torch.tensor([[0.0, 1.0]], device=device).repeat(batch_size, 1)
        program_logits = torch.zeros(batch_size, NUM_PROGRAMS, device=device)
        argument_logits = tuple(
            torch.zeros(batch_size, depth, device=device)
            for depth in ARGUMENT_DEPTHS
        )
        return end_logits, program_logits, argument_logits, states


class GraphExecutorTest(unittest.TestCase):
    def setUp(self):
        self.device = torch.device("cpu")
        self.model = GraphNPI().to(self.device)
        self.problem = GraphProblem(
            node_count=2,
            edges=((0, 1, 1),),
            source=0,
            family="directed",
        )

    def test_scalar_executor_reports_invalid_observation(self):
        executor = DijkstraExecutor(
            self.model,
            self.problem.node_count,
            self.problem.edges,
            self.problem.source,
            self.device,
        )
        with patch.object(
            executor.environment,
            "observe",
            side_effect=ValueError("invalid pointer"),
        ):
            with self.assertRaisesRegex(ExecutionFailure, "invalid observation"):
                executor.execute()

    def test_batched_executor_isolates_invalid_observation(self):
        valid_problem = GraphProblem(
            node_count=2,
            edges=((0, 1, 1),),
            source=1,
            family="directed",
        )
        original_observe = GraphEnvironment.observe

        def selective_observe(environment):
            if environment.pointer_registers[Symbol.SOURCE] == 0:
                raise ValueError("invalid pointer")
            return original_observe(environment)

        with patch.object(GraphEnvironment, "observe", new=selective_observe):
            outcomes = execute_dijkstra_batch(
                ImmediateReturnModel(),
                [self.problem, valid_problem],
                self.device,
            )

        self.assertIsNone(outcomes[0].result)
        self.assertIn("invalid observation", outcomes[0].failure)
        self.assertIsNone(outcomes[1].failure)
        self.assertIsNotNone(outcomes[1].result)


if __name__ == "__main__":
    unittest.main()
