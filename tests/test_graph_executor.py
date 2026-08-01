import unittest
from types import SimpleNamespace
from unittest.mock import patch

import tensorflow as tf

from npi.core.runtime import RuntimeProfile, execute_batch
from npi.tasks.graph.codec import CODEC
from npi.tasks.graph.environment import GraphEnvironment
from npi.tasks.graph.spec import SPEC, Symbol


class ImmediateReturnModel:
    config = SimpleNamespace(layers=1)

    def initial_state(self, batch_size):
        zeros = tf.zeros((batch_size, 1), tf.float32)
        return ((zeros, zeros),)

    def compiled_inference_step(self, use_xla=True):
        return tf.function(self.inference_step, jit_compile=use_xla)

    def inference_step(self, features, programs, states):
        batch = tf.shape(features)[0]
        end = tf.tile([[0.0, 1.0]], (batch, 1))
        program = tf.zeros((batch, SPEC.num_programs))
        arguments = tuple(tf.zeros((batch, depth)) for depth in SPEC.argument_depths)
        return end, program, arguments, states


class GraphExecutorTest(unittest.TestCase):
    def test_invalid_observation_isolated_from_valid_batch_member(self):
        bad = GraphEnvironment(2, [(0, 1, 1)], 0)
        good = GraphEnvironment(2, [(0, 1, 1)], 1)
        original = GraphEnvironment.observe

        def observe(environment):
            if environment.pointer_registers[Symbol.SOURCE] == 0:
                raise ValueError("invalid pointer")
            return original(environment)

        profile = RuntimeProfile()
        with patch.object(GraphEnvironment, "observe", new=observe):
            outcomes = execute_batch(
                ImmediateReturnModel(),
                SPEC,
                CODEC,
                [bad, good],
                [100, 100],
                use_xla=False,
                profile=profile,
            )
        self.assertIn("invalid observation", outcomes[0].failure)
        self.assertIsNone(outcomes[1].failure)
        self.assertIsNotNone(outcomes[1].result)
        self.assertEqual(profile.loop_iterations, 1)
        self.assertEqual(profile.observations, 1)
        self.assertEqual(profile.model_rows, 1)
        self.assertGreater(profile.inference_seconds, 0.0)
        self.assertGreater(profile.total_seconds, profile.inference_seconds)


if __name__ == "__main__":
    unittest.main()
