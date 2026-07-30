import unittest

import torch

from modern_npi.data import FEATURE_SIZE
from modern_npi.model import NeuralProgrammerInterpreter


class ModelTest(unittest.TestCase):
    def test_sequence_and_recursive_step_paths_share_parameters(self):
        torch.manual_seed(4)
        model = NeuralProgrammerInterpreter(hidden_size=32)
        model.eval()
        features = torch.randn(2, 5, FEATURE_SIZE)
        programs = torch.randint(0, 5, (2, 5))

        with torch.inference_mode():
            sequence_end = model(features, programs)[0]
            states = model.initial_state(2, torch.device("cpu"))
            step_end = []
            for time in range(features.shape[1]):
                outputs = model.inference_step(
                    features[:, time], programs[:, time], states
                )
                step_end.append(outputs[0])
                states = outputs[-1]

        torch.testing.assert_close(
            sequence_end,
            torch.stack(step_end, dim=1),
            rtol=1e-5,
            atol=1e-6,
        )


if __name__ == "__main__":
    unittest.main()
