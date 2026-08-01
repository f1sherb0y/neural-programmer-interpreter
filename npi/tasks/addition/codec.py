import numpy as np

from npi.core.codec import append_argument_one_hot
from npi.tasks.addition.spec import SPEC


class AdditionCodec:
    spec = SPEC

    def encode_observation(self, observation) -> np.ndarray:
        encoded = np.zeros(self.spec.observation_size, np.float32)
        offset = 0
        for digit in observation.digits:
            encoded[offset + digit] = 1.0
            offset += 10
        encoded[offset] = observation.at_most_significant_input
        return encoded

    def encode_feature(self, observation, arguments) -> np.ndarray:
        return append_argument_one_hot(
            self.encode_observation(observation), arguments, self.spec
        )


CODEC = AdditionCodec()
