import numpy as np

from npi.core.codec import append_argument_one_hot
from npi.tasks.graph.spec import SPEC


class GraphCodec:
    spec = SPEC

    def encode_observation(self, observation) -> np.ndarray:
        encoded = np.asarray(observation.as_tuple(), np.float32)
        if encoded.shape != (self.spec.observation_size,):
            raise ValueError(f"Unexpected graph observation shape: {encoded.shape}")
        return encoded

    def encode_feature(self, observation, arguments) -> np.ndarray:
        return append_argument_one_hot(
            self.encode_observation(observation), arguments, self.spec
        )


CODEC = GraphCodec()
