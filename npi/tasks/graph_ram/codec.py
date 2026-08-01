import numpy as np

from npi.core.codec import append_argument_one_hot
from npi.tasks.graph_ram.spec import SPEC


class RamGraphCodec:
    spec = SPEC

    def encode_observation(self, observation):
        encoded = np.asarray(observation.as_tuple(), np.float32)
        if encoded.shape != (self.spec.observation_size,):
            raise ValueError(f"Unexpected RAM observation shape: {encoded.shape}")
        return encoded

    def encode_feature(self, observation, arguments):
        return append_argument_one_hot(
            self.encode_observation(observation), arguments, self.spec
        )


CODEC = RamGraphCodec()
