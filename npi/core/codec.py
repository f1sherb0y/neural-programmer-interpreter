import numpy as np

from npi.core.spec import Arguments, TaskSpec


def append_argument_one_hot(
    observation: np.ndarray,
    arguments: Arguments,
    spec: TaskSpec,
) -> np.ndarray:
    if observation.shape != (spec.observation_size,):
        raise ValueError(
            f"Expected observation shape {(spec.observation_size,)}, got {observation.shape}"
        )
    if len(arguments) != len(spec.argument_depths):
        raise ValueError("Invocation argument count does not match task specification")
    feature = np.zeros(spec.feature_size, dtype=np.float32)
    feature[: spec.observation_size] = observation
    offset = spec.observation_size
    for argument, depth in zip(arguments, spec.argument_depths, strict=True):
        if not 0 <= int(argument) < depth:
            raise ValueError(f"Argument {argument} is outside depth {depth}")
        feature[offset + int(argument)] = 1.0
        offset += depth
    return feature
