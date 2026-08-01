from dataclasses import dataclass
from typing import Any, Protocol

import numpy as np

Arguments = tuple[int, ...]


@dataclass(frozen=True)
class TaskSpec:
    """Static neural interface for one task/action space.

    All dimensions are task-owned. The shared NPI core only consumes this
    contract, so adding programs or changing action arguments does not require
    changes to the model or trainer.
    """

    name: str
    observation_size: int
    num_programs: int
    argument_depths: tuple[int, ...]
    action_program: int
    root_program: int
    default_arguments: Arguments
    return_before_call: bool

    def __post_init__(self) -> None:
        if self.observation_size < 1 or self.num_programs < 1:
            raise ValueError("Observation and program dimensions must be positive")
        if not self.argument_depths or any(depth < 1 for depth in self.argument_depths):
            raise ValueError("Every argument head must have positive depth")
        if len(self.default_arguments) != len(self.argument_depths):
            raise ValueError("Default arguments must match argument heads")
        if not 0 <= self.action_program < self.num_programs:
            raise ValueError("Action program is outside the program vocabulary")
        if not 0 <= self.root_program < self.num_programs:
            raise ValueError("Root program is outside the program vocabulary")
        for value, depth in zip(
            self.default_arguments, self.argument_depths, strict=True
        ):
            if not 0 <= value < depth:
                raise ValueError("A default argument is outside its vocabulary")

    @property
    def argument_size(self) -> int:
        return sum(self.argument_depths)

    @property
    def feature_size(self) -> int:
        return self.observation_size + self.argument_size


class ObservationCodec(Protocol):
    spec: TaskSpec

    def encode_observation(self, observation: Any) -> np.ndarray:
        """Return one float32 vector with spec.observation_size elements."""

    def encode_feature(self, observation: Any, arguments: Arguments) -> np.ndarray:
        """Encode local observation and current invocation arguments."""


class RuntimeEnvironment(Protocol):
    def observe(self) -> Any: ...

    def execute(self, arguments: Arguments) -> None: ...

    def result(self) -> Any: ...
