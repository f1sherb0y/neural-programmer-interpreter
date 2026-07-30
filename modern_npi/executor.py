from dataclasses import dataclass

import numpy as np
import torch

from modern_npi.constants import DEFAULT_ARGS, Program
from modern_npi.data import encode_feature
from modern_npi.environment import AdditionEnvironment
from modern_npi.model import NeuralProgrammerInterpreter


class ExecutionFailure(RuntimeError):
    pass


@dataclass(frozen=True)
class ExecutionResult:
    value: str
    model_steps: int
    maximum_depth: int


class AdditionExecutor:
    def __init__(
        self,
        model: NeuralProgrammerInterpreter,
        first: str,
        second: str,
        device: torch.device,
    ):
        self.model = model
        self.device = device
        self.environment = AdditionEnvironment(first, second)
        self.model_steps = 0
        self.maximum_depth = 0
        self.step_limit = 40 * (self.environment.input_width + 2)

    @torch.inference_mode()
    def execute(self) -> ExecutionResult:
        self.model.eval()
        self._run(Program.ADD, tuple(map(int, DEFAULT_ARGS)), depth=1)
        return ExecutionResult(
            self.environment.result(),
            self.model_steps,
            self.maximum_depth,
        )

    def _run(
        self,
        program: Program,
        arguments: tuple[int, int, int],
        *,
        depth: int,
    ) -> None:
        self.maximum_depth = max(self.maximum_depth, depth)
        if depth > 16:
            raise ExecutionFailure("Program call depth exceeded 16")
        states = self.model.initial_state(1, self.device)

        while True:
            self.model_steps += 1
            if self.model_steps > self.step_limit:
                raise ExecutionFailure(f"Execution exceeded {self.step_limit} model steps")

            feature = encode_feature(self.environment.observe(), arguments)
            feature_tensor = torch.from_numpy(feature).unsqueeze(0).to(self.device)
            program_tensor = torch.tensor([int(program)], device=self.device)
            end_logits, program_logits, argument_logits, states = self.model.inference_step(
                feature_tensor,
                program_tensor,
                states,
            )
            should_end = bool(end_logits.argmax(dim=-1).item())

            if program == Program.ACT:
                if not should_end:
                    raise ExecutionFailure("ACT failed to return after one environment action")
                try:
                    self.environment.execute(arguments)
                except (ValueError, IndexError) as error:
                    raise ExecutionFailure(str(error)) from error
                return

            child = Program(program_logits.argmax(dim=-1).item())
            child_arguments = tuple(
                logits.argmax(dim=-1).item() for logits in argument_logits
            )
            self._run(child, child_arguments, depth=depth + 1)
            if should_end:
                return


def execute_addition(
    model: NeuralProgrammerInterpreter,
    first: str,
    second: str,
    device: torch.device,
) -> ExecutionResult:
    return AdditionExecutor(model, first, second, device).execute()
