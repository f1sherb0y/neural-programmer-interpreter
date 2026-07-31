from dataclasses import dataclass

import numpy as np
import torch

from modern_npi.graph.constants import ARGUMENT_DEPTHS, DEFAULT_ACTION, Program
from modern_npi.graph.data import encode_feature
from modern_npi.graph.environment import GraphEnvironment
from modern_npi.graph.model import GraphNPI


class ExecutionFailure(RuntimeError):
    pass


@dataclass(frozen=True)
class ExecutionResult:
    distances: list[int | None]
    parents: list[int | None]
    model_steps: int
    maximum_depth: int


class DijkstraExecutor:
    def __init__(
        self,
        model: GraphNPI,
        node_count: int,
        edges: list[tuple[int, int, int]] | tuple[tuple[int, int, int], ...],
        source: int,
        device: torch.device,
    ):
        self.model = model
        self.device = device
        self.environment = GraphEnvironment(node_count, list(edges), source)
        self.model_steps = 0
        self.maximum_depth = 0
        self.step_limit = 200 * (node_count * node_count + len(edges) + 1)

    @torch.inference_mode()
    def execute(self) -> ExecutionResult:
        self.model.eval()
        self._run(Program.DIJKSTRA, tuple(map(int, DEFAULT_ACTION)), depth=1)
        return ExecutionResult(
            self.environment.distances(),
            self.environment.parents(),
            self.model_steps,
            self.maximum_depth,
        )

    def _run(
        self,
        program: Program,
        arguments: tuple[int, ...],
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
            try:
                feature = encode_feature(self.environment.observe(), arguments)
            except (ValueError, IndexError) as error:
                raise ExecutionFailure(
                    f"step {self.model_steps}, invalid observation: {error}"
                ) from error
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
                    raise ExecutionFailure("ACT failed to return")
                try:
                    self.environment.execute(arguments)
                except (ValueError, IndexError) as error:
                    raise ExecutionFailure(
                        f"step {self.model_steps}, ACT arguments {arguments}: {error}"
                    ) from error
                return

            if should_end:
                return

            child = Program(program_logits.argmax(dim=-1).item())
            child_arguments = tuple(
                logits.argmax(dim=-1).item() for logits in argument_logits
            )
            if len(child_arguments) != len(ARGUMENT_DEPTHS):
                raise ExecutionFailure("Invalid argument count")
            self._run(child, child_arguments, depth=depth + 1)


def execute_dijkstra(
    model: GraphNPI,
    node_count: int,
    edges: list[tuple[int, int, int]] | tuple[tuple[int, int, int], ...],
    source: int,
    device: torch.device,
) -> ExecutionResult:
    return DijkstraExecutor(model, node_count, edges, source, device).execute()


@dataclass(frozen=True)
class BatchExecutionOutcome:
    result: ExecutionResult | None
    failure: str | None


@dataclass
class _Frame:
    program: Program
    arguments: tuple[int, ...]
    states: list[tuple[torch.Tensor, torch.Tensor]] | None = None


@dataclass
class _Execution:
    environment: GraphEnvironment
    stack: list[_Frame]
    step_limit: int
    model_steps: int = 0
    maximum_depth: int = 1
    failure: str | None = None


def _new_execution(
    node_count: int,
    edges: tuple[tuple[int, int, int], ...],
    source: int,
) -> _Execution:
    return _Execution(
        environment=GraphEnvironment(node_count, list(edges), source),
        stack=[_Frame(Program.DIJKSTRA, tuple(map(int, DEFAULT_ACTION)))],
        step_limit=200 * (node_count * node_count + len(edges) + 1),
    )


@torch.inference_mode()
def execute_dijkstra_batch(
    model: GraphNPI,
    problems,
    device: torch.device,
) -> list[BatchExecutionOutcome]:
    """Advance independent recursive interpreters together in one GPU batch."""
    model.eval()
    executions = [
        _new_execution(problem.node_count, problem.edges, problem.source)
        for problem in problems
    ]
    zero_state = model.initial_state(1, device)
    zero_state = [(hidden[0], memory[0]) for hidden, memory in zero_state]

    while True:
        active = [
            index
            for index, execution in enumerate(executions)
            if execution.stack and execution.failure is None
        ]
        if not active:
            break

        ready = []
        for index in active:
            execution = executions[index]
            execution.model_steps += 1
            if execution.model_steps > execution.step_limit:
                execution.failure = (
                    f"execution exceeded {execution.step_limit} model steps"
                )
            else:
                ready.append(index)
        if not ready:
            continue

        observable = []
        encoded_features = []
        for index in ready:
            execution = executions[index]
            frame = execution.stack[-1]
            try:
                encoded_features.append(
                    encode_feature(
                        execution.environment.observe(),
                        frame.arguments,
                    )
                )
                observable.append(index)
            except (ValueError, IndexError) as error:
                execution.failure = (
                    f"step {execution.model_steps}, invalid observation: {error}"
                )
        ready = observable
        if not ready:
            continue

        frames = [executions[index].stack[-1] for index in ready]
        features = np.stack(encoded_features)
        feature_tensor = torch.from_numpy(features).to(device, non_blocking=True)
        program_tensor = torch.tensor(
            [int(frame.program) for frame in frames],
            dtype=torch.long,
            device=device,
        )
        batched_states = []
        for layer in range(model.layers_count):
            batched_states.append(
                (
                    torch.stack(
                        [
                            zero_state[layer][0]
                            if frame.states is None
                            else frame.states[layer][0]
                            for frame in frames
                        ]
                    ),
                    torch.stack(
                        [
                            zero_state[layer][1]
                            if frame.states is None
                            else frame.states[layer][1]
                            for frame in frames
                        ]
                    ),
                )
            )

        end_logits, program_logits, argument_logits, next_states = (
            model.inference_step(feature_tensor, program_tensor, batched_states)
        )
        end_predictions = end_logits.argmax(dim=-1).cpu().tolist()
        program_predictions = program_logits.argmax(dim=-1).cpu().tolist()
        argument_predictions = [
            logits.argmax(dim=-1).cpu().tolist() for logits in argument_logits
        ]

        for batch_index, execution_index in enumerate(ready):
            execution = executions[execution_index]
            frame = execution.stack[-1]
            frame.states = [
                (hidden[batch_index], memory[batch_index])
                for hidden, memory in next_states
            ]
            should_end = bool(end_predictions[batch_index])

            if frame.program == Program.ACT:
                if not should_end:
                    execution.failure = (
                        f"step {execution.model_steps}: ACT failed to return"
                    )
                    continue
                try:
                    execution.environment.execute(frame.arguments)
                except (ValueError, IndexError) as error:
                    execution.failure = (
                        f"step {execution.model_steps}, ACT arguments "
                        f"{frame.arguments}: {error}"
                    )
                    continue
                execution.stack.pop()
                continue

            if should_end:
                execution.stack.pop()
                continue

            arguments = tuple(
                predictions[batch_index] for predictions in argument_predictions
            )
            execution.stack.append(
                _Frame(Program(program_predictions[batch_index]), arguments)
            )
            execution.maximum_depth = max(
                execution.maximum_depth, len(execution.stack)
            )
            if execution.maximum_depth > 16:
                execution.failure = "program call depth exceeded 16"

    outcomes = []
    for execution in executions:
        if execution.failure is not None:
            outcomes.append(BatchExecutionOutcome(None, execution.failure))
        else:
            outcomes.append(
                BatchExecutionOutcome(
                    ExecutionResult(
                        execution.environment.distances(),
                        execution.environment.parents(),
                        execution.model_steps,
                        execution.maximum_depth,
                    ),
                    None,
                )
            )
    return outcomes
