from dataclasses import dataclass
from typing import Any

import numpy as np
import tensorflow as tf

from npi.core.model import NeuralProgrammerInterpreter
from npi.core.spec import Arguments, ObservationCodec, RuntimeEnvironment, TaskSpec


class ExecutionFailure(RuntimeError):
    pass


@dataclass(frozen=True)
class ExecutionStats:
    model_steps: int
    maximum_depth: int


@dataclass(frozen=True)
class BatchOutcome:
    result: Any | None
    stats: ExecutionStats | None
    failure: str | None


@dataclass
class _Frame:
    program: int
    arguments: Arguments
    states: tuple[tuple[tf.Tensor, tf.Tensor], ...] | None = None
    return_after_child: bool = False


@dataclass
class _Execution:
    environment: RuntimeEnvironment
    stack: list[_Frame]
    step_limit: int
    model_steps: int = 0
    maximum_depth: int = 1
    failure: str | None = None


def _complete_frame(execution: _Execution) -> None:
    execution.stack.pop()
    while execution.stack and execution.stack[-1].return_after_child:
        execution.stack.pop()


class RecursiveExecutor:
    """Task-neutral recursive NPI runtime backed by TensorFlow/XLA inference."""

    def __init__(
        self,
        model: NeuralProgrammerInterpreter,
        spec: TaskSpec,
        codec: ObservationCodec,
        environment: RuntimeEnvironment,
        step_limit: int,
        maximum_depth: int = 16,
        use_xla: bool = True,
    ):
        self.model = model
        self.spec = spec
        self.codec = codec
        self.environment = environment
        self.step_limit = step_limit
        self.depth_limit = maximum_depth
        self.model_steps = 0
        self.maximum_depth = 0
        self._step = model.compiled_inference_step(use_xla)

    def execute(self) -> ExecutionStats:
        self._run(self.spec.root_program, self.spec.default_arguments, depth=1)
        return ExecutionStats(self.model_steps, self.maximum_depth)

    def _run(self, program: int, arguments: Arguments, *, depth: int) -> None:
        self.maximum_depth = max(self.maximum_depth, depth)
        if depth > self.depth_limit:
            raise ExecutionFailure(f"Program call depth exceeded {self.depth_limit}")
        states = self.model.initial_state(1)
        while True:
            self.model_steps += 1
            if self.model_steps > self.step_limit:
                raise ExecutionFailure(
                    f"Execution exceeded {self.step_limit} model steps"
                )
            try:
                feature = self.codec.encode_feature(
                    self.environment.observe(), arguments
                )
            except (ValueError, IndexError) as error:
                raise ExecutionFailure(
                    f"Step {self.model_steps}, invalid observation: {error}"
                ) from error
            outputs = self._step(
                tf.convert_to_tensor(feature[None, :], tf.float32),
                tf.convert_to_tensor([program], tf.int32),
                states,
            )
            end_logits, program_logits, argument_logits, states = outputs
            should_end = bool(tf.argmax(end_logits[0]).numpy())
            if program == self.spec.action_program:
                if not should_end:
                    raise ExecutionFailure("Action program failed to return")
                try:
                    self.environment.execute(arguments)
                except (ValueError, IndexError) as error:
                    raise ExecutionFailure(
                        f"Step {self.model_steps}, invalid action {arguments}: {error}"
                    ) from error
                return
            if should_end and self.spec.return_before_call:
                return
            child = int(tf.argmax(program_logits[0]).numpy())
            child_arguments = tuple(
                int(tf.argmax(logits[0]).numpy()) for logits in argument_logits
            )
            self._run(child, child_arguments, depth=depth + 1)
            if should_end:
                return


def execute_batch(
    model: NeuralProgrammerInterpreter,
    spec: TaskSpec,
    codec: ObservationCodec,
    environments: list[RuntimeEnvironment],
    step_limits: list[int],
    *,
    use_xla: bool = True,
    maximum_depth: int = 16,
) -> list[BatchOutcome]:
    """Advance heterogeneous recursive executions through shared TF calls."""
    if len(environments) != len(step_limits):
        raise ValueError("Every environment requires one step limit")
    executions = [
        _Execution(
            environment,
            [_Frame(spec.root_program, spec.default_arguments)],
            step_limit,
        )
        for environment, step_limit in zip(environments, step_limits, strict=True)
    ]
    step = model.compiled_inference_step(use_xla)
    zero_states = model.initial_state(1)
    zero_states = tuple((hidden[0], memory[0]) for hidden, memory in zero_states)

    while True:
        active = [
            index
            for index, execution in enumerate(executions)
            if execution.stack and execution.failure is None
        ]
        if not active:
            break
        ready = []
        features = []
        for index in active:
            execution = executions[index]
            execution.model_steps += 1
            if execution.model_steps > execution.step_limit:
                execution.failure = (
                    f"Execution exceeded {execution.step_limit} model steps"
                )
                continue
            frame = execution.stack[-1]
            try:
                features.append(
                    codec.encode_feature(
                        execution.environment.observe(), frame.arguments
                    )
                )
            except (ValueError, IndexError) as error:
                execution.failure = (
                    f"Step {execution.model_steps}, invalid observation: {error}"
                )
                continue
            ready.append(index)
        if not ready:
            continue

        frames = [executions[index].stack[-1] for index in ready]
        layer_states = []
        for layer in range(model.config.layers):
            layer_states.append(
                (
                    tf.stack(
                        [
                            zero_states[layer][0]
                            if frame.states is None
                            else frame.states[layer][0]
                            for frame in frames
                        ]
                    ),
                    tf.stack(
                        [
                            zero_states[layer][1]
                            if frame.states is None
                            else frame.states[layer][1]
                            for frame in frames
                        ]
                    ),
                )
            )
        outputs = step(
            tf.convert_to_tensor(np.stack(features), tf.float32),
            tf.convert_to_tensor([frame.program for frame in frames], tf.int32),
            tuple(layer_states),
        )
        end_logits, program_logits, argument_logits, next_states = outputs
        ends = tf.argmax(end_logits, axis=-1).numpy()
        programs = tf.argmax(program_logits, axis=-1).numpy()
        arguments = [tf.argmax(logits, axis=-1).numpy() for logits in argument_logits]

        for batch_index, execution_index in enumerate(ready):
            execution = executions[execution_index]
            frame = execution.stack[-1]
            frame.states = tuple(
                (hidden[batch_index], memory[batch_index])
                for hidden, memory in next_states
            )
            should_end = bool(ends[batch_index])
            if frame.program == spec.action_program:
                if not should_end:
                    execution.failure = "Action program failed to return"
                    continue
                try:
                    execution.environment.execute(frame.arguments)
                except (ValueError, IndexError) as error:
                    execution.failure = (
                        f"Step {execution.model_steps}, invalid action "
                        f"{frame.arguments}: {error}"
                    )
                    continue
                _complete_frame(execution)
                continue
            if should_end and spec.return_before_call:
                _complete_frame(execution)
                continue
            child_arguments = tuple(int(values[batch_index]) for values in arguments)
            frame.return_after_child = should_end and not spec.return_before_call
            execution.stack.append(_Frame(int(programs[batch_index]), child_arguments))
            execution.maximum_depth = max(execution.maximum_depth, len(execution.stack))
            if execution.maximum_depth > maximum_depth:
                execution.failure = f"Program call depth exceeded {maximum_depth}"
                continue

    outcomes = []
    for execution in executions:
        if execution.failure is not None:
            outcomes.append(BatchOutcome(None, None, execution.failure))
        else:
            outcomes.append(
                BatchOutcome(
                    execution.environment.result(),
                    ExecutionStats(execution.model_steps, execution.maximum_depth),
                    None,
                )
            )
    return outcomes
