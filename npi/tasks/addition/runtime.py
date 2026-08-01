from dataclasses import dataclass

from npi.core.runtime import RecursiveExecutor
from npi.tasks.addition.codec import CODEC
from npi.tasks.addition.environment import AdditionEnvironment
from npi.tasks.addition.spec import SPEC


@dataclass(frozen=True)
class AdditionResult:
    value: str
    model_steps: int
    maximum_depth: int


def execute_addition(model, first: str, second: str, *, use_xla: bool = True):
    environment = AdditionEnvironment(first, second)
    executor = RecursiveExecutor(
        model,
        SPEC,
        CODEC,
        environment,
        step_limit=40 * (environment.input_width + 2),
        use_xla=use_xla,
    )
    stats = executor.execute()
    return AdditionResult(environment.result(), stats.model_steps, stats.maximum_depth)
