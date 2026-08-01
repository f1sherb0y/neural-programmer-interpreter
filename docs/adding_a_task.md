# Adding a Task

Create `npi/tasks/<task>/` with the following modules.

## 1. Specification

```python
from npi.core.spec import TaskSpec

SPEC = TaskSpec(
    name="my_task",
    observation_size=24,
    num_programs=6,
    argument_depths=(4, 8, 3, 12, 2),
    action_program=5,
    root_program=0,
    default_arguments=(0, 0, 0, 0, 0),
    return_before_call=True,
)
```

Argument count is unrestricted. Each depth defines one categorical output head.

## 2. Environment

Implement:

```python
class MyEnvironment:
    def observe(self): ...
    def execute(self, arguments: tuple[int, ...]) -> None: ...
    def result(self): ...
```

Keep variable-size state in the environment. The model should select fixed symbols such as register identities, fields, directions, and opcodes rather than unbounded addresses.

## 3. Codec

Implement `encode_observation` and combine it with invocation arguments:

```python
from npi.core.codec import append_argument_one_hot

class MyCodec:
    spec = SPEC

    def encode_observation(self, observation): ...

    def encode_feature(self, observation, arguments):
        return append_argument_one_hot(
            self.encode_observation(observation), arguments, self.spec
        )
```

## 4. Traces

Emit generic `npi.core.traces.Episode` and `Decision` objects. Each program invocation is one episode with a fixed current-program ID and invocation arguments.

Use `has_child=False` for terminal decisions that do not supervise a child program or arguments.

## 5. Dataset and Training

```python
from npi.core.data import EpisodeDataset
from npi.core.experiment import train_epochs

training = EpisodeDataset(episodes, codec)
model, history = train_epochs(
    SPEC,
    training,
    validation,
    checkpoint,
    epochs=100,
    batch_size=256,
    learning_rate=3e-4,
    weight_decay=1e-4,
    seed=1,
    use_xla=True,
)
```

No shared model or trainer changes are required.

## 6. Execution

```python
from npi.core.runtime import RecursiveExecutor

executor = RecursiveExecutor(
    model,
    SPEC,
    codec,
    environment,
    step_limit=10000,
    use_xla=True,
)
executor.execute()
```

Use `execute_batch` for many independent environments.
