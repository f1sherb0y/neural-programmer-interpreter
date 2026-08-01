import json
from pathlib import Path

from npi.core.model import NeuralProgrammerInterpreter, NPIConfig
from npi.core.spec import TaskSpec


def save_model(
    model: NeuralProgrammerInterpreter,
    path: Path,
    metadata: dict | None = None,
) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    model.save_weights(path)
    if metadata is not None:
        path.with_suffix(".json").write_text(json.dumps(metadata, indent=2) + "\n")


def load_model(
    spec: TaskSpec,
    path: Path,
    config: NPIConfig | None = None,
) -> NeuralProgrammerInterpreter:
    model = NeuralProgrammerInterpreter(spec, config)
    model.build_for_task()
    model.load_weights(path)
    return model
