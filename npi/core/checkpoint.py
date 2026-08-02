import json
from pathlib import Path

import tensorflow as tf

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


def create_training_checkpoint(model, optimizer, directory, *, max_to_keep=1):
    """Create a strict checkpoint only after every optimizer slot exists."""
    optimizer.build(model.trainable_variables)
    checkpoint = tf.train.Checkpoint(model=model, optimizer=optimizer)
    manager = tf.train.CheckpointManager(
        checkpoint,
        str(directory),
        max_to_keep=max_to_keep,
    )
    return checkpoint, manager


def restore_training_checkpoint(
    optimizer,
    checkpoint,
    manager,
    expected_step: int,
) -> str:
    """Restore one complete TensorFlow training checkpoint, or fail loudly."""
    if manager.latest_checkpoint is None:
        raise ValueError("No TensorFlow training checkpoint is available")
    status = checkpoint.restore(manager.latest_checkpoint)
    status.assert_consumed()
    actual_step = int(optimizer.iterations.numpy())
    if actual_step != expected_step:
        raise ValueError(
            f"Optimizer checkpoint is at step {actual_step}, expected {expected_step}"
        )
    return manager.latest_checkpoint
