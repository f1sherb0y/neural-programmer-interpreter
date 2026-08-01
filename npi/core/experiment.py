import random
import time
from dataclasses import asdict, dataclass
from pathlib import Path

import numpy as np
import tensorflow as tf

from npi.core.checkpoint import load_model, save_model
from npi.core.data import EpisodeDataset
from npi.core.model import NeuralProgrammerInterpreter, NPIConfig
from npi.core.spec import TaskSpec
from npi.core.trainer import Trainer


@dataclass(frozen=True)
class EpochMetrics:
    epoch: int
    optimizer_step: int
    loss: float
    online_accuracy: float
    training_accuracy: float
    validation_accuracy: float
    learning_rate: float


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    tf.keras.utils.set_random_seed(seed)


def train_epochs(
    spec: TaskSpec,
    training_data: EpisodeDataset,
    validation_data: EpisodeDataset,
    checkpoint: Path,
    *,
    epochs: int,
    batch_size: int,
    learning_rate: float,
    weight_decay: float,
    l1_regularization: float,
    l2_regularization: float,
    seed: int,
    use_xla: bool = True,
    config: NPIConfig | None = None,
) -> tuple[NeuralProgrammerInterpreter, list[EpochMetrics]]:
    set_seed(seed)
    model = NeuralProgrammerInterpreter(spec, config)
    model.build_for_task()
    trainer = Trainer(
        model,
        learning_rate,
        use_xla=use_xla,
        weight_decay=weight_decay,
        l1_regularization=l1_regularization,
        l2_regularization=l2_regularization,
    )
    history = []
    best_validation = -1.0
    started = time.time()
    for epoch in range(1, epochs + 1):
        loss_sum = 0.0
        correct = 0
        decisions = 0
        batches = 0
        for batch in training_data.batches(batch_size, shuffle=True, seed=seed + epoch):
            metrics = trainer.train_batch(batch)
            loss_sum += metrics.loss
            correct += metrics.correct
            decisions += metrics.decisions
            batches += 1
        training_accuracy = trainer.accuracy(training_data, batch_size)
        validation_accuracy = trainer.accuracy(validation_data, batch_size)
        entry = EpochMetrics(
            epoch,
            int(trainer.optimizer.iterations.numpy()),
            loss_sum / batches,
            correct / decisions,
            training_accuracy,
            validation_accuracy,
            trainer.learning_rate,
        )
        history.append(entry)
        print(
            f"epoch {epoch:03d} step {entry.optimizer_step:6d}: "
            f"loss={entry.loss:.4f} train={training_accuracy:.6f} "
            f"validation={validation_accuracy:.6f} lr={entry.learning_rate:.2e}",
            flush=True,
        )
        if validation_accuracy >= best_validation:
            best_validation = validation_accuracy
            save_model(
                model,
                checkpoint,
                {
                    "framework": "tensorflow",
                    "tensorflow": tf.__version__,
                    "task": spec.name,
                    "seed": seed,
                    "training_invocations": training_data.size,
                    "training_decisions": training_data.decisions,
                    "optimizer": type(trainer.optimizer).__name__,
                    "learning_rate": trainer.learning_rate,
                    "weight_decay": trainer.weight_decay,
                    "l1_regularization": trainer.l1_regularization,
                    "l2_regularization": trainer.l2_regularization,
                    "history": [asdict(item) for item in history],
                },
            )
        if training_accuracy == 1.0 and validation_accuracy == 1.0:
            break
    print(f"training_seconds: {time.time() - started:.1f}")
    return load_model(spec, checkpoint, config), history
