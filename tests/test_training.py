import tempfile
import unittest
from pathlib import Path

import numpy as np
import tensorflow as tf

from npi.core.checkpoint import (
    create_training_checkpoint,
    restore_training_checkpoint,
)
from npi.core.model import NeuralProgrammerInterpreter, NPIConfig
from npi.core.trainer import Trainer, to_tensors
from npi.tasks.addition.data import make_dataset
from npi.tasks.addition.spec import SPEC


class TrainingTest(unittest.TestCase):
    def test_xla_training_updates_model(self):
        dataset = make_dataset(1, 1, 3)
        batch = next(dataset.batches(64, shuffle=False, seed=0))
        model = NeuralProgrammerInterpreter(
            SPEC,
            NPIConfig(
                state_size=16, program_size=8, key_size=8, hidden_size=16, layers=1
            ),
        )
        model.build_for_task()
        before = [value.numpy().copy() for value in model.trainable_variables]
        trainer = Trainer(
            model,
            learning_rate=3e-4,
            weight_decay=0.0,
            l1_regularization=1e-8,
            use_xla=True,
        )
        metrics = trainer.train_batch(batch)
        self.assertIs(type(trainer.optimizer), tf.keras.optimizers.Adam)
        self.assertAlmostEqual(float(trainer.optimizer.learning_rate.numpy()), 3e-4)
        self.assertEqual(trainer.weight_decay, 0.0)
        self.assertEqual(trainer.l1_regularization, 1e-8)
        self.assertEqual(trainer.l2_regularization, 0.0)
        plain_trainer = Trainer(model, l1_regularization=0.0, use_xla=False)
        regularized_loss, _, _ = trainer._outputs_and_loss(to_tensors(batch), False)
        plain_loss, _, _ = plain_trainer._outputs_and_loss(to_tensors(batch), False)
        expected_l1 = 1e-8 * sum(
            float(tf.reduce_sum(tf.abs(value))) for value in model.trainable_variables
        )
        self.assertAlmostEqual(
            float(regularized_loss - plain_loss), expected_l1, places=6
        )
        l2_trainer = Trainer(
            model,
            l1_regularization=0.0,
            l2_regularization=1e-9,
            use_xla=False,
        )
        l2_loss, _, _ = l2_trainer._outputs_and_loss(to_tensors(batch), False)
        expected_l2 = 1e-9 * sum(
            float(tf.reduce_sum(tf.square(value)))
            for value in model.trainable_variables
        )
        self.assertAlmostEqual(float(l2_loss - plain_loss), expected_l2, places=6)
        self.assertGreater(metrics.loss, 0)
        self.assertEqual(metrics.decisions, int(batch.sequence_mask.sum()))
        self.assertTrue(
            any(
                (old != new.numpy()).any()
                for old, new in zip(before, model.trainable_variables, strict=True)
            )
        )
        with self.assertRaises(ValueError):
            Trainer(model, weight_decay=1e-4, l1_regularization=1e-8)
        with self.assertRaises(ValueError):
            Trainer(model, l1_regularization=1e-8, l2_regularization=1e-9)

        with tempfile.TemporaryDirectory() as temporary:
            directory = Path(temporary)
            _, manager = create_training_checkpoint(
                model, trainer.optimizer, directory / "resume"
            )
            checkpoint_path = manager.save(
                checkpoint_number=int(trainer.optimizer.iterations.numpy())
            )
            checkpoint_names = [
                name for name, _ in tf.train.list_variables(checkpoint_path)
            ]
            self.assertEqual(
                sum(
                    name.startswith("model/") and name.endswith("VARIABLE_VALUE")
                    for name in checkpoint_names
                ),
                len(model.trainable_variables),
            )
            self.assertFalse(
                any(
                    name.startswith("optimizer/_trainable_variables")
                    for name in checkpoint_names
                )
            )

            restored_model = NeuralProgrammerInterpreter(
                SPEC,
                NPIConfig(
                    state_size=16,
                    program_size=8,
                    key_size=8,
                    hidden_size=16,
                    layers=1,
                ),
            )
            restored_model.build_for_task()
            restored_trainer = Trainer(
                restored_model,
                learning_rate=3e-4,
                weight_decay=0.0,
                l1_regularization=1e-8,
                use_xla=True,
            )
            restored_checkpoint, restored_manager = create_training_checkpoint(
                restored_model,
                restored_trainer.optimizer,
                directory / "resume",
            )
            restore_training_checkpoint(
                restored_trainer.optimizer,
                restored_checkpoint,
                restored_manager,
                int(trainer.optimizer.iterations.numpy()),
            )
            for actual, restored in zip(
                model.trainable_variables,
                restored_model.trainable_variables,
                strict=True,
            ):
                self.assertTrue(np.array_equal(actual.numpy(), restored.numpy()))
            for actual, restored in zip(
                trainer.optimizer.variables,
                restored_trainer.optimizer.variables,
                strict=True,
            ):
                self.assertTrue(np.array_equal(actual.numpy(), restored.numpy()))


if __name__ == "__main__":
    unittest.main()
