import unittest

from npi.core.model import NeuralProgrammerInterpreter, NPIConfig
from npi.core.trainer import Trainer
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
        trainer = Trainer(model, use_xla=True)
        metrics = trainer.train_batch(batch)
        self.assertGreater(metrics.loss, 0)
        self.assertEqual(metrics.decisions, int(batch.sequence_mask.sum()))
        self.assertTrue(
            any(
                (old != new.numpy()).any()
                for old, new in zip(before, model.trainable_variables, strict=True)
            )
        )


if __name__ == "__main__":
    unittest.main()
