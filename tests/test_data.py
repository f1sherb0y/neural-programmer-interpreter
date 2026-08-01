import unittest

import numpy as np

from npi.tasks.addition.data import make_dataset


class DatasetTest(unittest.TestCase):
    def test_bucketed_batches_mask_padding(self):
        dataset = make_dataset(2, 3, 1)
        decisions = 0
        shapes = set()
        for batch in dataset.batches(32, shuffle=False, seed=0):
            decisions += int(batch.sequence_mask.sum())
            shapes.add(batch.features.shape[1])
            self.assertTrue(
                np.all((batch.sequence_mask == 0) | (batch.sequence_mask == 1))
            )
        self.assertEqual(decisions, dataset.decisions)
        self.assertTrue(shapes.issubset({1, 2, 4, 8, 16}))


if __name__ == "__main__":
    unittest.main()
