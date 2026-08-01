import unittest

import numpy as np
import tensorflow as tf

from npi.core.model import NeuralProgrammerInterpreter, NPIConfig
from npi.core.spec import TaskSpec
from npi.tasks.addition.spec import SPEC
from npi.tasks.graph.weight_video import exact_raster_shape


class ModelTest(unittest.TestCase):
    def test_weight_video_raster_has_one_pixel_per_parameter(self):
        width, height = exact_raster_shape(1_046_040)
        self.assertEqual((width, height), (1137, 920))
        self.assertEqual(width * height, 1_046_040)

    def test_sequence_and_recursive_step_paths_share_parameters(self):
        tf.keras.utils.set_random_seed(4)
        model = NeuralProgrammerInterpreter(SPEC, NPIConfig(hidden_size=32))
        features = tf.random.normal((2, 5, SPEC.feature_size))
        programs = tf.random.uniform((2, 5), maxval=SPEC.num_programs, dtype=tf.int32)
        sequence = model((features, programs))[0]
        states = model.initial_state(2)
        steps = []
        for index in range(5):
            outputs = model.inference_step(
                features[:, index], programs[:, index], states
            )
            steps.append(outputs[0])
            states = outputs[-1]
        np.testing.assert_allclose(
            sequence.numpy(), tf.stack(steps, axis=1).numpy(), rtol=1e-5, atol=1e-6
        )

    def test_model_dimensions_come_from_task_spec(self):
        custom = TaskSpec(
            name="custom",
            observation_size=7,
            num_programs=3,
            argument_depths=(2, 4, 9, 3, 2),
            action_program=2,
            root_program=0,
            default_arguments=(0, 0, 0, 0, 0),
            return_before_call=True,
        )
        model = NeuralProgrammerInterpreter(custom, NPIConfig(hidden_size=16))
        outputs = model((tf.zeros((2, 3, 27)), tf.zeros((2, 3), tf.int32)))
        self.assertEqual(outputs[0].shape, (2, 3, 2))
        self.assertEqual(outputs[1].shape, (2, 3, 3))
        self.assertEqual([value.shape[-1] for value in outputs[2]], [2, 4, 9, 3, 2])


if __name__ == "__main__":
    unittest.main()
