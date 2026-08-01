from dataclasses import dataclass

import tensorflow as tf

from npi.core.spec import TaskSpec


@dataclass(frozen=True)
class NPIConfig:
    state_size: int = 128
    program_size: int = 64
    key_size: int = 32
    hidden_size: int = 256
    layers: int = 2


class NeuralProgrammerInterpreter(tf.keras.Model):
    """Task-parameterized NPI with shared sequence and recursive-step cells."""

    def __init__(
        self,
        spec: TaskSpec,
        config: NPIConfig | None = None,
        name: str = "neural_programmer_interpreter",
    ):
        super().__init__(name=name)
        self.spec = spec
        self.config = config or NPIConfig()
        config = self.config
        self.state_hidden = tf.keras.layers.Dense(
            config.state_size, activation="relu", name="state_hidden"
        )
        self.state_output = tf.keras.layers.Dense(
            config.state_size, name="state_output"
        )
        self.program_embeddings = tf.keras.layers.Embedding(
            spec.num_programs, config.program_size, name="program_embeddings"
        )
        self.cells = [
            tf.keras.layers.LSTMCell(config.hidden_size, name=f"lstm_cell_{index}")
            for index in range(config.layers)
        ]
        self.recurrent_core = tf.keras.layers.RNN(
            tf.keras.layers.StackedRNNCells(self.cells),
            return_sequences=True,
            name="recurrent_core",
        )
        self.end_head = tf.keras.layers.Dense(2, name="end_head")
        self.key_head = tf.keras.layers.Dense(config.key_size, name="key_head")
        self.argument_heads = tuple(
            tf.keras.layers.Dense(depth, name=f"argument_head_{index}")
            for index, depth in enumerate(spec.argument_depths)
        )
        self._compiled_steps = {}
        self.program_keys = self.add_weight(
            name="program_keys",
            shape=(spec.num_programs, config.key_size),
            initializer="glorot_uniform",
            trainable=True,
        )

    def _inputs(self, features: tf.Tensor, program_ids: tf.Tensor) -> tf.Tensor:
        state = self.state_output(self.state_hidden(features))
        program = self.program_embeddings(program_ids)
        return tf.concat((state, program), axis=-1)

    def _decode(self, hidden: tf.Tensor):
        end_logits = self.end_head(hidden)
        predicted_key = self.key_head(hidden)
        program_logits = tf.linalg.matmul(
            predicted_key, self.program_keys, transpose_b=True
        )
        argument_logits = tuple(head(hidden) for head in self.argument_heads)
        return end_logits, program_logits, argument_logits

    def call(self, inputs, training: bool = False):
        features, program_ids = inputs
        hidden = self.recurrent_core(
            self._inputs(features, program_ids), training=training
        )
        return self._decode(hidden)

    def initial_state(self, batch_size: tf.Tensor | int):
        return tuple(
            (
                tf.zeros((batch_size, self.config.hidden_size), tf.float32),
                tf.zeros((batch_size, self.config.hidden_size), tf.float32),
            )
            for _ in self.cells
        )

    def inference_step(self, features, program_ids, states):
        current = self._inputs(features, program_ids)
        next_states = []
        for cell, (hidden, memory) in zip(self.cells, states, strict=True):
            current, state = cell(current, states=(hidden, memory), training=False)
            next_states.append((state[0], state[1]))
        return (*self._decode(current), tuple(next_states))

    def compiled_inference_step(self, use_xla: bool = True):
        if use_xla not in self._compiled_steps:
            self._compiled_steps[use_xla] = tf.function(
                self.inference_step,
                jit_compile=use_xla,
                reduce_retracing=True,
            )
        return self._compiled_steps[use_xla]

    def build_for_task(self, batch_size: int = 1) -> None:
        self(
            (
                tf.zeros((batch_size, 1, self.spec.feature_size), tf.float32),
                tf.zeros((batch_size, 1), tf.int32),
            )
        )
