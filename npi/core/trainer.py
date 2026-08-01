from dataclasses import dataclass

import tensorflow as tf

from npi.core.data import EpisodeDataset, TrainingBatch
from npi.core.model import NeuralProgrammerInterpreter


@dataclass(frozen=True)
class StepMetrics:
    loss: float
    correct: int
    decisions: int


def to_tensors(batch: TrainingBatch):
    return (
        tf.convert_to_tensor(batch.features),
        tf.convert_to_tensor(batch.programs),
        tf.convert_to_tensor(batch.target_end),
        tf.convert_to_tensor(batch.target_program),
        tuple(tf.convert_to_tensor(value) for value in batch.target_arguments),
        tf.convert_to_tensor(batch.child_mask),
        tf.convert_to_tensor(batch.sequence_mask),
    )


def masked_loss(logits, targets, mask):
    losses = tf.nn.sparse_softmax_cross_entropy_with_logits(
        labels=targets, logits=logits
    )
    return tf.reduce_sum(losses * mask) / tf.maximum(tf.reduce_sum(mask), 1.0)


def exact_decisions(
    end_logits,
    program_logits,
    argument_logits,
    target_end,
    target_program,
    target_arguments,
    child_mask,
    sequence_mask,
):
    correct = tf.equal(tf.argmax(end_logits, axis=-1, output_type=tf.int32), target_end)
    child_correct = tf.equal(
        tf.argmax(program_logits, axis=-1, output_type=tf.int32), target_program
    )
    for logits, targets in zip(argument_logits, target_arguments, strict=True):
        child_correct = tf.logical_and(
            child_correct,
            tf.equal(tf.argmax(logits, axis=-1, output_type=tf.int32), targets),
        )
    correct = tf.logical_and(
        correct,
        tf.logical_or(tf.equal(child_mask, 0.0), child_correct),
    )
    correct = tf.logical_and(correct, tf.equal(sequence_mask, 1.0))
    return tf.reduce_sum(tf.cast(correct, tf.int32)), tf.cast(
        tf.reduce_sum(sequence_mask), tf.int32
    )


class Trainer:
    def __init__(
        self,
        model: NeuralProgrammerInterpreter,
        learning_rate: float = 3e-4,
        *,
        use_xla: bool = True,
        weight_decay: float = 0.0,
        l1_regularization: float = 1e-10,
        l2_regularization: float = 0.0,
    ):
        self.model = model
        self.learning_rate = float(learning_rate)
        self.weight_decay = float(weight_decay)
        self.l1_regularization = float(l1_regularization)
        self.l2_regularization = float(l2_regularization)
        coefficients = (
            self.weight_decay,
            self.l1_regularization,
            self.l2_regularization,
        )
        if any(value < 0.0 for value in coefficients):
            raise ValueError("Regularization coefficients cannot be negative")
        if sum(bool(value) for value in coefficients) > 1:
            raise ValueError("Choose only one regularization method")
        if self.weight_decay:
            self.optimizer = tf.keras.optimizers.AdamW(
                learning_rate=self.learning_rate,
                weight_decay=self.weight_decay,
            )
        else:
            self.optimizer = tf.keras.optimizers.Adam(
                learning_rate=self.learning_rate,
            )
        self.train_step = tf.function(
            self._train_step,
            jit_compile=use_xla,
            reduce_retracing=True,
        )
        self.eval_step = tf.function(
            self._eval_step,
            jit_compile=use_xla,
            reduce_retracing=True,
        )

    def _outputs_and_loss(self, tensors, training):
        (
            features,
            programs,
            target_end,
            target_program,
            target_arguments,
            child_mask,
            sequence_mask,
        ) = tensors
        outputs = self.model((features, programs), training=training)
        end_logits, program_logits, argument_logits = outputs
        child_sequence_mask = child_mask * sequence_mask
        loss = masked_loss(end_logits, target_end, sequence_mask)
        loss += masked_loss(program_logits, target_program, child_sequence_mask)
        for logits, targets in zip(argument_logits, target_arguments, strict=True):
            loss += masked_loss(logits, targets, child_sequence_mask)
        if self.l1_regularization:
            l1_norm = tf.add_n(
                [
                    tf.reduce_sum(tf.abs(value))
                    for value in self.model.trainable_variables
                ]
            )
            loss += tf.cast(self.l1_regularization, loss.dtype) * l1_norm
        if self.l2_regularization:
            squared_l2_norm = tf.add_n(
                [
                    tf.reduce_sum(tf.square(value))
                    for value in self.model.trainable_variables
                ]
            )
            loss += tf.cast(self.l2_regularization, loss.dtype) * squared_l2_norm
        correct, decisions = exact_decisions(
            end_logits,
            program_logits,
            argument_logits,
            target_end,
            target_program,
            target_arguments,
            child_mask,
            sequence_mask,
        )
        return loss, correct, decisions

    def _train_step(self, tensors):
        with tf.GradientTape() as tape:
            loss, correct, decisions = self._outputs_and_loss(tensors, True)
        gradients = tape.gradient(loss, self.model.trainable_variables)
        gradients, _ = tf.clip_by_global_norm(gradients, 1.0)
        self.optimizer.apply_gradients(
            zip(gradients, self.model.trainable_variables, strict=True)
        )
        return loss, correct, decisions

    def _eval_step(self, tensors):
        return self._outputs_and_loss(tensors, False)

    def train_batch(self, batch: TrainingBatch) -> StepMetrics:
        loss, correct, decisions = self.train_step(to_tensors(batch))
        return StepMetrics(float(loss), int(correct), int(decisions))

    def accuracy(self, dataset: EpisodeDataset, batch_size: int) -> float:
        correct = 0
        decisions = 0
        for batch in dataset.batches(batch_size, shuffle=False, seed=0):
            _, batch_correct, batch_decisions = self.eval_step(to_tensors(batch))
            correct += int(batch_correct)
            decisions += int(batch_decisions)
        return correct / decisions
